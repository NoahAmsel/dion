"""``train.weight_norms``: read-only per-parameter weight norms for wandb.

Logged at every validation to track where weights settle between weight decay and
the optimizer's updates -- which fraction and learning rate change. It must (1)
report the right numbers, (2) leave the model untouched, and (3) under FSDP report
the norm of the full parameter, not of one rank's shard.
"""

import copy
import math
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from models.gpt_model import GPT, GPTConfig
from train import weight_norms

CUDA = torch.cuda.device_count() if torch.cuda.is_available() else 0
SMALL = dict(sequence_len=16, vocab_size=64, n_layer=3, n_head=4, n_embd=32)


def _model(**kw):
    torch.manual_seed(0)
    m = GPT(GPTConfig(**{**SMALL, **kw}))
    m.init_weights()
    return m


@pytest.mark.parametrize("kw", [{}, dict(mlp_type="swiglu", attn_gate=True, norm_gain=True)])
def test_one_norm_per_parameter_with_correct_values(kw):
    model = _model(**kw)
    out = weight_norms(model)
    params = dict(model.named_parameters())
    per_param = {k[len("weight_norm/"):]: v for k, v in out.items() if k.startswith("weight_norm/")}
    assert set(per_param) == set(params)
    for name, p in params.items():
        assert math.isclose(per_param[name], p.detach().norm().item(), rel_tol=1e-6)


def test_rms_is_averaged_over_layers_by_kind():
    model = _model()
    out = weight_norms(model)
    L = SMALL["n_layer"]
    key = "weight_rms/transformer.h.*.attn.c_q.weight"
    assert key in out
    expected = sum(
        model.transformer.h[i].attn.c_q.weight.norm().item()
        / math.sqrt(model.transformer.h[i].attn.c_q.weight.numel())
        for i in range(L)
    ) / L
    assert math.isclose(out[key], expected, rel_tol=1e-6)
    # Non-block parameters keep their own name.
    assert "weight_rms/transformer.wte.weight" in out
    assert not any(".h.0." in k for k in out if k.startswith("weight_rms/"))


def test_does_not_modify_the_model():
    model = _model(norm_gain=True)
    before = copy.deepcopy(model.state_dict())
    weight_norms(model)
    for k, v in model.state_dict().items():
        assert torch.equal(v, before[k]), k


def _fsdp_worker(rank, world_size, port, out_path):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    from torch.distributed.fsdp import fully_shard

    ref = _model(norm_gain=True).cuda()
    sharded = copy.deepcopy(ref)
    for block in sharded.transformer.h:
        fully_shard(block)
    fully_shard(sharded)

    got = weight_norms(sharded)  # collective: every rank calls it
    if rank == 0:
        torch.save({"ref": weight_norms(ref), "sharded": got}, out_path)
    dist.destroy_process_group()


@pytest.mark.skipif(CUDA < 2, reason="needs at least 2 GPUs")
def test_fsdp_norms_are_full_parameter_norms(tmp_path):
    out = str(tmp_path / "norms.pt")
    mp.spawn(_fsdp_worker, args=(min(CUDA, 2), 29541, out), nprocs=min(CUDA, 2), join=True)
    got = torch.load(out)
    assert set(got["sharded"]) == set(got["ref"])
    for k, v in got["ref"].items():
        assert math.isclose(got["sharded"][k], v, rel_tol=1e-5), k
