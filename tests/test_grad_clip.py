"""``train.clip_grad_norm``: global-norm clipping across every parameter.

Clipping is off by default (``grad_clip: 0``), so these pin down two things:
disabled means gradients are untouched, and enabled means the same result as
``torch.nn.utils.clip_grad_norm_`` -- one norm over all parameters, one
coefficient applied to all of them, Muon-managed matrices included.

The sharded case runs the same check on FSDP2 DTensor gradients, where the norm
has to be computed across shards rather than per rank. It needs 2+ GPUs.
"""

import copy
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from train import clip_grad_norm


CUDA = torch.cuda.device_count() if torch.cuda.is_available() else 0


def _model_with_grads(scale):
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Embedding(16, 8),  # a non-matrix-optimizer group
        torch.nn.Linear(8, 8, bias=False),  # a Muon-style matrix
        torch.nn.Linear(8, 16, bias=False),
    )
    for p in model.parameters():
        p.grad = scale * torch.randn_like(p)
    return model


def _grads(model):
    return [p.grad.clone() for p in model.parameters()]


def test_disabled_leaves_grads_untouched_and_returns_norm():
    model = _model_with_grads(scale=10.0)
    before = _grads(model)
    norm = clip_grad_norm(model.parameters(), max_norm=0.0)

    for a, b in zip(before, _grads(model)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    expected = torch.linalg.vector_norm(torch.stack([g.norm() for g in before]))
    torch.testing.assert_close(norm, expected)


@pytest.mark.parametrize("scale", [10.0, 1e-3])
def test_matches_clip_grad_norm_(scale):
    """Clipped when the norm exceeds max_norm, untouched when it does not."""
    ours, ref = _model_with_grads(scale), _model_with_grads(scale)
    ours_norm = clip_grad_norm(ours.parameters(), max_norm=1.0)
    ref_norm = torch.nn.utils.clip_grad_norm_(ref.parameters(), max_norm=1.0)

    torch.testing.assert_close(ours_norm, ref_norm)
    for a, b in zip(_grads(ours), _grads(ref)):
        torch.testing.assert_close(a, b)


def test_clipped_global_norm_is_max_norm_and_logged_norm_is_pre_clip():
    model = _model_with_grads(scale=10.0)
    pre = clip_grad_norm(model.parameters(), max_norm=1.0)
    post = torch.nn.utils.get_total_norm(_grads(model))

    assert pre.item() > 1.0, "test premise: the norm must start above max_norm"
    torch.testing.assert_close(post, torch.tensor(1.0), atol=1e-5, rtol=0)


def test_one_coefficient_for_all_parameters():
    """Every gradient shrinks by the same factor -- not a per-tensor clip."""
    model = _model_with_grads(scale=10.0)
    before = _grads(model)
    clip_grad_norm(model.parameters(), max_norm=1.0)
    ratios = [(a / b).mean().item() for a, b in zip(_grads(model), before)]
    assert max(ratios) - min(ratios) < 1e-6


def test_params_without_grads_are_skipped():
    model = _model_with_grads(scale=10.0)
    frozen = next(model.parameters())
    frozen.grad = None
    clip_grad_norm(model.parameters(), max_norm=1.0)  # must not raise
    assert frozen.grad is None


# --------------------------------------------------------------------------
# FSDP2: DTensor gradients. The norm must be global across shards, and the
# clipped gradients must match an unsharded model clipped the same way.
# --------------------------------------------------------------------------


def _fsdp_worker(rank, world_size, port, out_path):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    from torch.distributed.fsdp import fully_shard

    torch.manual_seed(0)
    ref = torch.nn.Sequential(
        torch.nn.Linear(32, 64, bias=False), torch.nn.Linear(64, 32, bias=False)
    ).cuda()
    sharded = copy.deepcopy(ref)
    for layer in sharded:
        fully_shard(layer)
    fully_shard(sharded)

    torch.manual_seed(1)  # same batch on every rank, so the grads match ref
    x = 10.0 * torch.randn(8, 32, device="cuda")
    ref(x).square().sum().backward()
    sharded(x).square().sum().backward()
    # FSDP averages gradients across ranks; with identical inputs that is a no-op.

    ref_norm = clip_grad_norm(ref.parameters(), max_norm=1.0)
    sharded_norm = clip_grad_norm(sharded.parameters(), max_norm=1.0)
    full = [p.grad.full_tensor().cpu() for p in sharded.parameters()]

    if rank == 0:
        torch.save(
            {
                "ref_norm": ref_norm.cpu(),
                "sharded_norm": sharded_norm.full_tensor().cpu(),
                "ref_grads": [p.grad.cpu() for p in ref.parameters()],
                "sharded_grads": full,
            },
            out_path,
        )
    dist.destroy_process_group()


@pytest.mark.skipif(CUDA < 2, reason="needs at least 2 GPUs")
def test_fsdp_dtensor_grads_clip_like_unsharded(tmp_path):
    out = str(tmp_path / "fsdp_clip.pt")
    mp.spawn(_fsdp_worker, args=(min(CUDA, 2), 29531, out), nprocs=min(CUDA, 2), join=True)
    got = torch.load(out)

    assert got["ref_norm"].item() > 1.0, "test premise: the norm must start above max_norm"
    torch.testing.assert_close(got["sharded_norm"], got["ref_norm"], rtol=1e-4, atol=0)
    for a, b in zip(got["sharded_grads"], got["ref_grads"]):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)
