"""``train.set_seed`` must pin every RNG the run draws from, on every rank.

Without it ``train.py`` never seeds torch at all, so each run gets a different
initialization and init variance is folded into every optimizer comparison --
the effect sizes being measured between Muon variants are smaller than that.

Two properties matter, and they are tested separately:

* **Reproducible.** The same seed gives the same model, a different seed gives
  a different one. Single process, so this runs anywhere.
* **Identical across ranks.** Every rank must draw the same init and the same
  optimizer projections. DDP would paper over a mismatched init by
  broadcasting rank 0's parameters, but FSDP initializes each shard locally
  with no such broadcast, and Dion/DionSimple draw a random ``Q`` per
  parameter (``dion.py``, ``dion_simple.py``) that ranks holding pieces of the
  same matrix must agree on. That needs 2+ GPUs, so it is skipped otherwise.

Note what this does *not* claim: seeding does not make a run bit-for-bit
reproducible. cuBLAS split-k, NCCL reduction order and atomics stay
nondeterministic, so two runs with the same seed will drift apart. The point is
to remove initialization as a difference between arms, not to expect identical
loss curves.
"""

import os
import random

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from models.gpt_model import GPT, GPTConfig
from train import set_seed


CUDA = torch.cuda.device_count() if torch.cuda.is_available() else 0

# Small enough to build a few times on CPU; the properties under test do not
# depend on width or depth.
TEST_CONFIG = dict(sequence_len=64, vocab_size=512, n_layer=2, n_head=2, n_embd=32)


def _build(seed, **overrides):
    set_seed(seed)
    cfg = GPTConfig(**{**TEST_CONFIG, **overrides})
    model = GPT(cfg)
    model.init_weights()
    return model


def _flat(model):
    return torch.cat([p.detach().reshape(-1) for _, p in sorted(model.named_parameters())])


def test_set_seed_covers_every_rng_we_use():
    """random, numpy and torch all replay identically, and differ on a new seed."""
    set_seed(1234)
    first = (random.random(), float(np.random.rand()), torch.randn(4).tolist())
    set_seed(1234)
    assert (random.random(), float(np.random.rand()), torch.randn(4).tolist()) == first

    set_seed(4321)
    assert (random.random(), float(np.random.rand()), torch.randn(4).tolist()) != first


@pytest.mark.parametrize("tie_embeddings", [False, True])
def test_model_init_is_reproducible(tie_embeddings):
    same_a = _flat(_build(0, tie_embeddings=tie_embeddings))
    same_b = _flat(_build(0, tie_embeddings=tie_embeddings))
    different = _flat(_build(1, tie_embeddings=tie_embeddings))

    torch.testing.assert_close(same_a, same_b, rtol=0, atol=0)
    assert not torch.equal(same_a, different), "a new seed must give a new init"


def test_unseeded_runs_differ():
    """Guards the premise: without set_seed, two builds really are different.

    If torch ever started seeding itself deterministically this test would fail
    and the seeding work would be redundant -- worth knowing either way.
    """
    assert not torch.equal(_flat(_build_unseeded()), _flat(_build_unseeded()))


def _build_unseeded():
    torch.seed()  # re-seed from entropy, undoing any earlier set_seed
    model = GPT(GPTConfig(**TEST_CONFIG))
    model.init_weights()
    return model


# --------------------------------------------------------------------------
# Multi-rank: every rank must land on the same init and the same random draws.
# --------------------------------------------------------------------------


def _worker(rank, world_size, port, seed, out_path):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    set_seed(seed)
    model = GPT(GPTConfig(**TEST_CONFIG)).to(f"cuda:{rank}")
    model.init_weights()
    params = _flat(model).to(f"cuda:{rank}")

    # A device draw stands in for Dion's per-parameter Q: it must match too,
    # and it exercises the CUDA generator rather than the CPU one.
    device_draw = torch.randn(64, device=f"cuda:{rank}")

    gathered_params = [torch.empty_like(params) for _ in range(world_size)]
    gathered_draws = [torch.empty_like(device_draw) for _ in range(world_size)]
    dist.all_gather(gathered_params, params)
    dist.all_gather(gathered_draws, device_draw)

    if rank == 0:
        torch.save(
            {
                "params": [t.cpu() for t in gathered_params],
                "draws": [t.cpu() for t in gathered_draws],
            },
            out_path,
        )
    dist.destroy_process_group()


@pytest.mark.skipif(CUDA < 2, reason="needs at least 2 GPUs")
def test_all_ranks_initialize_identically(tmp_path):
    world_size = min(CUDA, 2)
    out = str(tmp_path / "ranks.pt")
    mp.spawn(_worker, args=(world_size, 29511, 0, out), nprocs=world_size, join=True)
    got = torch.load(out)

    for rank in range(1, world_size):
        torch.testing.assert_close(
            got["params"][0], got["params"][rank], rtol=0, atol=0
        )
        torch.testing.assert_close(got["draws"][0], got["draws"][rank], rtol=0, atol=0)


# --------------------------------------------------------------------------
# Sharded init must NOT tile. Identical seeds on every rank are correct here,
# but only because DTensor does not naively run the local RNG per shard: its
# OffsetBasedRNGTracker advances the Philox offset by each shard's coordinates,
# so each rank draws its own distinct slice of the seeded stream. That
# machinery broadcasts rank 0's RNG state and warns unless every rank called
# torch.manual_seed with the same value -- so differing per-rank seeds would be
# the bug, not the fix.
#
# Note what is NOT asserted: that gathering the shards reproduces a
# single-process draw. Measured on 2xH200 (job 16906547) at 64x8 with
# world_size 2, rank 0's shard matched a single-device draw exactly and rank
# 1's did not -- the per-rank offsets do not line up element-for-element with
# an unsharded draw at this granularity. Nothing here depends on that: it would
# only matter when comparing runs at *different* world sizes, and at a fixed
# world size the seed still fully determines the init.
# --------------------------------------------------------------------------

SHARD_ROWS, SHARD_COLS = 1024, 8


def _shard_worker(rank, world_size, port, seed, out_path):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    from torch.distributed.tensor import DeviceMesh, Shard, zeros

    set_seed(seed)
    mesh = DeviceMesh("cuda", list(range(world_size)))
    dt = zeros(SHARD_ROWS, SHARD_COLS, device_mesh=mesh, placements=[Shard(0)])
    torch.nn.init.normal_(dt, mean=0.0, std=1.0)

    torch.save({"local": dt.to_local().cpu()}, f"{out_path}.{rank}")
    dist.destroy_process_group()


def _run_sharded_init(tmp_path, world_size, port, seed, name):
    out = str(tmp_path / name)
    mp.spawn(
        _shard_worker, args=(world_size, port, seed, out), nprocs=world_size, join=True
    )
    return [torch.load(f"{out}.{r}")["local"] for r in range(world_size)]


@pytest.mark.skipif(CUDA < 2, reason="needs at least 2 GPUs")
def test_sharded_init_is_not_tiled_and_is_reproducible(tmp_path):
    world_size = min(CUDA, 2)
    first = _run_sharded_init(tmp_path, world_size, 29512, 0, "a")
    second = _run_sharded_init(tmp_path, world_size, 29513, 0, "b")

    # 1. No shard is a copy of another. This is the failure mode a shared seed
    #    would cause if each rank drew from offset 0 of its own stream.
    for r in range(1, world_size):
        assert not torch.equal(first[0], first[r]), (
            "shards are identical -- each rank drew the same values instead of "
            "its own slice of the seeded stream"
        )

    # 2. The same seed reproduces the same shards, rank by rank.
    for r in range(world_size):
        torch.testing.assert_close(first[r], second[r], rtol=0, atol=0)

    # 3. The draw is still standard normal overall. 8192 samples put the
    #    standard error near 0.011 for both statistics, so 0.06 is ~5 sigma.
    values = torch.cat([sh.reshape(-1) for sh in first])
    assert values.numel() == SHARD_ROWS * SHARD_COLS
    assert abs(values.mean().item()) < 0.06
    assert abs(values.std().item() - 1.0) < 0.06
