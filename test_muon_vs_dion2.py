#!/usr/bin/env python3
"""
Test that Muon and Dion2(fraction=1.0, ef_decay=mu) produce identical optimizer steps.

A standalone single-GPU test that:
  1. Builds a GPT model
  2. Deep-copies it 3 times (muon_model, dion2_model, control_model)
  3. Applies 1 optimizer step with shared random gradients
  4. Compares resulting weights

Usage:
  CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 test_muon_vs_dion2.py
"""

import copy
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from dion.muon import Muon
from dion.dion2 import Dion2
from models.gpt_model import GPT, GPTConfig


def build_model():
    """Build a small GPT model on CUDA."""
    config = GPTConfig(
        sequence_len=2048,
        vocab_size=100352,
        n_layer=20,
        n_head=16,
        n_embd=1536,
        use_bias=False,
    )
    model = GPT(config)
    model.to("cuda")
    model.init_weights()
    return model


def make_param_groups(model, n_head: int):
    """Separate params into orthogonal vs scalar groups (mirrors train.py logic)."""
    qkv_params = []
    other_matrix_params = []
    scalar_params = []
    qkv_names = {"c_q.weight", "c_k.weight", "c_v.weight"}

    for name, p in model.named_parameters():
        if any(q in name for q in qkv_names):
            qkv_params.append(p)
        elif p.ndim >= 2 and "wte" not in name and "lm_head" not in name:
            other_matrix_params.append(p)
        else:
            scalar_params.append(p)

    param_groups = [
        dict(params=other_matrix_params),
        dict(params=qkv_params, fraction=1.0, adjust_lr=None, num_heads=n_head),
        dict(params=scalar_params, algorithm="adamw", lr=0.001, betas=(0.95, 0.98), weight_decay=0),
    ]
    return param_groups


def make_muon(model, process_group, n_head: int, **kwargs):
    """Create a Muon optimizer."""
    groups = make_param_groups(model, n_head)
    return Muon(
        groups,
        distributed_mesh=process_group,
        lr=0.02,
        mu=0.95,
        weight_decay=0.01,
        nesterov=False,  #
        adjust_lr="spectral_norm",
        use_triton=True,
        use_polar_express=True,
        use_gram_newton_schulz=True,
        **kwargs,
    )


def make_dion2(model, process_group, n_head: int, **kwargs):
    """Create a Dion2 optimizer configured to be equivalent to Muon."""
    groups = make_param_groups(model, n_head)
    return Dion2(
        groups,
        distributed_mesh=process_group,
        lr=0.02,
        fraction=1.0,
        ef_decay=0.95,
        weight_decay=0.01,
        adjust_lr="spectral_norm",
        use_triton=True,
        use_polar_express=True,
        use_gram_newton_schulz=True,
        **kwargs,
    )


def assign_shared_gradients(models, seed: int = 42):
    """Assign identical random gradients to all models."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    # Generate grads for first model, then copy to others
    for p in models[0].parameters():
        if p.requires_grad:
            p.grad = torch.randn(p.shape, device="cuda", dtype=p.dtype, generator=gen)

    # Copy grads to other models
    for model in models[1:]:
        for p_src, p_dst in zip(models[0].parameters(), model.parameters()):
            if p_src.requires_grad:
                p_dst.grad = p_src.grad.clone()


def compare_weights(model_a, model_b, label: str):
    """Compare weights between two models, report max diff per param."""
    max_diff = 0.0
    diffs = {}
    for (name_a, p_a), (name_b, p_b) in zip(
        model_a.named_parameters(), model_b.named_parameters()
    ):
        assert name_a == name_b
        diff = (p_a - p_b).abs().max().item()
        diffs[name_a] = diff
        max_diff = max(max_diff, diff)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Max weight diff: {max_diff:.6e}")

    # Show top-5 largest diffs
    sorted_diffs = sorted(diffs.items(), key=lambda x: -x[1])[:5]
    for name, d in sorted_diffs:
        print(f"    {name}: {d:.6e}")

    return max_diff


def main():
    # Init distributed (required by Muon/Dion2)
    assert torch.cuda.is_available()
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(f"cuda:{local_rank}")

    print("Building model...")
    base_model = build_model()
    n_head = 16

    # Deep copy for each optimizer
    muon_model = copy.deepcopy(base_model)
    dion2_model = copy.deepcopy(base_model)
    control_model = copy.deepcopy(base_model)

    # Get a process group (required by Muon/Dion2 for distributed ortho)
    dummy_ddp = DDP(muon_model, device_ids=[local_rank])
    pg = dummy_ddp.process_group

    # Create optimizers
    muon_opt = make_muon(muon_model, pg, n_head)
    dion2_opt = make_dion2(dion2_model, pg, n_head)
    control_opt = make_muon(control_model, pg, n_head)

    num_steps = 3
    for step in range(num_steps):
        print(f"\n--- Step {step + 1} ---")

        # Assign shared random gradients (different seed each step)
        assign_shared_gradients(
            [muon_model, dion2_model, control_model], seed=42 + step
        )

        # Take optimizer steps
        muon_opt.step()
        dion2_opt.step()
        control_opt.step()

        muon_opt.zero_grad()
        dion2_opt.zero_grad()
        control_opt.zero_grad()

        # Compare
        compare_weights(muon_model, dion2_model, f"Step {step+1}: Muon vs Dion2")
        compare_weights(muon_model, control_model, f"Step {step+1}: Muon vs Control (Muon)")

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    final_muon_dion2 = compare_weights(muon_model, dion2_model, "Final: Muon vs Dion2")
    final_muon_control = compare_weights(muon_model, control_model, "Final: Muon vs Control")
    print()

    if final_muon_control == 0.0:
        print("✓ Control (Muon vs Muon) is bitwise identical — sanity check passed")
    else:
        print(f"✗ Control diverged! Max diff: {final_muon_control:.6e}")

    if final_muon_dion2 == 0.0:
        print("✓ Muon vs Dion2 is bitwise identical!")
    elif final_muon_dion2 < 1e-2:
        print(f"~ Muon vs Dion2 differs by {final_muon_dion2:.6e} (likely bf16 NS permutation sensitivity)")
    else:
        print(f"✗ Muon vs Dion2 diverged significantly: {final_muon_dion2:.6e}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
