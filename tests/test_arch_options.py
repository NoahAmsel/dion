"""Opt-in architecture changes in models/gpt_model.py: SwiGLU MLP, headwise attention
gate, and learnable RMSNorm gains.

All three default off, and off must mean *unchanged*: existing experiments, pending
jobs and saved checkpoints depend on the default model being the one they trained.
On, every new 2D weight must reach the orthogonalizing optimizer and every gain the
scalar optimizer -- a 1D gain in the Muon group would be a silent bug.
"""

import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from models.gpt_model import GPT, GPTConfig, CausalSelfAttention, SwiGLUMLP

SMALL = dict(sequence_len=16, vocab_size=64, n_layer=2, n_head=4, n_embd=32)
ALL_ON = dict(mlp_type="swiglu", attn_gate=True, norm_gain=True)


def _model(seed=0, **kw):
    torch.manual_seed(seed)
    model = GPT(GPTConfig(**{**SMALL, **kw}))
    model.init_weights()
    return model


# ---------------------------------------------------------------- defaults


def test_default_model_has_no_new_parameters():
    names = [n for n, _ in _model().named_parameters()]
    assert not any(s in n for n in names for s in ("norm", "c_gate", "c_up", "c_down"))
    assert all(m.weight is None for m in _model().modules() if isinstance(m, nn.RMSNorm))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_parameter_free_rmsnorm_equals_functional(dtype):
    """The default norms replaced F.rms_norm(x, (dim,)); they must compute the same."""
    x = torch.randn(3, 5, 32, dtype=dtype)
    torch.testing.assert_close(
        nn.RMSNorm(32, eps=None, elementwise_affine=False)(x),
        F.rms_norm(x, (32,)),
        rtol=0,
        atol=0,
    )


# ---------------------------------------------------------------- norm gain


def test_norm_gain_adds_one_gain_per_norm_initialized_to_one():
    model = _model(norm_gain=True)
    gains = {n: p for n, p in model.named_parameters() if p.ndim == 1}
    L, d, hd = SMALL["n_layer"], SMALL["n_embd"], SMALL["n_embd"] // SMALL["n_head"]
    assert len(gains) == 4 * L + 1  # attn_norm, mlp_norm, q_norm, k_norm per block + ln_f
    for n, p in gains.items():
        assert p.shape == ((hd,) if ("q_norm" in n or "k_norm" in n) else (d,)), n
        assert torch.equal(p, torch.ones_like(p)), n


def test_norm_gain_at_init_computes_the_same_function():
    """Unit gains change nothing, including moving QK-norm ahead of RoPE (a rotation
    preserves the norm, so the two orders agree up to float error)."""
    plain, gained = _model(seed=3), _model(seed=3, norm_gain=True)
    missing, unexpected = gained.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and all(k.endswith("norm.weight") or k.endswith("ln_f.weight") for k in missing)
    idx = torch.randint(0, SMALL["vocab_size"], (2, SMALL["sequence_len"]))
    torch.testing.assert_close(gained(idx, idx), plain(idx, idx), rtol=1e-5, atol=1e-5)


def test_norm_gain_is_live():
    model, idx = _model(norm_gain=True), torch.randint(0, SMALL["vocab_size"], (2, 16))
    before = model(idx, idx)
    with torch.no_grad():
        model.transformer.h[0].attn.q_norm.weight.mul_(3.0)
    assert not torch.allclose(model(idx, idx), before)


# ---------------------------------------------------------------- attention gate


def _capture_c_proj_input(attn, x):
    seen = {}
    handle = attn.c_proj.register_forward_hook(lambda m, inp, out: seen.update(y=inp[0]))
    attn(x)
    handle.remove()
    return seen["y"]


def _attention(gate):
    torch.manual_seed(0)
    cfg = GPTConfig(**{**SMALL, "attn_gate": gate})
    attn = CausalSelfAttention(cfg)
    for mod in attn.modules():
        if isinstance(mod, nn.Linear):
            nn.init.normal_(mod.weight, std=0.2)
    attn.rotary.init_inv_freq()
    return attn


def test_attn_gate_shape():
    attn = _attention(gate=True)
    assert attn.c_gate.weight.shape == (SMALL["n_head"], SMALL["n_embd"])


def test_attn_gate_scales_each_head_independently():
    """Gate logits of 0 halve a head; a large logit leaves it intact. Checked on the
    concatenated head outputs entering c_proj."""
    gated, plain = _attention(gate=True), _attention(gate=False)
    gated.load_state_dict(plain.state_dict(), strict=False)
    x = torch.randn(2, 8, SMALL["n_embd"])
    hd, H = SMALL["n_embd"] // SMALL["n_head"], SMALL["n_head"]

    with torch.no_grad():
        gated.c_gate.weight.zero_()
        # Head 1's gate logit = 1000 * x[..., 0]; pick x so that is huge positive.
        gated.c_gate.weight[1, 0] = 1000.0
    x[..., 0] = x[..., 0].abs() + 1.0

    y_plain = _capture_c_proj_input(plain, x).view(2, 8, H, hd)
    y_gated = _capture_c_proj_input(gated, x).view(2, 8, H, hd)
    for h in range(H):
        expected = y_plain[:, :, h] if h == 1 else 0.5 * y_plain[:, :, h]
        torch.testing.assert_close(y_gated[:, :, h], expected, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------- SwiGLU


def test_swiglu_matches_formula_and_shapes():
    cfg = GPTConfig(**SMALL)
    mlp = SwiGLUMLP(cfg)
    d = cfg.n_embd
    assert mlp.c_up.weight.shape == mlp.c_gate.weight.shape == (4 * d, d)
    assert mlp.c_down.weight.shape == (d, 4 * d)
    assert getattr(mlp.c_down, "_residual_proj", False)
    x = torch.randn(3, 5, d)
    expected = F.linear(
        F.silu(F.linear(x, mlp.c_gate.weight)) * F.linear(x, mlp.c_up.weight),
        mlp.c_down.weight,
    )
    torch.testing.assert_close(mlp(x), expected)


def test_unknown_mlp_type_raises():
    with pytest.raises(ValueError, match="mlp_type"):
        GPT(GPTConfig(**{**SMALL, "mlp_type": "gelu"}))


# ---------------------------------------------------------------- CLI + optimizer


def test_cli_flags_reach_hyperparameters(monkeypatch):
    import train

    monkeypatch.setattr(sys, "argv", ["train.py", "--mlp_type", "swiglu", "--attn_gate", "--norm_gain"])
    hp = train.override_args_from_cli(train.Hyperparameters(), train.parse_cli_args())
    assert (hp.mlp_type, hp.attn_gate, hp.norm_gain) == ("swiglu", True, True)

    monkeypatch.setattr(sys, "argv", ["train.py"])
    hp = train.override_args_from_cli(train.Hyperparameters(), train.parse_cli_args())
    assert (hp.mlp_type, hp.attn_gate, hp.norm_gain) == ("relu2", False, False)


@pytest.fixture
def gloo_world():
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29621")
        dist.init_process_group("gloo", rank=0, world_size=1)
        yield
        dist.destroy_process_group()
    else:
        yield


@pytest.mark.parametrize("kw", [{}, ALL_ON], ids=["default", "all_on"])
def test_optimizer_groups_matrices_and_gains(gloo_world, monkeypatch, kw):
    import train
    from torch.nn.parallel import DistributedDataParallel as DDP

    monkeypatch.setattr(sys, "argv", ["train.py", "--no_triton"])
    cli = train.parse_cli_args()
    hp = train.Hyperparameters()
    hp.optimizer, hp.scalar_opt = "normuon", "adamw"
    for k, v in kw.items():
        setattr(hp, k, v)
    model = _model(**kw)
    opt = train.init_optimizer(model, None, DDP(model), hp, cli)

    group_of = {
        id(p): g.get("algorithm", "matrix") for g in opt.param_groups for p in g["params"]
    }
    for name, p in model.named_parameters():
        assert id(p) in group_of, f"{name} is in no param group"
        in_blocks = name.startswith("transformer.h.")
        expected = "normuon" if (in_blocks and p.ndim == 2) else "adamw"
        assert group_of[id(p)] == expected, f"{name} {tuple(p.shape)} -> {group_of[id(p)]}"

    expected_groups = 4 if kw.get("norm_gain") else 3
    assert len(opt.param_groups) == expected_groups


# ---------------------------------------------------------------- scalar_lr


@pytest.mark.parametrize(
    "scalar_lr, expected", [(0.0, 0.02), (0.01, 0.01)],
    ids=["follows_lr", "decoupled"],
)
def test_scalar_lr_sets_only_the_adamw_groups(gloo_world, monkeypatch, scalar_lr, expected):
    """An eta/sqrt(f) sweep on one global lr moves the AdamW groups too, and that
    side carries most of the penalty. --scalar_lr holds them still; 0 keeps the
    old behaviour of following --lr."""
    import train
    from torch.nn.parallel import DistributedDataParallel as DDP

    monkeypatch.setattr(sys, "argv", ["train.py", "--no_triton"])
    cli = train.parse_cli_args()
    hp = train.Hyperparameters()
    hp.optimizer, hp.scalar_opt, hp.lr, hp.scalar_lr = "normuon", "adamw", 0.02, scalar_lr
    hp.norm_gain = True
    model = _model(**ALL_ON)
    opt = train.init_optimizer(model, None, DDP(model), hp, cli)

    kinds = {}
    names = {id(p): n for n, p in model.named_parameters()}
    for g in opt.param_groups:
        for p in g["params"]:
            kinds[names[id(p)]] = (g.get("algorithm", "matrix"), g.get("lr"))
    for name, (algorithm, lr) in kinds.items():
        if algorithm == "adamw":
            assert lr == pytest.approx(expected), f"{name} at {lr}"
    # The matrix group keeps the base lr whatever scalar_lr does; adjust_lr scales
    # it per parameter inside the optimizer, not here.
    assert opt.param_groups[0].get("lr", hp.lr) == pytest.approx(0.02)
