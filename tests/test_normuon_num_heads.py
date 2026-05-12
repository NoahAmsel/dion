"""
Test that NorMuon with num_heads=None and num_heads=1 produce identical results.

When _resolve_num_heads maps num_heads=1 → None, both paths are trivially
identical. When the shortcut is removed (num_heads=1 actually enters
_prepare_head_split), the 2D→3D reshape causes numerical divergence.

The isolation tests below pin-point which stage diverges:
  1. Pre-orthogonalize (momentum update)
  2. Newton-Schulz orthogonalization (2D vs batched-3D matmul)
  3. NorMuon normalization (stacking dimensions differ)
  4. Post-orthogonalize (weight update)

Usage:
  pytest tests/test_normuon_num_heads.py -v
"""

import pytest
import torch

CUDA_AVAILABLE = torch.cuda.is_available()
torch._dynamo.config.cache_size_limit = 64


def _make_normuon(params, num_heads, **kwargs):
    """Create a NorMuon optimizer with the given num_heads on the param group."""
    from dion import NorMuon

    groups = [dict(params=params)]
    if num_heads is not None:
        groups[0]["num_heads"] = num_heads

    return NorMuon(
        groups,
        lr=0.02,
        mu=0.95,
        muon_beta2=0.95,
        weight_decay=0.01,
        nesterov=False,
        adjust_lr="spectral_norm",
        use_polar_express=True,
        use_gram_newton_schulz=True,
        use_triton=True,
        **kwargs,
    )


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
class TestNorMuonNumHeads:
    """NorMuon with num_heads=None vs num_heads=1 must be bitwise identical."""

    @pytest.mark.parametrize("num_steps", [1, 3])
    def test_none_vs_one(self, num_steps):
        """Run multiple steps and confirm weights stay bitwise identical."""
        torch.manual_seed(0)
        shapes = [(64, 128), (128, 64)]
        base_params = [
            torch.nn.Parameter(torch.randn(s, device="cuda", dtype=torch.bfloat16))
            for s in shapes
        ]

        params_none = [torch.nn.Parameter(p.data.clone()) for p in base_params]
        params_one = [torch.nn.Parameter(p.data.clone()) for p in base_params]

        opt_none = _make_normuon(params_none, num_heads=None)
        opt_one = _make_normuon(params_one, num_heads=1)

        for step in range(num_steps):
            torch.manual_seed(100 + step)
            for p_n, p_o in zip(params_none, params_one):
                grad = torch.randn_like(p_n)
                p_n.grad = grad.clone()
                p_o.grad = grad.clone()

            opt_none.step()
            opt_one.step()

            opt_none.zero_grad()
            opt_one.zero_grad()

            for i, (p_n, p_o) in enumerate(zip(params_none, params_one)):
                diff = (p_n.data - p_o.data).abs().max().item()
                assert diff == 0.0, (
                    f"Step {step + 1}, param {i}: "
                    f"num_heads=None vs num_heads=1 differ by {diff:.6e}"
                )


# ---------------------------------------------------------------------------
# Isolation tests: locate exactly where 2D vs 3D (num_heads=1) diverge
#
# These tests call each stage of the NorMuon pipeline manually with 2D
# input vs the equivalent (1, H, W) 3D input. A stage that returns
# max_diff > 0 is a divergence source.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
class TestNumHeadsDivergenceIsolation:
    """Pin-point which pipeline stage causes 2D vs 3D divergence."""

    SHAPE = (64, 128)

    def _make_inputs(self, dtype=torch.bfloat16):
        torch.manual_seed(0)
        G = torch.randn(self.SHAPE, device="cuda", dtype=dtype)
        M = torch.randn(self.SHAPE, device="cuda", dtype=dtype)
        return G, M

    @staticmethod
    def _make_gns_func():
        """Build the same NS function that the optimizer uses with
        use_gram_newton_schulz=True, use_triton=True, use_polar_express=True."""
        from gram_newton_schulz import GramNewtonSchulz

        _gns = GramNewtonSchulz(
            ns_use_kernels=True,
            use_gram_newton_schulz=True,
            gram_newton_schulz_reset_iterations=[2],
            compile_kwargs=dict(fullgraph=True, mode="default"),
        )
        return lambda X, epsilon=None: _gns(X)

    def test_pre_orthogonalize(self):
        """Momentum update: foreach ops on 2D [G] vs 3D [G.unsqueeze(0)]."""
        from dion.muon import muon_update_pre_orthogonalize

        G, M = self._make_inputs()
        M2 = M.clone()

        U_2d = muon_update_pre_orthogonalize(
            G=[G.clone()], M=[M.clone()], momentum=torch.tensor(0.95), nesterov=False,
        )
        U_3d = muon_update_pre_orthogonalize(
            G=[G.clone().unsqueeze(0)],
            M=[M2.clone().unsqueeze(0)],
            momentum=torch.tensor(0.95),
            nesterov=False,
        )

        diff = (U_2d[0] - U_3d[0].squeeze(0)).abs().max().item()
        print(f"pre_orthogonalize: max diff = {diff:.6e}")
        # foreach ops are element-wise; this should be zero
        assert diff == 0.0, f"Pre-orthogonalize diverged: {diff:.6e}"

    def test_newton_schulz_gns(self):
        """GramNewtonSchulz: 2D (H, W) vs 3D (1, H, W)."""
        ns_func = self._make_gns_func()

        torch.manual_seed(0)
        M = torch.randn(self.SHAPE, device="cuda", dtype=torch.bfloat16)

        out_2d = ns_func(M)
        out_3d = ns_func(M.unsqueeze(0))

        diff = (out_2d - out_3d.squeeze(0)).abs().max().item()
        print(f"newton_schulz (GNS+triton): max diff = {diff:.6e}")

    def test_newton_schulz_gns_float32(self):
        """GramNewtonSchulz in float32: 2D vs 3D."""
        ns_func = self._make_gns_func()

        torch.manual_seed(0)
        M = torch.randn(self.SHAPE, device="cuda", dtype=torch.float32)

        out_2d = ns_func(M)
        out_3d = ns_func(M.unsqueeze(0))

        diff = (out_2d - out_3d.squeeze(0)).abs().max().item()
        print(f"newton_schulz (GNS+triton, f32): max diff = {diff:.6e}")

    def test_normuon_normalization(self):
        """NorMuon normalization: stacked (1, H, W) vs (1, 1, H, W)."""
        from dion.normuon import normuon_normalization_stacked

        torch.manual_seed(0)
        U = torch.randn(self.SHAPE, device="cuda", dtype=torch.bfloat16)
        V = torch.zeros(self.SHAPE[0], 1, device="cuda", dtype=torch.bfloat16)
        beta2 = torch.tensor(0.95)

        # 2D path: stack([U]) → (1, H, W), stack([V]) → (1, H, 1)
        U_out_2d, V_out_2d = normuon_normalization_stacked(
            U.unsqueeze(0).clone(), V.unsqueeze(0).clone(), beta2
        )

        # 3D path: stack([U.unsqueeze(0)]) → (1, 1, H, W), stack([V.unsqueeze(0)]) → (1, 1, H, 1)
        U_out_3d, V_out_3d = normuon_normalization_stacked(
            U.unsqueeze(0).unsqueeze(0).clone(), V.unsqueeze(0).unsqueeze(0).clone(), beta2
        )

        u_diff = (U_out_2d - U_out_3d.squeeze(1)).abs().max().item()
        v_diff = (V_out_2d - V_out_3d.squeeze(1)).abs().max().item()
        print(f"normuon_normalization U: max diff = {u_diff:.6e}")
        print(f"normuon_normalization V: max diff = {v_diff:.6e}")

    def test_post_orthogonalize(self):
        """Weight update: foreach ops on 2D vs 3D."""
        from dion.muon import muon_update_post_orthogonalize

        torch.manual_seed(0)
        X = torch.randn(self.SHAPE, device="cuda", dtype=torch.bfloat16)
        U = torch.randn(self.SHAPE, device="cuda", dtype=torch.bfloat16)
        lr = torch.tensor(0.02)
        adj_lr = torch.tensor(0.02 * (64 / 128) ** 0.5)
        wd = torch.tensor(0.01)

        X2d, X3d = X.clone(), X.clone().unsqueeze(0)

        muon_update_post_orthogonalize(
            X=[X2d], U=[U.clone()], base_lr=lr, adjusted_lr=adj_lr, weight_decay=wd,
        )
        muon_update_post_orthogonalize(
            X=[X3d], U=[U.clone().unsqueeze(0)], base_lr=lr, adjusted_lr=adj_lr, weight_decay=wd,
        )

        diff = (X2d - X3d.squeeze(0)).abs().max().item()
        print(f"post_orthogonalize: max diff = {diff:.6e}")
        # foreach ops are element-wise; this should be zero
        assert diff == 0.0, f"Post-orthogonalize diverged: {diff:.6e}"
