from itertools import repeat
import torch
from .newton_schulz_triton import ns_line_1, ns_line_2


coeffs_list = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),  # subsequent coeffs equal this numerically
]
# safety factor for numerical stability (but exclude last polynomial)
coeffs_list = [(a / 1.01, b / 1.01**3, c / 1.01**5)
                for (a, b, c) in coeffs_list[:-1]] + [coeffs_list[-1]]


@torch.compile(dynamic=False, fullgraph=True)
def PolarExpress(G: torch.Tensor, steps: int) -> torch.Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()  # for speed
    if G.size(-2) > G.size(-1): X = X.mT  # this reduces FLOPs
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-7)
    hs = coeffs_list[:steps] + list( 
        repeat(coeffs_list[-1], steps - len(coeffs_list)))
    for a, b, c in hs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X  # X <- aX + bX^3 + cX^5
    if G.size(-2) > G.size(-1): X = X.mT
    return X


@torch.compile(dynamic=False, fullgraph=True)
def PolarExpress_triton(G: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Direct copy of newton_schulz_triton, but with Polar Express coefficients
    """
    X = G.to(dtype=torch.bfloat16)
    if G.size(-2) > G.size(-1): X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-7)
    hs = coeffs_list[:steps] + list( 
        repeat(coeffs_list[-1], steps - len(coeffs_list)))

    # Allocate buffers
    X = X.contiguous()
    A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    C = torch.empty_like(X)
    ns_line_3 = torch.baddbmm if X.ndim > 2 else torch.addmm

    # Perform the NS iterations
    for a, b, c in hs:
        ns_line_1(X, out=A)  # A = X @ X.mT
        ns_line_2(A, alpha=c, beta=b, out=B)  # B = b * A + c * A @ A
        ns_line_3(X, B, X, beta=a, out=C)  # C = a * X + B @ X
        X, C = C, X  # Swap references to avoid unnecessary copies
    if G.size(-2) > G.size(-1): X = X.mT
    return X


def quadratic(R, a, b, c, out):
    """Computes out = aI + bR + cR^2"""
    # TODO! implement with a triton kernel
    ns_line_1(R, out=out)
    out.mul_(c).add_(R, alpha=b)
    I = torch.eye(R.shape[-2], device=R.device, dtype=R.dtype)  # TODO! don't create this every time
    out.add_(I, alpha=a)

    # NOTE BUG: below should be a better way to compute bR + cR^2
    # but if you use torch.compile and addmm, we get blow ups to infinite values...
    # R_copy = R.clone()
    # R_copy2 = R.clone()
    # addmm = torch.baddbmm if R.ndim > 2 else torch.addmm
    # addmm(R, R_copy, R_copy2, beta=b, alpha=c, out=out)  # bR + cR^2
    # END BUG

    # NOTE: below should be a better way to add aI to the diagonal of out in-place, but it gives compiler problems
    # with warnings.catch_warnings():
    #     warnings.filterwarnings("ignore", message=".*torch._prims_common.check.*", category=FutureWarning)
    #     out.diagonal(dim1=-2, dim2=-1).add_(a)


def symmetric_matmul(A, B, out):
    """Computes out = A @ B where A and B are symmetric matrices."""
    # TODO! Currently just does regular matmul.
    torch.matmul(A, B, out=out)


@torch.compile(dynamic=False, fullgraph=True)
def final_appF(G: torch.Tensor, steps: int, restart_interval: int, shift_eps: float = 0, first_restart: int = 0) -> torch.Tensor:
    assert G.ndim >= 2
    X = G.to(dtype=torch.bfloat16)
    if G.size(-2) > G.size(-1): X = X.mT  # this reduces FLOPs
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-7)
    hs = coeffs_list[:steps] + list( 
        repeat(coeffs_list[-1], steps - len(coeffs_list)))
    
    # Allocate buffers
    I = torch.eye(X.shape[-2], device=X.device, dtype=X.dtype)
    R = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    Q = torch.empty_like(R)
    M1 = torch.empty_like(R)
    M2 = torch.empty_like(R)

    for iter, (a, b, c) in enumerate(hs):
        if iter == 0:
            ns_line_1(X, out=R)  # R = X @ X.mT
            R.add_(I, alpha=shift_eps).mul_(1/(1+shift_eps))  # (R + eps*I) / (1 + eps) numerical stability
            quadratic(R, a, b, c, out=Q)  # Q = aI + bR + cR^2
            symmetric_matmul(R, Q, out=M2); symmetric_matmul(Q.mT, M2, out=R)  # R = Q R Q
        elif (iter >= first_restart) and ((iter - first_restart) % restart_interval == 0):
            X = Q @ X  # TODO: is this inplace?
            ns_line_1(X, out=R)  # R = X @ X.mT
            quadratic(R, a, b, c, out=Q)  # Q = aI + bR + cR^2
            symmetric_matmul(R, Q, out=M2); symmetric_matmul(Q.mT, M2, out=R)  # R = Q R Q
        else:
            quadratic(R, a, b, c, out=M1)  # M1 = aI + bR + cR^2
            symmetric_matmul(R, M1, out=M2); symmetric_matmul(M1.mT, M2, out=R)  # R = M1 R M1
            symmetric_matmul(M1, Q, out=M2); Q, M2 = M2, Q  # Q = M1 Q

    X = Q @ X  # TODO: is this inplace?
    if G.size(-2) > G.size(-1): X = X.mT
    return X


@torch.compile(dynamic=False, fullgraph=True)
def PolarExpressAdaptiveF(G: torch.Tensor):
    if max(G.shape[-2:]) > 2 * min(G.shape[-2:]):
        return final_appF(G, steps=5, restart_interval=3, shift_eps=1e-3)
    else:
        return PolarExpress_triton(G, steps=5)
