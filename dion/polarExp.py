from itertools import repeat
import torch
from .newton_schulz_triton import *


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
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 +1e-7)
    hs = coeffs_list[:steps] + list( 
        repeat(coeffs_list[-1], steps - len(coeffs_list)))
    # hs = [(1.5, -0.5, 0)] * steps
    for a, b, c in hs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X  # X <- aX + bX^3 + cX^5
    if G.size(-2) > G.size(-1): X = X.mT
    return X


def log(M, V):
    M = M.double()
    print(f"sym: {((M - M.mT).norm() / M.norm()).item():.6e}")
    Lambda = V.T @ M @ V
    print(f"correct eigvecs: {((Lambda - torch.diag(torch.diag(Lambda))).norm() / M.norm()).item():.6e}")
    eigs = torch.linalg.eigvalsh(M)
    print(f"eigs: min {eigs.min().item():.6e}, max {eigs.max().item():.6e}")
    return eigs

def mm(A, B):
    return (A @ B + B @ A)/2


@torch.compile(dynamic=False, fullgraph=True)
def appendix_F_safe(G: torch.Tensor, steps: int, restart_interval: int, shift_eps: float = 0, do_logs=False) -> torch.Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1): X = X.mT  # this reduces FLOPs
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-7)
    hs = coeffs_list[:steps] + list( 
        repeat(coeffs_list[-1], steps - len(coeffs_list)))
    # hs = [(1.875 / 1.05, -1.25 / 1.05**3, 0.375 / 1.05**5)] * steps
    # hs = [(1.5, -0.5, 0)] * steps
    I = torch.eye(X.shape[-2], device=X.device, dtype=X.dtype)
    Y = (X @ X.mT + shift_eps * I)/(1+shift_eps)  # numerical stability
    logs = dict(R=[], Q=[])
    if do_logs: 
        Lambda, V = torch.linalg.eigh(Y.double())
        print("--- Y ---")
        log(Y, V)
    Q = I.clone()
    for iter, (a, b, c) in enumerate(hs):
        if (iter % restart_interval == 0) and (iter > 0):
            X = Q @ X
            # Y = (X @ X.mT + shift_eps * I)/(1+shift_eps)  # numerical stability
            Y = X @ X.mT
            Q = I.clone()
        R = Q.mT @ Y @ Q
        # R = (mm(Q, mm(Y, Q)) + mm(Y, mm(Q, Q))) / 2
        # Q = Q @ (a*I + R @ (b*I + c*R))  # Q <- Q(aI + bR + cR^2)
        hR = a*I + mm(R, b*I + c*R)
        Q = mm(Q, hR)
        if do_logs:
            print(f"{iter} --- R ---")
            logs["R"].append(log(R, V))
            print(f"{iter} --- Q ---")
            logs["Q"].append(log(Q, V))
            if logs["R"][-1].max() > 1e1:
                break
    X = Q @ X
    # if (X.norm(dim=(-2, -1), keepdim=False) > 5 * I.shape[0]).any() or not (torch.isfinite(X).all()):
    #     warnings.warn("X.norm() is unusually large. Saving G to disk.")
    #     os.makedirs("bad_G", exist_ok=True)
    #     filename = f"bad_G_{uuid.uuid4().hex}.pt"
    #     torch.save(G, os.path.join("bad_G", filename))
    if G.size(-2) > G.size(-1): X = X.mT
    return X, logs


@torch.compile(dynamic=False, fullgraph=True)
def alt1(G: torch.Tensor, steps: int, restart_interval: int, shift_eps: float = 0) -> torch.Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1): X = X.mT  # this reduces FLOPs
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-7)
    hs = [(1.5, -0.5, 0)] * steps
    I = torch.eye(X.shape[-2], device=X.device, dtype=X.dtype)
    Y = (X @ X.mT + shift_eps * I)/(1+shift_eps)  # numerical stability
    Lambda, V = torch.linalg.eigh(Y.double())
    # print("--- Y ---")
    # log(Y, V)
    logs = dict(R=[], hR=[])
    R = Y.clone()
    for iter, (a, b, c) in enumerate(hs):
        if (iter % restart_interval == 0) and (iter > 0):
            Y = (X @ X.mT + shift_eps * I)/(1+shift_eps)  # numerical stability
            R = Y.clone()
        hR = a*I + mm(R, b*I + c*R)
        R = hR.mT @ R @ hR
        print(f"{iter} --- hR ---")
        logs["hR"].append(log(hR, V))
        print(f"{iter} --- R ---")
        logs["R"].append(log(R, V))
        X = hR @ X
    if G.size(-2) > G.size(-1): X = X.mT
    return X, logs


def alt2(G: torch.Tensor, steps: int, restart_interval: int, shift_eps: float = 0) -> torch.Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1): X = X.mT  # this reduces FLOPs
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-7)
    hs = [(1.5, -0.5, 0)] * steps
    I = torch.eye(X.shape[-2], device=X.device, dtype=X.dtype)
    R = X @ X.mT
    Lambda, V = torch.linalg.eigh(R.double())
    R = (R + shift_eps * I)/(1+shift_eps)  # numerical stability
    # print("--- Y ---")
    # log(Y, V)
    logs = dict(R=[], hR=[])
    R = (R + R.mT) / 2
    Q = I.clone()
    for iter, (a, b, c) in enumerate(hs):
        if (iter % restart_interval == 0) and (iter > 0):
        # if iter == restart_interval:  # only one restart
            X = Q @ X
            Q = I.clone()
            R = X @ X.mT
            # R = (R + shift_eps * I)/(1+shift_eps)  # numerical stability
        hR = a*I + mm(R, b*I + c*R)
        R = hR.mT @ R @ hR
        R = (R + R.mT) / 2
        Q = hR @ Q
        print(f"{iter} --- hR ---")
        logs["hR"].append(log(hR, V))
        print(f"{iter} --- R ---")
        logs["R"].append(log(R, V))
    X = Q @ X
    if G.size(-2) > G.size(-1): X = X.mT
    return X, logs

@torch.compile(dynamic=False, fullgraph=True)
def alt3(G: torch.Tensor, steps: int, restart_interval: int, shift_eps: float = 0, first_restart: int = 0) -> torch.Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1): X = X.mT  # this reduces FLOPs
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-7)
    hs = coeffs_list[:steps] + list( 
        repeat(coeffs_list[-1], steps - len(coeffs_list)))
    # hs = [(1.5, -0.5, 0)] * steps
    I = torch.eye(X.shape[-2], device=X.device, dtype=X.dtype)
    R = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    for iter, (a, b, c) in enumerate(hs):
        if iter == 0:
            ns_line_1(X, out=R)  # R = X @ X.mT
            R = (R + shift_eps * I)/(1+shift_eps)  # numerical stability
            Q = a*I + mm(R, b*I + c*R)
            R = Q.mT @ R @ Q
            R = (R + R.mT) / 2
        elif (iter >= first_restart) and ((iter - first_restart) % restart_interval == 0):
            X = Q @ X
            ns_line_1(X, out=R)  # R = X @ X.mT
            Q = a*I + mm(R, b*I + c*R)
            R = Q.mT @ R @ Q
            R = (R + R.mT) / 2
        else:
            hR = a*I + mm(R, b*I + c*R)
            R = hR.mT @ R @ hR
            R = (R + R.mT) / 2
            Q = hR @ Q
    X = Q @ X
    if G.size(-2) > G.size(-1): X = X.mT
    return X


@torch.compile(dynamic=False, fullgraph=True)
def final_appF(G: torch.Tensor, steps: int, restart_interval: int, shift_eps: float = 0, first_restart: int = 0) -> torch.Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1): X = X.mT  # this reduces FLOPs
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-7)
    hs = coeffs_list[:steps] + list( 
        repeat(coeffs_list[-1], steps - len(coeffs_list)))
    I = torch.eye(X.shape[-2], device=X.device, dtype=X.dtype)
    R = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    Q = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    M1 = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    M2 = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    addmm = torch.baddbmm if X.ndim > 2 else torch.addmm
    for iter, (a, b, c) in enumerate(hs):
        if iter == 0:
            ns_line_1(X, out=R)  # R = X @ X.mT
            R.add_(I, alpha=shift_eps).mul_(1/(1+shift_eps))  # numerical stability
            # Q = a*I + mm(R, b*I + c*R)
            # R = Q.mT @ R @ Q

            Q = b*R + c*(R @ R)
            # addmm(R, R, R, beta=b, alpha=c, out=Q)  # bR + cR^2
            # Q = (Q + Q.mT) / 2
            Q.add_(I, alpha=a)  # aI + bR + cR^2
            torch.matmul(R, Q, out=M2)
            torch.matmul(Q.mT, M2, out=R)
        elif (iter >= first_restart) and ((iter - first_restart) % restart_interval == 0):
            X = Q @ X
            ns_line_1(X, out=R)  # R = X @ X.mT
            # Q = a*I + mm(R, b*I + c*R)
            # R = Q.mT @ R @ Q

            Q = b*R + c*(R @ R)
            # NOTE BUG: below should be equivalent to previous line
            # but if you use torch.compile and addmm, we get blow ups to infinite values...
            # R_copy = R.clone()
            # R_copy2 = R.clone()
            # Q = addmm(R, R_copy, R_copy2, beta=b, alpha=c)  # bR + cR^2
            # END BUG
            # Q = (Q + Q.mT) / 2
            Q.add_(I, alpha=a)  # aI + bR + cR^2
            torch.matmul(R, Q, out=M2)
            torch.matmul(Q.mT, M2, out=R)
        else:
            # hR = a*I + mm(R, b*I + c*R)  # aI + bR + cR^2 =: aI + M1
            # R = hR.mT @ R @ hR  # = a^2 R + 2aRM1 + M1 R M1
            # Q = hR @ Q

            M1 = b*R + c*(R @ R)
            # addmm(R, R, R, beta=b, alpha=c, out=M1)  # bR + cR^2
            M1.add_(I, alpha=a)  # aI + bR + cR^2
            # M1 = (M1 + M1.mT) / 2
            torch.matmul(R, M1, out=M2)
            torch.matmul(M1.mT, M2, out=R)
            torch.matmul(M1, Q, out=M2)
            Q, M2 = M2, Q

            # addmm(R, R, R, beta=b, alpha=c, out=M1)  # M1 = bR + cR^2
            # torch.matmul(R, M1, out=M2)
            # R.mul_(a**2).add_(M2, alpha=2*a)
            # R.addmm_(M1, M2)  # R = a^2 R + 2aM2 + M1 M2
            # addmm(Q, M1, Q, beta=a, out=M2)  # hR @ Q = aQ + (bR + cR^2)Q
            # Q, M2 = M2, Q

    X = Q @ X
    if G.size(-2) > G.size(-1): X = X.mT
    return X


@torch.compile(dynamic=False, fullgraph=True)
def ALTWEIRD(G: torch.Tensor, steps: int, restart_interval: int, shift_eps: float = 0) -> torch.Tensor:
    assert G.ndim >= 2
    X = G.double()
    # X = G.bfloat16()
    if G.size(-2) > G.size(-1): X = X.mT  # this reduces FLOPs
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-7)
    hs = coeffs_list[:steps] + list( 
        repeat(coeffs_list[-1], steps - len(coeffs_list)))
    I = torch.eye(X.shape[-2], device=X.device, dtype=X.dtype)
    Y = X @ X.mT
    # R = (X @ X.mT + shift_eps * I)/(1+shift_eps)  # numerical stability
    Z = I.clone()
    Q = I.clone()
    Zs = []
    for iter, (a, b, c) in enumerate(hs):
        # if (iter % restart_interval == 0) and (iter > 0):
        #     # X = Q @ X
        #     # Y = X @ X.mT
        #     Qs.append(Q)
        #     Y = Q @ Y @ Q.mT
        #     Q = I.clone()
        # R = Q.mT @ Y @ Q
        # Q = Q @ (a*I + R @ (b*I + c*R))  # Q <- Q(aI + bR + cR^2)

        # if (iter % restart_interval == 0):
        #     R = ((Q @ X) @ (Q@X).mT + shift_eps * I)/(1+shift_eps)  # numerical stability
        #     Z = I.clone()
        # R = Z.mT @ R @ Z
        # Z = (a*I + R @ (b*I + c*R))
        # Zs.append(Z)
        # Q = Z @ Q

        R = Q.mT @ Y @ Q
        Z = (a*I + R @ (b*I + c*R))
        Zs.append(Z)
        Q = Z @ Q
    for Z in Zs:  # reversed(Zs):
        X = Z @ X
    # X = Q @ X
    # if (X.norm(dim=(-2, -1), keepdim=False) > 5 * I.shape[0]).any() or not (torch.isfinite(X).all()):
    #     warnings.warn("X.norm() is unusually large. Saving G to disk.")
    #     os.makedirs("bad_G", exist_ok=True)
    #     filename = f"bad_G_{uuid.uuid4().hex}.pt"
    #     torch.save(G, os.path.join("bad_G", filename))
    if G.size(-2) > G.size(-1): X = X.mT
    return X
