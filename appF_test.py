import torch
from dion.polarExp import *
import os
from tqdm import tqdm

# Extract all the spectra
# svs = []
# bad_g_dir = "bad_G"
# matrix_files = sorted(
#     [os.path.join(bad_g_dir, f) for f in os.listdir(bad_g_dir) if f.endswith(".pt")],
#     key=lambda x: os.path.getmtime(x)
# )
# for matrix_path in tqdm(matrix_files):
#     matrix = torch.load(matrix_path)
#     matrix = matrix.to(dtype=torch.bfloat16)
#     matrix /= matrix.norm()
#     ssss = torch.linalg.svdvals(matrix.double()).cpu()
#     if matrix.shape[0] < matrix.shape[1]:
#         matrix = matrix.mT
#     Y = matrix.T @ matrix
#     svs.append((ssss, torch.linalg.eigvalsh(Y.double()).cpu()))
#     # out = FastApplyPolarExpress(matrix, steps=5, restart_interval=3, shift_eps=1e-3)
# out_path = "spectra.pt"
# torch.save(svs, out_path)
# print(f"Saved {len(svs)} singular-value tensors to {out_path}")
# exit(0)


matrix_path = "bad_G/bad_G_fe57dd52c153467db334f70eb895c51f.pt"
matrix = torch.load(matrix_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
# print(S[:10])
# print(S[-10:])

S = torch.concat((
    # torch.logspace(0, -3, steps=len(S)//2, device=S.device, dtype=S.dtype),
    # torch.zeros(len(S) - len(S)//2, device=S.device, dtype=S.dtype)
    # torch.logspace(-5, -6, steps=len(S) - len(S)//2, device=S.device, dtype=S.dtype)

    torch.logspace(0, -0.5, steps=2, device=S.device, dtype=S.dtype),
    # torch.logspace(-2, -2, steps=len(S)-2, device=S.device, dtype=S.dtype)
    torch.zeros(len(S) - 2, device=S.device, dtype=S.dtype)
))
# S = torch.logspace(0, -3, steps=len(S), device=S.device, dtype=S.dtype)
matrix = U @ torch.diag(S) @ Vh


def eval(matrix, true_polar, estimate, verbose=True):
    assert estimate.isfinite().all()
    estimate = estimate.to(matrix.dtype)
    H = estimate.mT @ matrix
    H = (H + H.mT) / 2
    I = torch.eye(estimate.shape[0], device=estimate.device, dtype=estimate.dtype)

    # print(f"relative error: {((estimate - true_polar).norm() / (true_polar).norm()).item():.6f}")
    orthogonality_error = ((estimate @ estimate.mT - I).norm() / I.norm()).item()
    residual_error = ((estimate @ H - matrix).norm() / matrix.norm()).item()
    Heigs = torch.linalg.eigvalsh(H)
    psd_error = (Heigs[Heigs < 0]).norm() / (Heigs[Heigs > 0]).norm()
    nuc = torch.linalg.matrix_norm(matrix, ord='nuc')
    dual_obj = ((nuc - torch.trace(estimate.mT @ matrix))/nuc).item()
    bound_violation = max((torch.linalg.matrix_norm(estimate, ord=2) - 1).item(), 0)

    if verbose:
        print(f"orthogonality error: {orthogonality_error:.6f}")
        print(f"residual error: {residual_error:.6f}")
        print(f"psd error: {psd_error:.6f}")
        print(f"dual obj: {dual_obj:.6f}")
        print(f"bound violation: {bound_violation:.6f}")
    return dict(
        orth_error=orthogonality_error,
        residual_error=residual_error,
        psd_error=psd_error,
        dual_obj=dual_obj,
        bound_violation=bound_violation,
    )


# if matrix.shape[0] < matrix.shape[1]:
#     matrix = matrix.mT
# matrix /= matrix.norm()
# Y = matrix.bfloat16().T @ matrix.bfloat16()  # + 0e-3 * torch.eye(matrix.shape[1], device=matrix.device, dtype=torch.bfloat16)
# print(torch.linalg.eigvalsh(Y.double())[:20])
# exit(0)


STEPS = 15

print("*"*30)
print("reference")
eval(matrix, U @ Vh, PolarExpress(matrix, steps=STEPS))

# print("*"*30)
# print("Fast apply")
# out, logs = appendix_F_safe(matrix, steps=STEPS, restart_interval=100, shift_eps=0e-4)
# eval(matrix, U @ Vh, out)
# torch.save(logs, "intermediate.pt")

# print("*"*30)
# print("flaot32 fast apply")
# out, logs = appendix_F_float32(matrix, steps=STEPS)
# eval(matrix, U @ Vh, out)
# torch.save(logs, "intermediate_float32.pt")

# print("*"*30)
# print("Alt1")
# out, logs = alt1(matrix, steps=STEPS, restart_interval=100, shift_eps=1e-3)
# eval(matrix, U @ Vh, out)
# torch.save(logs, "intermediate_alt1.pt")

# print("*"*30)
# print("Alt2")
# out, logs = alt2(matrix, steps=30, restart_interval=5, shift_eps=1e-3)
# eval(matrix, U @ Vh, out)
# torch.save(logs, "intermediate_alt2.pt")

# print("*"*30)
# print("Alt3")
# out = alt3(matrix, steps=STEPS, restart_interval=5, shift_eps=1e-3)
# eval(matrix, U @ Vh, out)

print("*"*30)
print("Final")
out = final_appF(matrix, steps=STEPS, restart_interval=5, shift_eps=1e-3)
eval(matrix, U @ Vh, out)

# print("*"*30)
# print("Fast apply safe")
# eval(matrix, U @ Vh, appendix_F_safe(matrix, steps=STEPS, restart_interval=3, shift_eps=1e-2))
