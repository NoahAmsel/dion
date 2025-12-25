import torch
import triton
import triton.language as tl
from torch import Tensor


def _get_autotune_configs():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": 8,
                "LOWER_UPPER": 1,
            },
            num_stages=stages,
            num_warps=warps,
        )
        for bm in [64, 128]
        for bn in [64, 128, 256]
        for bk in [64, 128]
        for stages, warps in [(3, 4), (3, 8), (4, 4)]
        if bm // bn <= 2 and bn // bm <= 2
    ]


@triton.jit
def _pid_to_block(
    pid,
    M,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Helper function to map Triton program ID to (batch, row, col) of the output matrix.
    """
    # Split output matrix into blocks of size (BLOCK_SIZE_M, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_N)

    # Map PID to a single matrix in batch
    batch_idx = pid // (num_pid_m * num_pid_n)
    pid = pid % (num_pid_m * num_pid_n)

    # Map PID to 2D grid of blocks
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)

    m_idx = pid_m * BLOCK_SIZE_M
    n_idx = pid_n * BLOCK_SIZE_N

    return batch_idx, m_idx, n_idx


@triton.autotune(
    configs=_get_autotune_configs(),
    key=["M", "K", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"],
)
@triton.jit
def ns_line_1_kernel(
    A_ptr,
    C_ptr,
    M,
    K,
    a_stride_b,
    a_stride_r,
    a_stride_c,
    c_stride_b,
    c_stride_r,
    c_stride_c,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    """
    Input A has shape (M, K)
    Output C has shape (M, M)
    Compute C = A @ A.T
    """

    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # Create pointer arrays for A and A.T
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)


def ns_line_1(A: Tensor, *, out: Tensor = None):
    """
    Launch Triton kernel to compute C = A @ A.T
    """
    if A.ndim > 3 or A.ndim < 2:
        raise ValueError(f"Input tensor must be 2D or 3D, but got {A.ndim}D tensor.")

    M, K = A.shape[-2:]

    if out is None:
        out = torch.empty((*A.shape[:-1], M), device=A.device, dtype=A.dtype)
    assert out.size(-2) == M, "Output matrix has incorrect shape"
    assert out.size(-1) == M, "Output matrix has incorrect shape"

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    grid = lambda meta: (
        batch_size
        * triton.cdiv(M, meta["BLOCK_SIZE_M"])
        * triton.cdiv(M, meta["BLOCK_SIZE_N"]),
    )
    ns_line_1_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        K=K,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
    )

    return out


@triton.autotune(
    configs=_get_autotune_configs(),
    key=["M", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"],
)
@triton.jit
def ns_line_2_kernel(
    A_ptr,
    C_ptr,
    M,
    a_stride_b,
    a_stride_r,
    a_stride_c,
    c_stride_b,
    c_stride_r,
    c_stride_c,
    alpha,
    beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    """
    Input A is square matrix with shape (M, M)
    Output C has shape (M, M)
    Compute C = alpha * A @ A.T + beta * A
    """

    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # Create pointer arrays for A and A.T
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < M - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < M - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    # Load block of A to add (corresponds to the current block of C)
    offs_am = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_an = n_idx + tl.arange(0, BLOCK_SIZE_N)
    a_add_ptrs = A_ptr + (offs_am[:, None] * a_stride_r + offs_an[None, :] * a_stride_c)
    a_add_mask = (offs_am[:, None] < M) & (offs_an[None, :] < M)
    a_add = tl.load(a_add_ptrs, mask=a_add_mask, other=0.0).to(tl.float32)

    # Apply alpha and beta
    accumulator *= alpha
    accumulator += a_add * beta

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)


def ns_line_2(A: Tensor, alpha: float, beta: float, *, out: Tensor = None):
    """
    Launch Triton kernel to compute C = alpha * A @ A.T + beta * A
    """
    if A.ndim > 3 or A.ndim < 2:
        raise ValueError(f"Input tensor must be 2D or 3D, but got {A.ndim}D tensor.")

    M, K = A.shape[-2:]
    if M != K:
        raise ValueError(
            f"Input must be symmetric square matrix, but got shape {A.shape}"
        )

    if out is None:
        out = torch.empty((*A.shape[:-1], M), device=A.device, dtype=A.dtype)
    assert out.size(-2) == M, "Output matrix has incorrect shape"
    assert out.size(-1) == M, "Output matrix has incorrect shape"

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    grid = lambda meta: (
        batch_size
        * triton.cdiv(M, meta["BLOCK_SIZE_M"])
        * triton.cdiv(M, meta["BLOCK_SIZE_N"]),
    )
    ns_line_2_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
        alpha=alpha,
        beta=beta,
    )

    return out


@torch.compile(dynamic=False, fullgraph=True)
def zeropower_via_newtonschulz5(G: Tensor, epsilon: float = 1e-7):
    """
    Reference implementation of Newton-Schulz without Triton.
    """
    # Newton-Schulz constants
    ns_consts = [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]

    X = G.to(dtype=torch.bfloat16)
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + epsilon)

    for a, b, c in ns_consts:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


@torch.compile(dynamic=False, fullgraph=True)
def eig_polar(G: Tensor):
    X = G.to(dtype=torch.bfloat16)
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / X.norm(dim=(-2, -1), keepdim=True)

    A = ns_line_1(X)  # A = X @ X.mT
    Lambda, V = torch.linalg.eigh(A.float())
    A = V.bfloat16()
    A = (A * Lambda.rsqrt().bfloat16().unsqueeze(-2)) @ A.mT
    X = A @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


@torch.compile(dynamic=False, fullgraph=True)
def newton_schulz_triton(G: Tensor, epsilon: float = 1e-7):
    """
    Triton implementation of Newton-Schulz iteration
    """
    # Newton-Schulz constants
    ns_consts = [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]

    X = G.to(dtype=torch.bfloat16)
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + epsilon)

    # Allocate buffers
    X = X.contiguous()
    A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    C = torch.empty_like(X)

    ns_line_3 = torch.baddbmm if X.ndim > 2 else torch.addmm

    # Perform the NS iterations
    for a, b, c in ns_consts:
        ns_line_1(X, out=A)  # A = X @ X.mT
        ns_line_2(A, alpha=c, beta=b, out=B)  # B = b * A + c * A @ A
        ns_line_3(X, B, X, beta=a, out=C)  # C = a * X + B @ X
        X, C = C, X  # Swap references to avoid unnecessary copies

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


@torch.compile(dynamic=False, fullgraph=True)
def newton_schulz_appendixF_triton(G: Tensor, shift_epsilon: float = 1e-3, epsilon: float = 1e-7):
    """
    Triton implementation of Newton-Schulz iteration
    """
    # Newton-Schulz constants
    ns_consts = [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]

    X = G.to(dtype=torch.bfloat16)
    if G.size(-2) > G.size(-1):  # we want matrices to be fat
        X = X.mT

    # Ensure spectral norm is at most 1
    # TODO use Schatten-4
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + epsilon)

    # Allocate buffers
    X = X.contiguous()
    Y = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    Q = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    temp = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    temp2 = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)

    ns_line_3 = torch.baddbmm if X.ndim > 2 else torch.addmm

    a0, b0, c0 = ns_consts[0]
    ns_line_1(X, out=Y)  # Y = X @ X.mT
    # Y = ns_line_1(X.float()).bfloat16()  # Y = X @ X.mT
    if shift_epsilon != 0:
        Y += shift_epsilon * torch.eye(X.size(-2), X.size(-2), device=X.device, dtype=X.dtype)
    ns_line_2(Y, alpha=c0, beta=b0, out=Q)  # Q = b0 * Y + c0 * Y @ Y
    Q += a0*torch.eye(X.size(-2), X.size(-2), device=X.device, dtype=X.dtype)  # TODO: fix this, perhaps as below?
    # Q.diagonal(dim1=-2, dim2=-1).add_(a0)  # Q += a0*I

    # Perform the NS iterations
    for a, b, c in ns_consts[1:3]:
    # for a, b, c in ns_consts[1:]:
        R = Q.mT @ Y @ Q  # TODO: implement as triton kernel
        ns_line_2(R, alpha=c, beta=b, out=temp)
        ns_line_3(Q, Q, temp, beta=a, out=temp2)
        Q, temp2 = temp2, Q  # Swap references to avoid unnecessary copies

    X = Q @ X

    a3, b3, c3 = ns_consts[3]
    ns_line_1(X, out=Y)  # Y = X @ X.mT
    ns_line_2(Y, alpha=c3, beta=b3, out=Q)  # Q = b0 * Y + c0 * Y @ Y
    Q += a3*torch.eye(X.size(-2), X.size(-2), device=X.device, dtype=X.dtype)  # TODO: fix this, perhaps as below?
    # Q.diagonal(dim1=-2, dim2=-1).add_(a0)  # Q += a0*I

    # Perform the NS iterations
    for a, b, c in ns_consts[4:]:
        R = Q.mT @ Y @ Q  # TODO: implement as triton kernel
        ns_line_2(R, alpha=c, beta=b, out=temp)
        ns_line_3(Q, Q, temp, beta=a, out=temp2)
        Q, temp2 = temp2, Q  # Swap references to avoid unnecessary copies

    X = Q @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


@torch.compile(dynamic=False, fullgraph=True)
def newton_schulz_triton_adaptive(G: Tensor, epsilon: float = 1e-7):
    if max(G.shape[-2:]) > 2 * min(G.shape[-2:]):
        return newton_schulz_appendixF_triton(G, epsilon=epsilon)
    else:
        return newton_schulz_triton(G, epsilon=epsilon)


@torch.compile(dynamic=False, fullgraph=True)
def newton_schulz_stopper(G: torch.Tensor, ell, tol=.1, max_iter=20, epsilon: float = 1e-7):
    """
    Triton implementation of Newton-Schulz iteration
    - always does Appendix F trick with no restarting
    - chooses coefficients adaptively like Chen and Chow
    - stops 
    """

    X = G.to(dtype=torch.bfloat16)
    if G.size(-2) > G.size(-1):  # we want matrices to be fat
        X = X.mT

    # Ensure spectral norm is at most 1
    # TODO use Schatten-4
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + epsilon)

    # Allocate buffers
    X = X.contiguous()
    Y = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    Q = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    temp = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)

    ns_line_3 = torch.baddbmm if X.ndim > 2 else torch.addmm
    trace = lambda X: torch.einsum("...ii", X).min() if X.ndim > 2 else torch.trace

    ell = torch.tensor(ell, device=G.device, dtype=G.dtype)
    alpha = torch.sqrt(3 / (1 + ell * (1 + ell)))
    ns_line_1(X, out=Y)  # Y = X @ X.mT
    # Y = ns_line_1(X.float()).bfloat16()  # Y = X @ X.mT
    Q = (-0.5 * alpha**3) * Y + (1.5*alpha) * torch.eye(X.size(-2), X.size(-2), device=X.device, dtype=X.dtype)
    ell = 1.5 * (alpha * ell) - 0.5 * (alpha * ell)**3

    # Perform the NS iterations
    for _ in range(max_iter):
        R = Q.mT @ Y @ Q  # TODO: implement as triton kernel
        alpha = torch.sqrt(3 / (1 + ell * (1 + ell)))
        ns_line_3(Q, Q, R, beta=(1.5*alpha), alpha=(-0.5 * alpha**3), out=temp)
        Q, temp = temp, Q  # Swap references to avoid unnecessary copies
        # if trace(R) > X.size(-2) - tol:
        #     break
        ell = 1.5 * (alpha * ell) - 0.5 * (alpha * ell)**3

    X = Q @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X
