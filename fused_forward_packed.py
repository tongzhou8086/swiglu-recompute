"""PACKED fused SwiGLU forward for the recompute scheme.

Like `fused_forward.py`, but using the **packed** weight layout + warp
specialization -- the structure that lets the fused GEMM match cuBLAS (see
`bench_packed_vs_cublas.py`). The forward kernel emits BOTH outputs from one
warp-specialized launch:

    h            [M, H]    the activation, fed into the down-projection
    preact_packed[M, 2H]   raw preactivation, packed [left0|gate0|left1|...],
                           saved for the backward recompute (NOT factors)

It mirrors `_fused_swiglu_wide_packed_save_factors_kernel` from the `swiglu_fused`
repo, swapping the factor side-store for a raw-preact side-store. The packed
column order matches `x @ packed_weight`, so the backward reuses the packed
`grad_de`-from-preact kernel and packed GEMMs.

Self-contained: the packed scaffolding (TMA alloc, pid swizzle, pack/unpack, the
packed grad_de kernel) is copied here so `swiglu-recompute` has no cross-repo
path dependency.

Weight is stored packed as ``packed_weight`` of shape ``[K=D, 2H]``.
"""

from __future__ import annotations

import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl


# -----------------------------------------------------------------------------
# Launch config (matches the production packed kernel).
# -----------------------------------------------------------------------------
BLOCK_SIZE_M = 128
BLOCK_SIZE_N_HALF = 128
BLOCK_SIZE_K = 64
GROUP_SIZE_M = 32
NUM_WARPS = 8
NUM_STAGES = 4
WARP_SPECIALIZE = True


def _tma_alloc(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def _ensure_allocator() -> None:
    triton.set_allocator(_tma_alloc)


@functools.cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@triton.jit
def _compute_pid(tile_id, num_pid_in_group: tl.constexpr, num_pid_m: tl.constexpr,
                 GROUP_SIZE_M_: tl.constexpr):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M_
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M_)
    pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


# -----------------------------------------------------------------------------
# Packing utilities (torch). Packed layout: [left0|gate0|left1|gate1|...] over
# the output dim at block_n_half granularity.
# -----------------------------------------------------------------------------
def pack_swiglu_weight_chunked_torch(weight: torch.Tensor,
                                     block_n_half: int = BLOCK_SIZE_N_HALF) -> torch.Tensor:
    """Pack [K, 2N] (standard [left | gate]) into chunk-interleaved [K, 2N]."""
    assert weight.is_contiguous()
    k, n2 = weight.shape
    assert n2 % 2 == 0
    n_half = n2 // 2
    assert n_half % block_n_half == 0
    out = torch.empty_like(weight)
    chunks = n_half // block_n_half
    left = weight[:, :n_half].view(k, chunks, block_n_half)
    gate = weight[:, n_half:].view(k, chunks, block_n_half)
    packed = out.view(k, chunks, 2, block_n_half)
    packed[:, :, 0, :].copy_(left)
    packed[:, :, 1, :].copy_(gate)
    return out


def unpack_swiglu_weight_chunked_torch(packed_weight: torch.Tensor,
                                       block_n_half: int = BLOCK_SIZE_N_HALF) -> torch.Tensor:
    """Undo chunk-interleaved packing back to standard [K, 2N]."""
    assert packed_weight.is_contiguous()
    k, n2 = packed_weight.shape
    n_half = n2 // 2
    assert n_half % block_n_half == 0
    out = torch.empty_like(packed_weight)
    chunks = n_half // block_n_half
    packed = packed_weight.view(k, chunks, 2, block_n_half)
    out[:, :n_half].view(k, chunks, block_n_half).copy_(packed[:, :, 0, :])
    out[:, n_half:].view(k, chunks, block_n_half).copy_(packed[:, :, 1, :])
    return out


def pack_swiglu_linear_weight(linear_weight: torch.Tensor,
                              block_n_half: int = BLOCK_SIZE_N_HALF) -> torch.Tensor:
    """Pack a Linear weight [2N, K] into internal packed [K, 2N]."""
    assert linear_weight.ndim == 2
    return pack_swiglu_weight_chunked_torch(linear_weight.t().contiguous(), block_n_half)


def unpack_swiglu_linear_weight(packed_weight: torch.Tensor,
                                block_n_half: int = BLOCK_SIZE_N_HALF) -> torch.Tensor:
    """Return a Linear weight [2N, K] from internal packed storage."""
    return unpack_swiglu_weight_chunked_torch(
        packed_weight.contiguous(), block_n_half
    ).t().contiguous()


# -----------------------------------------------------------------------------
# Forward kernel: one wide dot -> (h, preact). Stores raw packed preact (not
# factors). Single dot => warp-specializable.
# -----------------------------------------------------------------------------
@triton.jit
def _fused_swiglu_wide_packed_save_preact_kernel(
    a_ptr, bp_ptr, c_ptr, preact_ptr,
    M: tl.constexpr, N_HALF: tl.constexpr, K: tl.constexpr, NUM_SMS: tl.constexpr,
    BLOCK_SIZE_M_: tl.constexpr, BLOCK_SIZE_N_HALF_: tl.constexpr,
    BLOCK_SIZE_K_: tl.constexpr, GROUP_SIZE_M_: tl.constexpr,
    WARP_SPECIALIZE_: tl.constexpr, FLATTEN: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    BLOCK_SIZE_N2: tl.constexpr = BLOCK_SIZE_N_HALF_ * 2

    num_pid_m: tl.constexpr = tl.cdiv(M, BLOCK_SIZE_M_)
    num_pid_n: tl.constexpr = tl.cdiv(N_HALF, BLOCK_SIZE_N_HALF_)
    k_tiles: tl.constexpr = tl.cdiv(K, BLOCK_SIZE_K_)
    num_tiles: tl.constexpr = num_pid_m * num_pid_n
    num_pid_in_group: tl.constexpr = GROUP_SIZE_M_ * num_pid_n

    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[K, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_K_])
    bp_desc = tl.make_tensor_descriptor(
        bp_ptr, shape=[K, N_HALF * 2], strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_K_, BLOCK_SIZE_N2])
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N_HALF], strides=[N_HALF, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_])
    preact_desc = tl.make_tensor_descriptor(
        preact_ptr, shape=[M, N_HALF * 2], strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_])

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS,
                            flatten=FLATTEN, warp_specialize=WARP_SPECIALIZE_):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M_)
        offs_m = pid_m * BLOCK_SIZE_M_
        offs_n = pid_n * BLOCK_SIZE_N_HALF_
        offs_n2 = pid_n * BLOCK_SIZE_N2

        acc = tl.zeros((BLOCK_SIZE_M_, BLOCK_SIZE_N2), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K_
            a = a_desc.load([offs_m, offs_k])
            b = bp_desc.load([offs_k, offs_n2])           # one wide load
            acc = tl.dot(a, b, acc, input_precision=INPUT_PRECISION)

        acc3 = tl.reshape(acc, (BLOCK_SIZE_M_, 2, BLOCK_SIZE_N_HALF_))
        acc3 = tl.permute(acc3, (0, 2, 1))
        left, gate = tl.split(acc3)
        h = left * (gate * tl.sigmoid(gate))

        # raw packed preact: [left chunk | gate chunk] at this n2 block
        preact_desc.store([offs_m, offs_n2], left.to(preact_ptr.dtype.element_ty))
        preact_desc.store([offs_m, offs_n2 + BLOCK_SIZE_N_HALF_],
                          gate.to(preact_ptr.dtype.element_ty))
        c_desc.store([offs_m, offs_n], h.to(c_ptr.dtype.element_ty))


def fused_swiglu_save_preact_packed(x, packed_weight):
    """One launch: (h [M,H], preact_packed [M,2H]) for ``x @ packed_weight``."""
    _ensure_allocator()
    assert x.is_cuda and packed_weight.is_cuda
    assert x.is_contiguous() and packed_weight.is_contiguous()
    m, k = x.shape
    k2, n2 = packed_weight.shape
    assert k == k2 and n2 % 2 == 0
    n_half = n2 // 2
    assert m % BLOCK_SIZE_M == 0 and n_half % BLOCK_SIZE_N_HALF == 0

    h = torch.empty((m, n_half), device=x.device, dtype=x.dtype)
    preact = torch.empty((m, n2), device=x.device, dtype=x.dtype)
    device_index = x.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    num_sms = _num_sms(device_index)
    grid = (min(num_sms, triton.cdiv(m, BLOCK_SIZE_M) * triton.cdiv(n_half, BLOCK_SIZE_N_HALF)),)

    if x.dtype == torch.float32:
        input_precision, block_k, num_stages, ws = "ieee", 32, 2, False
    else:
        input_precision, block_k, num_stages, ws = "tf32", BLOCK_SIZE_K, NUM_STAGES, WARP_SPECIALIZE

    _fused_swiglu_wide_packed_save_preact_kernel[grid](
        x, packed_weight, h, preact, m, n_half, k,
        NUM_SMS=num_sms, BLOCK_SIZE_M_=BLOCK_SIZE_M, BLOCK_SIZE_N_HALF_=BLOCK_SIZE_N_HALF,
        BLOCK_SIZE_K_=block_k, GROUP_SIZE_M_=GROUP_SIZE_M, WARP_SPECIALIZE_=ws,
        FLATTEN=True, INPUT_PRECISION=input_precision,
        num_warps=NUM_WARPS, num_stages=num_stages)
    return h, preact


# -----------------------------------------------------------------------------
# Backward kernel: packed grad_de from packed preact + dy (explicit SiLU deriv).
# -----------------------------------------------------------------------------
@triton.jit
def _swiglu_packed_grad_de_from_preact_kernel(
    preact_ptr, dy_ptr, grad_de_ptr,
    M: tl.constexpr, N_HALF: tl.constexpr,
    BLOCK_SIZE_M_: tl.constexpr, BLOCK_SIZE_N_HALF_: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    BLOCK_SIZE_N2: tl.constexpr = BLOCK_SIZE_N_HALF_ * 2
    num_pid_n: tl.constexpr = tl.cdiv(N_HALF, BLOCK_SIZE_N_HALF_)
    pid_m = pid // num_pid_n
    pid_n = pid - pid_m * num_pid_n
    offs_m = pid_m * BLOCK_SIZE_M_
    offs_n = pid_n * BLOCK_SIZE_N_HALF_
    offs_n2 = pid_n * BLOCK_SIZE_N2

    preact_desc = tl.make_tensor_descriptor(
        preact_ptr, shape=[M, N_HALF * 2], strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N2])
    dy_desc = tl.make_tensor_descriptor(
        dy_ptr, shape=[M, N_HALF], strides=[N_HALF, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_])
    grad_desc = tl.make_tensor_descriptor(
        grad_de_ptr, shape=[M, N_HALF * 2], strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N2])

    preact = preact_desc.load([offs_m, offs_n2]).to(tl.float32)
    preact3 = tl.reshape(preact, (BLOCK_SIZE_M_, 2, BLOCK_SIZE_N_HALF_))
    preact3 = tl.permute(preact3, (0, 2, 1))
    left, gate = tl.split(preact3)

    dy = dy_desc.load([offs_m, offs_n]).to(tl.float32)
    sig = tl.sigmoid(gate)
    silu = gate * sig
    silu_prime = sig + silu * (1.0 - sig)
    grad_left = dy * silu
    grad_gate = dy * left * silu_prime
    grad_de = tl.cat(grad_left, grad_gate, dim=1)
    grad_desc.store([offs_m, offs_n2], grad_de.to(grad_de_ptr.dtype.element_ty))


def swiglu_packed_grad_de_from_preact(preact, dy):
    """packed grad_de [M,2H] from packed preact [M,2H] + dy [M,H] (fresh out)."""
    _ensure_allocator()
    assert preact.is_contiguous()
    if not dy.is_contiguous():
        dy = dy.contiguous()
    m, n2 = preact.shape
    n_half = n2 // 2
    assert dy.shape == (m, n_half) and n_half % BLOCK_SIZE_N_HALF == 0
    grad_de = torch.empty_like(preact)
    grid = (triton.cdiv(m, BLOCK_SIZE_M) * triton.cdiv(n_half, BLOCK_SIZE_N_HALF),)
    _swiglu_packed_grad_de_from_preact_kernel[grid](
        preact, dy, grad_de, m, n_half,
        BLOCK_SIZE_M_=BLOCK_SIZE_M, BLOCK_SIZE_N_HALF_=BLOCK_SIZE_N_HALF,
        num_warps=NUM_WARPS)
    return grad_de


# h recomputed from packed preact (for grad_W2 = grad_y.t() @ h).
@torch.compile
def _h_from_packed_preact(preact, M, chunks, bnh):
    p = preact.view(M, chunks, 2, bnh)
    left = p[:, :, 0, :].reshape(M, chunks * bnh)
    gate = p[:, :, 1, :].reshape(M, chunks * bnh)
    return left * F.silu(gate)


# -----------------------------------------------------------------------------
# autograd.Function: packed fused forward, recompute-from-packed-preact backward.
# -----------------------------------------------------------------------------
class SwiGLUPackedFusedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed_weight, w2):
        if not x.is_contiguous():
            x = x.contiguous()
        if not packed_weight.is_contiguous():
            packed_weight = packed_weight.contiguous()
        h, preact = fused_swiglu_save_preact_packed(x, packed_weight)  # one kernel
        ctx.save_for_backward(x, packed_weight, w2, preact)            # save preact, NOT h
        y = h @ w2.t()
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, packed_weight, w2, preact = ctx.saved_tensors
        m, n2 = preact.shape
        n_half = n2 // 2
        chunks = n_half // BLOCK_SIZE_N_HALF
        h = _h_from_packed_preact(preact, m, chunks, BLOCK_SIZE_N_HALF)  # recompute
        grad_w2 = grad_y.t() @ h                       # [Dout, H]
        grad_h = grad_y @ w2                           # [M, H]
        grad_de = swiglu_packed_grad_de_from_preact(preact, grad_h)  # packed [M, 2H]
        grad_x = grad_de @ packed_weight.t()           # [M, K]
        grad_pw = x.t() @ grad_de                      # [K, 2H] (packed)
        return grad_x, grad_pw, grad_w2


class SwiGLUMLPPackedFused(nn.Module):
    """SwiGLU MLP using the packed warp-specialized fused forward + recompute bwd."""

    def __init__(self, dim, hidden_dim, out_dim=None, dtype=torch.float32):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim if out_dim is not None else dim
        assert hidden_dim % BLOCK_SIZE_N_HALF == 0
        # packed weight [K=dim, 2H]
        std = torch.empty(2 * hidden_dim, dim, dtype=dtype)
        nn.init.xavier_uniform_(std)
        self.packed_weight = nn.Parameter(pack_swiglu_linear_weight(std))
        self.w2 = nn.Parameter(torch.empty(self.out_dim, hidden_dim, dtype=dtype))
        nn.init.xavier_uniform_(self.w2)

    def forward(self, x):
        return SwiGLUPackedFusedFunction.apply(x, self.packed_weight, self.w2)


# -----------------------------------------------------------------------------
# Self-test vs ground truth (fp32 ieee + bf16).
# -----------------------------------------------------------------------------
def _selftest():
    from swiglu_recompute import SwiGLUMLPGroundTruth

    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    print(f"torch {torch.__version__} | triton {triton.__version__} | "
          f"{torch.cuda.get_device_name(0)}")
    D, H, Dout, M = 3584, 14336, 3584, 512

    def run(dtype, tol):
        torch.manual_seed(0)
        gt = SwiGLUMLPGroundTruth(D, H, Dout, dtype=dtype).cuda()
        pk = SwiGLUMLPPackedFused(D, H, Dout, dtype=dtype).cuda()
        with torch.no_grad():
            pk.packed_weight.copy_(pack_swiglu_linear_weight(gt.w1.contiguous()))
            pk.w2.copy_(gt.w2)
        x0 = torch.randn(M, D, device="cuda", dtype=dtype)
        xg = x0.detach().clone().requires_grad_(True)
        xp = x0.detach().clone().requires_grad_(True)
        yg = gt(xg)
        yp = pk(xp)
        gout = torch.randn_like(yg)
        gg = torch.autograd.grad(yg, [xg, gt.w1, gt.w2], gout)
        gp = torch.autograd.grad(yp, [xp, pk.packed_weight, pk.w2], gout)

        def rel(a, b):
            return ((a.float() - b.float()).abs().max()
                    / b.float().abs().max().clamp_min(1e-6)).item()

        grad_w1 = unpack_swiglu_linear_weight(gp[1])   # packed grad -> [2H, D]
        checks = {"out": rel(yp, yg), "grad_x": rel(gp[0], gg[0]),
                  "grad_W1": rel(grad_w1, gg[1]), "grad_W2": rel(gp[2], gg[2])}
        ok = all(v <= tol for v in checks.values())
        print(f"  [{dtype}] tol={tol:g}")
        for n, v in checks.items():
            print(f"      {n:<9s} rel_max={v:.3e}  {'OK' if v <= tol else 'FAIL'}")
        print(f"  => {'PASS' if ok else 'FAIL'}")
        return ok

    ok32 = run(torch.float32, tol=1e-4)
    okbf = run(torch.bfloat16, tol=2e-2)
    print("ALL PASS" if (ok32 and okbf) else "FAILURES PRESENT")


if __name__ == "__main__":
    _selftest()
