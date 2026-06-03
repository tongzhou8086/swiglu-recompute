"""Fused SwiGLU forward: one Triton kernel that outputs BOTH `preact` and `h`.

`SwiGLUMemoryOptimizedFunction.forward` runs the forward projection in two HBM
passes:

    preact = x @ W1.t()                 # GEMM, writes preact [M, 2H]
    h = _swiglu_forward_logic(preact)   # separate kernel: READS preact, writes h [M, H]

This module fuses them into a single matmul kernel that keeps the GEMM
accumulator in registers, computes `h = left * silu(gate)` from it, and emits
two outputs in one launch:

    preact [M, 2H]   (saved for the backward recompute; standard [left | gate] layout)
    h      [M, H]    (the activation, fed straight into the down-projection)

This mirrors the structure of `_fused_swiglu_wide_packed_save_factors_kernel`
in ~/projects/fused_swiglu_kernel, with two differences:

  1. The side output is the **raw preact** (this project recomputes the SiLU
     factors in backward; it does NOT precompute backward factors).
  2. **Standard (non-packed) weight layout.** The packed layout in that project
     exists so a CTA can load the matching left/gate column pair with one wide
     TMA load. Here we instead issue *two* narrow B-loads into two accumulators
     (`left` cols [0,H), `gate` cols [H,2H)) -- identical FLOPs/bytes, and it
     keeps `preact` in standard [M,2H] layout so the existing PyTorch backward
     (chunk(2) + standard GEMMs) works unchanged.

The win is forward bandwidth: the activation is computed from the in-register
accumulator, so `preact` is never read back from HBM (one fewer [M,2H] read,
~609 MiB at the project shape). Peak memory is unchanged -- both preact and h
are still materialized -- so this is a speed/bandwidth optimization, not a
memory one.

Weight is stored as ``Wt = W1.t()`` with shape ``[K=D, 2H]`` (left = Wt[:, :H],
gate = Wt[:, H:]); ``preact = x @ Wt``.
"""

from __future__ import annotations

import functools

import torch
import torch.nn as nn

import triton
import triton.language as tl

# Reuse this project's own elementwise helpers for the (unchanged) backward.
from swiglu_recompute import _swiglu_forward_logic, _swiglu_backward_logic


# -----------------------------------------------------------------------------
# Launch config (same block shape as the fused_swiglu save_factors kernel).
# -----------------------------------------------------------------------------
BLOCK_SIZE_M = 128
BLOCK_SIZE_N_HALF = 128   # block over the hidden dim H (= N_HALF)
BLOCK_SIZE_K = 64
GROUP_SIZE_M = 8
NUM_WARPS = 8
NUM_STAGES = 4
# Triton 3.7's automatic warp-specialization pass can't partition the two-dot /
# two-accumulator body (the standard-layout split), so leave it off. It's a
# perf knob, not correctness; the packed single-wide-dot kernel can use it.
WARP_SPECIALIZE = False
FLATTEN = True


def _tma_alloc(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def _ensure_allocator() -> None:
    # Required before any tl.make_tensor_descriptor (TMA) kernel launch.
    triton.set_allocator(_tma_alloc)


@functools.cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@triton.jit
def _compute_pid(
    tile_id,
    num_pid_in_group: tl.constexpr,
    num_pid_m: tl.constexpr,
    GROUP_SIZE_M_: tl.constexpr,
):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M_
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M_)
    pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


# -----------------------------------------------------------------------------
# Fused kernel: preact = x @ Wt ; h = left * silu(gate), emitted together.
# Standard layout: left = Wt[:, :H], gate = Wt[:, H:]  ->  two B-loads/tile.
# -----------------------------------------------------------------------------
@triton.jit
def _fused_matmul_swiglu_save_preact_kernel(
    a_ptr,            # x        [M, K]
    b_ptr,            # Wt       [K, 2*N_HALF]   (standard: [left | gate])
    preact_ptr,       # preact   [M, 2*N_HALF]   (standard: [left | gate])
    h_ptr,            # h        [M, N_HALF]
    M: tl.constexpr,
    N_HALF: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE_M_: tl.constexpr,
    BLOCK_SIZE_N_HALF_: tl.constexpr,
    BLOCK_SIZE_K_: tl.constexpr,
    GROUP_SIZE_M_: tl.constexpr,
    WARP_SPECIALIZE_: tl.constexpr,
    FLATTEN_: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    N2: tl.constexpr = N_HALF * 2

    num_pid_m: tl.constexpr = tl.cdiv(M, BLOCK_SIZE_M_)
    num_pid_n: tl.constexpr = tl.cdiv(N_HALF, BLOCK_SIZE_N_HALF_)
    k_tiles: tl.constexpr = tl.cdiv(K, BLOCK_SIZE_K_)
    num_tiles: tl.constexpr = num_pid_m * num_pid_n
    num_pid_in_group: tl.constexpr = GROUP_SIZE_M_ * num_pid_n

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_K_],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[K, N2],
        strides=[N2, 1],
        block_shape=[BLOCK_SIZE_K_, BLOCK_SIZE_N_HALF_],
    )
    preact_desc = tl.make_tensor_descriptor(
        preact_ptr,
        shape=[M, N2],
        strides=[N2, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )
    h_desc = tl.make_tensor_descriptor(
        h_ptr,
        shape=[M, N_HALF],
        strides=[N_HALF, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_],
    )

    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=FLATTEN_,
        warp_specialize=WARP_SPECIALIZE_,
    ):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M_
        )
        offs_m = pid_m * BLOCK_SIZE_M_
        offs_n = pid_n * BLOCK_SIZE_N_HALF_

        acc_left = tl.zeros((BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_), dtype=tl.float32)
        acc_gate = tl.zeros((BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K_
            a = a_desc.load([offs_m, offs_k])             # [BM, BK]
            b_left = b_desc.load([offs_k, offs_n])         # left cols [0, H)
            b_gate = b_desc.load([offs_k, N_HALF + offs_n])  # gate cols [H, 2H)
            acc_left = tl.dot(a, b_left, acc_left, input_precision=INPUT_PRECISION)
            acc_gate = tl.dot(a, b_gate, acc_gate, input_precision=INPUT_PRECISION)

        # h = left * silu(gate) = left * gate * sigmoid(gate), from registers.
        h = acc_left * (acc_gate * tl.sigmoid(acc_gate))

        # Store preact in standard [M, 2H] layout (left then gate) ...
        preact_desc.store([offs_m, offs_n], acc_left.to(preact_ptr.dtype.element_ty))
        preact_desc.store(
            [offs_m, N_HALF + offs_n], acc_gate.to(preact_ptr.dtype.element_ty)
        )
        # ... and the activation h.
        h_desc.store([offs_m, offs_n], h.to(h_ptr.dtype.element_ty))


def fused_matmul_swiglu_save_preact(
    x: torch.Tensor,
    wt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One launch: returns (preact [M, 2H], h [M, H]) for ``preact = x @ wt``.

    ``x`` is [M, K]; ``wt`` is [K, 2H] standard layout (left = wt[:, :H]).
    """
    _ensure_allocator()
    assert x.is_cuda and wt.is_cuda
    assert x.is_contiguous() and wt.is_contiguous()

    m, k = x.shape
    k2, n2 = wt.shape
    assert k == k2 and n2 % 2 == 0
    n_half = n2 // 2
    assert m % BLOCK_SIZE_M == 0, f"M={m} must be a multiple of {BLOCK_SIZE_M}"
    assert n_half % BLOCK_SIZE_N_HALF == 0, (
        f"H={n_half} must be a multiple of {BLOCK_SIZE_N_HALF}"
    )

    preact = torch.empty((m, n2), device=x.device, dtype=x.dtype)
    h = torch.empty((m, n_half), device=x.device, dtype=x.dtype)

    device_index = x.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    num_sms = _num_sms(device_index)
    grid = (
        min(
            num_sms,
            triton.cdiv(m, BLOCK_SIZE_M) * triton.cdiv(n_half, BLOCK_SIZE_N_HALF),
        ),
    )
    # fp32 inputs default to tf32 in tl.dot; force true fp32 for the fp32 path.
    # fp32 tiles are 2x the bytes of bf16, so the deep (stages=4, BK=64) pipeline
    # overflows the 232 KB B200 SMEM -- use a shallower/narrower config for fp32.
    if x.dtype == torch.float32:
        input_precision, block_k, num_stages = "ieee", 32, 2
    else:
        input_precision, block_k, num_stages = "tf32", BLOCK_SIZE_K, NUM_STAGES

    _fused_matmul_swiglu_save_preact_kernel[grid](
        x,
        wt,
        preact,
        h,
        m,
        n_half,
        k,
        NUM_SMS=num_sms,
        BLOCK_SIZE_M_=BLOCK_SIZE_M,
        BLOCK_SIZE_N_HALF_=BLOCK_SIZE_N_HALF,
        BLOCK_SIZE_K_=block_k,
        GROUP_SIZE_M_=GROUP_SIZE_M,
        WARP_SPECIALIZE_=WARP_SPECIALIZE,
        FLATTEN_=FLATTEN,
        INPUT_PRECISION=input_precision,
        num_warps=NUM_WARPS,
        num_stages=num_stages,
    )
    return preact, h


# -----------------------------------------------------------------------------
# autograd.Function: fused forward, recompute-h backward (same math as
# SwiGLUMemoryOptimizedFunction, but weight is Wt = W1.t() with shape [K, 2H]).
# -----------------------------------------------------------------------------
class SwiGLUFusedForwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, wt, w2):
        if not x.is_contiguous():
            x = x.contiguous()
        if not wt.is_contiguous():
            wt = wt.contiguous()
        preact, h = fused_matmul_swiglu_save_preact(x, wt)   # one kernel
        ctx.save_for_backward(x, wt, w2, preact)             # save preact, NOT h
        y = h @ w2.t()                                       # [M, Dout]
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, wt, w2, preact = ctx.saved_tensors
        h = _swiglu_forward_logic(preact)            # recompute h from preact
        grad_w2 = grad_y.t() @ h                     # [Dout, H]
        grad_h = grad_y @ w2                         # [M, H]
        grad_preact = _swiglu_backward_logic(preact, grad_h)  # [M, 2H], standard
        grad_wt = x.t() @ grad_preact                # [K, 2H]
        grad_x = grad_preact @ wt.t()                # [M, K]
        return grad_x, grad_wt, grad_w2


# -----------------------------------------------------------------------------
# nn.Module wrapper. Weight stored transposed (Wt = W1.t(), shape [D, 2H]).
# -----------------------------------------------------------------------------
class SwiGLUMLPFused(nn.Module):
    """SwiGLU MLP using the fused forward kernel + backward recompute of `h`."""

    def __init__(self, dim, hidden_dim, out_dim=None, dtype=torch.float32):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim if out_dim is not None else dim
        # Wt = W1.t(): [D, 2H]  (left = w1t[:, :H], gate = w1t[:, H:])
        self.w1t = nn.Parameter(torch.empty(dim, 2 * hidden_dim, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(self.out_dim, hidden_dim, dtype=dtype))
        nn.init.xavier_uniform_(self.w1t)
        nn.init.xavier_uniform_(self.w2)

    def forward(self, x):
        return SwiGLUFusedForwardFunction.apply(x, self.w1t, self.w2)


# -----------------------------------------------------------------------------
# Self-test: fused vs ground truth (fp32 ieee + bf16), plus preact/h sanity.
# -----------------------------------------------------------------------------
def _selftest():
    import torch.nn.functional as F
    from swiglu_recompute import SwiGLUMLPGroundTruth

    assert torch.cuda.is_available(), "needs a CUDA GPU"
    torch.cuda.set_device(0)
    print(f"torch {torch.__version__} | triton {triton.__version__} | "
          f"{torch.cuda.get_device_name(0)}")

    D, H, Dout = 3584, 14336, 3584
    M = 512

    def run(dtype, tol):
        torch.manual_seed(0)
        gt = SwiGLUMLPGroundTruth(D, H, Dout, dtype=dtype).cuda()
        fused = SwiGLUMLPFused(D, H, Dout, dtype=dtype).cuda()
        with torch.no_grad():
            fused.w1t.copy_(gt.w1.t().contiguous())   # Wt = W1.t()
            fused.w2.copy_(gt.w2)

        x0 = torch.randn(M, D, device="cuda", dtype=dtype)
        xg = x0.detach().clone().requires_grad_(True)
        xf = x0.detach().clone().requires_grad_(True)
        yg = gt(xg)
        yf = fused(xf)
        gout = torch.randn_like(yg)
        gg = torch.autograd.grad(yg, [xg, gt.w1, gt.w2], gout)
        gf = torch.autograd.grad(yf, [xf, fused.w1t, fused.w2], gout)

        def rel(a, b):
            denom = b.float().abs().max().clamp_min(1e-6)
            return ((a.float() - b.float()).abs().max() / denom).item()

        checks = {
            "out": rel(yf, yg),
            "grad_x": rel(gf[0], gg[0]),
            "grad_W1": rel(gf[1].t(), gg[1]),   # grad wrt Wt -> compare transpose
            "grad_W2": rel(gf[2], gg[2]),
        }
        ok = all(v <= tol for v in checks.values())
        print(f"  [{dtype}] tol={tol:g}")
        for name, v in checks.items():
            print(f"      {name:<9s} rel_max={v:.3e}  {'OK' if v <= tol else 'FAIL'}")
        print(f"  => {'PASS' if ok else 'FAIL'}")
        return ok

    # preact / h sanity (fp32, ieee) vs torch reference.
    torch.manual_seed(1)
    wt = torch.randn(D, 2 * H, device="cuda", dtype=torch.float32) * (D ** -0.5)
    x = torch.randn(M, D, device="cuda", dtype=torch.float32) * (D ** -0.5)
    preact, h = fused_matmul_swiglu_save_preact(x, wt)
    ref_preact = x @ wt
    left, gate = ref_preact.chunk(2, dim=-1)
    ref_h = left * F.silu(gate)
    pe = (preact - ref_preact).abs().max().item()
    he = (h - ref_h).abs().max().item()
    print(f"  [sanity fp32] preact max_abs={pe:.3e}  h max_abs={he:.3e}")

    ok32 = run(torch.float32, tol=1e-4)
    okbf = run(torch.bfloat16, tol=2e-2)
    print("ALL PASS" if (ok32 and okbf) else "FAILURES PRESENT")


if __name__ == "__main__":
    _selftest()
