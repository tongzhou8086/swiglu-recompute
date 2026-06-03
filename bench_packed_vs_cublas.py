"""Verify: does the PACKED warp-specialized fused forward beat cuBLAS GEMM + a
separate (compiled) activation?

Loads the production packed kernel from the swiglu_fused repo and compares its
no-save forward `fused_swiglu_wide_packed` (one wide dot, WARP_SPECIALIZE=True,
GROUP_SIZE_M=32) against cuBLAS `x @ packed_weight` + a @torch.compile activation
over the packed preactivation. Same shape as the rest of this project.

This is the apples-to-apples check the standard-layout `fused_forward.py` could
not win, because its two-dot body cannot warp-specialize.

Run:  python bench_packed_vs_cublas.py
"""

from __future__ import annotations

import importlib.util
import math
import pathlib

import torch
import torch.nn.functional as F
import triton.testing as tt

PKG = pathlib.Path(
    "/data/home/tong/projects/swiglu_fused/swiglu/swiglu_layer/fused_swiglu_wide_packed.py"
)
spec = importlib.util.spec_from_file_location("fused_swiglu_wide_packed", PKG)
fs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fs)

BNH = fs.BLOCK_SIZE_N_HALF


def main():
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    M, K, N = 11136, 3584, 14336      # M, d_model, d_ff per gated branch
    dtype = torch.bfloat16
    print(f"torch {torch.__version__} | triton {fs.triton.__version__} | "
          f"{torch.cuda.get_device_name(0)}")
    print(f"M={M} K={K} N={N} dtype={dtype}  "
          f"(packed: WS={fs.WARP_SPECIALIZE}, GROUP_SIZE_M={fs.GROUP_SIZE_M})\n")

    torch.manual_seed(0)
    x = (torch.randn(M, K, device="cuda", dtype=dtype) * (K ** -0.5)).contiguous()
    w_normal = (torch.randn(K, 2 * N, device="cuda", dtype=dtype) * (K ** -0.5)).contiguous()
    w_packed = fs.pack_swiglu_weight_chunked_torch(w_normal)

    chunks = N // BNH

    # Fairest baseline: STANDARD weight -> standard preact, cheap chunk activation
    # (no gather). This is what you'd run if you weren't using the fused kernel.
    @torch.compile
    def act_std(preact):                # preact [M, 2N] standard [left | gate]
        left, gate = preact.chunk(2, dim=-1)
        return left * F.silu(gate)

    def baseline_std():
        return act_std(x @ w_normal)    # cuBLAS GEMM (standard) + cheap activation

    # Also show the packed-preact baseline (activation must gather -> upper bound).
    @torch.compile
    def act_packed(preact):             # preact packed [left0|gate0|left1|...]
        p = preact.view(M, chunks, 2, BNH)
        return (p[:, :, 0, :].reshape(M, N)) * F.silu(p[:, :, 1, :].reshape(M, N))

    def baseline_packed():
        return act_packed(x @ w_packed)

    def packed_fused():
        return fs.fused_swiglu_wide_packed(x, w_packed)

    # correctness vs the standard baseline
    ref = baseline_std()
    out = packed_fused()
    rel = ((out.float() - ref.float()).abs().max()
           / ref.float().abs().max().clamp_min(1e-6)).item()
    print(f"correctness: packed-fused vs cuBLAS+act  rel_max={rel:.3e}  "
          f"{'OK' if rel <= 2e-2 else 'FAIL'}\n")

    for _ in range(20):
        baseline_std(); baseline_packed(); packed_fused()
    torch.cuda.synchronize()

    q = (0.5, 0.0, 1.0)
    t_gemm, _, _ = tt.do_bench(lambda: x @ w_normal, warmup=50, rep=500, quantiles=q)
    t_std, _, _ = tt.do_bench(baseline_std, warmup=50, rep=500, quantiles=q)
    t_pk, _, _ = tt.do_bench(baseline_packed, warmup=50, rep=500, quantiles=q)
    t_fused, _, _ = tt.do_bench(packed_fused, warmup=50, rep=500, quantiles=q)

    flops = 2 * M * K * (2 * N)
    def tflops(ms):
        return flops / (ms / 1e3) / 1e12

    print(f"{'variant':<40} {'ms (median)':>12} {'TFLOP/s':>9}")
    print("-" * 64)
    print(f"{'cuBLAS GEMM only (floor)':<40} {t_gemm:>12.3f} {tflops(t_gemm):>9.0f}")
    print(f"{'cuBLAS GEMM + @compile act (standard)':<40} {t_std:>12.3f} {tflops(t_std):>9.0f}")
    print(f"{'cuBLAS GEMM + @compile act (packed/gather)':<40} {t_pk:>12.3f} {tflops(t_pk):>9.0f}")
    print(f"{'packed fused (WS, one wide dot)':<40} {t_fused:>12.3f} {tflops(t_fused):>9.0f}")
    print(f"\npacked-fused vs cuBLAS + standard act : {t_std / t_fused:.3f}x  (fairest baseline)")
    print(f"packed-fused vs cuBLAS + packed act   : {t_pk / t_fused:.3f}x")
    print(f"packed-fused vs cuBLAS GEMM-only      : {t_gemm / t_fused:.3f}x")


if __name__ == "__main__":
    main()
