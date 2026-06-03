# Fused-forward kernel result (B200, torch 2.12.0+cu130 / triton 3.7.0, bf16)

`fused_forward.py` fuses the SwiGLU forward projection + activation into one
Triton kernel that emits both `preact [M,2H]` and `h [M,H]` from a single matmul
(the GEMM accumulator stays in registers; `h = left·silu(gate)` is computed from
it). This mirrors `_fused_swiglu_wide_packed_save_factors_kernel` in
`~/projects/fused_swiglu_kernel`, but uses the **standard (non-packed) weight
layout** — two narrow `B`-loads into two accumulators instead of one wide load —
so `preact` stays in standard `[M,2H]` form and the existing PyTorch backward
(recompute `h` from `preact`) works unchanged.

Shape `M=11136 D=3584 H=14336 Dout=3584`.

## Correctness — PASS

vs `SwiGLUMLPGroundTruth`: fp32 `rel_max ≤ 7e-6` (kernel uses `input_precision="ieee"`),
bf16 `rel_max ≤ 5.5e-3`. (out, grad_x, grad_W1, grad_W2 all checked; grad is
w.r.t. `Wt = W1.t()`, compared transposed.)

## Memory + speed (one fwd+bwd)

| Variant | Peak | fwd ms | full ms |
|---|---|---|---|
| ground_truth (cuBLAS GEMM + eager pointwise) | 2002 MiB | 3.004 | 9.351 |
| recompute (cuBLAS GEMM + `@torch.compile` pointwise) | 2275 MiB | **2.499** | **8.451** |
| **fused** (Triton GEMM + in-register activation) | 2275 MiB | 3.087 | 8.885 |
| **fused vs recompute** | +0 MiB | **0.81×** | **0.95×** |

## Finding: correct & memory-neutral, but a forward *slowdown* on B200

- **Memory is unchanged** vs the recompute path (2275 MiB, exact). Expected: the
  fused kernel still materializes `preact[M,2H]` and `h[M,H]`. The fusion only
  removes the *re-read* of `preact` from HBM — a bandwidth/launch saving, never a
  memory one.
- **The fusion is a net speed loss here.** Fusing eliminates the separate
  activation pass (one `[M,2H]` read + `[M,H]` write ≈ 0.9 GiB) — but our
  standard-layout Triton GEMM is slower than cuBLAS by *more* than that saving.
  Result: fused fwd is 0.81× the recompute path (and even slower than the
  ground-truth eager fwd, 3.087 vs 3.004 ms), so the W1 GEMM itself is the
  bottleneck, not the activation.
- **Why the GEMM trails cuBLAS:** the standard-layout split needs two `tl.dot`s
  per K-step (one for `left` cols `[0,H)`, one for `gate` cols `[H,2H)`).
  Triton 3.7's automatic warp-specialization pass **cannot partition that
  two-dot body** (it fails to compile with `warp_specialize=True`), so the
  kernel runs without the producer/consumer warp split that B200 needs to hide
  TMA + MMA latency. This matches `fused_swiglu`'s own measurement: even its
  tuned (packed, warp-specialized) fused kernel reaches only ~58% of B200 BF16
  peak vs cuBLAS's ~73% — i.e. a hand-written fused GEMM does not beat cuBLAS on
  raw throughput at this shape.

## Takeaway

The fusion is correct and clean, and the *idea* (skip the `preact` re-read) is
sound, but on B200 at this shape the saving is small relative to cuBLAS's GEMM
advantage. To make the fused forward a win you'd need the GEMM to be competitive
with cuBLAS, which requires **warp specialization** — and that, in turn, wants
the **packed single-wide-dot layout** (`fused_swiglu`'s approach), not the
standard two-dot split. With the standard layout, the cuBLAS-GEMM +
`@torch.compile`-pointwise path (the existing `recompute` variant) is faster.

(The packed-layout variant — which *could* use warp specialization — was not
implemented; this experiment deliberately stayed in the standard layout.)
