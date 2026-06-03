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

Full MLP (W1-proj + activation + W2-proj), one fwd+bwd, `GROUP_SIZE_M=16` (tuned):

| Variant | Peak | fwd ms | full ms |
|---|---|---|---|
| ground_truth (cuBLAS GEMM + eager pointwise) | 2002 MiB | 2.986 | 9.199 |
| recompute (cuBLAS GEMM + `@torch.compile` pointwise) | 2275 MiB | **2.633** | **8.407** |
| **fused** (Triton GEMM + in-register activation) | 2275 MiB | 3.060 | 8.790 |
| **fused vs recompute** | +0 MiB | **0.86×** | **0.96×** |

## Finding: correct & memory-neutral, but a forward *slowdown* on B200

- **Memory is unchanged** vs the recompute path (2275 MiB, exact). Expected: the
  fused kernel still materializes `preact[M,2H]` and `h[M,H]`. The fusion only
  removes the *re-read* of `preact` from HBM — a bandwidth/launch saving, never a
  memory one.
- **The fusion is a net speed loss here.** Fusing eliminates the separate
  activation pass (one `[M,2H]` read + `[M,H]` write ≈ 0.9 GiB) — but our
  standard-layout Triton GEMM is slower than cuBLAS by *more* than that saving.

## Tuning: `GROUP_SIZE_M` and warp specialization (`tune_fused_forward.py`)

Isolating just the W1-projection + activation (no W2), vs cuBLAS baselines:

```
baseline  cuBLAS GEMM + @compile pointwise : 1.534 ms
baseline  cuBLAS GEMM only (floor)         : 1.556 ms

WS      G | status     fwd ms  vs base
False   1 | ok          2.475   0.620x
False   4 | ok          2.057   0.746x
False   8 | ok          1.983   0.774x
False  16 | ok          1.979   0.775x   <- best
True  1-16| ws-fail        -        -
```

- **`GROUP_SIZE_M=16` is best** (1.979 ms; vs 1.983 at 8, 2.057 at 4, 2.475 at 1) —
  a real but small L2-swizzle win. Now the default.
- **Warp specialization cannot be enabled.** `warp_specialize=True` fails to
  compile for the standard-layout **two-dot** body (`TritonGPUAutomaticWarpSpecialization`
  MLIR error). A single-wide-dot variant (build the `[BK,2·BN]` tile with
  `tl.cat`) — which *would* be WS-partitionable — also fails: `tl.cat`
  concatenates along axis 0, so it can't form the wide column tile without the
  pre-packed weight layout. So on the standard layout, WS is unavailable in
  triton 3.7.
- **WS wouldn't close the gap anyway.** cuBLAS GEMM-only is 1.556 ms at ~73% of
  B200 BF16 peak, so a Triton kernel at `fused_swiglu`'s WS-tuned ~58% of peak
  would take ≈ 1.556 × (73/58) ≈ **1.96 ms** — essentially what our *non-WS*
  two-dot kernel at `G=16` already hits (1.98 ms). We're already at the
  throughput a warp-specialized Triton kernel reaches at this shape; cuBLAS is
  simply faster than Triton for this GEMM. Even the best fused config is slower
  than cuBLAS **GEMM-alone** (1.98 vs 1.56 ms) — before counting the activation.

## Takeaway

The fusion is correct, clean, and memory-neutral, and the *idea* (skip the
`preact` re-read) is sound — but on B200 at this shape the bottleneck is GEMM
throughput, and a hand-written Triton GEMM does not beat cuBLAS here. Warp
specialization is not the missing lever (we already match its achievable
throughput, and it can't be enabled on the standard layout anyway). The
cuBLAS-GEMM + `@torch.compile`-pointwise path (the existing `recompute` variant)
remains the fastest forward.

(The packed single-wide-dot layout — `fused_swiglu`'s approach — was not
implemented; this experiment deliberately stayed in the standard layout. Given
`fused_swiglu`'s own ~58%-vs-73% result, it would likely also trail cuBLAS at
this shape, with its real payoff being the *factor side-store* that cheapens the
backward, not the forward.)
