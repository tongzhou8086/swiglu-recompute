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
- **Warp specialization cannot be enabled *on the standard layout*.**
  `warp_specialize=True` fails to compile for the **two-dot** body
  (`TritonGPUAutomaticWarpSpecialization` MLIR error). A single-wide-dot variant
  built with `tl.cat` — which *would* be WS-partitionable — also fails: `tl.cat`
  concatenates along axis 0, so it can't form the `[BK,2·BN]` wide tile without
  the pre-packed weight layout. So WS is unavailable here in triton 3.7.

## The packed layout *does* win — verified (`bench_packed_vs_cublas.py`)

> [!IMPORTANT]
> An earlier version of this note claimed WS "wouldn't close the gap anyway"
> (citing cuBLAS ~73% / Triton ~58% of peak). **That was wrong.** Measured
> directly on B200, the packed warp-specialized kernel from `fused_swiglu`
> (`fused_swiglu_wide_packed`: one wide dot, `WARP_SPECIALIZE=True`,
> `GROUP_SIZE_M=32`) **beats cuBLAS + a separate activation** at this shape:

| variant (W1-proj + activation only) | ms | TFLOP/s | % of ~2250 peak |
|---|---|---|---|
| cuBLAS GEMM only (floor) | 1.587 | 1442 | ~64% |
| cuBLAS GEMM + `@compile` act (standard, fairest) | 1.785 | 1282 | — |
| cuBLAS GEMM + `@compile` act (packed/gather) | 1.790 | 1279 | — |
| **packed fused (WS, one wide dot)** | **1.677** | 1364 | ~61% |

- **packed-fused is 1.064× faster than cuBLAS + a separate activation** (vs the
  *fairest* standard-weight baseline; the packed/gather baseline is the same
  1.067×), and only 5.4% slower than the raw cuBLAS GEMM — the activation is
  absorbed almost for free in the epilogue. (The standard vs packed activation
  baselines are within noise, 1.785 vs 1.790 — the gather isn't the cost; the
  separate-kernel launch + `preact` round-trip is.)
- cuBLAS (~63% peak) and the packed-WS Triton GEMM (~61%) are **nearly equal** on
  raw throughput; the 58/73 figures cited before did not apply to this kernel.
  Fusing the activation more than covers the tiny GEMM gap.
- The packed kernel (1.667 ms) is **1.19× faster than our standard-layout best**
  (1.979 ms): the single wide dot is what makes WS legal, and WS is what closes
  the gap to cuBLAS.

## Integrated packed variant in this project (`fused_forward_packed.py`)

`fused_forward_packed.py` brings the packed kernel into the recompute scheme: the
forward emits `(h, preact_packed)` from one warp-specialized launch (storing raw
packed `preact`, not factors), and the backward recomputes `h` and `grad_preact`
from the packed `preact` (reusing the packed `grad_de`-from-preact kernel + packed
GEMMs). Full-MLP 4-way comparison, one fwd+bwd (`bench_fused_forward.py`):

| variant | peak | fwd ms | full ms | vs recompute (fwd / full) |
|---|---|---|---|---|
| ground_truth (cuBLAS + eager) | 2002 MiB | 2.964 | 9.350 | — |
| recompute (cuBLAS + `@compile`) | 2275 MiB | 2.625 | 8.478 | 1.00× / 1.00× |
| fused-std (standard layout) | 2275 MiB | 3.045 | 8.819 | 0.86× / 0.96× |
| **fused-packed (WS)** | 2275 MiB | **2.597** | **8.454** | **1.01× / 1.00×** |

Correctness passes for both fused variants (fp32 ≤ 7.8e-6, bf16 ≤ 6.9e-3; packed
grad compared after unpacking).

- **The packed variant reaches parity / a slight edge** over the cuBLAS+compiled
  recompute path (1.011× fwd, 1.003× full) — and it is **memory-neutral**.
- This is the clean flip from the standard-layout variant (0.86× fwd): same
  fusion, same recompute backward, only the **layout + warp specialization**
  differ. The forward-only edge (~1.06× on W1-proj+activation, see above) dilutes
  to ~1.01× across the full MLP because the shared W2 GEMM and the (identical)
  backward dominate the rest.

## Takeaway (corrected)

The fusion idea (skip the `preact` re-read / absorb the activation in the
epilogue) **is** a real forward win — but **only with the packed layout +
warp specialization**, which lets the GEMM match cuBLAS while folding in the
activation. The **standard layout cannot get there**: its two-dot body can't
warp-specialize, so its GEMM trails cuBLAS by more than the fusion saves, and the
cuBLAS-GEMM + `@torch.compile`-pointwise path beats it. The packed variant
(`fused_forward_packed.py`) closes the gap — parity-to-slightly-faster on the full
MLP, memory-neutral. So the layout choice was the deciding factor, not the fusion
idea itself. (Memory is unchanged across all variants; this is purely a forward
speed/bandwidth story — the recompute memory win still comes from depth, per the
main README.)
