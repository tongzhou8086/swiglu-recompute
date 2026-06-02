# swiglu-recompute

A standalone experiment: a SwiGLU MLP that **recomputes the activation
`h = left · silu(gate)` in the backward pass** instead of saving it, to probe
whether selective activation recomputation reduces peak training memory.

Pure PyTorch — no external project dependencies. Requires a CUDA GPU.

```
x [M, D]  --W1[2H,D]-->  preact [M, 2H]  --SwiGLU-->  h [M, H]  --W2[Dout,H]-->  y [M, Dout]
```

The forward saves `preact` but not `h`; the backward recomputes `h` (and the
SiLU factors) from `preact`. `preact` is saved as an internal local (never
returned as a graph output), avoiding the autograd "graph-output gets a
protected copy" pitfall.

## Files

- `swiglu_recompute.py` — `SwiGLUMemoryOptimizedFunction` (the recompute
  `autograd.Function`), `SwiGLUMLPCustom`, and a standard `SwiGLUMLPGroundTruth`.
- `bench_swiglu_recompute.py` — correctness (fp32 + bf16) vs the standard MLP,
  plus peak-memory and timing at a configurable shape.

## Run

```bash
python bench_swiglu_recompute.py
# or a custom shape:
python bench_swiglu_recompute.py --m 11136 --d 3584 --h 14336 --dout 3584
```

## Status / finding (H800, torch 2.10, bf16, M=11136 D=3584 H=14336 Dout=3584)

**Correctness: passes** (fp32 rel_max ≤ 4e-7, bf16 ≤ 7e-3).

**Memory + speed (one fwd+bwd):**

| Variant | Peak Δ | Full fwd+bwd |
|---|---|---|
| ground-truth (standard autograd, saves `h`) | 2002 MiB | 16.9 ms |
| recompute — eager helpers | 3526 MiB | 19.5 ms |
| recompute — `@torch.compile` helpers | **2275 MiB** | **15.9 ms** |

Two findings:

1. **The naive eager recompute uses *more* memory, not less** (3526 vs 2002 MiB).
   Not saving `h` avoids ~305 MiB, but the hand-written backward materializes
   ~5–6 full-size `[M, H]` temporaries at once (`sigmoid`, `silu`, `silu'`,
   `grad_left`, `grad_gate`, plus a `torch.cat` to `[M, 2H]`) — a transient
   spike of ~1.5 GiB that dwarfs the saving.

2. **`@torch.compile` on the two elementwise helpers fixes most of it.** Inductor
   fuses each pointwise chain into a single kernel, so those temporaries are
   never materialized: peak drops 3526 → 2275 MiB and the block becomes ~7%
   *faster* than standard autograd (full 1.07x). Two decorators, no other code
   change.

**Takeaway:** even compiled, recompute still uses ~273 MiB *more* peak memory
than plain autograd at this shape — not saving `h` (305 MiB) is offset by
recomputing it (plus `grad_h`, `grad_preact`) in backward. So here recompute is
a **speed win, not a memory win**. Note this is a *single block* (fwd then
immediately bwd); recomputation's real memory payoff is across **many layers**
(gradient checkpointing — not keeping every layer's activations alive through a
deep forward), which a one-block micro-benchmark does not capture. To actually
beat autograd on memory for one block you'd need a fused backward (e.g. a Triton
kernel writing `grad_preact` in place with no intermediate buffers).

For *why* forward-saved activations (not the backward) set the training-memory
peak — and why this single-block benchmark is misleading — see
[`training_memory_forward_vs_backward.md`](./training_memory_forward_vs_backward.md).

## Stacked-depth result (the real win)

`bench_stacked_blocks.py` stacks `N` blocks (forward through all, then one
backward) and measures peak memory. The recompute win is invisible at one block
but grows ~linearly with depth:

| N | gt mem | rc mem | mem saving | gt ms | rc ms | speedup |
|---|---|---|---|---|---|---|
| 1 | 2002 MiB | 2275 MiB | −14% (worse) | 16.8 | 15.6 | 1.08× |
| 2 | 3297 MiB | 2961 MiB | +10% | 33.8 | 31.2 | 1.08× |
| 4 | 5888 MiB | 4334 MiB | +26% | 69.4 | 64.6 | 1.07× |
| 8 | 11068 MiB | 7078 MiB | +36% | 124.4 | 115.3 | 1.08× |
| 16 | 21429 MiB | 12567 MiB | **+41%** | 243.0 | 223.7 | 1.09× |

```bash
python bench_stacked_blocks.py            # default N = 1,2,4,8,16
```

- **Memory (vs *eager* autograd):** recompute loses at one block but the win
  grows with depth (41% less at N=16). Per-block: ~1295 MiB (eager) vs ~686 MiB
  (recompute) — recompute avoids ~610 MiB/block ≈ *two* `[M,H]` tensors (eager
  saves both `silu(gate)` and `h`; recompute rebuilds both from `preact`).

> [!IMPORTANT]
> The table above compares against an **eager** baseline. That is *not*
> apples-to-apples on speed, because recompute uses `@torch.compile`d helpers.
> See the fair comparison below.

## Fair comparison: compile the baseline too (`bench_fair_compile.py`)

When you `torch.compile` a *standard* module, AOTAutograd + Inductor generate the
backward and the min-cut partitioner **automatically recomputes** cheap
activations. Against that fair baseline (H800, bf16):

| N | gt-eager | **gt-compiled** | recompute | recompute vs gt-compiled |
|---|---|---|---|---|
| 1 | 2002 MiB / 16.9 ms | 1699 / 15.3 | 2275 / 15.7 | +34% mem, 0.97× speed |
| 4 | 5887 / 70.4 | 4670 / 62.7 | 4334 / 65.3 | −7% mem, 0.96× speed |
| 8 | 11068 / 124.3 | 8633 / 113.7 | 7078 / 115.2 | **−18% mem**, 0.99× speed |

- **Speed: recompute is *not* faster** — it is ~1–4% *slower* than the compiled
  baseline. The earlier "~8% faster" was purely *compiled-vs-eager*.
- **`torch.compile` alone** already cuts memory a lot (N=8: −22% vs eager) for
  free — Inductor auto-recomputes ~one `[M,H]`/block (slope ~991 vs eager 1295).
- **Manual recompute** still saves *more* at depth (−18% at N=8, growing): it
  recomputes *both* `[M,H]` tensors (slope ~686) where Inductor keeps one. Its
  value is extra memory at depth, at a small speed cost — not a free speedup.

See [`training_memory_forward_vs_backward.md`](./training_memory_forward_vs_backward.md)
for the full breakdown (what autograd saves, the affine slope/intercept model,
the reduction-% asymptote, and this fair-comparison analysis).
