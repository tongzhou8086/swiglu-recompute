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
