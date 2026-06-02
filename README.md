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

- **Correctness: passes** (fp32 rel_max ≤ 3e-7, bf16 ≤ 5e-3).
- **Memory: the naive recompute uses *more* peak memory, not less.**

  | | Peak Δ (one fwd+bwd) |
  |---|---|
  | ground-truth (standard autograd, saves `h`) | ~2002 MiB |
  | recompute (saves `preact`, recomputes `h`) | ~3526 MiB |

  Not saving `h` avoids ~305 MiB, but the hand-written backward materializes
  ~5–6 full-size `[M, H]` temporaries at once (`sigmoid`, `silu`, `silu'`,
  `grad_left`, `grad_gate`, plus a `torch.cat` to `[M, 2H]`), a transient spike
  of ~1.5 GiB. PyTorch's autograd backward uses fused kernels that avoid those
  temporaries, so the naive recompute loses more than it saves.

**Takeaway:** recomputation only pays off if the backward is itself
memory-lean. Next steps: write the backward to minimize/reuse buffers (in-place
SiLU factors, write directly into a preallocated `grad_preact` instead of
`torch.cat`, fuse the elementwise chain), and/or move it to a fused kernel.
