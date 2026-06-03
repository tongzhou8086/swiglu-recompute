# B200 results (torch 2.12.0+cu130, bf16)

Re-run of the three benchmarks on an **NVIDIA B200** (vs the original H800
numbers in `README.md`, which were torch 2.10). Shape unchanged:
`M=11136 D=3584 H=14336 Dout=3584`. Run via `srunpy <script>.py` on the
`dedicated` B200 partition.

Timing is the **min over reps** of the mean-per-iter (min is the cleanest
estimator of intrinsic compute time — contention only ever adds time), with
`reps=10` for the stacked/fair benchmarks. Memory is peak alloc delta, min over
reps.

## Correctness

Passes (identical to H800):

| dtype | out | grad_x | grad_W1 | grad_W2 |
|---|---|---|---|---|
| fp32 (tol 1e-5) | 4.5e-7 | 3.0e-7 | 3.3e-7 | 2.3e-7 |
| bf16 (tol 2e-2) | 5.2e-3 | 6.9e-3 | 5.5e-3 | 5.5e-3 |

## Single block (`bench_swiglu_recompute.py`)

| Variant | Peak Δ | fwd | full fwd+bwd |
|---|---|---|---|
| ground-truth (saves `h`) | 2002 MiB | 2.99 ms | 9.37 ms |
| recompute (compiled helpers) | 2275 MiB | 2.64 ms | **8.43 ms** |
| recompute vs gt | +14% mem | 1.13× | **1.11×** |

Memory is **identical to H800** (allocations are arch-independent). Speed: the
whole block is ~1.8× faster than H800 (full 9.4 ms vs 16.9 ms), and recompute's
edge over *eager* gt is a bit larger here (1.11× vs 1.07× on H800).

## Stacked depth (`bench_stacked_blocks.py`, vs eager baseline)

| N | gt mem | rc mem | mem saving | gt ms | rc ms | speedup |
|---|---|---|---|---|---|---|
| 1 | 2002 MiB | 2275 MiB | −14% (worse) | 8.9 | 8.0 | 1.12× |
| 2 | 3297 MiB | 2961 MiB | +10% | 18.2 | 16.3 | 1.12× |
| 4 | 5888 MiB | 4334 MiB | +26% | 37.1 | 32.9 | 1.13× |
| 8 | 11068 MiB | 7078 MiB | +36% | 63.5 | 55.7 | 1.14× |
| 16 | 21429 MiB | 12567 MiB | **+41%** | 120.7 | 104.7 | **1.15×** |

Memory figures match H800 exactly. The recompute speedup vs eager grows with
depth on B200 (up to 1.15× at N=16), a touch higher than H800's flat ~1.08×.

## Fair comparison: compile the baseline too (`bench_fair_compile.py`)

Each variant cell is **peak memory (MiB) / full fwd+bwd time (ms)**. The last
column compares recompute against the compiled baseline: negative mem = recompute
uses *less* memory; speed >1× = recompute *faster*.

| N | gt-eager (MiB / ms) | gt-compiled (MiB / ms) | recompute (MiB / ms) | recompute vs gt-compiled |
|---|---|---|---|---|
| 1 | 2002 / 9.1 | 1699 / 7.8 | 2275 / 7.6 | +34% mem, 1.02× speed |
| 4 | 5888 / 37.5 | 4670 / 32.9 | 4334 / 33.4 | −7% mem, 0.98× speed |
| 8 | 11068 / 63.7 | 8633 / 55.4 | 7078 / 55.7 | −18% mem, 0.99× speed |
| 16 | 21429 / 121.0 | 16558 / 103.8 | 12567 / 104.7 | **−24% mem**, 0.99× speed |

Same qualitative story as H800:

- **Speed: recompute is essentially at parity with the compiled baseline** —
  ~0.98–1.02× (within timing noise, not a real speedup); the earlier "faster" was
  compiled-vs-eager.
- **`torch.compile` alone** already cuts memory for free (Inductor auto-recomputes
  ~one `[M,H]`/block).
- **Manual recompute's value is extra memory at depth** — the reduction vs
  gt-compiled grows monotonically with depth (−7% at N=4, −18% at N=8, −24% at
  N=16), at no meaningful speed cost (parity) — but it is extra memory, not a
  free speedup.

## H800 → B200 summary

- **Memory: unchanged** (same allocations; the conclusions about save-vs-recompute
  and the depth crossover are architecture-independent).
- **Speed: ~1.8× faster wall-clock** across the board on B200.
- **Conclusions identical:** recompute is a memory win only at depth; against a
  compiled baseline it's a memory-at-depth play, not a speed win.
