# B200 results (torch 2.12.0+cu130, bf16)

Re-run of the three benchmarks on an **NVIDIA B200** (vs the original H800
numbers in `README.md`, which were torch 2.10). Shape unchanged:
`M=11136 D=3584 H=14336 Dout=3584`. Run via `srunpy <script>.py` on the
`dedicated` B200 partition.

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
| 1 | 2002 MiB | 2275 MiB | −14% (worse) | 9.0 | 8.0 | 1.13× |
| 2 | 3297 MiB | 2961 MiB | +10% | 18.3 | 16.5 | 1.11× |
| 4 | 5888 MiB | 4334 MiB | +26% | 37.4 | 33.3 | 1.12× |
| 8 | 11068 MiB | 7078 MiB | +36% | 63.5 | 55.4 | 1.15× |
| 16 | 21429 MiB | 12567 MiB | **+41%** | 121.4 | 104.0 | **1.17×** |

Memory figures match H800 exactly. The recompute speedup vs eager grows with
depth on B200 (up to 1.17× at N=16), a touch higher than H800's flat ~1.08×.

## Fair comparison: compile the baseline too (`bench_fair_compile.py`)

| N | gt-eager | gt-compiled | recompute | recompute vs gt-compiled |
|---|---|---|---|---|
| 1 | 2002 MiB / 9.0 ms | 1699 / 7.8 | 2275 / 8.2 | +34% mem, 0.95× speed |
| 4 | 5888 / 37.3 | 4670 / 32.7 | 4334 / 33.1 | −7% mem, 0.99× speed |
| 8 | 11068 / 63.7 | 8633 / 55.5 | 7078 / 56.0 | −18% mem, 0.99× speed |
| 16 | 21429 / 120.8 | 16558 / 102.7 | 12567 / 104.3 | **−24% mem**, 0.99× speed |

Same qualitative story as H800:

- **Speed: recompute is not faster than the compiled baseline** — ~0.95–0.99×
  (slightly slower), the earlier "faster" was compiled-vs-eager.
- **`torch.compile` alone** already cuts memory for free (Inductor auto-recomputes
  ~one `[M,H]`/block).
- **Manual recompute's value is extra memory at depth** — the reduction vs
  gt-compiled grows monotonically with depth (−7% at N=4, −18% at N=8, −24% at
  N=16), at a ~1% speed cost — not a free speedup.

## H800 → B200 summary

- **Memory: unchanged** (same allocations; the conclusions about save-vs-recompute
  and the depth crossover are architecture-independent).
- **Speed: ~1.8× faster wall-clock** across the board on B200.
- **Conclusions identical:** recompute is a memory win only at depth; against a
  compiled baseline it's a memory-at-depth play, not a speed win.
