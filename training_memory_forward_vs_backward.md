# Why forward-saved activations are the training-memory bottleneck (not the backward)

A note on *which* memory matters in training, and why optimizing the backward
pass barely moves peak memory while shrinking forward-saved activations is the
real lever. This is the principle behind activation/gradient checkpointing — and
it explains why the single-block benchmark in this repo was misleading.

## TL;DR

- Peak training memory ≈ **(sum of *all* layers' saved activations)** + **(one
  layer's backward temporaries)**.
- The first term scales with depth (`L ×`) and is the bottleneck. The second is
  a bounded `1 ×` constant.
- So a wasteful backward costs you a constant; a wasteful (large) *saved*
  activation costs you `L ×`. Optimize what the **forward saves**.
- Recomputing the SwiGLU activation instead of saving it cuts the per-layer
  saved footprint by **~610 MiB** (two `[M,H]` tensors — `h` *and* `silu(gate)`;
  see below) → a win that grows with depth, invisible at a single block.
- With the elementwise helpers `@torch.compile`d, recompute is also ~8% *faster*
  than standard autograd at every depth — so at depth it is a Pareto win (less
  memory *and* faster).

## Two kinds of memory, with very different lifetimes

1. **Saved activations (from the forward pass).** Tensors autograd stashes during
   forward to reuse in backward. They are **pinned alive from when they are
   computed until backward consumes them.**
2. **Backward temporaries.** Gradients and scratch created while computing a
   layer's backward. **Transient**: born when that layer is processed, freed as
   soon as backward moves on.

The decisive difference is **accumulation**.

## The timeline

Backward runs in **reverse layer order**, so every layer's saved activation
piles up during forward and **cannot be freed until backward reaches that
layer** — for layer 1, that's the very end. At the forward→backward boundary,
**all `L` layers' saved activations are alive at once.**

```
memory
  │                              ┌─ PEAK: all L layers' saved activations
  │                         ____/   + the 1st backward layer's temporaries
  │                    ____/    \___
  │               ____/             \___        backward frees saved
  │          ____/                      \___    activations as it goes,
  │     ____/                               \__ faster than it allocates
  │ ___/                                       \___ temporaries
  └────────────────────────┼───────────────────────► time
     forward L1→L2→…→LL     │     backward LL→…→L2→L1
```

- **Forward** is a *staircase up*: each layer adds its saved activation to a pool
  that grows to `Σ (per-layer saved) ≈ L × per-layer`.
- **Backward** is a *staircase down*: each layer frees its (now-consumed) saved
  activation and allocates transient gradients. Because it frees roughly as much
  as it allocates per step, total memory **decreases**.

So:

```
peak ≈ Σ_layers (saved activations)   +   max_layer (backward temporaries)
        \__________ L × ___________/       \________ 1 × ________/
              the bottleneck                   a bounded constant
```

## Why the backward "doesn't matter"

Two reasons:

1. **It doesn't accumulate.** At any instant you are doing exactly *one* layer's
   backward; its scratch is freed before the next layer. So over-allocating in
   backward adds a `1 ×` constant, not an `L ×` term.
2. **It reuses freed space.** By the time you allocate a layer's backward
   temporaries, you have **already freed that layer's saved activation**. The
   backward runs in the room the forward is giving back.

That is the colleague's point precisely: a layer's backward results are thrown
away immediately, so being wasteful there is cheap; a saved activation must
survive the *entire* forward and most of the backward, and it is multiplied by
depth.

## This is why gradient checkpointing exists

Activation checkpointing trades **more backward recompute** (transient, cheap in
the memory accounting) for **not storing forward activations** (persistent, the
expensive `L ×` term). Same principle: spend backward to shrink forward-saved.

## Reconciling with this repo's single-block benchmark

`bench_swiglu_recompute.py` measures a **single block** (`L = 1`): forward, then
*immediately* backward. With `L = 1` there is no staircase — the saved pool
(`1 ×`) and the backward temporaries (`1 ×`) are the same order of magnitude, so
the backward dominated the measured peak. That is an artifact of `L = 1`, not the
real tradeoff.

Per-layer accounting for this block (our shape, bf16; full derivation in
"What the forward actually saves" below):

| | Saved (persistent, scales with `L`) | per-layer backward transient (`1 ×`) |
|---|---|---|
| standard autograd | `x` 76 + `preact` 609 + `silu(gate)` 305 + `h` 305 = **~1295 MiB** | ~1× |
| recompute (save `x`, `preact`; recompute the rest) | `x` 76 + `preact` 609 = **~686 MiB** | ~1× (a bit larger) |

Recompute cuts the **persistent, depth-multiplied** term by **~610 MiB per layer**
(it stops saving both `h` and `silu(gate)`), at the cost of a slightly larger
**one-layer** transient:

- **`L = 1`** (this benchmark): saves ~610 once, costs ~880 once → a *wash, or
  worse*. This is exactly the "recompute uses more memory" result we saw.
- **`L = 32`** (a real model): saves `~610 × 32 ≈ 19 GiB` of persistent
  activation, costs ~880 MiB once → a large net win.

So the recompute idea **is** a memory win — at *model* scale, where the
saved-activation pool is the bottleneck. A one-block micro-benchmark structurally
cannot show it.

## Empirical confirmation: stacking N blocks

`bench_stacked_blocks.py` stacks `N` SwiGLU MLP blocks (each `dim -> hidden ->
dim`), runs one forward then one backward, and measures peak alloc for the
standard-autograd stack vs the recompute stack (H800, torch 2.10, bf16,
M=11136 D=3584 H=14336):

```
  N |   gt mem    rc mem    saving    % |   gt ms   rc ms  rc/gt
  1 |    2002M     2275M     -273M  -14% |   16.8    15.6   1.08x
  2 |    3297M     2961M      336M   10% |   33.8    31.2   1.08x
  4 |    5888M     4334M     1554M   26% |   69.4    64.6   1.07x
  8 |   11068M     7078M     3990M   36% |  124.4   115.3   1.08x
 16 |   21429M    12567M     8862M   41% |  243.0   223.7   1.09x
```

- **Memory:** at `N = 1` recompute *loses* (−14%) — the single-block peak is
  dominated by the backward transient, not saved activations. It crosses over at
  `N = 2` and the win grows ~linearly with depth — **41% less peak at 16 blocks**,
  and the fraction keeps rising (toward the asymptote derived below).
- **Speed:** recompute is **~7–9% faster at every depth** (`rc/gt ≈ 1.08x`). The
  `@torch.compile`d elementwise helpers fuse each pointwise chain into one kernel
  (fewer launches, less HBM traffic), which outweighs the extra recompute work —
  a per-block edge that holds as you stack. So at depth, recompute is a **Pareto
  win: faster *and* lower peak memory.**

This is the staircase argument made concrete: the memory win is invisible at one
block and emerges only once the persistent saved-activation pool (which scales
with depth) dominates the peak.

## What the forward actually saves (deriving 1295 vs 686)

Per-block peak grows ~linearly: ~**1295 MiB/block** (ground-truth) vs ~**686
MiB/block** (recompute). Both numbers fall straight out of *what autograd saves
for backward* — which is not obvious from the source, because **a tensor op under
autograd does two things**: the numeric op (the NumPy-like part) *and* graph
bookkeeping that **saves whichever operands its backward formula needs**. What's
saved is decided by each op's derivative, not by the names in your code:

| forward op | backward needs | so it saves |
|---|---|---|
| `preact = x @ W1ᵀ` | `grad_W1 = grad_preactᵀ @ x` | `x` |
| `s = silu(gate)` | `grad_gate = g·silu'(gate)` | `gate` (a view of `preact`) |
| `h = left * s` | `grad_left = g·s`, `grad_s = g·left` | **`left` *and* `s = silu(gate)`** |
| `y = h @ W2ᵀ` | `grad_W2 = grad_yᵀ @ h` | `h` |

The non-obvious one is `silu(gate)`: it is anonymous in `h = left * F.silu(gate)`,
but the **multiply's backward needs both operands**, so autograd pins it alive.
(Verified with `torch.autograd.graph.saved_tensors_hooks`: the saved set contains
a distinct `[M,H]` storage that is neither `x`, `preact`, `h`, nor a weight.)

Summing the distinct saved **activation** storages (bf16, our shape):

| saved activation | size |
|---|---|
| `x` `[M,D]` | 76.1 MiB |
| `preact` `[M,2H]` (held alive via the `left`/`gate` views) | 609.0 MiB |
| `silu(gate)` `[M,H]` | 304.5 MiB |
| `h` `[M,H]` | 304.5 MiB |
| **ground-truth total** | **1294.1 ≈ 1295** |
| recompute keeps only `x` + `preact` | **685.1 ≈ 686** |

Difference = `silu(gate)` + `h` = `2 × [M,H]` = **609 MiB/block** — exactly the
measured slope gap. Recompute keeps just `preact` and rebuilds `silu(gate)` and
`h` from it in backward with cheap (fused) elementwise math.

## Why the reduction % grows with depth — and where it saturates

Peak memory is **affine** in N (a slope *and* an intercept), so the reduction
*fraction* is a ratio of two lines, not a constant. Fitting the data:

```
peak_ground_truth(N) ≈ 1295·N + 709     MiB
peak_recompute(N)    ≈  686·N + 1591    MiB
saving(N)            ≈  609·N − 882      = (slope gap)·N − (fixed tax)
reduction%(N)        = saving / peak_gt
```

- **Slope** = per-block *forward-saved* activation (the breakdown above). Recompute's
  is smaller (686 vs 1295) → this is the depth-multiplied win.
- **Intercept** = the N-independent part ≈ the **transient working set of one
  block's backward** (the block being processed when the peak occurs) + small
  fixed buffers. Recompute's is *larger* (1591 vs 709) because its single-block
  backward recomputes `h`/`silu` and builds `grad_preact`, where autograd just
  reads saved tensors. That `+882` is a **fixed tax paid once**.

So at small N the (worse) intercept dominates → recompute loses; as N grows the
(better) slope dominates and the fraction climbs toward the **slope ratio**:

```
reduction%(N) → 609 / 1295 ≈ 47%   as N → ∞
```

matching the trend (−14% → +10% → +26% → +36% → +41% → ~47%). Two consequences:

- The percentage **saturates at ~47%, not 100%** — recompute still keeps `preact`
  + `x` (~686 MiB/block); you cannot save what you still store.
- **Absolute** saving grows linearly forever (~609 MiB/block). Practically you
  eventually OOM; counting params (~392 MiB/block), recompute fits roughly **1.5×
  more layers** before that. To push the asymptote higher, recompute *more* (save
  only `x`, recompute `preact` too — full activation checkpointing), trading the
  `W1` matmul recompute for a smaller slope.

## One precision

"The backward doesn't matter *at all*" is ~99% true but not literal: the peak does
include one layer's backward temporaries (the `+ 1 ×` term). It simply does not
*accumulate*, so it is a constant, not the bottleneck.

## Practical takeaways

- To cut training peak memory, reduce **what the forward saves per layer** (don't
  store `h`; store the smaller `preact`; or checkpoint and store almost nothing).
- Do **not** spend effort micro-optimizing backward temporaries for memory — they
  are a bounded constant and reuse freed space. (Optimize backward for *speed* if
  needed, a separate axis.)
- To *measure* a recomputation memory win, stack `N` blocks (a realistic forward
  followed by one backward) so the saved-activation staircase appears; a single
  block hides it.
