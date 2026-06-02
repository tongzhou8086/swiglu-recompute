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
- Recomputing `h` instead of saving it cuts the per-layer saved footprint by
  ~305 MiB → a win that grows with depth, invisible at a single block.

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

Per-layer accounting for this block (our shape, bf16):

| | Saved (persistent, scales with `L`) | per-layer backward transient (`1 ×`) |
|---|---|---|
| standard autograd | `preact` 609 + `h` 305 = **914 MiB** | ~1× |
| recompute (save `preact`, recompute `h`) | `preact` **609 MiB** | ~1× (a bit larger) |

Recompute cuts the **persistent, depth-multiplied** term by **305 MiB per layer**
(it stops saving `h`), at the cost of a slightly larger **one-layer** transient:

- **`L = 1`** (this benchmark): saves 305 once, costs ~305 once → a wash (or worse
  in eager). This is exactly the "recompute uses more memory" result we saw.
- **`L = 32`** (a real model): saves `305 × 32 ≈ 9.5 GiB` of persistent activation,
  costs `~305 MiB` once → a large net win.

So the recompute idea **is** a memory win — at *model* scale, where the
saved-activation pool is the bottleneck. A one-block micro-benchmark structurally
cannot show it.

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
