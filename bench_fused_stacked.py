"""Stacked-depth fair comparison INCLUDING the fused variants.

A single block is misleading on memory: the recompute/fused variants save
`preact[M,2H]` per block, so they look *worse* at N=1 but win as depth grows.
This stacks N blocks (forward through all, one backward) and reports peak memory
+ full fwd+bwd time for five variants against two baselines (cf.
`bench_fair_compile.py`):

  gt-eager     : standard SwiGLU MLP, eager autograd               (saves ~2 [M,H]/block)
  gt-compiled  : torch.compile(standard MLP)  <- fair baseline     (Inductor auto-recompute)
  recompute    : SwiGLUMLPCustom (save preact, recompute h)        (saves preact [M,2H]/block)
  fused-std    : SwiGLUMLPFused (standard-layout fused fwd)          "
  fused-packed : SwiGLUMLPPackedFused (packed + warp-specialized)    "

The three save-preact variants should track each other on memory and beat
gt-compiled at depth; fused-packed should also be competitive on time.

Run:  python bench_fused_stacked.py            # N = 1,2,4,8,16
"""

from __future__ import annotations

import argparse
import sys
import pathlib

import torch

THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from swiglu_recompute import SwiGLUMLPCustom, SwiGLUMLPGroundTruth  # noqa: E402
from fused_forward import SwiGLUMLPFused  # noqa: E402
from fused_forward_packed import SwiGLUMLPPackedFused  # noqa: E402
from bench_stacked_blocks import Stack, measure_peak, time_full, sync_clean, mib  # noqa: E402

VARIANTS = ["gt-eager", "gt-compiled", "recompute", "fused-std", "fused-packed"]
BLOCK_CLS = {
    "gt-eager": SwiGLUMLPGroundTruth,
    "gt-compiled": SwiGLUMLPGroundTruth,
    "recompute": SwiGLUMLPCustom,
    "fused-std": SwiGLUMLPFused,
    "fused-packed": SwiGLUMLPPackedFused,
}


def build(name, N, D, H, dtype):
    stack = Stack(BLOCK_CLS[name], N, D, H, dtype).cuda()
    if name == "gt-compiled":
        return torch.compile(stack)
    return stack


def measure_for_N(N, M, D, H, dtype):
    x0 = torch.randn(M, D, device="cuda", dtype=dtype)
    gout = torch.randn(M, D, device="cuda", dtype=dtype)
    mem, t = {}, {}
    for name in VARIANTS:
        model = build(name, N, D, H, dtype)
        mem[name] = measure_peak(model, x0, gout)
        t[name] = time_full(model, x0, gout, reps=5)
        del model
        sync_clean()
    del x0, gout
    sync_clean()
    return mem, t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=11136)
    p.add_argument("--d", type=int, default=3584)
    p.add_argument("--h", type=int, default=14336)
    p.add_argument("--ns", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    args = p.parse_args()

    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    dtype = torch.bfloat16
    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
    print(f"M={args.m} D={args.d} H={args.h} dtype={dtype}")
    print(f"per-block preact [M,2H] = {mib(args.m * 2 * args.h * dtype.itemsize):.0f} MiB\n")

    mems, times = {}, {}
    for N in args.ns:
        mems[N], times[N] = measure_for_N(N, args.m, args.d, args.h, dtype)

    def table(title, data, unit, fmt):
        print(f"  [{title}]")
        hdr = f"{'N':>3} | " + " ".join(f"{v:>13}" for v in VARIANTS)
        print(hdr)
        print("-" * len(hdr))
        for N in args.ns:
            row = " ".join(f"{fmt.format(data[N][v]):>13}" for v in VARIANTS)
            print(f"{N:>3} | {row}")
        print()

    table("peak memory (MiB)", mems, "MiB", "{:.0f}")
    table("full fwd+bwd (ms)", times, "ms", "{:.1f}")

    # Headline: fused-packed vs the two baselines, per N.
    print("  [fused-packed vs baselines]")
    hdr = (f"{'N':>3} | {'mem vs eager':>13} {'mem vs comp':>12} | "
           f"{'time vs eager':>13} {'time vs comp':>12}")
    print(hdr)
    print("-" * len(hdr))
    for N in args.ns:
        m, t = mems[N], times[N]
        fp_m, fp_t = m["fused-packed"], t["fused-packed"]
        me = 100 * (m["gt-eager"] - fp_m) / m["gt-eager"]
        mc = 100 * (m["gt-compiled"] - fp_m) / m["gt-compiled"]
        te = t["gt-eager"] / fp_t
        tc = t["gt-compiled"] / fp_t
        print(f"{N:>3} | {me:>+12.0f}% {mc:>+11.0f}% | {te:>12.3f}x {tc:>11.3f}x")
    print("  (mem: + = fused-packed uses less; time: >1 = fused-packed faster)")


if __name__ == "__main__":
    main()
