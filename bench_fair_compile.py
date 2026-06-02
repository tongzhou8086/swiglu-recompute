"""Apples-to-apples: compile the baseline too.

`bench_stacked_blocks.py` compared a torch.compile'd recompute path against an
EAGER autograd baseline, so part of recompute's speed edge was just
"compiled vs eager". Here we add a torch.compile'd standard SwiGLU MLP. When you
compile a standard module, AOTAutograd traces forward+backward and Inductor's
min-cut partitioner makes its own save-vs-recompute decisions -- so this is both
the fair speed baseline and a check on whether Inductor already recomputes
activations on its own.

Three variants, peak memory + full fwd+bwd time, at a few depths:
  gt-eager     : standard SwiGLU MLP, eager autograd
  gt-compiled  : torch.compile(standard SwiGLU MLP)   <- fair baseline
  recompute    : our autograd.Function (saves preact, recomputes h; helpers compiled)

Run:  python bench_fair_compile.py
"""

from __future__ import annotations

import argparse
import sys
import pathlib

import torch

THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from swiglu_recompute import SwiGLUMLPCustom, SwiGLUMLPGroundTruth  # noqa: E402
from bench_stacked_blocks import Stack, measure_peak, time_full, sync_clean  # noqa: E402


def run_N(N, M, D, H, dtype):
    x0 = torch.randn(M, D, device="cuda", dtype=dtype)
    gout = torch.randn(M, D, device="cuda", dtype=dtype)
    rows = []

    # measure one variant at a time so stacks don't coexist in memory
    m = Stack(SwiGLUMLPGroundTruth, N, D, H, dtype).cuda()
    rows.append(("gt-eager", measure_peak(m, x0, gout), time_full(m, x0, gout)))
    del m
    sync_clean()

    m = torch.compile(Stack(SwiGLUMLPGroundTruth, N, D, H, dtype).cuda())
    rows.append(("gt-compiled", measure_peak(m, x0, gout), time_full(m, x0, gout)))
    del m
    sync_clean()

    m = Stack(SwiGLUMLPCustom, N, D, H, dtype).cuda()
    rows.append(("recompute", measure_peak(m, x0, gout), time_full(m, x0, gout)))
    del m, x0, gout
    sync_clean()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=11136)
    p.add_argument("--d", type=int, default=3584)
    p.add_argument("--h", type=int, default=14336)
    p.add_argument("--ns", type=int, nargs="+", default=[1, 4, 8])
    args = p.parse_args()

    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    dtype = torch.bfloat16
    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
    print(f"M={args.m} D={args.d} H={args.h} dtype={dtype}\n")

    for N in args.ns:
        print(f"N = {N}")
        rows = run_N(N, args.m, args.d, args.h, dtype)
        base_t = next(t for name, _, t in rows if name == "gt-compiled")
        for name, mem, t in rows:
            print(f"   {name:<12} peak={mem:8.1f} MiB   full={t:7.1f} ms   "
                  f"vs gt-compiled: {base_t / t:.3f}x")
        print()


if __name__ == "__main__":
    main()
