"""Stacked N-block SwiGLU MLP: show the saved-activation staircase and the
recompute peak-memory win growing with depth.

Each block is dim -> hidden -> dim (Dout=D), so blocks compose. We run a forward
through all N blocks (one graph) then a single backward; PyTorch frees each
block's saved tensors as backward passes through it (no manual freeing). We
measure peak alloc for the standard-autograd stack vs the recompute stack and
watch the gap grow ~linearly in N (each block that doesn't save `h` saves one
[M, H] tensor of persistent activation).

Run:  python bench_stacked_blocks.py
"""

from __future__ import annotations

import argparse
import gc
import sys
import pathlib

import torch
import torch.nn as nn

THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from swiglu_recompute import SwiGLUMLPCustom, SwiGLUMLPGroundTruth  # noqa: E402

MIB = 1024 * 1024


def mib(n: int) -> float:
    return n / MIB


def sync_clean():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


class Stack(nn.Module):
    def __init__(self, block_cls, N, D, H, dtype):
        super().__init__()
        self.blocks = nn.ModuleList(
            [block_cls(D, H, D, dtype=dtype) for _ in range(N)]
        )

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def measure_peak(model, x0, gout, reps=3):
    params = list(model.parameters())
    # warmup (also triggers torch.compile of the recompute helpers)
    for _ in range(2):
        x = x0.detach().requires_grad_(True)
        y = model(x)
        torch.autograd.grad(y, [x] + params, gout)
        del x, y
    sync_clean()

    best = None
    for _ in range(reps):
        sync_clean()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        x = x0.detach().requires_grad_(True)
        y = model(x)              # forward through all N blocks -> one graph
        torch.autograd.grad(y, [x] + params, gout)  # single backward
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        best = (peak - base) if best is None else min(best, peak - base)
        del x, y
    return mib(best)


def time_full(model, x0, gout, iters=8, warmup=4, reps=3):
    """Full fwd+bwd time per step (ms), min over reps."""
    params = list(model.parameters())

    def step():
        x = x0.detach().requires_grad_(True)
        y = model(x)
        torch.autograd.grad(y, [x] + params, gout)

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    vals = []
    for _ in range(reps):
        s = torch.cuda.Event(True)
        e = torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            step()
        e.record()
        torch.cuda.synchronize()
        vals.append(s.elapsed_time(e) / iters)
    return min(vals)


def measure_for_N(N, M, D, H, dtype):
    x0 = torch.randn(M, D, device="cuda", dtype=dtype)
    gout = torch.randn(M, D, device="cuda", dtype=dtype)  # stack output is [M, D]

    # Build + measure one stack at a time so the two don't coexist in memory.
    gt = Stack(SwiGLUMLPGroundTruth, N, D, H, dtype).cuda()
    gt_mem = measure_peak(gt, x0, gout)
    gt_t = time_full(gt, x0, gout)
    del gt
    sync_clean()

    rc = Stack(SwiGLUMLPCustom, N, D, H, dtype).cuda()
    rc_mem = measure_peak(rc, x0, gout)
    rc_t = time_full(rc, x0, gout)
    del rc, x0, gout
    sync_clean()
    return gt_mem, rc_mem, gt_t, rc_t


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
    h_mib = mib(args.m * args.h * dtype.itemsize)
    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
    print(f"M={args.m} D={args.d} H={args.h} dtype={dtype}")
    print(f"per-block `h` [M,H] = {h_mib:.1f} MiB  (the persistent activation recompute avoids)\n")

    hdr = (f"{'N':>3} | {'gt mem':>10} {'rc mem':>10} {'saving':>10} {'%':>6} | "
           f"{'gt ms':>8} {'rc ms':>8} {'rc/gt':>6}")
    print(hdr)
    print("-" * len(hdr))
    for N in args.ns:
        gt_mem, rc_mem, gt_t, rc_t = measure_for_N(N, args.m, args.d, args.h, dtype)
        saving = gt_mem - rc_mem
        print(f"{N:>3} | {gt_mem:>9.0f}M {rc_mem:>9.0f}M {saving:>9.0f}M "
              f"{100 * saving / gt_mem:>5.0f}% | {gt_t:>7.1f} {rc_t:>7.1f} {gt_t / rc_t:>5.2f}x")


if __name__ == "__main__":
    main()
