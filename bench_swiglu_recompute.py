"""Correctness + peak-memory + timing test for the recompute-`h` SwiGLU MLP.

Compares the recomputation variant (SwiGLUMLPCustom) against a standard PyTorch
SwiGLU MLP (SwiGLUMLPGroundTruth) at the project benchmark shape.

  default shape:  M=11136, D=3584, H=14336, Dout=3584   (D=K, H=N from the
  fused-swiglu benchmark; Dout=D for the down-projection)

Run:  python bench_swiglu_recompute.py
"""

from __future__ import annotations

import argparse
import gc
import sys
import pathlib

import torch

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


def rel_max(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = b.float().abs().max().clamp_min(1e-6)
    return ((a.float() - b.float()).abs().max() / denom).item()


def make_pair(D, H, Dout, dtype):
    """Two MLPs (ground-truth, recompute) with identical synced weights."""
    gt = SwiGLUMLPGroundTruth(D, H, Dout, dtype=dtype).cuda()
    rc = SwiGLUMLPCustom(D, H, Dout, dtype=dtype).cuda()
    with torch.no_grad():
        rc.w1.copy_(gt.w1)
        rc.w2.copy_(gt.w2)
    return gt, rc


def run_correctness(D, H, Dout, dtype, tol):
    torch.manual_seed(42)
    M = 512
    gt, rc = make_pair(D, H, Dout, dtype)
    x0 = torch.randn(M, D, device="cuda", dtype=dtype)

    xg = x0.detach().clone().requires_grad_(True)
    xr = x0.detach().clone().requires_grad_(True)
    yg = gt(xg)
    yr = rc(xr)
    gout = torch.randn_like(yg)

    gg = torch.autograd.grad(yg, [xg, gt.w1, gt.w2], gout)
    gr = torch.autograd.grad(yr, [xr, rc.w1, rc.w2], gout)

    print(f"  [correctness {dtype}]  M={M} D={D} H={H} Dout={Dout}  (tol={tol})")
    checks = {
        "out": rel_max(yr, yg),
        "grad_x": rel_max(gr[0], gg[0]),
        "grad_W1": rel_max(gr[1], gg[1]),
        "grad_W2": rel_max(gr[2], gg[2]),
    }
    ok = True
    for name, r in checks.items():
        passed = r <= tol
        ok = ok and passed
        print(f"      {name:<9s} rel_max={r:.3e}  {'OK' if passed else 'FAIL'}")
    print(f"  => {'PASS' if ok else 'FAIL'}")
    del gt, rc, xg, xr, yg, yr, gg, gr, x0
    sync_clean()
    return ok


def measure_peak(model, x0, gout, reps=3):
    """Peak alloc delta for one fwd+bwd, min over reps. Warms up first."""
    # warmup (autograd.grad, no .grad accumulation)
    for _ in range(2):
        x = x0.detach().requires_grad_(True)
        y = model(x)
        torch.autograd.grad(y, [x, model.w1, model.w2], gout)
        del x, y
    sync_clean()

    best = None
    for _ in range(reps):
        sync_clean()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        x = x0.detach().requires_grad_(True)
        y = model(x)
        torch.autograd.grad(y, [x, model.w1, model.w2], gout)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        best = (peak - base) if best is None else min(best, peak - base)
        del x, y
    return mib(best)


def bench_time(model, x0, gout, iters=15, warmup=8, reps=5):
    """(forward_ms, full_fwd_bwd_ms), min over reps."""
    def fwd():
        x = x0.detach().requires_grad_(True)
        return model(x)

    def full():
        x = x0.detach().requires_grad_(True)
        y = model(x)
        torch.autograd.grad(y, [x, model.w1, model.w2], gout)

    def timeit(fn):
        vals = []
        for _ in range(reps):
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            s = torch.cuda.Event(True); e = torch.cuda.Event(True)
            s.record()
            for _ in range(iters):
                fn()
            e.record(); torch.cuda.synchronize()
            vals.append(s.elapsed_time(e) / iters)
        return min(vals)

    return timeit(fwd), timeit(full)


def run_bench(M, D, H, Dout, dtype):
    print(f"\n=== shape M={M} D={D} H={H} Dout={Dout}  dtype={dtype} ===")
    print(f"    preact [M,2H] = {mib(M * 2 * H * dtype.itemsize):.1f} MiB   "
          f"h [M,H] = {mib(M * H * dtype.itemsize):.1f} MiB")
    torch.manual_seed(0)
    x0 = torch.randn(M, D, device="cuda", dtype=dtype)
    gt, rc = make_pair(D, H, Dout, dtype)
    gout = torch.randn(M, Dout, device="cuda", dtype=dtype)

    print("\n  [peak memory: one fwd+bwd]")
    gt_mem = measure_peak(gt, x0, gout)
    rc_mem = measure_peak(rc, x0, gout)
    print(f"      ground_truth (saves h) : {gt_mem:8.1f} MiB")
    print(f"      recompute    (no h)    : {rc_mem:8.1f} MiB")
    print(f"      saving                 : {gt_mem - rc_mem:8.1f} MiB "
          f"({100 * (gt_mem - rc_mem) / gt_mem:+.1f}%)")

    print("\n  [timing]")
    gt_f, gt_full = bench_time(gt, x0, gout)
    rc_f, rc_full = bench_time(rc, x0, gout)
    print(f"      ground_truth : fwd={gt_f:.3f} ms  full={gt_full:.3f} ms")
    print(f"      recompute    : fwd={rc_f:.3f} ms  full={rc_full:.3f} ms")
    print(f"      recompute vs ground_truth: fwd {gt_f / rc_f:.3f}x  full {gt_full / rc_full:.3f}x")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=11136)
    p.add_argument("--d", type=int, default=3584)
    p.add_argument("--h", type=int, default=14336)
    p.add_argument("--dout", type=int, default=3584)
    args = p.parse_args()

    assert torch.cuda.is_available(), "needs a CUDA GPU"
    torch.cuda.set_device(0)
    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}")

    # Correctness: fp32 (tight) and bf16 (looser, explicit-derivative vs autograd).
    run_correctness(args.d, args.h, args.dout, torch.float32, tol=1e-5)
    run_correctness(args.d, args.h, args.dout, torch.bfloat16, tol=2e-2)

    # Memory + timing at the project shape, bf16.
    run_bench(args.m, args.d, args.h, args.dout, torch.bfloat16)


if __name__ == "__main__":
    main()
