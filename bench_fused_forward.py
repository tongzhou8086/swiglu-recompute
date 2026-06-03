"""Benchmark the fused-forward SwiGLU kernel vs ground-truth and the PyTorch
backward-recompute variant.

Three variants (identical weights, one fwd+bwd):
  ground_truth : standard SwiGLU MLP, plain autograd (saves h)              [W1: 2H,D]
  recompute    : SwiGLUMLPCustom -- saves preact, recomputes h in backward  [W1: 2H,D]
  fused        : SwiGLUMLPFused  -- one Triton kernel emits (preact, h);     [Wt: D,2H]
                 backward recomputes h (same math as `recompute`)

The fused kernel computes `h = left*silu(gate)` from the GEMM accumulator in
registers, so `preact` is never read back from HBM. This is a *forward
bandwidth* win, not a memory win: `fused` allocates the same preact[M,2H] and
h[M,H] as `recompute`, so peak memory should match -- only the forward time
should drop (one kernel, one fewer [M,2H] HBM read).

Run:  python bench_fused_forward.py
      python bench_fused_forward.py --m 11136 --d 3584 --h 14336 --dout 3584
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
from fused_forward import SwiGLUMLPFused  # noqa: E402
from fused_forward_packed import SwiGLUMLPPackedFused, pack_swiglu_linear_weight  # noqa: E402

MIB = 1024 * 1024


def mib(n: int) -> float:
    return n / MIB


def sync_clean():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def rel_max(a, b):
    denom = b.float().abs().max().clamp_min(1e-6)
    return ((a.float() - b.float()).abs().max() / denom).item()


def make_quad(D, H, Dout, dtype):
    """Ground-truth, recompute, standard-fused, packed-fused -- identical weights."""
    gt = SwiGLUMLPGroundTruth(D, H, Dout, dtype=dtype).cuda()
    rc = SwiGLUMLPCustom(D, H, Dout, dtype=dtype).cuda()
    fu = SwiGLUMLPFused(D, H, Dout, dtype=dtype).cuda()
    pk = SwiGLUMLPPackedFused(D, H, Dout, dtype=dtype).cuda()
    with torch.no_grad():
        rc.w1.copy_(gt.w1)
        rc.w2.copy_(gt.w2)
        fu.w1t.copy_(gt.w1.t().contiguous())   # Wt = W1.t()
        fu.w2.copy_(gt.w2)
        pk.packed_weight.copy_(pack_swiglu_linear_weight(gt.w1.contiguous()))
        pk.w2.copy_(gt.w2)
    return gt, rc, fu, pk


def run_correctness(D, H, Dout, dtype, tol):
    from fused_forward_packed import unpack_swiglu_linear_weight
    torch.manual_seed(42)
    M = 512
    gt, rc, fu, pk = make_quad(D, H, Dout, dtype)
    x0 = torch.randn(M, D, device="cuda", dtype=dtype)

    xg = x0.detach().clone().requires_grad_(True)
    xf = x0.detach().clone().requires_grad_(True)
    xp = x0.detach().clone().requires_grad_(True)
    yg = gt(xg)
    yf = fu(xf)
    yp = pk(xp)
    gout = torch.randn_like(yg)
    gg = torch.autograd.grad(yg, [xg, gt.w1, gt.w2], gout)
    gf = torch.autograd.grad(yf, [xf, fu.w1t, fu.w2], gout)
    gp = torch.autograd.grad(yp, [xp, pk.packed_weight, pk.w2], gout)

    ok = True
    for label, y, g, gW1 in (
        ("fused (standard)", yf, gf, gf[1].t()),                       # grad wrt Wt
        ("packed", yp, gp, unpack_swiglu_linear_weight(gp[1])),        # packed grad -> [2H,D]
    ):
        print(f"  [correctness {dtype}]  {label} vs ground_truth  (tol={tol})")
        checks = {"out": rel_max(y, yg), "grad_x": rel_max(g[0], gg[0]),
                  "grad_W1": rel_max(gW1, gg[1]), "grad_W2": rel_max(g[2], gg[2])}
        for name, r in checks.items():
            passed = r <= tol
            ok = ok and passed
            print(f"      {name:<9s} rel_max={r:.3e}  {'OK' if passed else 'FAIL'}")
    print(f"  => {'PASS' if ok else 'FAIL'}")
    del gt, rc, fu, pk, xg, xf, xp, yg, yf, yp, gg, gf, gp, x0
    sync_clean()
    return ok


def measure_peak(model, params, x0, gout, reps=3):
    """Peak alloc delta for one fwd+bwd, min over reps. Warms up first."""
    for _ in range(3):
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
        y = model(x)
        torch.autograd.grad(y, [x] + params, gout)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        best = (peak - base) if best is None else min(best, peak - base)
        del x, y
    return mib(best)


def bench_time(model, params, x0, gout, iters=15, warmup=8, reps=10):
    """(forward_ms, full_fwd_bwd_ms), min over reps."""
    def fwd():
        x = x0.detach().requires_grad_(True)
        return model(x)

    def full():
        x = x0.detach().requires_grad_(True)
        y = model(x)
        torch.autograd.grad(y, [x] + params, gout)

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
    gt, rc, fu, pk = make_quad(D, H, Dout, dtype)
    gout = torch.randn(M, Dout, device="cuda", dtype=dtype)

    variants = [
        ("ground_truth", gt, [gt.w1, gt.w2]),
        ("recompute", rc, [rc.w1, rc.w2]),
        ("fused-std", fu, [fu.w1t, fu.w2]),
        ("fused-packed", pk, [pk.packed_weight, pk.w2]),
    ]

    print("\n  [peak memory: one fwd+bwd]")
    mems = {}
    for name, m, params in variants:
        mems[name] = measure_peak(m, params, x0, gout)
        print(f"      {name:<14s}: {mems[name]:8.1f} MiB")

    print("\n  [timing: min over reps]")
    print(f"      {'variant':<14s} {'fwd ms':>9} {'full ms':>9}")
    times = {}
    for name, m, params in variants:
        f, full = bench_time(m, params, x0, gout)
        times[name] = (f, full)
        print(f"      {name:<14s} {f:>9.3f} {full:>9.3f}")

    rf, rfull = times["recompute"]
    print("\n  vs recompute (cuBLAS GEMM + @compile pointwise):")
    for name in ("fused-std", "fused-packed"):
        f, full = times[name]
        print(f"      {name:<14s}: fwd {rf / f:.3f}x   full {rfull / full:.3f}x")
    print("  (all fused variants are memory-neutral vs recompute; the win is forward bandwidth)")


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

    run_correctness(args.d, args.h, args.dout, torch.float32, tol=1e-4)
    run_correctness(args.d, args.h, args.dout, torch.bfloat16, tol=2e-2)
    run_bench(args.m, args.d, args.h, args.dout, torch.bfloat16)


if __name__ == "__main__":
    main()
