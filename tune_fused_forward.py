"""Tune the fused-forward kernel: sweep GROUP_SIZE_M x warp_specialize.

For each config: compile + correctness-check (vs torch x@Wt + SwiGLU) + time the
forward (min over reps). Reports which configs compile, which are correct, and
the fastest -- against the cuBLAS GEMM + @torch.compile-pointwise baseline and a
raw cuBLAS GEMM-only floor.

warp_specialize=True does NOT compile for the standard-layout two-dot body in
triton 3.7 (the automatic warp-specialization pass cannot partition two dots);
it is swept here to document that.

Run:  python tune_fused_forward.py
"""

from __future__ import annotations

import argparse
import gc

import torch
import torch.nn.functional as F

import fused_forward as ff


def sync_clean():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def time_fn(fn, iters=20, warmup=10, reps=10):
    """min over reps of mean-per-iter ms."""
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=11136)
    p.add_argument("--d", type=int, default=3584)
    p.add_argument("--h", type=int, default=14336)
    p.add_argument("--groups", type=int, nargs="+", default=[1, 4, 8, 16])
    args = p.parse_args()

    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    M, D, H = args.m, args.d, args.h
    dtype = torch.bfloat16
    print(f"torch {torch.__version__} | triton {ff.triton.__version__} | "
          f"{torch.cuda.get_device_name(0)}")
    print(f"M={M} D={D} H={H} dtype={dtype}\n")

    torch.manual_seed(0)
    x = (torch.randn(M, D, device="cuda", dtype=dtype) * (D ** -0.5)).contiguous()
    wt = (torch.randn(D, 2 * H, device="cuda", dtype=dtype) * (D ** -0.5)).contiguous()

    # Reference (torch) for correctness.
    ref_preact = x @ wt
    rleft, rgate = ref_preact.chunk(2, dim=-1)
    ref_h = rleft * F.silu(rgate)

    def correctness(preact, h):
        pe = ((preact.float() - ref_preact.float()).abs().max()
              / ref_preact.float().abs().max().clamp_min(1e-6)).item()
        he = ((h.float() - ref_h.float()).abs().max()
              / ref_h.float().abs().max().clamp_min(1e-6)).item()
        return pe, he

    # --- baselines -----------------------------------------------------------
    @torch.compile
    def act(preact):
        l, g = preact.chunk(2, dim=-1)
        return l * F.silu(g)

    def baseline_cublas_compiled():
        preact = x @ wt
        return act(preact)

    def baseline_gemm_only():
        return x @ wt

    for _ in range(10):
        baseline_cublas_compiled(); baseline_gemm_only()
    sync_clean()
    t_base = time_fn(baseline_cublas_compiled)
    t_gemm = time_fn(baseline_gemm_only)
    print(f"baseline  cuBLAS GEMM + @compile pointwise : {t_base:7.3f} ms")
    print(f"baseline  cuBLAS GEMM only (floor)         : {t_gemm:7.3f} ms\n")

    # --- sweep ---------------------------------------------------------------
    hdr = f"{'WS':<5} {'G':>3} | {'status':<12} {'preact_rel':>11} {'h_rel':>9} {'fwd ms':>9} {'vs base':>8}"
    print(hdr)
    print("-" * len(hdr))

    results = []
    for ws in (False, True):
        for g in args.groups:
            def run():
                return ff.fused_matmul_swiglu_save_preact(
                    x, wt, group_size_m=g, warp_specialize=ws
                )
            status, pe, he, t = "ok", None, None, None
            try:
                preact, h = run()       # compile + run once
                pe, he = correctness(preact, h)
                if max(pe, he) > 2e-2:
                    status = "WRONG"
                for _ in range(10):
                    run()
                sync_clean()
                t = time_fn(run)
            except Exception as ex:
                status = "ws-fail" if ("PassManager" in str(ex) or "MLIR" in str(ex)) else "compile-fail"
            pe_s = f"{pe:.2e}" if pe is not None else "-"
            he_s = f"{he:.2e}" if he is not None else "-"
            t_s = f"{t:.3f}" if t is not None else "-"
            vs = f"{t_base / t:.3f}x" if t else "-"
            print(f"{str(ws):<5} {g:>3} | {status:<12} {pe_s:>11} {he_s:>9} {t_s:>9} {vs:>8}")
            if t is not None and status == "ok":
                results.append((t, ws, g))
            sync_clean()

    if results:
        results.sort()
        t, ws, g = results[0]
        print(f"\nbest fused config: WS={ws} GROUP_SIZE_M={g} -> "
              f"{t:.3f} ms ({t_base / t:.3f}x vs cuBLAS+compiled, "
              f"{t_gemm / t:.3f}x vs cuBLAS GEMM-only floor)")
    else:
        print("\nno fused config both compiled and was correct")


if __name__ == "__main__":
    main()
