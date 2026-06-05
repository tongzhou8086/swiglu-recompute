"""Matmul-only Triton vs cuBLAS at the production FFN shape.

Strips the SwiGLU activation + preact side-store out of the canonical
packed kernel, so we measure ONLY the A @ packed_weight matmul cost.
Comparison target is `torch.matmul(A, packed_weight)`.

Two Triton variants, same canonical Blackwell form:

  triton-mm-base      persistent + warp-spec + FLATTEN + EPILOGUE_SUBTILE
  triton-mm-canon     same + tile_id_c deferral

Both compute C[M, 2H] = A[M, K] @ B[K, 2H], bf16.  Same launch config as
the swiglu-recompute packed kernels: BM=128, BN=256 (= 2 * BN_HALF=128),
BK=64, GSM=32, NW=8, NS=4, WS=True.

This is the apples-to-apples decomposition of the swiglu-fused bench:
the gap (or absence of gap) between matmul-only and SwiGLU-fused tells
us how much the activation + preact side-store costs.

Run:  python bench_matmul_only.py
"""

from __future__ import annotations

import argparse

import torch
import triton
import triton.language as tl

from fused_forward_packed import (
    BLOCK_SIZE_M, BLOCK_SIZE_N_HALF, BLOCK_SIZE_K, GROUP_SIZE_M,
    NUM_WARPS, NUM_STAGES, WARP_SPECIALIZE,
    _ensure_allocator, _num_sms, _compute_pid,
)

BLOCK_SIZE_N = BLOCK_SIZE_N_HALF * 2   # 256 — matches the packed kernel's wide dim


@triton.jit
def _matmul_only_kernel(
    a_ptr, b_ptr, c_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, NUM_SMS: tl.constexpr,
    BM_: tl.constexpr, BN_: tl.constexpr, BK_: tl.constexpr,
    GSM_: tl.constexpr,
    WARP_SPECIALIZE_: tl.constexpr, FLATTEN: tl.constexpr,
    EPILOGUE_SUBTILE: tl.constexpr, USE_TILE_ID_C: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m: tl.constexpr = tl.cdiv(M, BM_)
    num_pid_n: tl.constexpr = tl.cdiv(N, BN_)
    k_tiles: tl.constexpr = tl.cdiv(K, BK_)
    num_tiles: tl.constexpr = num_pid_m * num_pid_n
    num_pid_in_group: tl.constexpr = GSM_ * num_pid_n

    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BM_, BK_])
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[K, N], strides=[N, 1], block_shape=[BK_, BN_])
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[N, 1],
        block_shape=[BM_, BN_ // 2 if EPILOGUE_SUBTILE else BN_])

    tile_id_c = start_pid - NUM_SMS

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS,
                            flatten=FLATTEN, warp_specialize=WARP_SPECIALIZE_):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GSM_)
        offs_m = pid_m * BM_
        offs_n = pid_n * BN_

        acc = tl.zeros((BM_, BN_), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BK_
            a = a_desc.load([offs_m, offs_k])
            b = b_desc.load([offs_k, offs_n])
            acc = tl.dot(a, b, acc, input_precision=INPUT_PRECISION)

        if USE_TILE_ID_C:
            tile_id_c += NUM_SMS
            pid_m_c, pid_n_c = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GSM_)
            offs_cm = pid_m_c * BM_
            offs_cn = pid_n_c * BN_
        else:
            offs_cm = offs_m
            offs_cn = offs_n

        if EPILOGUE_SUBTILE:
            acc3 = tl.reshape(acc, (BM_, 2, BN_ // 2))
            acc3 = tl.permute(acc3, (0, 2, 1))
            acc0, acc1 = tl.split(acc3)
            c_desc.store([offs_cm, offs_cn], acc0.to(c_ptr.dtype.element_ty))
            c_desc.store([offs_cm, offs_cn + BN_ // 2], acc1.to(c_ptr.dtype.element_ty))
        else:
            c_desc.store([offs_cm, offs_cn], acc.to(c_ptr.dtype.element_ty))


def triton_matmul_only(A, B, use_tile_id_c: bool):
    """C = A @ B, canonical Blackwell form, optional tile_id_c deferral."""
    _ensure_allocator()
    assert A.is_contiguous() and B.is_contiguous()
    M, K = A.shape
    K2, N = B.shape
    assert K == K2 and M % BLOCK_SIZE_M == 0 and N % BLOCK_SIZE_N == 0

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    di = A.device.index if A.device.index is not None else torch.cuda.current_device()
    num_sms = _num_sms(di)
    grid = (min(num_sms, triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N)),)

    if A.dtype == torch.float32:
        input_precision, block_k, num_stages, ws = "ieee", 32, 2, False
    else:
        input_precision, block_k, num_stages, ws = "tf32", BLOCK_SIZE_K, NUM_STAGES, WARP_SPECIALIZE

    _matmul_only_kernel[grid](
        A, B, C, M, N, K,
        NUM_SMS=num_sms, BM_=BLOCK_SIZE_M, BN_=BLOCK_SIZE_N, BK_=block_k, GSM_=GROUP_SIZE_M,
        WARP_SPECIALIZE_=ws, FLATTEN=True, EPILOGUE_SUBTILE=True,
        USE_TILE_ID_C=use_tile_id_c, INPUT_PRECISION=input_precision,
        num_warps=NUM_WARPS, num_stages=num_stages)
    return C


def bench_time(fn, iters=15, warmup=8, reps=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
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
    p.add_argument("--k", type=int, default=3584)
    p.add_argument("--n", type=int, default=28672, help="output width (= 2*H for packed)")
    args = p.parse_args()

    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    M, K, N = args.m, args.k, args.n
    print(f"torch {torch.__version__} | triton {triton.__version__} | "
          f"{torch.cuda.get_device_name(0)}")
    print(f"shape  M={M}  K={K}  N={N}  bf16")
    print(f"FLOPs  = {2 * M * N * K / 1e12:.2f} TFLOP")

    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)

    # Correctness check
    C_ref = (A.float() @ B.float()).to(torch.bfloat16)
    for label, fn in (
        ("triton-mm-base",  lambda: triton_matmul_only(A, B, use_tile_id_c=False)),
        ("triton-mm-canon", lambda: triton_matmul_only(A, B, use_tile_id_c=True)),
    ):
        C = fn()
        rel = (C.float() - C_ref.float()).abs().max().item() / C_ref.float().abs().max().item()
        print(f"  [correctness] {label:<18s} rel_max={rel:.3e}  {'OK' if rel <= 5e-2 else 'FAIL'}")
    print()

    # Bench
    cases = [
        ("cublas",          lambda: torch.matmul(A, B)),
        ("triton-mm-base",  lambda: triton_matmul_only(A, B, use_tile_id_c=False)),
        ("triton-mm-canon", lambda: triton_matmul_only(A, B, use_tile_id_c=True)),
    ]
    flops = 2.0 * M * N * K
    print(f"  {'variant':<18s} {'ms':>9} {'TFLOPS':>9} {'/cublas':>10}")
    times_ms = {}
    for name, fn in cases:
        t = bench_time(fn)
        times_ms[name] = t
        tf = flops / (t * 1e-3) / 1e12
        print(f"  {name:<18s} {t:>9.3f} {tf:>9.1f}")
    print()
    print(f"  {'variant':<18s} {'ms':>9} {'TFLOPS':>9} {'/cublas':>10}")
    for name, _ in cases:
        t = times_ms[name]
        tf = flops / (t * 1e-3) / 1e12
        print(f"  {name:<18s} {t:>9.3f} {tf:>9.1f}   {times_ms['cublas']/t:>8.3f}x")


if __name__ == "__main__":
    main()
