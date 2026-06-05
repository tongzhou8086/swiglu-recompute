"""Canonical Blackwell-standard variant of fused_forward_packed.

Same SwiGLU forward as `fused_forward_packed.py`, but the matmul kernel
adds the **tile_id_c deferral** trick (Triton tutorials/09
matmul_kernel_descriptor_persistent): the store offsets are derived
from a second counter that lags `tile_id` by one outer iteration,
which lets the compiler interleave "K-loop for T+1" with "epilogue
stores for T" → K-loop / epilogue overlap.

EPILOGUE_SUBTILE was already present in the original (the
reshape/permute/split into separate left/gate/c TMA stores), so the
remaining canonical-form piece is just the deferral added here.

Side-by-side with `fused_forward_packed.py` so we can directly
bench the deferral's impact at the production FFN shape.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import triton
import triton.language as tl

from fused_forward_packed import (
    BLOCK_SIZE_M, BLOCK_SIZE_N_HALF, BLOCK_SIZE_K, GROUP_SIZE_M,
    NUM_WARPS, NUM_STAGES, WARP_SPECIALIZE,
    _ensure_allocator, _num_sms, _compute_pid,
    pack_swiglu_linear_weight, unpack_swiglu_linear_weight,
    swiglu_packed_grad_de_from_preact, _h_from_packed_preact,
)


@triton.jit
def _fused_swiglu_wide_packed_save_preact_kernel_canon(
    a_ptr, bp_ptr, c_ptr, preact_ptr,
    M: tl.constexpr, N_HALF: tl.constexpr, K: tl.constexpr, NUM_SMS: tl.constexpr,
    BLOCK_SIZE_M_: tl.constexpr, BLOCK_SIZE_N_HALF_: tl.constexpr,
    BLOCK_SIZE_K_: tl.constexpr, GROUP_SIZE_M_: tl.constexpr,
    WARP_SPECIALIZE_: tl.constexpr, FLATTEN: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    BLOCK_SIZE_N2: tl.constexpr = BLOCK_SIZE_N_HALF_ * 2

    num_pid_m: tl.constexpr = tl.cdiv(M, BLOCK_SIZE_M_)
    num_pid_n: tl.constexpr = tl.cdiv(N_HALF, BLOCK_SIZE_N_HALF_)
    k_tiles: tl.constexpr = tl.cdiv(K, BLOCK_SIZE_K_)
    num_tiles: tl.constexpr = num_pid_m * num_pid_n
    num_pid_in_group: tl.constexpr = GROUP_SIZE_M_ * num_pid_n

    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, K], strides=[K, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_K_])
    bp_desc = tl.make_tensor_descriptor(
        bp_ptr, shape=[K, N_HALF * 2], strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_K_, BLOCK_SIZE_N2])
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N_HALF], strides=[N_HALF, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_])
    preact_desc = tl.make_tensor_descriptor(
        preact_ptr, shape=[M, N_HALF * 2], strides=[N_HALF * 2, 1],
        block_shape=[BLOCK_SIZE_M_, BLOCK_SIZE_N_HALF_])

    # Deferred counter — numerically equal to tile_id every iter, but
    # placing the pid_c recompute below the K-loop lets the compiler
    # overlap tile T's epilogue with tile T+1's K-loop.
    tile_id_c = start_pid - NUM_SMS

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS,
                            flatten=FLATTEN, warp_specialize=WARP_SPECIALIZE_):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M_)
        offs_m = pid_m * BLOCK_SIZE_M_
        offs_n2 = pid_n * BLOCK_SIZE_N2

        acc = tl.zeros((BLOCK_SIZE_M_, BLOCK_SIZE_N2), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K_
            a = a_desc.load([offs_m, offs_k])
            b = bp_desc.load([offs_k, offs_n2])           # one wide load
            acc = tl.dot(a, b, acc, input_precision=INPUT_PRECISION)

        acc3 = tl.reshape(acc, (BLOCK_SIZE_M_, 2, BLOCK_SIZE_N_HALF_))
        acc3 = tl.permute(acc3, (0, 2, 1))
        left, gate = tl.split(acc3)
        h = left * (gate * tl.sigmoid(gate))

        tile_id_c += NUM_SMS
        pid_m_c, pid_n_c = _compute_pid(
            tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M_)
        offs_m_c = pid_m_c * BLOCK_SIZE_M_
        offs_n_c = pid_n_c * BLOCK_SIZE_N_HALF_
        offs_n2_c = pid_n_c * BLOCK_SIZE_N2

        preact_desc.store([offs_m_c, offs_n2_c],
                          left.to(preact_ptr.dtype.element_ty))
        preact_desc.store([offs_m_c, offs_n2_c + BLOCK_SIZE_N_HALF_],
                          gate.to(preact_ptr.dtype.element_ty))
        c_desc.store([offs_m_c, offs_n_c], h.to(c_ptr.dtype.element_ty))


def fused_swiglu_save_preact_packed_canon(x, packed_weight):
    """One launch (canonical-form): (h [M,H], preact_packed [M,2H])."""
    _ensure_allocator()
    assert x.is_cuda and packed_weight.is_cuda
    assert x.is_contiguous() and packed_weight.is_contiguous()
    m, k = x.shape
    k2, n2 = packed_weight.shape
    assert k == k2 and n2 % 2 == 0
    n_half = n2 // 2
    assert m % BLOCK_SIZE_M == 0 and n_half % BLOCK_SIZE_N_HALF == 0

    h = torch.empty((m, n_half), device=x.device, dtype=x.dtype)
    preact = torch.empty((m, n2), device=x.device, dtype=x.dtype)
    device_index = x.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    num_sms = _num_sms(device_index)
    grid = (min(num_sms,
                triton.cdiv(m, BLOCK_SIZE_M) * triton.cdiv(n_half, BLOCK_SIZE_N_HALF)),)

    if x.dtype == torch.float32:
        input_precision, block_k, num_stages, ws = "ieee", 32, 2, False
    else:
        input_precision, block_k, num_stages, ws = "tf32", BLOCK_SIZE_K, NUM_STAGES, WARP_SPECIALIZE

    _fused_swiglu_wide_packed_save_preact_kernel_canon[grid](
        x, packed_weight, h, preact, m, n_half, k,
        NUM_SMS=num_sms, BLOCK_SIZE_M_=BLOCK_SIZE_M, BLOCK_SIZE_N_HALF_=BLOCK_SIZE_N_HALF,
        BLOCK_SIZE_K_=block_k, GROUP_SIZE_M_=GROUP_SIZE_M, WARP_SPECIALIZE_=ws,
        FLATTEN=True, INPUT_PRECISION=input_precision,
        num_warps=NUM_WARPS, num_stages=num_stages)
    return h, preact


class SwiGLUPackedFusedFunctionCanon(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed_weight, w2):
        if not x.is_contiguous():
            x = x.contiguous()
        if not packed_weight.is_contiguous():
            packed_weight = packed_weight.contiguous()
        h, preact = fused_swiglu_save_preact_packed_canon(x, packed_weight)
        ctx.save_for_backward(x, packed_weight, w2, preact)
        return h @ w2.t()

    @staticmethod
    def backward(ctx, grad_y):
        x, packed_weight, w2, preact = ctx.saved_tensors
        m, n2 = preact.shape
        n_half = n2 // 2
        chunks = n_half // BLOCK_SIZE_N_HALF
        h = _h_from_packed_preact(preact, m, chunks, BLOCK_SIZE_N_HALF)
        grad_w2 = grad_y.t() @ h
        grad_h = grad_y @ w2
        grad_de = swiglu_packed_grad_de_from_preact(preact, grad_h)
        grad_x = grad_de @ packed_weight.t()
        grad_pw = x.t() @ grad_de
        return grad_x, grad_pw, grad_w2


class SwiGLUMLPPackedFusedCanon(nn.Module):
    """SwiGLU MLP with the canonical-form (tile_id_c deferred) fused forward."""

    def __init__(self, dim, hidden_dim, out_dim=None, dtype=torch.float32):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim if out_dim is not None else dim
        assert hidden_dim % BLOCK_SIZE_N_HALF == 0
        std = torch.empty(2 * hidden_dim, dim, dtype=dtype)
        nn.init.xavier_uniform_(std)
        self.packed_weight = nn.Parameter(pack_swiglu_linear_weight(std))
        self.w2 = nn.Parameter(torch.empty(self.out_dim, hidden_dim, dtype=dtype))
        nn.init.xavier_uniform_(self.w2)

    def forward(self, x):
        return SwiGLUPackedFusedFunctionCanon.apply(x, self.packed_weight, self.w2)


def _selftest():
    from swiglu_recompute import SwiGLUMLPGroundTruth

    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    print(f"torch {torch.__version__} | triton {triton.__version__} | "
          f"{torch.cuda.get_device_name(0)}")
    D, H, Dout, M = 3584, 14336, 3584, 512

    def run(dtype, tol):
        torch.manual_seed(0)
        gt = SwiGLUMLPGroundTruth(D, H, Dout, dtype=dtype).cuda()
        pk = SwiGLUMLPPackedFusedCanon(D, H, Dout, dtype=dtype).cuda()
        with torch.no_grad():
            pk.packed_weight.copy_(pack_swiglu_linear_weight(gt.w1.contiguous()))
            pk.w2.copy_(gt.w2)
        x0 = torch.randn(M, D, device="cuda", dtype=dtype)
        xg = x0.detach().clone().requires_grad_(True)
        xp = x0.detach().clone().requires_grad_(True)
        yg = gt(xg); yp = pk(xp)
        gout = torch.randn_like(yg)
        gg = torch.autograd.grad(yg, [xg, gt.w1, gt.w2], gout)
        gp = torch.autograd.grad(yp, [xp, pk.packed_weight, pk.w2], gout)
        def rel(a, b):
            return ((a.float() - b.float()).abs().max()
                    / b.float().abs().max().clamp_min(1e-6)).item()
        grad_w1 = unpack_swiglu_linear_weight(gp[1])
        checks = {"out": rel(yp, yg), "grad_x": rel(gp[0], gg[0]),
                  "grad_W1": rel(grad_w1, gg[1]), "grad_W2": rel(gp[2], gg[2])}
        ok = all(v <= tol for v in checks.values())
        print(f"  [{dtype}] tol={tol:g}")
        for n, v in checks.items():
            print(f"      {n:<9s} rel_max={v:.3e}  {'OK' if v <= tol else 'FAIL'}")
        print(f"  => {'PASS' if ok else 'FAIL'}")
        return ok

    ok32 = run(torch.float32, tol=1e-4)
    okbf = run(torch.bfloat16, tol=2e-2)
    print("ALL PASS" if (ok32 and okbf) else "FAILURES PRESENT")


if __name__ == "__main__":
    _selftest()
