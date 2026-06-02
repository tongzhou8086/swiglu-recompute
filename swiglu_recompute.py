"""SwiGLU MLP with backward-pass recomputation of the activation `h`.

Strategy (selective activation recomputation): the forward saves the
pre-activation `preact = x @ W1.t()` but NOT `h = left * silu(gate)`.  The
backward recomputes `h` (and the SiLU factors) from `preact` on the fly.  This
trades a cheap element-wise recompute for not storing the `[M, H]` activation.

`preact` is kept purely internal (saved via ctx, never returned as a graph
output), so this avoids the graph-output peak-memory pitfall: exposing a saved
tensor as a graph output makes autograd protect a copy of it.

Shapes:  x [M, D]   W1 [2H, D]   W2 [Dout, H]
         preact [M, 2H]   h [M, H]   y [M, Dout]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Core SwiGLU math helpers
# -----------------------------------------------------------------------------
def _swiglu_forward_logic(preact: torch.Tensor) -> torch.Tensor:
    """h = left * silu(gate), from preact [M, 2H] -> h [M, H]."""
    left, gate = preact.chunk(2, dim=-1)
    return left * F.silu(gate)


def _swiglu_backward_logic(preact: torch.Tensor, grad_h: torch.Tensor) -> torch.Tensor:
    """grad_preact [M, 2H] from preact and grad_h [M, H], via the explicit SiLU derivative.

    silu'(z) = sigmoid(z) + silu(z) * (1 - sigmoid(z))
    grad_left = grad_h * silu(gate)
    grad_gate = grad_h * left * silu'(gate)
    """
    left, gate = preact.chunk(2, dim=-1)

    sigmoid_gate = torch.sigmoid(gate)
    silu_gate = gate * sigmoid_gate
    dsilu_gate = sigmoid_gate + silu_gate * (1.0 - sigmoid_gate)

    grad_left = grad_h * silu_gate
    grad_gate = grad_h * left * dsilu_gate
    return torch.cat([grad_left, grad_gate], dim=-1)


# -----------------------------------------------------------------------------
# Custom autograd.Function: recompute `h` in backward instead of saving it
# -----------------------------------------------------------------------------
class SwiGLUMemoryOptimizedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W1, W2):
        preact = x @ W1.t()                  # [M, 2H]
        # Save preact (internal local) but NOT h -> h's [M, H] memory is freed.
        ctx.save_for_backward(x, W1, W2, preact)
        h = _swiglu_forward_logic(preact)    # [M, H]  (transient; not retained)
        y = h @ W2.t()                       # [M, Dout]
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, W1, W2, preact = ctx.saved_tensors
        h = _swiglu_forward_logic(preact)    # recompute h from saved preact

        grad_W2 = grad_y.t() @ h             # [Dout, H]
        grad_h = grad_y @ W2                 # [M, H]
        grad_preact = _swiglu_backward_logic(preact, grad_h)  # [M, 2H]
        grad_W1 = grad_preact.t() @ x        # [2H, D]
        grad_x = grad_preact @ W1            # [M, D]
        return grad_x, grad_W1, grad_W2


# -----------------------------------------------------------------------------
# nn.Module wrapper
# -----------------------------------------------------------------------------
class SwiGLUMLPCustom(nn.Module):
    """SwiGLU MLP (dim -> hidden -> out_dim) using backward recomputation of `h`."""

    def __init__(self, dim, hidden_dim, out_dim=None, dtype=torch.float32):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.w1 = nn.Parameter(torch.empty(2 * hidden_dim, dim, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(self.out_dim, hidden_dim, dtype=dtype))
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)

    def forward(self, x):
        return SwiGLUMemoryOptimizedFunction.apply(x, self.w1, self.w2)


class SwiGLUMLPGroundTruth(nn.Module):
    """Standard SwiGLU MLP relying on PyTorch autograd (saves `h` and `preact`)."""

    def __init__(self, dim, hidden_dim, out_dim=None, dtype=torch.float32):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.w1 = nn.Parameter(torch.empty(2 * hidden_dim, dim, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(self.out_dim, hidden_dim, dtype=dtype))
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)

    def forward(self, x):
        preact = x @ self.w1.t()
        left, gate = preact.chunk(2, dim=-1)
        h = left * F.silu(gate)
        return h @ self.w2.t()
