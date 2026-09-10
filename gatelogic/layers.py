"""Pure-PyTorch reimplementation of differentiable logic gate networks.

API mirrors Petersen et al.'s `difflogic` package (LogicLayer / GroupSum) so
that models built here can later be ported to the CUDA implementation on a
GPU machine. Gate i (0..15) is defined by its 4-entry truth table over
(a,b) in [(0,0),(0,1),(1,0),(1,1)], which is exactly the binary expansion
of i. The soft (relaxed) forward is

    out = sum_{p in 4 patterns} P(pattern p | a, b) * E_g[ table[g][p] ]

where the expectation over gates g uses softmax(weights). This is
algebraically identical to difflogic's per-gate real-valued relaxation but
needs only a (16,4) table instead of 16 hand-written expressions.

`forward_hard` runs the discretized circuit (argmax gate per neuron) on
boolean tensors — this is the "gate 固定 → ビット演算推論" path.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# truth_table[i] = outputs of gate i for (a,b) = (0,0),(0,1),(1,0),(1,1)
_TRUTH_TABLE = torch.tensor(
    [[(i >> 3) & 1, (i >> 2) & 1, (i >> 1) & 1, i & 1] for i in range(16)],
    dtype=torch.float32,
)
# NOTE: with this bit order, gate index reads as t00 t01 t10 t11 from MSB:
#   0b0001=AND is index 1? Here index i has t00=(i>>3)&1 ... t11=i&1, so
#   AND (0001) = 1, XOR (0110) = 6, OR (0111) = 7, XNOR (1001) = 9,
#   NAND (1110) = 14, A (0011) = 3, B (0101) = 5, FALSE = 0, TRUE = 15.
GATE_NAMES = [
    "FALSE", "AND", "A_AND_NOT_B", "A", "NOT_A_AND_B", "B", "XOR", "OR",
    "NOR", "XNOR", "NOT_B", "A_OR_NOT_B", "NOT_A", "NOT_A_OR_B", "NAND", "TRUE",
]


class LogicLayer(nn.Module):
    """A layer of `out_dim` two-input logic gates with fixed random wiring."""

    def __init__(self, in_dim: int, out_dim: int, generator: torch.Generator | None = None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        idx_a = torch.randint(0, in_dim, (out_dim,), generator=generator)
        idx_b = torch.randint(0, in_dim, (out_dim,), generator=generator)
        if in_dim > 1:  # avoid a == b so binary gates stay expressive
            clash = idx_a == idx_b
            idx_b[clash] = (idx_b[clash] + 1 + torch.randint(0, in_dim - 1, (int(clash.sum()),), generator=generator)) % in_dim
        self.register_buffer("idx_a", idx_a)
        self.register_buffer("idx_b", idx_b)
        self.register_buffer("table", _TRUTH_TABLE.clone())
        self.weights = nn.Parameter(torch.randn(out_dim, 16, generator=generator))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Soft forward. x: (..., in_dim) floats in [0,1] -> (..., out_dim)."""
        a = x[..., self.idx_a]
        b = x[..., self.idx_b]
        patterns = torch.stack(
            [(1 - a) * (1 - b), (1 - a) * b, a * (1 - b), a * b], dim=-1
        )  # (..., out_dim, 4)
        w4 = F.softmax(self.weights, dim=-1) @ self.table  # (out_dim, 4)
        return (patterns * w4).sum(-1)

    @torch.no_grad()
    def forward_hard(self, x: torch.Tensor) -> torch.Tensor:
        """Discretized forward. x: (..., in_dim) bool -> (..., out_dim) bool."""
        a = x[..., self.idx_a].long()
        b = x[..., self.idx_b].long()
        gate = self.weights.argmax(-1)  # (out_dim,)
        pattern = 2 * a + b  # (..., out_dim) in {0,1,2,3}
        table_g = self.table.bool()[gate]  # (out_dim, 4)
        return torch.gather(table_g.expand(*pattern.shape[:-1], -1, -1), -1, pattern.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def chosen_gates(self) -> list[str]:
        return [GATE_NAMES[int(g)] for g in self.weights.argmax(-1)]


class GroupSum(nn.Module):
    """Sum the outputs in k groups (class scores), divided by tau."""

    def __init__(self, k: int, tau: float = 1.0):
        super().__init__()
        self.k = k
        self.tau = tau

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *lead, d = x.shape
        assert d % self.k == 0
        return x.reshape(*lead, self.k, d // self.k).sum(-1) / self.tau


class GateEncoder(nn.Module):
    """A stack of LogicLayers, e.g. dims=[6, 64, 6] maps 6 bits -> 6-bit code."""

    def __init__(self, dims: list[int], generator: torch.Generator | None = None):
        super().__init__()
        self.stack = nn.ModuleList(
            LogicLayer(dims[i], dims[i + 1], generator) for i in range(len(dims) - 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.stack:
            x = layer(x)
        return x

    @torch.no_grad()
    def forward_hard(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.stack:
            x = layer.forward_hard(x)
        return x
