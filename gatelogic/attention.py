"""Gate-based attention for associative recall.

Both designs share the same skeleton:

    code_K_i = GateEncoder(key_i)          (learned logic circuit)
    code_Q   = GateEncoder(query)          (learned logic circuit)
    match_i  = agreement(code_K_i, code_Q) (fixed XNOR-based structure)
    out      = OR_i ( match_i AND value_i )

The aggregation is position-wise and parameter-free, so the module accepts
any sequence length N — length generalization is architectural, the question
is whether the *encoders* can be learned with gradient descent.

Design A ("hash match"): match_i = AND over XNOR bits (exact code match).
Design B ("popcount threshold"): match_i = [popcount(XNOR) >= theta],
relaxed with a sigmoid during training; theta is learned.
"""

import math

import torch
import torch.nn as nn

from .layers import GateEncoder


def _soft_xnor(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return p * q + (1 - p) * (1 - q)


class _GateAttentionBase(nn.Module):
    def __init__(self, key_bits: int, val_bits: int, code_bits: int,
                 hidden: int, share_qk: bool = False,
                 generator: torch.Generator | None = None):
        super().__init__()
        self.code_bits = code_bits
        self.k_enc = GateEncoder([key_bits, hidden, code_bits], generator)
        self.q_enc = self.k_enc if share_qk else GateEncoder([key_bits, hidden, code_bits], generator)

    def _codes(self, keys, query, hard: bool):
        # keys: (B, N, key_bits), query: (B, key_bits)
        enc_k = self.k_enc.forward_hard if hard else self.k_enc
        enc_q = self.q_enc.forward_hard if hard else self.q_enc
        ck = enc_k(keys)               # (B, N, c)
        cq = enc_q(query).unsqueeze(1)  # (B, 1, c)
        return ck, cq

    @staticmethod
    def _aggregate_soft(match, values):
        # OR_i (match_i AND v_ij), relaxed: 1 - prod_i (1 - match_i * v_ij)
        gated = match.unsqueeze(-1) * values          # (B, N, vb)
        return 1 - (1 - gated).prod(dim=1)            # (B, vb)

    @staticmethod
    def _aggregate_hard(match, values):
        gated = match.unsqueeze(-1) & values          # bool (B, N, vb)
        return gated.any(dim=1)                       # (B, vb)


class GateAttentionA(_GateAttentionBase):
    """Exact-match routing: AND over XNOR'd code bits."""

    def forward(self, keys, values, query):
        ck, cq = self._codes(keys, query, hard=False)
        match = _soft_xnor(ck, cq).prod(-1)           # (B, N)
        return self._aggregate_soft(match, values)

    @torch.no_grad()
    def forward_hard(self, keys, values, query):
        ck, cq = self._codes(keys, query, hard=True)
        match = (ck == cq).all(-1)                    # (B, N) bool
        return self._aggregate_hard(match, values)


class GateAttentionB(_GateAttentionBase):
    """Threshold routing: popcount(XNOR) >= theta, theta learned."""

    def __init__(self, *args, beta: float = 4.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.theta = nn.Parameter(torch.tensor(self.code_bits - 0.5))

    def forward(self, keys, values, query):
        ck, cq = self._codes(keys, query, hard=False)
        score = _soft_xnor(ck, cq).sum(-1)            # (B, N) in [0, c]
        match = torch.sigmoid(self.beta * (score - self.theta))
        return self._aggregate_soft(match, values)

    @torch.no_grad()
    def forward_hard(self, keys, values, query):
        ck, cq = self._codes(keys, query, hard=True)
        score = (ck == cq).sum(-1).float()            # popcount of XNOR
        match = score >= self.theta
        return self._aggregate_hard(match, values)


class MajorityNorm(nn.Module):
    """Phase-2 placeholder: k-out-of-n majority gate as a normalization
    primitive (popcount >= n/2). Included so the API is settled; unused in
    Phase 1."""

    def forward(self, x):
        n = x.shape[-1]
        return torch.sigmoid(4.0 * (x.sum(-1) - n / 2))

    @torch.no_grad()
    def forward_hard(self, x):
        n = x.shape[-1]
        return x.sum(-1) * 2 >= n
