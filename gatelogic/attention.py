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
                 generator: torch.Generator | None = None,
                 residual_init: bool = False):
        super().__init__()
        self.code_bits = code_bits
        self.k_enc = GateEncoder([key_bits, hidden, code_bits], generator, residual_init)
        self.q_enc = self.k_enc if share_qk else GateEncoder([key_bits, hidden, code_bits], generator, residual_init)

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


class CountingGateAttention(nn.Module):
    """Causal induction-head attention over a token sequence.

    Fixed structure, learned code: position t's query is enc(x_t); position
    j < t offers key enc(x_j) with value x_{j+1} (the token that followed).
    Exact code match gates a one-hot of the value, and the readout is the
    per-vocab COUNT of matched values — softmax replaced by count + argmax,
    i.e. a multi-way majority vote (the gate-native normalization). The
    prev-token wiring is built in; what is learned is only the matching
    circuit. `forward` returns soft logits (scale * soft counts);
    `forward_hard` returns integer counts from the discretized circuit.
    """

    def __init__(self, token_bits: int, vocab: int, code_bits: int, hidden: int,
                 generator: torch.Generator | None = None, residual_init: bool = True):
        super().__init__()
        self.vocab = vocab
        self.enc = GateEncoder([token_bits, hidden, code_bits], generator, residual_init)
        self.scale = nn.Parameter(torch.tensor(2.0))

    @staticmethod
    def _shifted_value_onehot(tokens_int: torch.Tensor, vocab: int) -> torch.Tensor:
        v = nn.functional.one_hot(tokens_int, vocab).float()  # (B, T, V)
        out = torch.zeros_like(v)
        out[:, :-1] = v[:, 1:]                                # value at j is x_{j+1}
        return out

    def forward(self, tokens_bits: torch.Tensor, tokens_int: torch.Tensor) -> torch.Tensor:
        # tokens_bits (B,T,tb) float, tokens_int (B,T) long -> logits (B,T,V)
        c = self.enc(tokens_bits)                                       # (B,T,cb)
        match = _soft_xnor(c.unsqueeze(2), c.unsqueeze(1)).prod(-1)     # (B,Tq,Tk)
        mask = torch.ones(match.shape[-2:], device=match.device).tril(-1)
        counts = (match * mask) @ self._shifted_value_onehot(tokens_int, self.vocab)
        return self.scale * counts

    @torch.no_grad()
    def forward_hard(self, tokens_bits: torch.Tensor, tokens_int: torch.Tensor) -> torch.Tensor:
        c = self.enc.forward_hard(tokens_bits)                          # bool (B,T,cb)
        match = (c.unsqueeze(2) == c.unsqueeze(1)).all(-1)              # (B,Tq,Tk)
        mask = torch.ones(match.shape[-2:], device=match.device, dtype=torch.bool).tril(-1)
        counts = (match & mask).float() @ self._shifted_value_onehot(tokens_int, self.vocab)
        return counts.long()                                            # integer votes


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
