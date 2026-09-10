"""Comparison models for Phase 1: same-scale softmax attention and MLP-only."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxAttentionBaseline(nn.Module):
    """Single-head dot-product attention over the (key, value) pairs.

    Position-independent parameters, so it accepts any N (like the gate
    designs). Outputs per-bit logits for the value bits.
    """

    def __init__(self, key_bits: int, val_bits: int, d_model: int = 32):
        super().__init__()
        self.d = d_model
        self.k_proj = nn.Linear(key_bits, d_model)
        self.q_proj = nn.Linear(key_bits, d_model)
        self.v_proj = nn.Linear(val_bits, d_model)
        self.out = nn.Linear(d_model, val_bits)

    def forward(self, keys, values, query):
        # keys (B,N,kb), values (B,N,vb), query (B,kb) -> (B,vb) logits
        k = self.k_proj(keys)                      # (B,N,d)
        q = self.q_proj(query).unsqueeze(1)        # (B,1,d)
        v = self.v_proj(values)                    # (B,N,d)
        att = torch.softmax((q @ k.transpose(1, 2)) / math.sqrt(self.d), dim=-1)
        return self.out((att @ v).squeeze(1))


class MLPBaseline(nn.Module):
    """Fixed-length MLP over the concatenated bits (no attention). Serves as
    the 'is attention even needed for this task?' control; cannot run on
    lengths other than the one it was built for."""

    def __init__(self, n_pairs: int, key_bits: int, val_bits: int, hidden: int = 256):
        super().__init__()
        in_dim = n_pairs * (key_bits + val_bits) + key_bits
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, val_bits),
        )

    def forward(self, keys, values, query):
        B = keys.shape[0]
        x = torch.cat([keys.reshape(B, -1), values.reshape(B, -1), query], dim=-1)
        return self.net(x)
