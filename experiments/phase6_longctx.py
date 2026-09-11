"""Phase 6-2: do the vote heads pay off at longer contexts?

The vote heads have no position parameters, so a trained GateLM runs at
ANY window length without retraining. This script loads a checkpoint and
evaluates the hardened circuit at T = 64 / 128 / 256:

  - full hard accuracy,
  - votes-only hard accuracy (alpha*bigram + alpha2*unigram),
  - static-only hard accuracy (per-position, so T-independent — a control),
  - copy availability: the fraction of positions whose bigram context
    already occurred earlier in the window (the ceiling on what
    in-context copying can even address).

Usage: python experiments/phase6_longctx.py [--ckpt phase6_gatelm_v04.pt]
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.phase3_charlm import GateLM, load_corpus, sample_windows

RESULTS = Path(__file__).resolve().parent.parent / "results"


@torch.no_grad()
def eval_T(model, ids, T, n_batches=12, batch=16):
    g = torch.Generator().manual_seed(42)
    stats = {"full": 0.0, "votes": 0.0, "static": 0.0, "copyable": 0.0, "n": 0}
    a1 = float(model.alpha.detach())
    a2 = float(model.alpha2.detach())
    b = float(model.beta.detach())
    for _ in range(n_batches):
        x = sample_windows(ids, batch, T, g)
        pb = model._pair_bits(x).bool()
        votes = a1 * model._votes(model.att_enc.forward_hard(pb), x, hard=True)
        if model.uni_enc is not None:
            votes = votes + a2 * model._votes(
                model.uni_enc.forward_hard(model._bits(x).bool()), x, hard=True)
        static = b * model.group(model.static.forward_hard(pb).float())
        sl_in, sl_out = slice(1, T - 1), slice(2, T)
        tg = x[:, sl_out].reshape(-1)
        for key, logits in [("full", votes + static), ("votes", votes), ("static", static)]:
            pred = logits[:, sl_in].reshape(-1, model.vocab).argmax(-1)
            stats[key] += (pred == tg).float().sum().item()
        # copy availability: exact bigram (x_{t-1}, x_t) seen at some j < t
        big = x[:, :-1] * model.vocab + x[:, 1:]              # (B, T-1), bigram ending at t>=1
        same = (big.unsqueeze(2) == big.unsqueeze(1))          # (B, T-1, T-1)
        mask = torch.ones(T - 1, T - 1, dtype=torch.bool).tril(-1)
        seen = (same & mask).any(-1)                           # (B, T-1) for t = 1..T-1
        stats["copyable"] += seen[:, :-1].reshape(-1).float().sum().item()  # t = 1..T-2
        stats["n"] += tg.numel()
    n = stats.pop("n")
    return {k: round(v / n, 4) for k, v in stats.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="phase6_gatelm_v04.pt")
    args = ap.parse_args()

    train_ids, val_ids, chars = load_corpus()
    g = torch.Generator().manual_seed(0)
    model = GateLM(len(chars), g, static_layers=2, two_heads=True)
    model.load_state_dict(torch.load(RESULTS / args.ckpt))

    report = {}
    for T in [64, 128, 256]:
        report[f"T={T}"] = eval_T(model, val_ids, T)
        print(f"T={T}:", report[f"T={T}"])

    (RESULTS / "phase6_longctx.json").write_text(json.dumps(report, indent=2))
    print("wrote results/phase6_longctx.json")
