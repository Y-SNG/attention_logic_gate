"""Phase 5-1: isolate and close GateLM's soft->hard gap.

Step 1 (diagnosis): load the v0.3 checkpoint and evaluate each branch in
isolation — static circuit, bigram-vote head, unigram-vote head — soft
(at the training temperature) and hardened, to see which branch loses
accuracy under discretization.

Step 2 (remedy): fine-tune the checkpoint with straight-through gates
(forward = argmax gate, i.e. the hardened circuit itself; backward =
softmax gradient). The training objective then IS the hard circuit's
behavior, so the gap closes by construction if optimization cooperates.

Usage: python experiments/phase5_hardgap.py [--ckpt phase3_charlm_v03.pt]
       [--steps 4000]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gatelogic.layers import LogicLayer
from experiments.phase3_charlm import (GateLM, load_corpus, sample_windows,
                                       loss_fn, evaluate, generate,
                                       set_temperature, gate_saturation)

RESULTS = Path(__file__).resolve().parent.parent / "results"


def set_ste(module, on: bool):
    for m in module.modules():
        if isinstance(m, LogicLayer):
            m.ste = on


@torch.no_grad()
def branch_eval(model, ids, g, n_batches=20, batch=32, T=64):
    """Per-branch soft and hard next-char accuracy."""
    def acc_from(logits_fn):
        tot, n = 0.0, 0
        gg = torch.Generator().manual_seed(42)  # same windows for all branches
        for _ in range(n_batches):
            x = sample_windows(ids, batch, T, gg)
            lo = logits_fn(x)[:, 1:-1].reshape(-1, model.vocab)
            tg = x[:, 2:].reshape(-1)
            tot += (lo.argmax(-1) == tg).float().sum().item()
            n += tg.numel()
        return round(tot / n, 4)

    out = {}
    # static branch
    out["static_soft"] = acc_from(lambda x: model.group(model.static(model._pair_bits(x))))
    out["static_hard"] = acc_from(
        lambda x: model.group(model.static.forward_hard(model._pair_bits(x).bool()).float()))
    # bigram-vote branch
    out["bigram_soft"] = acc_from(
        lambda x: model._votes(model.att_enc(model._pair_bits(x)), x, hard=False))
    out["bigram_hard"] = acc_from(
        lambda x: model._votes(model.att_enc.forward_hard(model._pair_bits(x).bool()), x, hard=True))
    if model.uni_enc is not None:
        out["unigram_soft"] = acc_from(
            lambda x: model._votes(model.uni_enc(model._bits(x)), x, hard=False))
        out["unigram_hard"] = acc_from(
            lambda x: model._votes(model.uni_enc.forward_hard(model._bits(x).bool()), x, hard=True))
    out["full_soft"] = acc_from(lambda x: model(x))
    out["full_hard"] = acc_from(lambda x: model.forward_hard(x))
    out["saturation"] = {name: round(gate_saturation(m), 4) for name, m in
                         [("att_enc", model.att_enc), ("static", model.static)]
                         + ([("uni_enc", model.uni_enc)] if model.uni_enc is not None else [])}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="phase3_charlm_v03.pt")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    train_ids, val_ids, chars = load_corpus()
    V = len(chars)
    g = torch.Generator().manual_seed(2000)
    torch.manual_seed(2000)
    model = GateLM(V, g, static_layers=2, two_heads=True)
    model.load_state_dict(torch.load(RESULTS / args.ckpt))
    set_temperature(model, 0.25)  # the temperature v0.3 trained at

    report = {}
    report["diagnosis_v03"] = branch_eval(model, val_ids, g)
    print("diagnosis (v0.3):", json.dumps(report["diagnosis_v03"], indent=1))

    # ---- STE fine-tune: train the hard circuit directly ----
    set_ste(model, True)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    t0 = time.time()
    for step in range(args.steps):
        x = sample_windows(train_ids, args.batch, 64, g)
        loss = loss_fn(model(x), x)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 1000 == 0:
            print(f"  ste step {step+1}/{args.steps} loss {loss.item():.3f} ({time.time()-t0:.0f}s)")
    set_ste(model, False)

    report["after_ste"] = branch_eval(model, val_ids, g)
    print("after STE fine-tune:", json.dumps(report["after_ste"], indent=1))

    bpc, acc = evaluate(model, val_ids, g)
    hbpc, hacc = evaluate(model, val_ids, g, hard=True)
    report["final"] = {"bpc": round(bpc, 4), "acc": round(acc, 4),
                       "hard_bpc": round(hbpc, 4), "hard_acc": round(hacc, 4)}
    print("final:", report["final"])
    sample = generate(model, val_ids, chars, g, top_k=5, temp=0.8)
    report["sample_hard_topk5"] = sample
    print("--- hardened top-k sample ---"); print(sample)

    torch.save(model.state_dict(), RESULTS / "phase5_gatelm_ste.pt")
    (RESULTS / "phase5_hardgap.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("wrote results/phase5_hardgap.json")
