"""Phase 3b: a first boolean-circuit character LM (tiny Shakespeare).

Architecture "GateLM v0" — two branches over a T-char window, combined:

  induction branch: CountingGateAttention on BIGRAM keys — the key at
    position j is the learned code of (x_{j-1}, x_j), its value is x_{j+1};
    the query at t is the code of (x_{t-1}, x_t). Exact code match casts an
    integer vote for the character that followed the earlier occurrence:
    in-context trigram copying, no position parameters, any T at inference.
  static branch: a logic-gate circuit over the same bigram bits with a
    GroupSum readout — a learned static trigram-ish predictor in circuit
    form (the difflogic-native classifier head).

  logits = alpha * votes + GroupSum(static) / tau, alpha learned.

Hardened inference is fully discrete: integer votes + integer group sums,
combined by a frozen affine readout, then argmax.

References reported alongside: count-based bigram/trigram models with
Laplace smoothing (the classical ceiling for static prediction) and a
2-layer transformer at the same context length. Metrics: soft bpc and
top-1 next-char accuracy on held-out text; hard top-1 for the gate model;
plus a greedy generation sample from the hardened circuit.

Usage: python experiments/phase3_charlm.py [--steps 6000]
"""

import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gatelogic.layers import GateEncoder, GroupSum, LogicLayer
from gatelogic.attention import _soft_xnor


def set_temperature(model: nn.Module, temp: float):
    for m in model.modules():
        if isinstance(m, LogicLayer):
            m.temp = temp


@torch.no_grad()
def gate_saturation(model: nn.Module) -> float:
    """Mean max softmax prob over all gates — 1.0 means fully committed."""
    probs = [F.softmax(m.weights, -1).max(-1).values.mean().item()
             for m in model.modules() if isinstance(m, LogicLayer)]
    return sum(probs) / len(probs)

RESULTS = Path(__file__).resolve().parent.parent / "results"
DATA = Path(__file__).resolve().parent.parent / "data"
CORPUS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

T_TRAIN = 64
CODE_BITS, ATT_HIDDEN = 16, 128
STATIC_HIDDEN, STATIC_PER_CLASS = 1024, 16


# --------------------------------------------------------------------- data

def load_corpus():
    DATA.mkdir(exist_ok=True)
    path = DATA / "tinyshakespeare.txt"
    if not path.exists():
        urllib.request.urlretrieve(CORPUS_URL, path)
    text = path.read_text()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(len(ids) * 0.9)
    return ids[:n], ids[n:], chars


def sample_windows(ids: torch.Tensor, batch: int, T: int, g: torch.Generator):
    starts = torch.randint(0, len(ids) - T - 1, (batch,), generator=g)
    return torch.stack([ids[s:s + T] for s in starts])


# -------------------------------------------------------------------- model

class GateLM(nn.Module):
    def __init__(self, vocab: int, g: torch.Generator,
                 static_layers: int = 1, two_heads: bool = False):
        super().__init__()
        self.vocab = vocab
        self.tb = (vocab - 1).bit_length()
        pair = 2 * self.tb
        self.att_enc = GateEncoder([pair, ATT_HIDDEN, CODE_BITS], g, residual_init=True)
        self.uni_enc = (GateEncoder([self.tb, ATT_HIDDEN, CODE_BITS], g, residual_init=True)
                        if two_heads else None)
        dims = [pair] + [STATIC_HIDDEN] * static_layers + [vocab * STATIC_PER_CLASS]
        self.static = GateEncoder(dims, g, residual_init=True)
        self.group = GroupSum(vocab, tau=4.0)
        self.alpha = nn.Parameter(torch.tensor(3.0))   # bigram-vote scale
        self.alpha2 = nn.Parameter(torch.tensor(1.0))  # unigram-vote scale
        self.beta = nn.Parameter(torch.tensor(1.0))    # static-branch scale

    def _bits(self, x):
        shifts = torch.arange(self.tb - 1, -1, -1)
        return ((x.unsqueeze(-1) >> shifts) & 1).float()

    def _pair_bits(self, x):
        b = self._bits(x)                       # (B,T,tb)
        prev = torch.zeros_like(b); prev[:, 1:] = b[:, :-1]
        return torch.cat([prev, b], dim=-1)     # (B,T,2tb); t=0 has zero prev

    def _votes(self, codes, x, hard: bool):
        # codes (B,T,cb); value at j is x_{j+1}; query t may match j < t
        if hard:
            match = (codes.unsqueeze(2) == codes.unsqueeze(1)).all(-1).float()
        else:
            match = _soft_xnor(codes.unsqueeze(2), codes.unsqueeze(1)).prod(-1)
        T = x.shape[1]
        mask = torch.ones(T, T).tril(-1)
        v = F.one_hot(x, self.vocab).float()
        val = torch.zeros_like(v); val[:, :-1] = v[:, 1:]
        return (match * mask) @ val             # (B,T,V)

    def forward(self, x, static_only: bool = False):
        pb = self._pair_bits(x)
        static = self.beta * self.group(self.static(pb))
        if static_only:  # stage 1: let the static circuit mature undominated
            return static
        out = static + self.alpha * self._votes(self.att_enc(pb), x, hard=False)
        if self.uni_enc is not None:
            out = out + self.alpha2 * self._votes(self.uni_enc(self._bits(x)), x, hard=False)
        return out

    @torch.no_grad()
    def forward_hard(self, x):
        pb = self._pair_bits(x).bool()
        votes = self._votes(self.att_enc.forward_hard(pb), x, hard=True)
        static = self.group(self.static.forward_hard(pb).float())
        # frozen affine readout over integer-derived quantities
        out = (float(self.alpha.detach()) * votes
               + float(self.beta.detach()) * static)
        if self.uni_enc is not None:
            uv = self._votes(self.uni_enc.forward_hard(self._bits(x).bool()), x, hard=True)
            out = out + float(self.alpha2.detach()) * uv
        return out


class TinyTransformer(nn.Module):
    def __init__(self, vocab: int, T: int, d: int = 64):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(T, d)
        layer = nn.TransformerEncoderLayer(d, nhead=2, dim_feedforward=128,
                                           batch_first=True, dropout=0.0)
        self.blocks = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        B, T = x.shape
        h = self.emb(x) + self.pos(torch.arange(T)).unsqueeze(0)
        return self.head(self.blocks(h, mask=nn.Transformer.generate_square_subsequent_mask(T)))


# ------------------------------------------------------------- count models

def ngram_bpc(train_ids, val_ids, vocab, order, alpha=0.5):
    """order=1: bigram P(x_t | x_{t-1}); order=2: trigram."""
    if order == 1:
        counts = torch.zeros(vocab, vocab)
        counts.index_put_((train_ids[:-1], train_ids[1:]),
                          torch.ones(len(train_ids) - 1), accumulate=True)
        probs = (counts + alpha) / (counts + alpha).sum(-1, keepdim=True)
        p = probs[val_ids[:-1], val_ids[1:]]
        acc = (probs[val_ids[:-1]].argmax(-1) == val_ids[1:]).float().mean().item()
    else:
        ctx = train_ids[:-2] * vocab + train_ids[1:-1]
        counts = torch.zeros(vocab * vocab, vocab)
        counts.index_put_((ctx, train_ids[2:]), torch.ones(len(ctx)), accumulate=True)
        probs = (counts + alpha) / (counts + alpha).sum(-1, keepdim=True)
        vctx = val_ids[:-2] * vocab + val_ids[1:-1]
        p = probs[vctx, val_ids[2:]]
        acc = (probs[vctx].argmax(-1) == val_ids[2:]).float().mean().item()
    return -p.log2().mean().item(), acc


# ----------------------------------------------------------- train and eval

def loss_fn(logits, x):
    # predict x_{t+1} from positions t = 1 .. T-2 (t=0 lacks a real bigram)
    return F.cross_entropy(logits[:, 1:-1].reshape(-1, logits.shape[-1]),
                           x[:, 2:].reshape(-1))


@torch.no_grad()
def evaluate(model, ids, g, hard=False, n_batches=20, batch=32, T=T_TRAIN):
    tot_nll, tot_acc, n = 0.0, 0.0, 0
    for _ in range(n_batches):
        x = sample_windows(ids, batch, T, g)
        logits = model.forward_hard(x) if hard else model(x)
        lo, tg = logits[:, 1:-1].reshape(-1, logits.shape[-1]), x[:, 2:].reshape(-1)
        tot_nll += F.cross_entropy(lo, tg, reduction="sum").item()
        tot_acc += (lo.argmax(-1) == tg).float().sum().item()
        n += tg.numel()
    return tot_nll / n / math.log(2), tot_acc / n


@torch.no_grad()
def generate(model, ids, chars, g, prime_len=64, gen_len=200, top_k=0, temp=0.8):
    s = int(torch.randint(0, len(ids) - prime_len, (1,), generator=g))
    x = ids[s:s + prime_len].unsqueeze(0).clone()
    for _ in range(gen_len):
        # the last position's bigram is valid and predicts the next unseen char
        logits = model.forward_hard(x[:, -T_TRAIN:])[:, -1]
        if top_k > 0:
            vals, idx = logits.topk(top_k, dim=-1)
            p = F.softmax(vals / temp, dim=-1)
            nxt = idx.gather(-1, torch.multinomial(p, 1, generator=g))
        else:
            nxt = logits.argmax(-1, keepdim=True)
        x = torch.cat([x, nxt], dim=1)
    text = "".join(chars[int(i)] for i in x[0])
    return text[:prime_len] + " ▌ " + text[prime_len:]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--staged", action="store_true",
                    help="stage 1 (50%%): static branch only; stage 2: full model"
                         " - the Phase 3a co-adaptation lesson applied to the LM")
    ap.add_argument("--temp-anneal", action="store_true",
                    help="cool gate softmax temperature 1.0 -> 0.25 between"
                         " 30%% and 60%% of training, then hold, to close the"
                         " soft->hard gap")
    ap.add_argument("--static-layers", type=int, default=1)
    ap.add_argument("--two-heads", action="store_true",
                    help="add a unigram-key vote head alongside the bigram one")
    ap.add_argument("--out", type=str, default="phase3_charlm.json")
    ap.add_argument("--skip-baselines", action="store_true")
    args = ap.parse_args()

    train_ids, val_ids, chars = load_corpus()
    V = len(chars)
    print(f"corpus: {len(train_ids)} train / {len(val_ids)} val chars, vocab {V}")
    results = {"vocab": V, "T": T_TRAIN}

    if not args.skip_baselines:
        for order, name in [(1, "bigram-count"), (2, "trigram-count")]:
            bpc, acc = ngram_bpc(train_ids, val_ids, V, order)
            results[name] = {"bpc": round(bpc, 4), "acc": round(acc, 4)}
            print(name, results[name])

    for name in (["gate-lm"] if args.skip_baselines else ["gate-lm", "transformer-2L"]):
        torch.manual_seed(1000)
        g = torch.Generator().manual_seed(1000)
        model = (GateLM(V, g, static_layers=args.static_layers, two_heads=args.two_heads)
                 if name == "gate-lm" else TinyTransformer(V, T_TRAIN))
        opt = torch.optim.Adam(model.parameters(), lr=0.03 if name == "gate-lm" else 1e-3)
        t0 = time.time()
        for step in range(args.steps):
            x = sample_windows(train_ids, args.batch, T_TRAIN, g)
            if name == "gate-lm" and args.temp_anneal:
                # cool between 30% and 60%, then TRAIN at the low temperature
                # for the remaining 40% so the circuit adapts to it
                frac = step / args.steps
                set_temperature(model, 1.0 - 0.75 * min(1.0, max(0.0, (frac - 0.3) / 0.3)))
            static_only = (name == "gate-lm" and args.staged
                           and step < args.steps // 2)
            logits = model(x, static_only=static_only) if name == "gate-lm" else model(x)
            loss = loss_fn(logits, x)
            opt.zero_grad(); loss.backward(); opt.step()
            if (step + 1) % 1000 == 0:
                print(f"  {name} step {step+1}/{args.steps} loss {loss.item():.3f} "
                      f"({time.time()-t0:.0f}s)")
        bpc, acc = evaluate(model, val_ids, g)
        entry = {"bpc": round(bpc, 4), "acc": round(acc, 4),
                 "train_s": round(time.time() - t0, 1)}
        if name == "gate-lm":
            hbpc, hacc = evaluate(model, val_ids, g, hard=True)
            entry["hard_bpc"], entry["hard_acc"] = round(hbpc, 4), round(hacc, 4)
            entry["alpha"] = round(float(model.alpha.detach()), 3)
            entry["beta"] = round(float(model.beta.detach()), 3)
            entry["gate_saturation"] = round(gate_saturation(model), 4)
            sample = generate(model, val_ids, chars, g)
            entry["sample_hard"] = sample
            print("--- hardened greedy sample (prime ▌ generation) ---")
            print(sample)
            sample_k = generate(model, val_ids, chars, g, top_k=5, temp=0.8)
            entry["sample_hard_topk5"] = sample_k
            print("--- hardened top-k=5 sample ---")
            print(sample_k)
        results[name] = entry
        print(name, entry)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"wrote results/{args.out}")
