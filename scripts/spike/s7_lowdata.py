"""Spike step 7: does pretraining pay off when labels are scarce?

On the composition-matched benchmark (Slot 7's), train with n labelled sequences
(n = 100 ... all) and compare, on the same 3,000-sequence test set:
  - CNN from scratch (Slot 2 architecture; epochs scaled so every n gets >= ~1,500 steps, early
    stopping on a 10% validation split of the n sequences)
  - 5-mer counts + logistic regression
  - Mistral-DNA frozen embeddings + linear probe
  - Mistral-DNA RANDOM-weight embeddings + linear probe (the "random features" control)
3 random draws of the n sequences per n. Writes out/s7_lowdata.json.
"""
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn, optim

from common import OUT, encode_batch, kmer_counts, load_split, norm, save_json

DEVICE = torch.device("cuda")
NS = [100, 250, 500, 1000, 2500, 5000, 12000, 30000]
DRAWS = 3
FM = "RaphaelMourad/Mistral-DNA-v1-17M-hg38"

seqs, y = load_split("train")
seqs = [norm(s) for s in seqs]


def gc_u(S):
    return np.array([((s.count("G") + s.count("C")) / len(s), s.count("U") / len(s)) for s in S])


def matched_indices(S, y, seed=0):  # identical to Slot 7
    rng = np.random.default_rng(seed)
    C = gc_u(S)
    L = np.array([len(s) for s in S])
    key = (np.clip((C[:, 0] * 12 / 0.8).astype(int), 0, 11) * 100
           + np.clip((C[:, 1] * 12 / 0.6).astype(int), 0, 11) * 10 + np.clip((L - 130) // 60, 0, 4))
    keep = []
    for k in np.unique(key[y == 1]):
        p = np.nonzero((key == k) & (y == 1))[0]
        n = np.nonzero((key == k) & (y == 0))[0]
        m = min(len(p), len(n))
        keep += list(rng.choice(p, m, replace=False)) + list(rng.choice(n, m, replace=False))
    keep = np.array(keep)
    rng.shuffle(keep)
    return keep


idx = matched_indices(seqs, y)
te, pool = idx[:3000], idx[3000:]
S_te, y_te = [seqs[i] for i in te], y[te]
print(f"matched pool {len(pool):,} | test {len(te):,}")

# ---- precompute features for the pool (first 30k) and the test set ----
pool = pool[:max(NS)]
from transformers import AutoConfig, AutoModel, AutoTokenizer
tok = AutoTokenizer.from_pretrained(FM, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token or tok.unk_token


@torch.no_grad()
def embed(model, S, bs=256):
    out = []
    for i in range(0, len(S), bs):
        e = tok([s.replace("U", "T") for s in S[i:i + bs]], return_tensors="pt", padding=True, truncation=True, max_length=512)
        e = {k: e[k].to(DEVICE) for k in ("input_ids", "attention_mask")}
        with torch.autocast("cuda", dtype=torch.float16):
            h = model(**e, output_hidden_states=True).hidden_states[-1].float()
        m = e["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
    return np.concatenate(out)


S_pool = [seqs[i] for i in pool]
t = time.time()
fm = AutoModel.from_pretrained(FM).to(DEVICE).eval()
E_pool, E_te = embed(fm, S_pool), embed(fm, S_te)
torch.manual_seed(0)
fm0 = AutoModel.from_config(AutoConfig.from_pretrained(FM)).to(DEVICE).eval()
R_pool, R_te = embed(fm0, S_pool), embed(fm0, S_te)
K_pool, K_te = kmer_counts(S_pool, 5), kmer_counts(S_te, 5)
print(f"features in {time.time() - t:.0f}s")
X_pool, X_te = encode_batch(S_pool), encode_batch(S_te)
y_pool = y[pool]


def lr_auc(Ftr, ytr, Fte, C):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=C))
    clf.fit(Ftr, ytr)
    return roc_auc_score(y_te, clf.predict_proba(Fte)[:, 1])


class GlobalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 10, (4, 10)), nn.ReLU(), nn.MaxPool2d((1, 3)),
            nn.Conv2d(10, 20, (1, 5)), nn.ReLU(), nn.MaxPool2d((1, 3)),
            nn.Flatten(), nn.Linear(20 * 53, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


@torch.no_grad()
def cnn_scores(model, X):
    model.eval()
    return np.concatenate([torch.sigmoid(model(X[i:i + 2048].to(DEVICE))).cpu().numpy() for i in range(0, len(X), 2048)])


def cnn_auc(sel, seed):
    """Train from scratch on `sel`; early-stop on a 10% split of `sel` (no peeking at the test set)."""
    rng = np.random.default_rng(seed)
    sel = rng.permutation(sel)
    nv = max(len(sel) // 10, 20)
    va, tr = sel[:nv], sel[nv:]
    torch.manual_seed(seed)
    model = GlobalCNN().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    Xtr, ytr = X_pool[tr], torch.tensor(y_pool[tr], dtype=torch.float32)
    bs = min(256, max(16, len(tr) // 8))
    steps_per_epoch = max(len(tr) // bs, 1)
    epochs = int(np.clip(1500 / steps_per_epoch, 6, 200))
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tr))
        for i in range(0, len(tr), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            crit(model(Xtr[b].to(DEVICE)), ytr[b].to(DEVICE)).backward()
            opt.step()
        v = roc_auc_score(y_pool[va], cnn_scores(model, X_pool[va])) if len(set(y_pool[va])) > 1 else 0
        if v > best:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 10:
                break
    model.load_state_dict(best_state)
    return roc_auc_score(y_te, cnn_scores(model, X_te))


res = {}
for n in NS:
    row = {k: [] for k in ("cnn", "kmer5", "fm_probe", "random_probe")}
    for d in range(DRAWS):
        sel = np.random.default_rng(100 + d).choice(len(pool), n, replace=False)
        C = 0.1 if n >= 2500 else 0.01
        row["fm_probe"].append(lr_auc(E_pool[sel], y_pool[sel], E_te, C))
        row["random_probe"].append(lr_auc(R_pool[sel], y_pool[sel], R_te, C))
        row["kmer5"].append(lr_auc(K_pool[sel], y_pool[sel], K_te, C))
        row["cnn"].append(cnn_auc(sel, d))
    res[n] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v)), "runs": v} for k, v in row.items()}
    print(f"n={n:6d} | " + " | ".join(f"{k} {res[n][k]['mean']:.3f}±{res[n][k]['sd']:.3f}" for k in row), flush=True)
save_json(res, "s7_lowdata.json")
