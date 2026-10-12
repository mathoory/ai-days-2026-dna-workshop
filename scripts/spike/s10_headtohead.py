"""Spike step 10: a fair head-to-head on the composition-matched benchmark.

Same labelled subsets (the 20,000 and the 500 that s9_finetune.run draws with rng(0)), same
3,000-sequence test set, 3 seeds each:
  - CNN from scratch (Slot 2 architecture, early stopping on 10% of the labels)
  - NT-500M frozen + linear probe
  - NT-500M + LoRA (s9_finetune recipe)
Writes out/s10_headtohead.json. Run with no arguments (importing s9_finetune must not start its loop).
"""
import json
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn, optim

import s9_finetune as s9
from common import OUT, encode_batch
from s8_models import MODELS, embed, load

DEVICE = torch.device("cuda")
NAME = "NT-500M (human ref)"
SEEDS = [0, 1, 2]
S_te, y_te = s9.S_te, s9.y_te
seqs, y, m_tr = s9.seqs, s9.y, s9.m_tr


def subset(n):
    rng = np.random.default_rng(0)
    return rng.choice(m_tr, n, replace=False) if n < len(m_tr) else m_tr


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
    return np.concatenate([torch.sigmoid(model(X[i:i + 2048].to(DEVICE))).cpu().numpy()
                           for i in range(0, len(X), 2048)])


X_te = encode_batch(S_te)


def cnn_run(sel, seed):
    rng = np.random.default_rng(seed)
    sel = rng.permutation(sel)
    nv = max(len(sel) // 10, 20)
    va, tr = sel[:nv], sel[nv:]
    Xtr, ytr = encode_batch([seqs[i] for i in tr]), torch.tensor(y[tr], dtype=torch.float32)
    Xva = encode_batch([seqs[i] for i in va])
    torch.manual_seed(seed)
    model = GlobalCNN().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    bs = min(256, max(16, len(tr) // 8))
    epochs = int(np.clip(1500 / max(len(tr) // bs, 1), 6, 200))
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tr))
        for i in range(0, len(tr), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            crit(model(Xtr[b].to(DEVICE)), ytr[b].to(DEVICE)).backward()
            opt.step()
        v = roc_auc_score(y[va], cnn_scores(model, Xva))
        if v > best:
            best, bad, best_state = v, 0, {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 10:
                break
    model.load_state_dict(best_state)
    return roc_auc_score(y_te, cnn_scores(model, X_te))


R = {}
for n, epochs, pat in ((20000, 2, None), (500, 30, 5)):
    sel = subset(n)
    row = {"cnn": [cnn_run(sel, s) for s in SEEDS]}
    print(n, "cnn", row["cnn"], flush=True)
    tok, m = load(MODELS[NAME])
    Etr, Ete = embed(tok, m, [seqs[i] for i in sel]), embed(tok, m, S_te)
    del m
    torch.cuda.empty_cache()
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.1 if n >= 2500 else 0.01))
    clf.fit(Etr, y[sel])
    row["frozen_probe"] = [roc_auc_score(y_te, clf.predict_proba(Ete)[:, 1])]
    print(n, "frozen", row["frozen_probe"], flush=True)
    row["lora"] = []
    for s in SEEDS:
        row["lora"].append(s9.run(NAME, n, epochs, pat, seed=s)["auc"])
        torch.cuda.empty_cache()
        print(n, "lora", row["lora"], flush=True)
    R[n] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v)), "runs": v} for k, v in row.items()}
    (OUT / "s10_headtohead.json").write_text(json.dumps(R, indent=1))
print(json.dumps(R, indent=1))
