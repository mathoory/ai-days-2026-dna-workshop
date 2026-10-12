"""Spike step 5: the 'fix' - composition-matched negatives, core-only embeddings, LoRA fine-tuning.

Rebuild negatives so positives and negatives have the same GC and U content (2-D bins) and
length, then rerun every model on both the original and the matched benchmark:
  GC-only LR, 5-mer LR (core / full), CNN (10 epochs), FM probe (full seq / core only),
  LoRA fine-tune of Mistral-DNA-17M.
"""
import os
import sys
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from common import OUT, core_of, encode_batch, kmer_counts, load_split, norm, save_json, timer

DEVICE = torch.device("cuda")
steps = sys.argv[1:] or ["build", "baselines", "cnn", "probe", "lora"]
T, R = {}, {}
rng = np.random.default_rng(0)

tr_seqs, tr_y = load_split("train")
ls_seqs, ls_y = load_split("ls")


def gc_u(s):
    s = norm(s)
    n = max(len(s), 1)
    return (s.count("G") + s.count("C")) / n, s.count("U") / n


# ---------------- build matched benchmark ----------------
def build_matched():
    feats = np.array([gc_u(s) for s in tr_seqs])
    L = np.array([len(s) for s in tr_seqs])
    pos = np.nonzero(tr_y == 1)[0]
    neg = np.nonzero(tr_y == 0)[0]
    bins = 12
    gb = np.clip((feats[:, 0] * bins / 0.8).astype(int), 0, bins - 1)  # GC 0..0.8
    ub = np.clip((feats[:, 1] * bins / 0.6).astype(int), 0, bins - 1)  # U 0..0.6
    lb = np.clip(((L - 130) // 60).astype(int), 0, 4)
    key = gb * 100 + ub * 10 + lb
    keep_pos, keep_neg = [], []
    for k in np.unique(key[pos]):
        p = pos[key[pos] == k]
        n = neg[key[neg] == k]
        m = min(len(p), len(n))
        if m == 0:
            continue
        keep_pos += list(rng.choice(p, m, replace=False))
        keep_neg += list(rng.choice(n, m, replace=False))
    idx = np.array(keep_pos + keep_neg)
    rng.shuffle(idx)
    k = len(idx) // 10
    return idx[k:], idx[:k]


if "build" in steps or not (OUT / "matched_split.npz").exists():
    with timer(T, "build_matched"):
        m_tr, m_te = build_matched()
    np.savez_compressed(OUT / "matched_split.npz", tr=m_tr, te=m_te)
d = np.load(OUT / "matched_split.npz")
m_tr, m_te = d["tr"], d["te"]
R["matched_sizes"] = {"train": int(len(m_tr)), "test": int(len(m_te)),
                      "pos_kept_frac": float((tr_y[np.concatenate([m_tr, m_te])] == 1).sum() / (tr_y == 1).sum())}
print(R["matched_sizes"])

# original benchmark: random 90/10 of the same file (same size as matched test for fairness of noise)
perm = rng.permutation(len(tr_seqs))
o_te, o_tr = perm[: len(m_te)], perm[len(m_te):]
BENCH = {"original": (o_tr, o_te), "matched": (m_tr, m_te)}


def lr_auc(Ftr, ytr, Fte, yte):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    clf.fit(Ftr, ytr)
    return roc_auc_score(yte, clf.predict_proba(Fte)[:, 1])


if "baselines" in steps:
    out = {}
    for b, (a, z) in BENCH.items():
        S = tr_seqs
        F_gc = np.array([gc_u(s) for s in S])
        out[b] = {"gc_u": lr_auc(F_gc[a], tr_y[a], F_gc[z], tr_y[z])}
        cores = [core_of(s) for s in S]
        K5c = kmer_counts(cores, 5)
        K5f = kmer_counts(S, 5)
        out[b]["k5_core"] = lr_auc(K5c[a], tr_y[a], K5c[z], tr_y[z])
        out[b]["k5_full"] = lr_auc(K5f[a], tr_y[a], K5f[z], tr_y[z])
        print(b, out[b], flush=True)
    R["baselines"] = out


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
def cnn_score(model, X, bs=1024):
    model.eval()
    return np.concatenate([torch.sigmoid(model(X[i:i + bs].to(DEVICE))).cpu().numpy() for i in range(0, len(X), bs)])


if "cnn" in steps:
    X = encode_batch(tr_seqs)
    y = torch.tensor(tr_y, dtype=torch.float32)
    out = {}
    for b, (a, z) in BENCH.items():
        torch.manual_seed(0)
        model = GlobalCNN().to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss()
        dl = DataLoader(TensorDataset(X[a], y[a]), batch_size=256, shuffle=True, drop_last=True)
        t = time.time()
        for ep in range(10):
            model.train()
            for xb, yb in dl:
                opt.zero_grad()
                crit(model(xb.to(DEVICE)), yb.to(DEVICE)).backward()
                opt.step()
        out[b] = {"auc": roc_auc_score(tr_y[z], cnn_score(model, X[z])), "train_s": round(time.time() - t, 1)}
        torch.save(model.state_dict(), OUT / f"cnn_{b}.pt")
        print("cnn", b, out[b], flush=True)
    R["cnn"] = out

# ---------------- FM probe: full vs core, original vs matched ----------------
FM = {"mistral17m": "RaphaelMourad/Mistral-DNA-v1-17M-hg38",
      "nt500m": "InstaDeepAI/nucleotide-transformer-500m-human-ref"}


def load_fm(repo):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    m = AutoModel.from_pretrained(repo, trust_remote_code=True).to(DEVICE).eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    return tok, m


@torch.no_grad()
def embed(tok, m, seqs, bs=128):
    out = []
    for i in range(0, len(seqs), bs):
        e = tok(seqs[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=512)
        e = {k: e[k].to(DEVICE) for k in ("input_ids", "attention_mask")}
        with torch.autocast("cuda", dtype=torch.float16):
            h = m(**e, output_hidden_states=True).hidden_states[-1].float()
        mk = e["attention_mask"].unsqueeze(-1).float()
        out.append(((h * mk).sum(1) / mk.sum(1)).cpu().numpy())
    return np.concatenate(out)


if "probe" in steps:
    out = {}
    N = 12000  # cap training size like the workshop would
    for name, repo in FM.items():
        tok, m = load_fm(repo)
        out[name] = {}
        for region in ("full", "core"):
            S = [(core_of(s) if region == "core" else s) for s in tr_seqs]
            S = [norm(s).replace("U", "T") for s in S]
            for b, (a, z) in BENCH.items():
                a = a[:N]
                t = time.time()
                Ea, Ez = embed(tok, m, [S[i] for i in a]), embed(tok, m, [S[i] for i in z])
                emb_s = time.time() - t
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))
                clf.fit(Ea, tr_y[a])
                out[name][f"{region}:{b}"] = {"auc": roc_auc_score(tr_y[z], clf.predict_proba(Ez)[:, 1]),
                                             "embed_s": round(emb_s, 1), "n_train": int(len(a))}
                print(name, region, b, out[name][f"{region}:{b}"], flush=True)
        del m
        torch.cuda.empty_cache()
    R["probe"] = out

# ---------------- LoRA fine-tuning of Mistral-DNA-17M ----------------
if "lora" in steps:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    repo = FM["mistral17m"]
    out = {}
    for b, (a, z) in BENCH.items():
        a = a[:20000]
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token or tok.unk_token
        model = AutoModelForSequenceClassification.from_pretrained(repo, num_labels=1, trust_remote_code=True)
        model.config.pad_token_id = tok.pad_token_id
        cfg = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="SEQ_CLS",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        model = get_peft_model(model, cfg).to(DEVICE)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        S = [norm(s).replace("U", "T") for s in tr_seqs]
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4)
        crit = nn.BCEWithLogitsLoss()

        @torch.no_grad()
        def evaluate():
            model.eval()
            ps = []
            for i in range(0, len(z), 256):
                e = tok([S[j] for j in z[i:i + 256]], return_tensors="pt", padding=True, truncation=True, max_length=256)
                e = {k: e[k].to(DEVICE) for k in ("input_ids", "attention_mask")}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ps.append(model(**e).logits.float().squeeze(-1).cpu().numpy())
            model.train()
            return roc_auc_score(tr_y[z], np.concatenate(ps))

        hist, t0 = [], time.time()
        for ep in range(3):
            pa = rng.permutation(a)
            model.train()
            for i in range(0, len(pa), 32):
                bi = pa[i:i + 32]
                e = tok([S[j] for j in bi], return_tensors="pt", padding=True, truncation=True, max_length=256)
                e = {k: e[k].to(DEVICE) for k in ("input_ids", "attention_mask")}
                yb = torch.tensor(tr_y[bi], dtype=torch.float32, device=DEVICE)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logit = model(**e).logits.float().squeeze(-1)
                loss = crit(logit, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
            hist.append({"epoch": ep + 1, "auc": evaluate(), "min": round((time.time() - t0) / 60, 2)})
            print("lora", b, hist[-1], flush=True)
        out[b] = {"trainable_params": int(trainable), "n_train": int(len(a)), "history": hist}
        del model
        torch.cuda.empty_cache()
    R["lora_mistral17m"] = out

R["timings"] = T
save_json(R, f"s5_fix_{'_'.join(steps)}.json")
