"""Spike step 8: is it just Mistral-DNA? Compare many DNA foundation models, frozen + linear probe.

Run in the `aidays-tf4` env (transformers 4.57.3, as in the KAUST workshop) because DNABERT-2,
HyenaDNA and NT-v2 ship custom code that breaks on transformers 5.

Per model, same splits as the workshop:
  ls      : train 12k random from the training file, test = held-out `ls` (Slots 5-6)
  shuffle : bound `ls` sequences dinucleotide-shuffled vs real unbound (Slot 6), probe from `ls`
  matched : composition-matched benchmark, train 12k, test 3,000 (Slot 7)
  n100/n500: matched, 100 / 500 labels, 3 draws (Slot 7 label curve)
Writes out/s8_models.json incrementally.
"""
import json
import os
import sys
import time
import traceback

if os.environ.get("NO_TRITON") == "1":
    # DNABERT-2's custom code uses a triton flash-attention kernel that fails to compile with modern triton;
    # hiding triton makes it fall back to plain PyTorch attention (same maths, slower).
    sys.modules["triton"] = None

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import OUT, dinuc_shuffle, load_split, norm

DEVICE = torch.device("cuda")
MODELS = {
    "Mistral-DNA-17M (hg38)": "RaphaelMourad/Mistral-DNA-v1-17M-hg38",
    "DNABERT-2-117M (multi-species)": "zhihan1996/DNABERT-2-117M",
    "HyenaDNA-small-32k (hg38)": "LongSafari/hyenadna-small-32k-seqlen-hf",
    "HyenaDNA-medium-160k (hg38)": "LongSafari/hyenadna-medium-160k-seqlen-hf",
    "NT-v2-50M (multi-species)": "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
    "NT-v2-100M (multi-species)": "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
    "NT-v2-250M (multi-species)": "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species",
    "NT-v2-500M (multi-species)": "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
    "NT-500M (human ref)": "InstaDeepAI/nucleotide-transformer-500m-human-ref",
    "NT-2.5B (multi-species)": "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
}
which = sys.argv[1:] or list(MODELS)

# ---------------- data: identical to the workshop notebooks ----------------
seqs, y = load_split("train")
seqs = [norm(s) for s in seqs]
ls_seqs, ls_y = load_split("ls")
ls_seqs = [norm(s) for s in ls_seqs]
tr12 = np.random.default_rng(0).permutation(len(seqs))[:12000]          # workshop_data()


def gc_u(S):
    return np.array([((s.count("G") + s.count("C")) / len(s), s.count("U") / len(s)) for s in S])


def matched_indices(S, y, seed=0):
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


m_idx = matched_indices(seqs, y)
m_te, m_tr = m_idx[:3000], m_idx[3000:3000 + 12000]
rng = np.random.default_rng(0)
bound_ls = [s for s, l in zip(ls_seqs, ls_y) if l == 1]
unbound_ls = [s for s, l in zip(ls_seqs, ls_y) if l == 0]
shuf = [dinuc_shuffle(s, rng) for s in bound_ls]
y_shuf = np.array([1] * len(shuf) + [0] * len(unbound_ls))

ALL = {"tr12": [seqs[i] for i in tr12], "ls": ls_seqs, "shuf": shuf + unbound_ls,
       "m_tr": [seqs[i] for i in m_tr], "m_te": [seqs[i] for i in m_te]}


# ---------------- generic loader + embedder ----------------
def load(repo):
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    if "DNABERT-2" in repo:
        # KAUST's workaround: AutoModel clashes with DNABERT-2's bundled BertConfig on transformers>=4.40,
        # so load its custom BertModel class directly.
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        cfg = AutoConfig.from_pretrained(repo, trust_remote_code=True)
        m = get_class_from_dynamic_module("bert_layers.BertModel", repo).from_pretrained(repo, config=cfg)
    elif "hyenadna" in repo:
        m = AutoModel.from_pretrained(repo, trust_remote_code=True)
    else:
        try:
            m = AutoModelForMaskedLM.from_pretrained(repo, trust_remote_code=True)
        except Exception:
            m = AutoModel.from_pretrained(repo, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    return tok, m.to(DEVICE).eval()


@torch.no_grad()
def embed(tok, m, S, bs=64, fp16=True):
    out = []
    for i in range(0, len(S), bs):
        e = tok([s.replace("U", "T") for s in S[i:i + bs]], return_tensors="pt", padding=True,
                truncation=True, max_length=512)
        ids = e["input_ids"].to(DEVICE)
        mask = e["attention_mask"].to(DEVICE) if "attention_mask" in e else (ids != tok.pad_token_id).long()
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
            try:
                o = m(input_ids=ids, attention_mask=mask, output_hidden_states=True)
                h = o.hidden_states[-1] if getattr(o, "hidden_states", None) is not None else o[0]
            except TypeError:
                o = m(input_ids=ids)
                h = o[0] if isinstance(o, (tuple, list)) else o.last_hidden_state
        h = h.float()
        mk = mask.unsqueeze(-1).float()
        out.append(((h * mk).sum(1) / mk.sum(1)).cpu().numpy())
    return np.concatenate(out)


def probe(Etr, ytr, Ete, yte, C=0.1):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=C))
    clf.fit(Etr, ytr)
    return roc_auc_score(yte, clf.predict_proba(Ete)[:, 1]), clf


if __name__ == "__main__":
    out_path = OUT / "s8_models.json"
    R = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in which:
        repo = MODELS[name]
        r = {"repo": repo}
        t0 = time.time()
        try:
            tok, m = load(repo)
            r["load_s"] = round(time.time() - t0, 1)
            r["params_M"] = round(sum(p.numel() for p in m.parameters()) / 1e6, 1)
            t = time.time()
            E = {k: embed(tok, m, v) for k, v in ALL.items()}
            r["embed_s"] = round(time.time() - t, 1)
            r["ls"], clf = probe(E["tr12"], y[tr12], E["ls"], ls_y)
            r["shuffle"] = roc_auc_score(y_shuf, clf.predict_proba(E["shuf"])[:, 1])
            r["matched"], _ = probe(E["m_tr"], y[m_tr], E["m_te"], y[m_te])
            for n in (100, 500):
                vals = []
                for d in range(3):
                    sel = np.random.default_rng(100 + d).choice(len(m_tr), n, replace=False)
                    vals.append(probe(E["m_tr"][sel], y[m_tr][sel], E["m_te"], y[m_te], C=0.01)[0])
                r[f"n{n}"] = float(np.mean(vals))
            del m
            torch.cuda.empty_cache()
        except Exception:
            r["error"] = traceback.format_exc()[-1500:]
        R[name] = r
        out_path.write_text(json.dumps(R, indent=1))
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "repo"}, flush=True)
