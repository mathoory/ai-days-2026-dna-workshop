"""Spike step 3: foundation-model embeddings + linear probe on AGO2.

For each model: fresh-cache download time, load time, embed time per 1k seqs, peak GPU mem,
probe AUROC (held-out + ls), dinucleotide-shuffle control, and a random-init twin
(same architecture, no pretraining). Embeddings are saved for later steps.
"""
import os
import shutil
import sys
import time
import traceback

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import OUT, dinuc_shuffle, load_split, norm, save_json, timer

MODELS = {
    "nt50m": "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
    "nt500m": "InstaDeepAI/nucleotide-transformer-500m-human-ref",
    "mistral17m": "RaphaelMourad/Mistral-DNA-v1-17M-hg38",
}
which = sys.argv[1:] or list(MODELS)
DEVICE = torch.device("cuda")
N_TRAIN = int(os.environ.get("N_TRAIN", 10000))

rng = np.random.default_rng(0)
tr_seqs, tr_y = load_split("train")
ls_seqs, ls_y = load_split("ls")
perm = rng.permutation(len(tr_seqs))
tr_i, te_i = perm[:N_TRAIN], perm[N_TRAIN:N_TRAIN + 2000]
to_dna = lambda s: norm(s).replace("U", "T")
S_tr = [to_dna(tr_seqs[i]) for i in tr_i]
S_te = [to_dna(tr_seqs[i]) for i in te_i]
S_ls = [to_dna(s) for s in ls_seqs]
pos_ls = [s for s, l in zip(S_ls, ls_y) if l == 1]
S_shuf = [dinuc_shuffle(s, rng).replace("U", "T") for s in pos_ls]
np.savez_compressed(OUT / "fm_split.npz", tr_i=tr_i, te_i=te_i)


def load(name, repo, random_init=False):
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer
    kw = {"trust_remote_code": True}
    tok = AutoTokenizer.from_pretrained(repo, **kw)
    if random_init:
        cfg = AutoConfig.from_pretrained(repo, **kw)
        try:
            m = AutoModelForMaskedLM.from_config(cfg, **kw)
        except Exception:
            m = AutoModel.from_config(cfg, **kw)
    else:
        try:
            m = AutoModelForMaskedLM.from_pretrained(repo, **kw)
        except Exception as e:
            print("MaskedLM load failed, trying AutoModel:", repr(e)[:300])
            m = AutoModel.from_pretrained(repo, **kw)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    return tok, m.to(DEVICE).eval()


@torch.no_grad()
def embed(tok, m, seqs, bs=64):
    out = []
    for i in range(0, len(seqs), bs):
        enc = tok(seqs[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=512)
        enc = {k: v.to(DEVICE) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        with torch.autocast("cuda", dtype=torch.float16):
            o = m(**enc, output_hidden_states=True)
        h = o.hidden_states[-1].float()
        mask = enc["attention_mask"].unsqueeze(-1).float()
        out.append(((h * mask).sum(1) / mask.sum(1)).cpu().numpy())
    return np.concatenate(out)


def probe(Etr, ytr, Ete, yte, Els, Esh, Els_neg):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))
    t = time.time()
    clf.fit(Etr, ytr)
    fit_s = time.time() - t
    p_sh = clf.predict_proba(Esh)[:, 1]
    p_neg = clf.predict_proba(Els_neg)[:, 1]
    return {
        "auc_heldout": roc_auc_score(yte, clf.predict_proba(Ete)[:, 1]),
        "auc_ls": roc_auc_score(ls_y, clf.predict_proba(Els)[:, 1]),
        "auc_shuffledpos_vs_neg": roc_auc_score([1] * len(p_sh) + [0] * len(p_neg), np.concatenate([p_sh, p_neg])),
        "probe_fit_s": round(fit_s, 2),
    }


R = {}
for name in which:
    repo = MODELS[name]
    r, T = {"repo": repo}, {}
    hf_home = f"/tmp/hf_fresh_{name}"
    shutil.rmtree(hf_home, ignore_errors=True)
    os.environ["HF_HOME"] = hf_home
    try:
        from huggingface_hub import snapshot_download
        with timer(T, "download_fresh"):
            path = snapshot_download(repo)
        r["download_MB"] = round(sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(hf_home) for f in fs) / 1e6, 1)
        with timer(T, "load_to_gpu"):
            tok, m = load(name, repo)
        r["n_params_M"] = round(sum(p.numel() for p in m.parameters()) / 1e6, 1)
        r["n_tokens_example"] = len(tok(S_ls[0])["input_ids"])
        print(name, "tokens for one seq:", r["n_tokens_example"], tok.convert_ids_to_tokens(tok(S_ls[0])["input_ids"])[:8])
        torch.cuda.reset_peak_memory_stats()
        with timer(T, "embed_1k"):
            embed(tok, m, S_tr[:1000])
        with timer(T, "embed_all"):
            Etr, Ete, Els, Esh = embed(tok, m, S_tr), embed(tok, m, S_te), embed(tok, m, S_ls), embed(tok, m, S_shuf)
        r["peak_gpu_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        r["n_embedded"] = len(S_tr) + len(S_te) + len(S_ls) + len(S_shuf)
        np.savez_compressed(OUT / f"emb_{name}.npz", tr=Etr, te=Ete, ls=Els, shuf=Esh)
        Els_neg = Els[ls_y == 0]
        r["probe_pretrained"] = probe(Etr, tr_y[tr_i], Ete, tr_y[te_i], Els, Esh, Els_neg)
        print(name, "pretrained", r["probe_pretrained"], flush=True)
        del m
        torch.cuda.empty_cache()
        with timer(T, "random_init_embed"):
            tok, m0 = load(name, repo, random_init=True)
            torch.manual_seed(0)
            E0 = [embed(tok, m0, S) for S in (S_tr, S_te, S_ls, S_shuf)]
        r["probe_random_init"] = probe(E0[0], tr_y[tr_i], E0[1], tr_y[te_i], E0[2], E0[3], E0[2][ls_y == 0])
        print(name, "random init", r["probe_random_init"], flush=True)
        del m0
        torch.cuda.empty_cache()
    except Exception:
        r["error"] = traceback.format_exc()[-3000:]
        print(r["error"])
    r["timings"] = T
    R[name] = r
    save_json(R, f"s3_fm_{'_'.join(which)}.json")
