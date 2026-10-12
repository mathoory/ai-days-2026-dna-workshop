"""Spike step 4: Galaxy 'verysmall' pretraining recipe - timing, loss curve, and does it help the AGO2 probe?

Follows the Galaxy genomic-llm-pretraining tutorial (CC-BY, Mourad & Batut): Mixtral config from
the Mistral-DNA repo, DNABERT-2 BPE tokenizer, hg38 200-bp 'verysmall' set, causal-LM objective.
Differences: sdpa attention instead of flash-attn, a plain training loop, a wall-clock budget.
"""
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import OUT, load_split, norm, save_json, timer

BUDGET_MIN = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
DEVICE = torch.device("cuda")
T, R = {}, {"budget_min": BUDGET_MIN}
repo_dir = OUT / "Mistral-DNA"
if not repo_dir.exists():
    with timer(T, "git_clone_Mistral-DNA"):
        subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/raphaelmourad/Mistral-DNA.git", str(repo_dir)], check=True)
R["repo_MB"] = round(sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(repo_dir) for f in fs) / 1e6, 1)

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

cfg = AutoConfig.from_pretrained(repo_dir / "data/models/Mixtral-8x7B-v0.1")
R["config"] = {k: getattr(cfg, k, None) for k in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                                                    "intermediate_size", "num_local_experts", "vocab_size")}
tok = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
tok.padding_side = "left"
torch.manual_seed(0)
model = AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa").to(DEVICE)
R["n_params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
print(R)

with timer(T, "load_and_tokenize"):
    df = pd.read_csv(repo_dir / "data/genome_sequences/hg38/sequences_hg38_200b_verysmall.csv.gz")
    texts = df.iloc[:, 0].astype(str).tolist() if "text" not in df else df["text"].astype(str).tolist()
    enc = tok(texts, padding="longest", truncation=True, max_length=128, return_tensors="pt")
R["n_pretrain_seqs"] = len(texts)
ids, att = enc["input_ids"], enc["attention_mask"]
n_val = len(ids) // 10
val_ids, val_att, tr_ids, tr_att = ids[:n_val], att[:n_val], ids[n_val:], att[n_val:]


@torch.no_grad()
def val_loss():
    model.eval()
    ls = []
    for i in range(0, len(val_ids), 64):
        x, a = val_ids[i:i + 64].to(DEVICE), val_att[i:i + 64].to(DEVICE)
        lab = x.masked_fill(a == 0, -100)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ls.append(model(input_ids=x, attention_mask=a, labels=lab).loss.item())
    model.train()
    return float(np.mean(ls))


opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
log, step, t0, epoch = [], 0, time.time(), 0
log.append({"step": 0, "min": 0.0, "val_loss": val_loss()})
print(log[-1], flush=True)
done = False
while not done:
    epoch += 1
    perm = torch.randperm(len(tr_ids))
    for i in range(0, len(perm), 32):
        b = perm[i:i + 32]
        x, a = tr_ids[b].to(DEVICE), tr_att[b].to(DEVICE)
        lab = x.masked_fill(a == 0, -100)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=x, attention_mask=a, labels=lab).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        if step % 100 == 0:
            log.append({"step": step, "epoch": epoch, "min": round((time.time() - t0) / 60, 2),
                        "train_loss": loss.item(), "val_loss": val_loss()})
            print(log[-1], flush=True)
        if time.time() - t0 > BUDGET_MIN * 60:
            done = True
            break
R["log"], R["steps"], R["epochs_started"] = log, step, epoch
R["steps_per_epoch"] = (len(tr_ids) + 31) // 32

# ---- does this tiny pretraining help the AGO2 probe? ----
rng = np.random.default_rng(0)
tr_seqs, tr_y = load_split("train")
ls_seqs, ls_y = load_split("ls")
split = np.load(OUT / "fm_split.npz")
tr_i = split["tr_i"]
S_tr = [norm(tr_seqs[i]).replace("U", "T") for i in tr_i]
S_ls = [norm(s).replace("U", "T") for s in ls_seqs]


@torch.no_grad()
def embed(seqs, bs=64):
    model.eval()
    out = []
    tok.padding_side = "right"
    for i in range(0, len(seqs), bs):
        e = tok(seqs[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=256)
        e = {k: e[k].to(DEVICE) for k in ("input_ids", "attention_mask")}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(**e, output_hidden_states=True).hidden_states[-1].float()
        m = e["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
    return np.concatenate(out)


clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))
clf.fit(embed(S_tr), tr_y[tr_i])
R["probe_auc_ls_after_pretraining"] = roc_auc_score(ls_y, clf.predict_proba(embed(S_ls))[:, 1])
print("probe after our pretraining:", R["probe_auc_ls_after_pretraining"])
R["timings"] = T
save_json(R, f"s4_pretrain_{int(BUDGET_MIN)}min.json")
