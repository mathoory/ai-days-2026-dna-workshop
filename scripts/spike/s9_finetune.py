"""Spike step 9: LoRA fine-tuning of several DNA foundation models, same recipe for all.

Recipe (identical across models): LoRA r=16 on every linear layer of the backbone
(peft target_modules="all-linear"), a new linear head on the mean-pooled last layer, AdamW lr 5e-4
(head 1e-3), bf16 autocast, batch 32, max 256 tokens.
Settings, both on the composition-matched benchmark (Slot 7), test = the same 3,000 sequences:
  n20000 : 20k labels, 2 epochs
  n500   : 500 labels, up to 30 epochs, early stopping on 10% of the 500 (never the test set)
Usage: python s9_finetune.py "<model name>" ...   (names as in s8_models.MODELS)
Writes out/s9_finetune.json incrementally.
"""
import json
import os
import sys
import time
import traceback

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from s8_models import ALL, MODELS, load, m_te, m_tr as _m_tr12, matched_indices, seqs, y  # noqa: F401
from common import OUT

DEVICE = torch.device("cuda")
BATCH = int(os.environ.get("BATCH", 32))
m_idx = matched_indices(seqs, y)
m_tr = m_idx[3000:3000 + 20000]
S_te, y_te = [seqs[i] for i in m_te], y[m_te]


class Classifier(nn.Module):
    def __init__(self, backbone, hidden):
        super().__init__()
        self.backbone, self.head = backbone, nn.Linear(hidden, 1)

    def forward(self, ids, mask):
        try:
            o = self.backbone(input_ids=ids, attention_mask=mask, output_hidden_states=True)
            h = o.hidden_states[-1] if getattr(o, "hidden_states", None) is not None else o[0]
        except TypeError:
            o = self.backbone(input_ids=ids)
            h = o[0] if isinstance(o, (tuple, list)) else o.last_hidden_state
        mk = mask.unsqueeze(-1).to(h.dtype)
        return self.head(((h * mk).sum(1) / mk.sum(1)).float()).squeeze(-1)


def run(name, n, epochs, patience=None, seed=0):
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed)
    tok, bb = load(MODELS[name])
    if os.environ.get("BF16_WEIGHTS") == "1":   # NT-2.5B does not fit a 24 GB GPU in fp32 with gradients
        bb = bb.to(torch.bfloat16)
    hidden = getattr(bb.config, "hidden_size", None) or getattr(bb.config, "d_model")
    # every nn.Linear inside the backbone, by name (peft's "all-linear" mis-fires on HyenaDNA's custom blocks)
    # full module paths, not last names: HyenaDNA has Linear layers named "0", "1" that would also match whole blocks
    lin = [n for n, mod in bb.named_modules()
           if isinstance(mod, nn.Linear) and not any(k in n for k in ("lm_head", "classifier", "score"))]
    bb = get_peft_model(bb, LoraConfig(r=16, lora_alpha=16, lora_dropout=0.05, target_modules=lin))
    model = Classifier(bb, hidden).to(DEVICE)
    lora_params = [p for p in model.backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": lora_params, "lr": 5e-4}, {"params": model.head.parameters(), "lr": 1e-3}])
    crit = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(0)            # the labelled subset is the same for every seed and method
    order_rng = np.random.default_rng(seed)   # the seed changes init and data order only
    sel = rng.choice(m_tr, n, replace=False) if n < len(m_tr) else m_tr
    if patience:
        nv = max(n // 10, 20)
        va, tr = sel[:nv], sel[nv:]
    else:
        va, tr = None, sel

    def enc(idx_or_seqs):
        S = [seqs[i] for i in idx_or_seqs] if not isinstance(idx_or_seqs[0], str) else idx_or_seqs
        e = tok([s.replace("U", "T") for s in S], return_tensors="pt", padding=True, truncation=True, max_length=256)
        ids = e["input_ids"].to(DEVICE)
        mask = e["attention_mask"].to(DEVICE) if "attention_mask" in e else (ids != tok.pad_token_id).long()
        return ids, mask

    @torch.no_grad()
    def scores(S):
        model.eval()
        out = []
        for i in range(0, len(S), 128):
            ids, mask = enc(S[i:i + 128])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(model(ids, mask).float().cpu().numpy())
        model.train()
        return np.concatenate(out)

    t0, hist, best, best_state, bad = time.time(), [], -1, None, 0
    model.train()
    for ep in range(epochs):
        perm = order_rng.permutation(tr)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            ids, mask = enc(b)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = crit(model(ids, mask).float(), torch.tensor(y[b], dtype=torch.float32, device=DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
        if patience:
            v = roc_auc_score(y[va], scores([seqs[i] for i in va]))
            if v > best:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items() if "lora" in k or "head" in k}
            else:
                bad += 1
                if bad >= patience:
                    break
        else:
            hist.append(roc_auc_score(y_te, scores(S_te)))
    if patience and best_state:
        model.load_state_dict(best_state, strict=False)
    auc = roc_auc_score(y_te, scores(S_te))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    del model, bb
    torch.cuda.empty_cache()
    return {"auc": auc, "epochs_run": ep + 1, "minutes": round((time.time() - t0) / 60, 2),
            "trainable_frac": round(trainable / total, 4), "per_epoch_auc": hist}


out_path = OUT / os.environ.get("S9_OUT", "s9_finetune.json")
R = json.loads(out_path.read_text()) if out_path.exists() else {}
for name in sys.argv[1:]:
    R.setdefault(name, {})
    for key, (n, epochs, pat) in {"n20000": (20000, 2, None), "n500": (500, 30, 5)}.items():
        try:
            R[name][key] = run(name, n, epochs, pat)
        except Exception:
            R[name][key] = {"error": traceback.format_exc()[-1500:]}
        out_path.write_text(json.dumps(R, indent=1, default=float))
        print(name, key, {k: v for k, v in R[name][key].items() if k != "per_epoch_auc"}, flush=True)
