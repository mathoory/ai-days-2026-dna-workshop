"""Spike step 6: (a) masked-base fill-in demo timing with NT-500m (standard HF code),
(b) can NT-v2-50m (custom code) be loaded on transformers 5.x with a config patch?"""
import time
import traceback

import torch

from common import save_json

DEVICE = torch.device("cuda")
R = {}

# (a) MLM demo - KAUST 02_nucleotide_transformer_architecture, adapted
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

t = time.time()
repo = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
tok = AutoTokenizer.from_pretrained(repo)
m = AutoModelForMaskedLM.from_pretrained(repo).to(DEVICE).eval()
R["nt500m_load_cached_s"] = round(time.time() - t, 2)
seq = "TGTGGGAAGGGCTATACCACTGCCTCATACTTTCTTCTACATTTGCACAAAGG"  # contains a miR-19 site (TTTGCAC)
ids = tok(seq, return_tensors="pt")["input_ids"]
toks = tok.convert_ids_to_tokens(ids[0])
pos = next(i for i, x in enumerate(toks) if "TTGCAC" in x or "TTTGCA" in x or i == 6)
true_tok = toks[pos]
ids_m = ids.clone()
ids_m[0, pos] = tok.mask_token_id
t = time.time()
with torch.no_grad():
    logits = m(input_ids=ids_m.to(DEVICE)).logits[0, pos]
top = torch.topk(logits.softmax(-1), 5)
R["mlm_demo"] = {"tokens": toks, "masked_pos": pos, "true": true_tok,
                 "top5": [(tok.convert_ids_to_tokens(int(i)), float(p)) for p, i in zip(top.values, top.indices)],
                 "infer_s": round(time.time() - t, 3)}
print(R["mlm_demo"])

# (b) NT-v2-50m on transformers 5 with rope_theta patch
try:
    repo2 = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
    cfg = AutoConfig.from_pretrained(repo2, trust_remote_code=True)
    if not hasattr(cfg, "rope_theta"):
        cfg.rope_theta = 10000.0
    m2 = AutoModelForMaskedLM.from_pretrained(repo2, config=cfg, trust_remote_code=True).to(DEVICE).eval()
    tok2 = AutoTokenizer.from_pretrained(repo2, trust_remote_code=True)
    with torch.no_grad():
        o = m2(**{k: v.to(DEVICE) for k, v in tok2(seq, return_tensors="pt").items()}, output_hidden_states=True)
    R["nt50m_patch"] = {"ok": True, "hidden": list(o.hidden_states[-1].shape)}
except Exception:
    R["nt50m_patch"] = {"ok": False, "err": traceback.format_exc()[-1500:]}
print(R["nt50m_patch"])
import transformers
R["transformers"] = transformers.__version__
save_json(R, "s6_mlm.json")
