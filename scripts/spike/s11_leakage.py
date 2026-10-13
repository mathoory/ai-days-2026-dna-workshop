"""Does the random train/test split leak? Score the CNN (and the 5-mer probe) on held-out sequences that do and
do not have a training window centred within 150 nt on the same chromosome and strand.

Headers look like ">ID;chr19,38230198,38230238,-". Run with torch available (hw1-00660121 locally):
    python scripts/spike/s11_leakage.py
Writes scripts/spike/out/s11_leakage.json.
"""
import bisect
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "slides" / "figures"))
from intuition import cnn_scores, load_cnn  # noqa: E402

DATA = ROOT / "data"


def read(name):
    recs, hdr = [], None
    for line in open(DATA / name):
        line = line.strip()
        if line.startswith(">"):
            hdr = line[1:]
        elif hdr:
            loc = hdr.split(";")[1].split(",")
            recs.append((loc[0], loc[3], (int(loc[1]) + int(loc[2])) // 2, line.upper().replace("T", "U")))
            hdr = None
    return recs


def split(kind):
    p, n = read(f"CLIPSEQ_AGO2.{kind}.positives.fa"), read(f"CLIPSEQ_AGO2.{kind}.negatives.fa")
    return p + n, np.array([1] * len(p) + [0] * len(n))


tr, ty = split("train")
ls, ly = split("ls")
centres = defaultdict(list)
for chrom, strand, c, _ in tr:
    centres[(chrom, strand)].append(c)
for k in centres:
    centres[k].sort()


def near(chrom, strand, c, d=150):
    a = centres.get((chrom, strand), [])
    i = bisect.bisect_left(a, c - d)
    return i < len(a) and a[i] <= c + d


ov = np.array([near(ch, st, c) for ch, st, c, _ in ls])
model = load_cnn()
s = cnn_scores(model, [x[3] for x in ls])
out = {"n": len(ls), "overlap_bound": float(ov[ly == 1].mean()), "overlap_unbound": float(ov[ly == 0].mean()),
       "cnn_all": roc_auc_score(ly, s)}
# AUROC on held-out sequences WITHOUT a nearby training window (both classes), and with one
for name, m in (("cnn_no_nearby_train", ~ov), ("cnn_nearby_train", ov)):
    if len(set(ly[m])) == 2:
        out[name] = roc_auc_score(ly[m], s[m]); out[name + "_n"] = int(m.sum())
# Fair comparison: drop every held-out sequence with a nearby training window
print(json.dumps(out, indent=1))
(ROOT / "scripts" / "spike" / "out" / "s11_leakage.json").write_text(json.dumps(out, indent=1))
