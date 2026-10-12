"""Spike step 1: data stats, shortcut baselines, k-mer enrichment, miRNA seed sites."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import core_of, holdout, kmer_counts, load_split, norm, save_json, timer

T = {}
with timer(T, "load"):
    tr_seqs, tr_y = load_split("train")
    ls_seqs, ls_y = load_split("ls")

res = {"n_train": len(tr_seqs), "n_ls": len(ls_seqs), "pos_frac_train": float(tr_y.mean())}


def stats(seqs, y, tag):
    L = np.array([len(s) for s in seqs])
    C = np.array([len(core_of(s)) for s in seqs])
    full = [norm(s) for s in seqs]
    gc = np.array([(s.count("G") + s.count("C")) / max(len(s), 1) for s in full])
    u = np.array([s.count("U") / max(len(s), 1) for s in full])
    out = {}
    for lab in (1, 0):
        m = y == lab
        out[f"{tag}_label{lab}"] = {
            "len_mean": L[m].mean(), "len_min": int(L[m].min()), "len_max": int(L[m].max()),
            "core_len_mean": C[m].mean(), "gc_mean": gc[m].mean(), "u_mean": u[m].mean(),
        }
    return out


res["stats"] = {**stats(tr_seqs, tr_y, "train"), **stats(ls_seqs, ls_y, "ls")}
print(res["stats"])

tr_idx, te_idx = holdout(len(tr_seqs))


def feats(seqs, kind, region):
    S = [core_of(s) if region == "core" else s for s in seqs]
    if kind == "length":
        return np.array([[len(s)] for s in S], dtype=float)
    Sn = [norm(s) for s in S]
    if kind == "gc":
        return np.array([[(s.count("G") + s.count("C")) / max(len(s), 1)] for s in Sn])
    if kind == "u":
        return np.array([[s.count("U") / max(len(s), 1)] for s in Sn])
    if kind == "k1":
        return kmer_counts(S, 1)
    if kind == "k2":
        return kmer_counts(S, 2)
    if kind == "k3":
        return kmer_counts(S, 3)
    if kind == "k4":
        return kmer_counts(S, 4)
    if kind == "k5":
        return kmer_counts(S, 5)
    raise ValueError(kind)


base = {}
for region in ("full", "core"):
    for kind in ("length", "gc", "u", "k1", "k2", "k3", "k4", "k5"):
        key = f"{region}:{kind}"
        with timer(T, f"feat_{key}"):
            Ftr = feats(tr_seqs, kind, region)
            Fls = feats(ls_seqs, kind, region)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(Ftr[tr_idx], tr_y[tr_idx])
        base[key] = {
            "auc_heldout": roc_auc_score(tr_y[te_idx], clf.predict_proba(Ftr[te_idx])[:, 1]),
            "auc_ls": roc_auc_score(ls_y, clf.predict_proba(Fls)[:, 1]),
        }
        print(key, base[key], flush=True)
res["baselines"] = base

# ---- k-mer enrichment (7-mers, core and full) and miRNA seed sites ----
SITES = {  # 7mer-m8 site = reverse complement of miRNA nt 2-8 (RNA alphabet)
    "let-7": "CUACCUC", "miR-21": "AUAAGCU", "miR-17/20/106": "GCACUUU", "miR-19": "UUUGCAC",
    "miR-92/25": "GUGCAAU", "miR-15/16/195": "UGCUGCU", "miR-103/107": "AUGCUGC",
    "miR-30": "UGUUUAC", "miR-26": "UACUUGA", "miR-1/206": "ACAUUCC", "miR-122": "ACACUCC",
    "miR-124": "GUGCCUU", "miR-302/372": "AGCACUU", "miR-148/152": "UGCACUG",
    "miR-7": "GUCUUCC", "miR-181": "UGAAUGU", "miR-125": "CUCAGGG", "miR-27": "ACUGUGA",
    "miR-23": "AAUGUGA", "miR-221/222": "AUGUAGC", "miR-24": "CUGAGCC",
}


def frac_with(seqs, site):
    return float(np.mean([site in s for s in seqs]))


pos_full = [norm(s) for s, l in zip(tr_seqs, tr_y) if l == 1]
neg_full = [norm(s) for s, l in zip(tr_seqs, tr_y) if l == 0]
pos_core = [norm(core_of(s)) for s, l in zip(tr_seqs, tr_y) if l == 1]
neg_core = [norm(core_of(s)) for s, l in zip(tr_seqs, tr_y) if l == 0]
sites = {}
for name, site in SITES.items():
    pf, nf = frac_with(pos_full, site), frac_with(neg_full, site)
    pc, nc = frac_with(pos_core, site), frac_with(neg_core, site)
    sites[name] = {"site": site, "pos_full": pf, "neg_full": nf, "ratio_full": pf / max(nf, 1e-9),
                   "pos_core": pc, "neg_core": nc, "ratio_core": pc / max(nc, 1e-9)}
res["mirna_sites"] = sites
for k, v in sorted(sites.items(), key=lambda kv: -kv[1]["ratio_core"]):
    print(f"{k:15s} {v['site']}  core {v['pos_core']:.3f}/{v['neg_core']:.3f} = {v['ratio_core']:.2f}"
          f"   full {v['pos_full']:.3f}/{v['neg_full']:.3f} = {v['ratio_full']:.2f}")

ALPH = "ACGU"


def code2kmer(c, k):
    s = []
    for _ in range(k):
        s.append(ALPH[c % 4])
        c //= 4
    return "".join(reversed(s))


for region, P, N in (("core", pos_core, neg_core), ("full", pos_full, neg_full)):
    K = 7
    Mp = (kmer_counts(P, K, normalise=False) > 0).mean(0)
    Mn = (kmer_counts(N, K, normalise=False) > 0).mean(0)
    ratio = (Mp + 1e-3) / (Mn + 1e-3)
    ok = (Mp > 0.01)
    order = np.argsort(-ratio * ok)[:25]
    top = [{"kmer": code2kmer(int(c), K), "pos": float(Mp[c]), "neg": float(Mn[c]),
            "ratio": float(ratio[c])} for c in order]
    res[f"top7_{region}"] = top
    print(f"\nTop enriched 7-mers ({region}):")
    for t in top:
        hit = [n for n, s in SITES.items() if s == t["kmer"]]
        print(f"  {t['kmer']}  {t['pos']:.3f} vs {t['neg']:.3f}  x{t['ratio']:.2f}  {hit}")

res["timings"] = T
save_json(res, "s1_shortcuts.json")
