"""Third batch: reading results (topic 17), the real-data hero, the recap, and fixes after the audits.

python intuition3.py [hero auroc_ci splits probe_trap recap]
"""
import sys

import matplotlib.image as mpimg
import numpy as np
from matplotlib.patches import Rectangle
from sklearn.metrics import roc_auc_score

from deckstyle import *  # noqa: F401,F403
from intuition import ALPH, DATA, cnn_scores, load_cnn, onehot, split

CUSTOM = OUT


def hero(model, pos):
    s = next(x for x in pos if "UGCUGCU" in x[150:200])
    k = s.index("UGCUGCU", 150)
    w = s[k - 9:k + 19]
    score = float(cnn_scores(model, [s])[0])
    fig, ax = canvas()
    x0, dx = 10, 5.0
    for i, ch in enumerate(w):
        label(ax, x0 + i * dx, 74, ch, fs=26, color=BASE[ch], bold=True)
        for r, b in enumerate(ALPH):
            ax.add_patch(Rectangle((x0 + i * dx - 2.4, 56 - r * 5), 4.8, 4.6,
                                   fc=BASE[b] if ch == b else "white", ec="#cfcfcf", lw=1))
    for r, b in enumerate(ALPH):
        label(ax, x0 - 6, 58.3 - r * 5, b, fs=18, color=BASE[b], bold=True)
    label(ax, 80, 86, "one bound RNA stretch from the AGO2 data (28 of its 330 letters)", fs=22, color=GREY)
    label(ax, 80, 31, f"the CNN you'll train today scores the whole sequence {score:.2f}", fs=26, color=BLUE, bold=True)
    label(ax, 80, 18, "what in these letters gave it that number?", fs=26, color=NAVY)
    save(fig, "hero")


def bootstrap_ci(y, s, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        out.append(roc_auc_score(y[i], s[i]))
    return np.percentile(out, [2.5, 97.5])


def auroc_ci(model, tr, ty, ls, ly):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    idx = np.random.default_rng(0).permutation(len(tr))[:12000]
    rows = []
    for name, f, c in (("Nucleotide Transformer (486M) + probe", "emb_nt500m.npz", PURPLE),
                       ("Mistral-DNA (16M) + probe", "emb_mistral17m.npz", PURPLE)):
        d = np.load(DATA / f)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1)).fit(d["train"], ty[idx])
        rows.append((name, clf.predict_proba(d["test"])[:, 1], c))
    rows.append(("CNN from Slot 2", cnn_scores(model, ls), BLUE))
    fig = plt.figure(figsize=(16, 9), dpi=120)
    ax = fig.add_axes([0.36, 0.2, 0.6, 0.6])
    for i, (n, s, c) in enumerate(rows):
        a = roc_auc_score(ly, s)
        lo, hi = bootstrap_ci(ly, s)
        ax.plot([lo, hi], [i, i], color=c, lw=6, alpha=.45, solid_capstyle="round")
        ax.scatter(a, i, s=300, color=c, zorder=3)
        ax.text(hi + 0.006, i, f"{a:.2f}  [{lo:.2f}–{hi:.2f}]", va="center", fontsize=21, color=c)
        print(n, a, lo, hi)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=21)
    ax.set_xlim(0.72, 0.95); ax.set_ylim(-0.6, len(rows) - 0.4); ax.tick_params(labelsize=18)
    ax.set_xlabel("held-out AUROC on 1,000 sequences · bar = 95% bootstrap range", fontsize=20)
    fig.text(0.5, 0.88, "on a 1,000-sequence test set an AUROC moves about ±0.025 by chance",
             ha="center", fontsize=23, color=NAVY, fontweight="bold")
    fig.text(0.5, 0.07, "overlapping bars: no evidence of a difference · separate bars: a real gap",
             ha="center", fontsize=20, color=GREY)
    save(fig, "auroc_ci")


def splits():
    fig, ax = canvas()
    box(ax, 6, 48, 96, 16, "train · fit the parameters", fc=BLUE_L, ec=BLUE, fs=24, r=0.6)
    box(ax, 103, 48, 24, 16, "validation\nchoose settings", fc=GOLD_L, ec=GOLD, fs=19, r=0.6)
    box(ax, 128, 48, 26, 16, "test\nreport once", fc=RED_L, ec=RED, fs=19, r=0.6, bold=True)
    for x, t in ((54, "92,346 sequences in the training file"), (115, "peeked at while training"),
                 (141, "1,000 held-out (ls)")):
        label(ax, x, 40, t, fs=17, color=GREY)
    label(ax, 80, 24, "today: the 1,000 held-out sequences are the test set in every slot", fs=22, color=NAVY)
    label(ax, 80, 15, "Slot 2 also watches them every epoch, so small gains on them are not evidence", fs=19, color=GREY)
    save(fig, "splits")


def probe_trap_fix():
    # re-draw only the legend placement of intuition2.probe_trap
    from intuition2 import probe_trap
    probe_trap(*split("train"))


def recap():
    tiles = [("seed_pairing", "the seed rule"), ("filter_dot", "a filter is a dot product"),
             ("planting_bars", "the CNN found miR-16"), ("position_scan", "…and learned where"),
             ("emb_pca", "embeddings track G+C"), ("shuffle_scores", "the shuffle test"),
             ("dumbbell", "matched negatives"), ("headtohead", "fine-tuning wins")]
    fig, axes = plt.subplots(2, 4, figsize=(16, 9), dpi=120)
    for ax, (n, t) in zip(axes.flat, tiles):
        ax.imshow(mpimg.imread(CUSTOM / f"{n}.png")); ax.axis("off")
        ax.set_title(t, fontsize=20, color=NAVY, fontweight="bold")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.08, wspace=0.04, hspace=0.25)
    save(fig, "recap")


if __name__ == "__main__":
    which = sys.argv[1:] or ["hero", "auroc_ci", "splits", "probe_trap"]
    tr, ty = split("train"); ls, ly = split("ls")
    pos = [s for s, l in zip(tr, ty) if l == 1]
    model = load_cnn()
    for w in which:
        {"hero": lambda: hero(model, pos), "auroc_ci": lambda: auroc_ci(model, tr, ty, ls, ly),
         "splits": splits, "probe_trap": probe_trap_fix, "recap": recap}[w]()
