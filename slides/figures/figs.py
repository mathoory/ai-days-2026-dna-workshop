"""Every figure used in the decks, in one style (deckstyle.py).

Rules: real data only; no caption sentences inside figures; sentence-case labels; label the
data directly instead of using a legend where possible; one colour = one meaning.

Run in an env with torch + transformers (hw1-00660121 locally):
    python figs.py            # all
    python figs.py hero auroc # some
"""
import json
import re
import sys

import matplotlib.image as mpimg
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from sklearn.metrics import roc_auc_score, roc_curve

from deckstyle import *  # noqa: F401,F403
from intuition import ALPH, DATA, DEV, acts, cnn_scores, dinuc_shuffle, load_cnn, onehot, split
from intuition2 import MIR16, draw_logo, gc, windows_pwm
from intuition3 import bootstrap_ci
import intuition4 as i4
from deck05_08 import SPIKE, gc_u, matched_indices, train_set

REPO = ROOT.parent
EXEC = REPO / "notebooks" / "executed"
UNB = UNBOUND


def heat(ax, W, vmax=None, rows=True):
    vmax = vmax or np.abs(W).max()
    im = ax.imshow(W, cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(range(4) if rows else [])
    if rows:
        ax.set_yticklabels(list(ALPH), fontsize=19, fontweight="medium")
    ax.set_xticks([]); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def nb_text(stem):
    path = EXEC / f"{stem}_solution.ipynb"
    if not path.exists():                      # runs of the old slot notebooks were moved to the archive
        path = EXEC.parent / "archive" / "executed" / f"{stem}_solution.ipynb"
    nb = json.load(open(path, encoding="utf-8"))
    return "".join("".join(o.get("text", "")) for c in nb["cells"] for o in c.get("outputs", []))


def _kmer_matrix(S, k=5):
    from itertools import product
    idx = {"".join(p): i for i, p in enumerate(product("ACGU", repeat=k))}
    M = np.zeros((len(S), len(idx)), dtype=np.float32)
    for r, s in enumerate(S):
        for i in range(len(s) - k + 1):
            j = idx.get(s[i:i + k])
            if j is not None:
                M[r, j] += 1
    return M / np.maximum(M.sum(1, keepdims=True), 1)


# ============================================================================ deck 01
def hero(model, pos):
    s = next(x for x in pos if "UGCUGCU" in x[150:200])
    k = s.index("UGCUGCU", 150)
    w = s[k - 9:k + 19]
    fig, ax = canvas()
    x0, dx = 14, 4.9
    for i, ch in enumerate(w):
        label(ax, x0 + i * dx, 70, ch, fs=28, color=INK, family=MONO)
        for r, b in enumerate(ALPH):
            ax.add_patch(Rectangle((x0 + i * dx - 2.3, 52 - r * 5.2), 4.6, 4.6,
                                   fc=INK if ch == b else FAINT, ec="white", lw=1.5))
    for r, b in enumerate(ALPH):
        label(ax, x0 - 6, 54.3 - r * 5.2, b, fs=20, color=MUTED)
    save(fig, "hero")
    return float(cnn_scores(model, [s])[0])


def seed_pairing(pos):
    site = "UGCUGCU"
    s = next(x for x in pos if site in x[60:-60])
    k = s.index(site, 60)
    w0 = k - 9
    win = s[w0:w0 + 26]
    kw = k - w0
    L = len(MIR16)
    mir_rev = MIR16[::-1]
    fig, ax = canvas()
    x0, dx = 28, 4.6
    label(ax, 4, 60, "mRNA", fs=22, color=MUTED, ha="left")
    label(ax, 4, 34, "miR-16", fs=22, color=MUTED, ha="left")
    label(ax, x0 - 5, 60, "5′", fs=20, color=MUTED); label(ax, x0 + len(win) * dx, 60, "3′", fs=20, color=MUTED)
    for i, ch in enumerate(win):
        label(ax, x0 + i * dx, 60, ch, fs=28, color=INK, bold=kw <= i < kw + 7, family=MONO)
    xs, first = [], None
    for m, ch in enumerate(mir_rev):
        i = kw + (m - (L - 8))
        if i < -1 or i > len(win):
            continue
        first = i if first is None else first
        x = x0 + i * dx
        paired = L - 8 <= m <= L - 2           # microRNA letters 2..8
        label(ax, x, 34, ch, fs=28, color=INK if paired else LIGHT, bold=paired, family=MONO)
        if paired:
            ax.plot([x, x], [39, 51.5], color=MUTED, lw=1.8)
            xs.append(x)
    label(ax, x0 + (first - 1) * dx, 34, "3′", fs=20, color=MUTED)
    label(ax, x0 + (kw + 8) * dx, 34, "5′", fs=20, color=MUTED)
    label(ax, np.mean(xs), 24, "microRNA letters 2–8 (seed 2–7, plus 8)", fs=20, color=MUTED)
    save(fig, "seed_pairing")


def labels_raster(tr, ty):
    import matplotlib
    rng = np.random.default_rng(0)
    pos = [s for s, l in zip(tr, ty) if l == 1]; neg = [s for s, l in zip(tr, ty) if l == 0]
    P = [pos[i] for i in rng.choice(len(pos), 150, replace=False)]
    N = [neg[i] for i in rng.choice(len(neg), 150, replace=False)]
    cmap = {b: matplotlib.colors.to_rgb(BASE[b]) for b in ALPH}

    def img(S):
        S = sorted(S, key=gc)
        M = np.ones((len(S), 330, 3))
        for r, s in enumerate(S):
            for c, ch in enumerate(s[:330]):
                M[r, c] = cmap.get(ch, (1, 1, 1))
        return M

    fig, axes = plt.subplots(1, 2, figsize=(16, 9), dpi=120)
    titles = (f"Bound: {len(pos):,} sites measured by CLIP-seq", f"Unbound: {len(neg):,} random positions in the same genes")
    for ax, S, t in ((axes[0], P, titles[0]), (axes[1], N, titles[1])):
        ax.imshow(img(S), aspect="auto", interpolation="nearest")
        ax.set_title(t, fontsize=22, pad=46); ax.set_xlabel("Position in the sequence"); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        for x in (150, 190):                                    # the measured site, letters 150 to 190
            ax.axvline(x, color=INK, lw=3.5)
        ax.annotate("", (150, -4), (190, -4), annotation_clip=False,
                    arrowprops=dict(arrowstyle="<->", color=INK, lw=2, shrinkA=0, shrinkB=0))
        ax.text(170, -7, "measured site", ha="center", va="bottom", fontsize=19, color=INK, fontweight="bold",
                clip_on=False)
        ax.set_xticks([0, 50, 100, 150, 190, 250, 300])
    axes[0].set_ylabel("150 of them, sorted by G+C")
    fig.legend(handles=[Patch(color=BASE[b], label=b) for b in ALPH], loc="upper right", ncol=4,
               bbox_to_anchor=(0.98, 0.98), handlelength=1, columnspacing=1.2)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.82, bottom=0.1, wspace=0.06)
    save(fig, "labels_raster")


def lr_training(tr, ty):
    rng = np.random.default_rng(1)
    idx = rng.choice(len(tr), 3000, replace=False)
    X = np.array([[gc(tr[i]), tr[i].count("U") / len(tr[i])] for i in idx]); y = ty[idx].astype(float)
    mu, sd = X.mean(0), X.std(0); Z = (X - mu) / sd
    w, b, lr = np.array([-1.0, -1.0]), 0.8, 0.5        # a deliberately bad start so the line visibly moves
    snaps, losses = {}, []
    for step in range(301):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        losses.append(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))
        if step in (0, 300):
            snaps[step] = (w.copy(), b)
        g = p - y
        w -= lr * Z.T @ g / len(y); b -= lr * g.mean()
    fig = newfig()
    o = rng.permutation(len(y))
    for k, step in enumerate((0, 300)):
        ax = fig.add_axes([0.07 + k * 0.31, 0.14, 0.27, 0.7])
        ax.scatter(X[o, 0], X[o, 1], c=np.where(y[o] == 1, BOUND, MID), s=30, alpha=.6, lw=0)
        ws, bs = snaps[step]
        xs = np.linspace(X[:, 0].min(), X[:, 0].max(), 50)
        zy = -(ws[0] * (xs - mu[0]) / sd[0] + bs) / ws[1]
        ax.plot(xs, zy * sd[1] + mu[1], color=HELD_C, lw=3)
        ax.set_ylim(X[:, 1].min(), X[:, 1].max())
        ax.set_title(f"Step {step}"); ax.set_xlabel("G+C fraction")
        if k == 0:
            ax.set_ylabel("U fraction")
        else:
            ax.set_yticklabels([])
    ax = fig.add_axes([0.73, 0.14, 0.24, 0.7])
    ax.plot(losses, color=TRAIN_C, lw=3)
    for st in (0, 300):
        ax.scatter(st, losses[st], s=70, color=TRAIN_C, zorder=3)
        ax.annotate(f"{losses[st]:.2f}", (st, losses[st]), xytext=(10, 8) if st == 0 else (-50, 10), textcoords="offset points")
    ax.set_title("Loss"); ax.set_xlabel("Step"); ygrid(ax)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=10, color=BOUND, label="Bound"),
                        Line2D([], [], marker="o", ls="", ms=10, color=MID, label="Unbound"),
                        Line2D([], [], color=HELD_C, lw=3, label="p = 0.5")],
               loc="upper right", ncol=3, bbox_to_anchor=(0.98, 1.0))
    save(fig, "lr_training")


def auroc(model, ls, ly):
    s = cnn_scores(model, ls)
    fig = newfig()
    a1 = fig.add_axes([0.07, 0.14, 0.46, 0.66]); a2 = fig.add_axes([0.62, 0.14, 0.34, 0.72])
    bins = np.linspace(0, 1, 30)
    a1.hist(s[ly == 0], bins, color=UNB, alpha=.9, label="Unbound")
    a1.hist(s[ly == 1], bins, color=BOUND, alpha=.6, label="Bound")
    a1.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    a1.set_xlabel("CNN score"); a1.set_ylabel("Sequences"); ygrid(a1)
    fpr, tpr, _ = roc_curve(ly, s)
    a2.plot([0, 1], [0, 1], color=HAIR, lw=1.5, ls="--")
    a2.plot(fpr, tpr, color=CNN_C, lw=3); a2.fill_between(fpr, tpr, color=CNN_C, alpha=.07)
    a2.text(0.42, 0.2, f"AUROC {roc_auc_score(ly, s):.2f}", fontsize=28, color=CNN_C)
    a2.set_xlabel("False positive rate"); a2.set_ylabel("True positive rate")
    a2.set_xlim(0, 1); a2.set_ylim(0, 1.01); a2.set_aspect("equal")
    save(fig, "auroc")


def auroc_ci(model, tr, ty, ls, ly):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    gcu = lambda S: np.array([[gc(s), s.count("U") / len(s)] for s in S])
    lr2 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(gcu(tr), ty)
    idx = np.random.default_rng(0).permutation(len(tr))[:12000]          # same 12,000 as the part 3 notebook
    lr5 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1)).fit(
        _kmer_matrix([tr[i] for i in idx]), ty[idx])
    rows = [("G+C and U only", lr2.predict_proba(gcu(ls))[:, 1], SHORTCUT_C),
            ("5-letter word counts", lr5.predict_proba(_kmer_matrix(ls))[:, 1], SHORTCUT_C),
            ("CNN", cnn_scores(model, ls), CNN_C)]
    fig = newfig()
    ax = fig.add_axes([0.27, 0.16, 0.66, 0.72])
    for i, (n, s, c) in enumerate(rows):
        a = roc_auc_score(ly, s); lo, hi = bootstrap_ci(ly, s)
        ax.plot([lo, hi], [i, i], color=c, lw=6, alpha=.35, solid_capstyle="round")
        ax.scatter(a, i, s=220, color=c, zorder=3)
        ax.text(hi + 0.008, i, f"{a:.2f}  ({lo:.2f}–{hi:.2f})", va="center", color=INK)
        print("auroc_ci", n, round(a, 3), round(lo, 3), round(hi, 3))
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlim(0.6, 0.95); ax.set_ylim(-0.6, len(rows) - 0.4); xgrid(ax)
    ax.set_xlabel("Held-out AUROC, 95% bootstrap interval (1,000 sequences)")
    save(fig, "auroc_ci")


def probe_trap(tr, ty):
    """Fit vs held-out with a model students already know: logistic regression on 5-letter word counts."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    ls, ly = split("ls")
    idx = np.random.default_rng(0).permutation(len(tr))[:12000]
    Xtr, ytr = _kmer_matrix([tr[i] for i in idx]), ty[idx]
    Xls = _kmer_matrix(ls)
    good = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.01)).fit(Xtr, ytr)
    over = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=100)).fit(Xtr[:600], ytr[:600])
    a_tr = roc_auc_score(ytr, good.predict_proba(Xtr)[:, 1]); a_te = roc_auc_score(ly, good.predict_proba(Xls)[:, 1])
    o_tr = roc_auc_score(ytr[:600], over.predict_proba(Xtr[:600])[:, 1]); o_te = roc_auc_score(ly, over.predict_proba(Xls)[:, 1])
    fig = newfig()
    ax = fig.add_axes([0.09, 0.13, 0.86, 0.7])
    x = np.array([0, 1])
    ax.bar(x - 0.17, [a_tr, o_tr], 0.32, color=TRAIN_C, label="Scored on its own training sequences")
    ax.bar(x + 0.17, [a_te, o_te], 0.32, color=HELD_C, label="Scored on the held-out sequences")
    for xi, a, b in zip(x, (a_tr, o_tr), (a_te, o_te)):
        ax.text(xi - 0.17, a + 0.008, f"{a:.2f}", ha="center")
        ax.text(xi + 0.17, b + 0.008, f"{b:.2f}", ha="center")
    ax.set_xticks(x); ax.set_xticklabels(["12,000 sequences, regularised", "600 sequences, barely regularised"])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0.5, 1.06); ax.set_ylabel("AUROC"); ygrid(ax)
    ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    save(fig, "probe_trap")
    print("probe_trap (5-mer LR)", round(a_tr, 3), round(a_te, 3), round(o_tr, 3), round(o_te, 3))



def kmer_enrichment():
    """7-letter words most enriched in the 40-letter measured sites of bound vs unbound sequences."""
    import collections

    def cores(name):
        out, h = [], None
        for l in open(DATA / name):
            l = l.strip()
            if l.startswith(">"):
                h = l
            elif h:
                m = re.search(r"[ACGTU]+", l); out.append(m.group(0).replace("T", "U")); h = None
        return out

    P, N = cores("CLIPSEQ_AGO2.train.positives.fa"), cores("CLIPSEQ_AGO2.train.negatives.fa")

    def presence(S, k=7):
        c = collections.Counter()
        for x in S:
            c.update({x[i:i + k] for i in range(len(x) - k + 1)})
        return c

    cp, cn = presence(P), presence(N)
    rows = sorted(((a / len(P)) / ((cn.get(w, 0) + 1) / len(N)), w, a / len(P)) for w, a in cp.items() if a >= 150)[::-1][:16]
    known = {"UUUGCAC": "miR-19 seed match", "UGUUUAC": "miR-30 seed match", "UGCUGCU": "miR-15/16 seed match",
             "GCUGCUA": "miR-15/16 seed match (letters 2–7 + A)", "CUACCUC": "let-7 seed match"}
    fig = newfig()
    ax = fig.add_axes([0.16, 0.12, 0.62, 0.78])
    y = np.arange(len(rows))
    ax.barh(y, [r[0] for r in rows], color=[INK if r[1] in known else LIGHT for r in rows], height=0.7)
    ax.set_yticks(y); ax.set_yticklabels([r[1] for r in rows], family=MONO); ax.invert_yaxis()
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    for yi, (ratio, w, fb) in zip(y, rows):
        ax.text(ratio + 0.08, yi, f"{fb * 100:.1f}% of bound" + (f"   {known[w]}" if w in known else ""),
                va="center", fontsize=17, color=INK if w in known else MUTED)
    ax.set_xlabel("Enrichment in bound sites (fraction containing the word, bound / unbound)"); xgrid(ax)
    save(fig, "kmer_enrichment")
    print("kmer top", [(w, round(r, 2)) for r, w, _ in rows[:6]])


# ============================================================================ deck 02
def onehot_big():
    s = "GGACAGAAGGGUAUAG"
    fig, ax = canvas()
    x0, dx = 26, 7.4
    for i, ch in enumerate(s):
        label(ax, x0 + i * dx, 74, ch, fs=32, color=INK, family=MONO)
        for r, b in enumerate(ALPH):
            on = ch == b
            ax.add_patch(Rectangle((x0 + i * dx - 3.4, 54 - r * 9), 6.8, 8.4, fc=BASE[b] if on else FAINT,
                                   ec="white", lw=2))
            label(ax, x0 + i * dx, 58.2 - r * 9, "1" if on else "0", fs=20, color="white" if on else LIGHT)
    for r, b in enumerate(ALPH):
        label(ax, x0 - 9, 58.2 - r * 9, b, fs=24, color=MUTED)
    save(fig, "onehot_big")


def filter_dot(model, pos, neg):
    ap = acts(model, pos).max(2).mean(0); an = acts(model, neg).max(2).mean(0)
    f = int(np.argmax(ap - an))
    W = model.net[0].weight[f, 0].detach().cpu().numpy(); b = float(model.net[0].bias[f])
    A = acts(model, pos[:3000])[:, f, :]
    i, j = np.unravel_index(np.argmax(A), A.shape)
    win = pos[i][j:j + 10]
    X = onehot([win], L=10)[0, 0].numpy()
    picked = (W * X).sum(0)                      # the weight of the letter present at each position
    v = np.abs(W).max()
    fig = newfig()
    a1 = fig.add_axes([0.05, 0.4, 0.38, 0.45]); a2 = fig.add_axes([0.55, 0.4, 0.38, 0.45])
    im = heat(a1, W, v); a1.set_title(f"Filter {f}: weights")
    cb = fig.colorbar(im, cax=fig.add_axes([0.44, 0.45, 0.008, 0.35])); cb.outline.set_visible(False)
    cb.set_ticks([-round(v, 1), 0, round(v, 1)])
    from matplotlib.colors import to_rgb
    a2.imshow(np.array([[to_rgb(BASE[ALPH[r]]) if X[r, c] else (0.945, 0.953, 0.957) for c in range(10)] for r in range(4)]), aspect="auto")
    a2.set_yticks(range(4)); a2.set_yticklabels(list(ALPH)); a2.tick_params(length=0)
    a2.set_xticks(range(10)); a2.set_xticklabels(list(win), family=MONO)
    for t_ in a2.get_xticklabels():
        t_.set_color(BASE[t_.get_text()]); t_.set_fontweight("bold")
    for sp in a2.spines.values():
        sp.set_visible(False)
    a2.set_title("A real window (one-hot)")
    fig.text(0.505, 0.62, "×", fontsize=44, ha="center", va="center", color=MUTED)
    for c, p_ in enumerate(picked):                        # each picked weight under its letter
        a2.text(c, 4.35, f"{p_:.2f}", ha="center", va="top", fontsize=15, color=INK, clip_on=False)
    a2.text(-0.6, 4.35, "weight", ha="right", va="top", fontsize=18, color=MUTED, clip_on=False)
    fig.text(0.55, 0.1, f"Sum {picked.sum():.2f}  +  bias {b:+.2f}  =  {picked.sum() + b:.2f}", fontsize=28, color=INK)
    save(fig, "filter_dot")
    print("filter_dot filter", f, "ratio", ap[f] / an[f])
    return f


def activation_track(model, f, pos):
    A = acts(model, pos[:2000])[:, f, :]
    i = int(np.argmax(A.max(1)))
    s, a = pos[i], A[i][:len(pos[i]) - 9]
    j = int(np.argmax(a))
    fig = newfig()
    ax = fig.add_axes([0.08, 0.14, 0.88, 0.7])
    ax.bar(range(len(a)), a, color=[ACCENT if k == j else TRAIN_L for k in range(len(a))], width=1.0)
    ax.set_xlim(-1, len(a)); ax.set_ylabel(f"Filter {f} output"); ax.set_xlabel("Position along one bound sequence")
    ax.annotate(s[j:j + 10], (j, a[j]), xytext=(j + 20 if j < 0.6 * len(a) else j - 120, a[j] * 0.97),
                fontsize=24, family=MONO, color=INK, va="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.2))
    save(fig, "activation_track")


def last_layer(model):
    W = model.net[-1].weight.detach().cpu().numpy().reshape(20, 53)
    v = np.abs(W).max()
    fig = newfig()
    ax = fig.add_axes([0.07, 0.16, 0.82, 0.66])
    im = ax.imshow(W, cmap=DIVERGING, vmin=-v, vmax=v, aspect="auto")
    ax.axvspan(40.5, 52.5, color="white", alpha=.55, lw=0)
    ax.text(46.5, -1.2, "padding", ha="center", color=MUTED)
    ax.set_xlabel("Pooled position (column j covers letters ≈ 9j to 9j + 29)"); ax.set_ylabel("Channel")
    ax.set_yticks([]); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, cax=fig.add_axes([0.91, 0.16, 0.012, 0.66])); cb.outline.set_visible(False)
    cb.set_label("Weight (blue +, grey −)")
    save(fig, "last_layer")


def filter_learning(tr, ty, ls, ly):
    from torch import nn
    from intuition import GlobalCNN
    torch.manual_seed(0)
    model = GlobalCNN().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    X = onehot(tr); y = torch.tensor(ty, dtype=torch.float32)
    snaps = {0: model.net[0].weight.detach().cpu().numpy().copy()}
    aucs = {0: roc_auc_score(ly, cnn_scores(model, ls))}
    for ep in range(1, 11):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 256):
            bi = perm[i:i + 256]
            opt.zero_grad(); crit(model(X[bi].to(DEV)), y[bi].to(DEV)).backward(); opt.step()
        model.eval()
        if ep in (1, 3, 10):
            snaps[ep] = model.net[0].weight.detach().cpu().numpy().copy()
            aucs[ep] = roc_auc_score(ly, cnn_scores(model, ls))
    f = int(np.argmax(np.abs(snaps[10] - snaps[0]).reshape(10, -1).sum(1)))
    vmax = max(np.abs(snaps[e][f, 0]).max() for e in snaps)
    fig = newfig()
    axes = [fig.add_axes([0.04 + k * 0.225, 0.2, 0.2, 0.5]) for k in range(4)]
    for k, (ax, e) in enumerate(zip(axes, (0, 1, 3, 10))):
        im = heat(ax, snaps[e][f, 0], vmax, rows=(k == 0))
        ax.set_title(f"Epoch {e}\nAUROC {aucs[e]:.2f}", loc="center")
    fig.text(0.04, 0.84, f"Filter {f} during one training run", fontsize=24, color=INK)
    cb = fig.colorbar(im, cax=fig.add_axes([0.945, 0.2, 0.01, 0.5])); cb.outline.set_visible(False)
    for k in range(4):                       # frames for the slide: one more epoch per click
        for j, ax in enumerate(axes):
            ax.set_visible(j <= k)
        fig.savefig(OUT / f"filter_learning_{k}.png", dpi=120, facecolor="white")
    for ax in axes:
        ax.set_visible(True)
    save(fig, "filter_learning")
    print("filter_learning filter", f, {e: round(a, 3) for e, a in aucs.items()})


def learning_curve():
    ref = [float(x) for x in re.findall(r"held-out AUROC = ([0-9.]+)", nb_text("02_build_cnn"))][:10]
    seeds = json.load(open(SPIKE / "s2_cnn.json"))["curves"]
    fig = newfig()
    ax = fig.add_axes([0.09, 0.14, 0.8, 0.72])
    ax.axhline(0.5, color=HAIR, lw=1.5, ls="--"); ax.text(10, 0.515, "Chance", ha="right", color=MUTED)
    for k, c in seeds.items():
        ax.plot(range(1, 11), c["auc"][:10], color=LIGHT, lw=2)
    ax.plot(range(1, len(ref) + 1), ref, "o-", color=CNN_C, lw=3, ms=9)
    ax.legend(handles=[Line2D([], [], color=CNN_C, lw=3, marker="o", label="Workshop run"),
                       Line2D([], [], color=LIGHT, lw=2, label="Three other seeds")], loc="lower right")
    ax.set_ylim(0.45, 0.95); ax.set_xlim(0.5, 10.5); ax.set_xticks(range(1, 11))
    ax.set_xlabel("Epoch"); ax.set_ylabel("Held-out AUROC"); ygrid(ax)
    save(fig, "learning_curve")


# ============================================================================ deck 03
def filter_logo(model, pos, f=0):
    W = model.net[0].weight[f, 0].detach().cpu().numpy()
    pwm, n = windows_pwm(model, f, pos[:5000])
    fig = newfig()
    a1 = fig.add_axes([0.06, 0.18, 0.38, 0.6]); a2 = fig.add_axes([0.56, 0.18, 0.4, 0.6])
    im = heat(a1, W); a1.set_xticks(range(10)); a1.set_xticklabels(range(1, 11))
    a1.set_title(f"Filter {f}: weights"); a1.set_xlabel("Position in window")
    cb = fig.colorbar(im, cax=fig.add_axes([0.45, 0.24, 0.008, 0.48])); cb.outline.set_visible(False)
    draw_logo(a2, pwm); a2.set_ylabel("Bits"); a2.set_xticks(np.arange(10) + 0.5); a2.set_xticklabels(range(1, 11))
    a2.set_title(f"Logo of its {n:,} highest-output windows"); a2.set_xlabel("Position in window")
    save(fig, "filter_logo")
    print("filter_logo windows", n)


def logo_grid(model, pos):
    fig, axes = plt.subplots(2, 5, figsize=(16, 9), dpi=120)
    for f, ax in enumerate(axes.flat):
        pwm, n = windows_pwm(model, f, pos[:3000])
        draw_logo(ax, pwm); ax.set_title(f"Filter {f}"); ax.set_ylabel(""); ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    fig.text(0.01, 0.5, "Logo height: 0 to 2 bits", rotation=90, va="center", color=MUTED)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.9, bottom=0.06, wspace=0.1, hspace=0.4)
    save(fig, "logo_grid")


def logo_bound_unbound(model, pos, neg, f=0):
    pb, nb = windows_pwm(model, f, pos[:5000]); pu, nu = windows_pwm(model, f, neg[:5000])
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), dpi=120)
    draw_logo(axes[0], pb); axes[0].set_title(f"Bound sequences ({nb:,} windows)")
    draw_logo(axes[1], pu); axes[1].set_title(f"Unbound sequences ({nu:,} windows)")
    for ax in axes:
        ax.set_ylabel("Bits"); ax.set_xticks(np.arange(10) + 0.5); ax.set_xticklabels(range(1, 11))
        ax.set_xlabel("Position in window")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.78, bottom=0.2, wspace=0.18)
    save(fig, "logo_bound_unbound")
    return "".join(ALPH[i] for i in pb.argmax(1))


def dose_response(model, motif):
    canvas_ = i4.random_rna(1000)
    base = cnn_scores(model, canvas_).mean()
    shuf = "".join(i4.rng.permutation(list(motif)))
    copies = list(range(1, 8))                  # 7 copies fit in a 330-letter sequence (as in the notebook)
    real = [cnn_scores(model, i4.plant(canvas_, motif, c)).mean() for c in copies]
    ctrl = [cnn_scores(model, i4.plant(canvas_, shuf, c)).mean() for c in copies]
    fig = newfig()
    ax = fig.add_axes([0.09, 0.16, 0.86, 0.7])
    ax.axhline(base, color=MUTED, lw=1.5, ls=":")
    ax.plot(copies, real, "o-", color=INK, lw=3, ms=10)
    ax.plot(copies, ctrl, "o--", color=MID, lw=2.5, ms=9)
    ax.legend(handles=[Line2D([], [], color=INK, lw=3, marker="o", label=f"Motif {motif}"),
                       Line2D([], [], color=MID, lw=2.5, ls="--", marker="o", label=f"Same letters shuffled ({shuf})"),
                       Line2D([], [], color=MUTED, lw=1.5, ls=":", label="Nothing inserted")], loc="upper left")
    ax.set_xticks(copies); ax.set_xticklabels([f"{c}\n@{40 + (c - 1) * 45}" for c in copies])
    ax.set_xlabel("Copies inserted (and where the last one starts)"); ax.set_ylabel("Mean CNN score")
    ax.set_ylim(0.3, max(real) + 0.12); ygrid(ax)
    save(fig, "dose_response")
    return shuf


def position_scan(model, pos, shuf, f=0):
    pwm, _ = windows_pwm(model, f, pos[:5000])
    motif = "".join(ALPH[i] for i in pwm.argmax(1))
    rng = np.random.default_rng(0)
    canv = ["".join(ALPH[i] for i in row) for row in rng.integers(0, 4, (400, 330))]
    base = cnn_scores(model, canv).mean()
    xs = list(range(0, 321, 4))

    def scan(m):
        return np.array([cnn_scores(model, [c[:x] + m + c[x + len(m):] for c in canv]).mean() - base for x in xs])

    r, sh = scan(motif), scan(shuf)
    fig = newfig()
    ax = fig.add_axes([0.09, 0.16, 0.86, 0.7])
    ax.axvspan(150, 190, color=FAINT, zorder=0)
    ax.axhline(0, color=MUTED, lw=1)
    ax.plot(xs, r, color=INK, lw=3); ax.plot(xs, sh, color=MID, lw=2.5, ls="--")
    ax.set_ylim(min(r.min(), sh.min()) - 0.06, r.max() * 1.35)
    ax.text(170, r.max() * 1.12, "Measured site\n(letters 150–190)", ha="center", color=MUTED, va="bottom")
    ax.legend(handles=[Line2D([], [], color=INK, lw=3, label=f"Motif {motif}"),
                       Line2D([], [], color=MID, lw=2.5, ls="--", label=f"Same letters shuffled ({shuf})")],
              loc="upper left")
    for c, x in enumerate((40, 85, 130, 175, 220, 265, 310), 1):     # where plant() writes copies 1 to 7
        ax.plot([x, x], [0, 0.04], color=INK, lw=2, transform=ax.get_xaxis_transform(), clip_on=False)
        ax.text(x, 0.055, str(c), transform=ax.get_xaxis_transform(), ha="center", color=INK, fontsize=18)
    ax.set_xlabel("Insertion position (numbers: where copies 1–7 go in the previous test)")
    ax.set_ylabel("Change in mean CNN score")
    save(fig, "position_scan")


def planting_bars(model, ls, ly):
    sites = {"miR-15/16\nUGCUGCU": "UGCUGCU", "miR-19\nUUUGCAC": "UUUGCAC", "miR-30\nUGUUUAC": "UGUUUAC",
             "let-7\nCUACCUC": "CUACCUC"}
    canv = [s for s, y in zip(ls, ly) if y == 0]
    base = cnn_scores(model, canv).mean()
    r, m, sd = [], [], []
    for site in sites.values():
        r.append(cnn_scores(model, i4.plant(canv, site, 6)).mean())
        sh = [cnn_scores(model, i4.plant(canv, "".join(i4.rng.permutation(list(site))), 6)).mean() for _ in range(10)]
        m.append(np.mean(sh)); sd.append(np.std(sh))
    fig = newfig()
    ax = fig.add_axes([0.09, 0.16, 0.86, 0.68])
    x = np.arange(4)
    ax.axhline(base, color=MUTED, lw=1.5, ls=":", label="Nothing inserted")
    ax.bar(x - 0.18, r, 0.34, color=INK, label="Seed site, 6 copies")
    ax.bar(x + 0.18, m, 0.34, yerr=sd, capsize=7, color=LIGHT, error_kw=dict(ecolor=MUTED, lw=1.8),
           label="Same letters shuffled (mean ± SD of 10)")
    for xi, a, mm, s_ in zip(x, r, m, sd):
        ax.text(xi - 0.18, a + 0.012, f"{a:.2f}", ha="center")
        ax.text(xi + 0.18, mm + s_ + 0.012, f"{mm:.2f}", ha="center", color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels(list(sites)); ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Mean CNN score"); ax.set_ylim(0, 0.66); ygrid(ax)
    ax.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0), fontsize=18)
    save(fig, "planting_bars")
    print("planting", [round(v, 3) for v in r], [round(v, 3) for v in m], [round(v, 3) for v in sd], round(base, 3))


# ============================================================================ deck 04
def tokens(pos):
    from transformers import AutoTokenizer
    s = next(x for x in pos if "UUUGCAC" in x[40:-40])
    k = s.index("UUUGCAC", 40)
    w = s[k - 12:k + 18].replace("U", "T")
    site = 12
    rows = [("Single letters\n(one-hot, the CNN)", None, w),
            ("Mistral-DNA, DNABERT-2\n(learned BPE vocabulary)", "RaphaelMourad/Mistral-DNA-v1-17M-hg38", w),
            ("Nucleotide Transformer\n(6-letter tokens)", "InstaDeepAI/nucleotide-transformer-500m-human-ref", w),
            ("Nucleotide Transformer,\nshifted by one letter", "InstaDeepAI/nucleotide-transformer-500m-human-ref", w[1:])]
    fig, ax = canvas()
    dx, x0 = 3.9, 44
    for i, ch in enumerate(w):
        label(ax, x0 + i * dx, 80, ch, fs=24, color=INK, family=MONO)
    ax.plot([x0 + site * dx - 1.8, x0 + (site + 6) * dx + 1.8], [75.5, 75.5], color=INK, lw=2.5)
    label(ax, x0 + (site + 3) * dx, 72, "miR-19 site", fs=18, color=MUTED)
    label(ax, 4, 80, "Sequence (DNA letters)", fs=20, color=MUTED, ha="left")
    for r, (name, repo, s_) in enumerate(rows):
        if repo is None:
            t = list(s_)
        else:
            tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
            t = tok.convert_ids_to_tokens(tok(s_, add_special_tokens=False)["input_ids"])
        y = 60 - r * 15
        label(ax, 4, y, name, fs=19, color=MUTED, ha="left")
        p = 0 if s_ == w else 1
        for j, tk in enumerate(t):
            tk = tk.replace("▁", "").replace("Ġ", "")
            if not tk or tk[0] in "<[":
                continue
            box(ax, x0 + p * dx - 1.85, y - 4.5, len(tk) * dx - 0.45, 9, fc=[FAINT, HAIR][j % 2], r=0.6)
            for q, ch in enumerate(tk):
                label(ax, x0 + (p + q) * dx, y, ch, fs=21, color=INK, family=MONO)
            p += len(tk)
    save(fig, "tokens")


def pipeline_real(pos):
    from transformers import AutoModel, AutoTokenizer
    name = "RaphaelMourad/Mistral-DNA-v1-17M-hg38"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    fm = AutoModel.from_pretrained(name, trust_remote_code=True).eval()
    s = pos[3][:60].replace("U", "T")
    enc = tok(s, return_tensors="pt")
    with torch.no_grad():
        H = fm(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], output_hidden_states=True).hidden_states[-1][0].numpy()
    toks = tok.convert_ids_to_tokens(enc["input_ids"][0])
    fig = newfig()
    a1 = fig.add_axes([0.17, 0.32, 0.78, 0.54])
    v = np.percentile(np.abs(H), 97)
    a1.imshow(H, cmap=DIVERGING, vmin=-v, vmax=v, aspect="auto", interpolation="nearest")
    a1.set_yticks(range(len(toks))); a1.set_yticklabels(toks, fontsize=18, family=MONO); a1.set_xticks([])
    a1.tick_params(length=0)
    a1.set_title(f"Last layer: one row of {H.shape[1]} numbers per token")
    a2 = fig.add_axes([0.17, 0.14, 0.78, 0.07])
    a2.imshow(H.mean(0, keepdims=True), cmap=DIVERGING, vmin=-v, vmax=v, aspect="auto", interpolation="nearest")
    a2.set_yticks([0]); a2.set_yticklabels(["Mean"]); a2.set_xticks([0, 64, 128, 192, 255])
    a2.tick_params(length=0); a2.set_xlabel("Embedding dimension")
    for a in (a1, a2):
        for sp in a.spines.values():
            sp.set_visible(False)
    fig.text(0.56, 0.245, "↓", ha="center", fontsize=30, color=MUTED)
    save(fig, "pipeline_real")


def next_token(pos):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "RaphaelMourad/Mistral-DNA-v1-17M-hg38"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    lm = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True).eval()
    ids = tok(pos[7].replace("U", "T"), return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    cut = 14
    with torch.no_grad():
        p = torch.softmax(lm(input_ids=ids[None, :cut]).logits[0, -1], -1)
    top = torch.topk(p, 8); true = int(ids[cut])
    items = [(tok.convert_ids_to_tokens(int(i)), float(v), int(i) == true) for v, i in zip(top.values, top.indices)]
    if true not in top.indices.tolist():
        items.append((tok.convert_ids_to_tokens(true), float(p[true]), True))
    prefix = "".join(tok.convert_ids_to_tokens(ids[:cut]))
    fig = newfig()
    ax = fig.add_axes([0.22, 0.12, 0.7, 0.62])
    ax.barh(range(len(items)), [v for _, v, _ in items], color=[HELD_C if t else TRAIN_L for *_, t in items], height=0.7)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([n + ("  (true)" if t else "") for n, _, t in items], family=MONO)
    ax.invert_yaxis(); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlabel("Predicted probability of the next token"); xgrid(ax)
    fig.text(0.04, 0.86, "Context", color=MUTED)
    fig.text(0.13, 0.86, "…" + prefix[-52:], family=MONO, color=INK)
    save(fig, "next_token")


def landscape():
    pts = [("DNABERT", 2021.0, 110e6, "M"), ("Nucleotide Transformer", 2023.0, 2.5e9, "M"), ("NT-50M", 2023.0, 50e6, "M"),
           ("DNABERT-2", 2023.4, 117e6, "M"), ("HyenaDNA", 2023.5, 6.6e6, "A"), ("Mistral-DNA", 2024.0, 17e6, "A"),
           ("Evo", 2024.2, 7e9, "A"), ("Evo 2", 2025.1, 40e9, "A")]
    fig = newfig()
    ax = fig.add_axes([0.1, 0.13, 0.86, 0.75])
    ax.plot([2023.0, 2023.0], [50e6, 2.5e9], color=LIGHT, lw=2)
    for n, x, p, kind in pts:
        ax.scatter(x, p, s=200, zorder=3, color=ACCENT if kind == "M" else HELD_C, edgecolor="white", lw=1.5)
        ax.text(x + 0.07, p * 1.3, n, fontsize=17)
    ax.set_yscale("log"); ax.set_xlim(2020.6, 2025.9); ax.set_xticks([2021, 2022, 2023, 2024, 2025])
    ax.set_ylabel("Parameters"); ygrid(ax)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=12, color=ACCENT, label="Masked (fill in the blank)"),
                       Line2D([], [], marker="o", ls="", ms=12, color=HELD_C, label="Autoregressive (predict next)")],
              loc="upper left")
    save(fig, "landscape")


# ============================================================================ deck 05
def results_part2():
    rows = [("105M model, random weights", 0.554, LIGHT), ("105M model, 3 min pretraining", 0.717, ACCENT_L),
            ("Mistral-DNA (16M), probe", 0.782, ACCENT), ("NT-500M (human reference), probe", 0.788, ACCENT),
            ("CNN, part 1", 0.861, GOOD_C)]
    fig = newfig()
    ax = fig.add_axes([0.34, 0.14, 0.6, 0.74])
    for i, (n, v, c) in enumerate(rows):
        ax.barh(i, v - 0.5, left=0.5, color=c, height=0.62)
        ax.text(v + 0.004, i, f"{v:.2f}", va="center")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
    ax.tick_params(axis="y", length=0); ax.set_xlim(0.5, 0.9); xgrid(ax)
    ax.set_xlabel("Held-out AUROC (bars start at 0.5 = chance)")
    fig.text(0.34, 0.92, "Probes: 12,000 labelled sequences.  CNN: 92,000.", color=MUTED)
    save(fig, "results_part2")


def pretrain_loss():
    from transformers import AutoTokenizer
    t = nb_text("05_pretrained_model")
    st = [(int(a), float(b)) for a, b in re.findall(r"step\s+(\d+) \|.*?loss ([0-9.]+)", t)]
    tok = AutoTokenizer.from_pretrained("RaphaelMourad/Mistral-DNA-v1-17M-hg38", trust_remote_code=True)
    tr, _ = split("train")
    cnt = np.bincount(np.concatenate([tok(s.replace("U", "T"), add_special_tokens=False)["input_ids"] for s in tr[:5000]]),
                      minlength=len(tok)).astype(float)
    pr = cnt[cnt > 0] / cnt.sum()
    unigram = float(-(pr * np.log(pr)).sum())
    fig = newfig()
    ax = fig.add_axes([0.09, 0.14, 0.82, 0.72])
    ax.axhline(np.log(4096), color=MUTED, lw=1.5, ls="--")
    ax.text(st[-1][0], np.log(4096) - 0.1, "Uniform guess (ln 4,096)", ha="right", va="top", color=MUTED)
    ax.axhline(unigram, color=MUTED, lw=1.5, ls=":")
    ax.text(st[-1][0], unigram + 0.06, "Token frequencies only (no context)", ha="right", va="bottom", color=MUTED)
    ax.plot([s for s, _ in st], [l for _, l in st], "o-", color=INK, lw=3, ms=8)
    ax.set_xlabel("Training step (3 minutes on one GPU)"); ax.set_ylabel("Next-token loss")
    ax.set_ylim(min(6, unigram - 0.3), 8.6); ygrid(ax)
    save(fig, "pretrain_loss")
    print("pretrain unigram entropy", round(unigram, 3))


def emb_pca(tr, ty):
    from matplotlib.colors import LinearSegmentedColormap
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    E = np.load(DATA / "emb_mistral17m.npz")["train"]
    idx = np.random.default_rng(0).permutation(len(tr))[:12000]
    S = [tr[i] for i in idx]; y = ty[idx]
    g = np.array([gc(s) for s in S])
    P = PCA(2).fit_transform(StandardScaler().fit_transform(E))
    greys = LinearSegmentedColormap.from_list("grey_ink", ["#E8EAED", INK])
    fig = newfig()
    a1 = fig.add_axes([0.05, 0.1, 0.38, 0.74]); a2 = fig.add_axes([0.6, 0.1, 0.38, 0.74])
    o = np.random.default_rng(1).permutation(len(y))
    sc = a1.scatter(P[o, 0], P[o, 1], c=g[o], cmap=greys, s=5, alpha=.8, lw=0)
    cb = fig.colorbar(sc, cax=fig.add_axes([0.45, 0.2, 0.01, 0.5])); cb.outline.set_visible(False); cb.set_label("G+C fraction")
    a2.scatter(P[o, 0], P[o, 1], c=np.where(y[o] == 1, BOUND, UNB), s=4, alpha=.45, lw=0)
    a2.legend(handles=[Line2D([], [], marker="o", ls="", ms=10, color=BOUND, label="Bound"),
                       Line2D([], [], marker="o", ls="", ms=10, color=UNB, label="Unbound")], loc="upper right")
    a1.set_title("Coloured by G+C"); a2.set_title("Coloured by label")
    for a in (a1, a2):
        a.set_xticks([]); a.set_yticks([]); a.set_xlabel("PCA direction 1"); a.set_ylabel("PCA direction 2")
    save(fig, "emb_pca")
    return np.corrcoef(P[:, 0], g)[0, 1], np.corrcoef(P[:, 1], g)[0, 1]


# ============================================================================ deck 06
def gc_hist():
    S, y = train_set()
    g = gc_u(S)[:, 0]
    fig = newfig()
    ax = fig.add_axes([0.08, 0.14, 0.88, 0.7])
    bins = np.linspace(0.15, 0.8, 60)
    ax.hist(g[y == 0], bins, color=MID, histtype="step", lw=2.5, density=True)
    ax.hist(g[y == 1], bins, color=BOUND, alpha=.45, density=True)
    ax.axvline(g[y == 1].mean(), color=BOUND, lw=1.8, ls="--"); ax.axvline(g[y == 0].mean(), color=MID, lw=1.8, ls="--")
    ax.legend(handles=[Patch(color=BOUND, alpha=.45, label=f"Bound (mean {g[y == 1].mean():.2f})"),
                       Line2D([], [], color=MID, lw=2.5, label=f"Unbound (mean {g[y == 0].mean():.2f})")], loc="upper right")
    ax.set_xlabel("G+C fraction"); ax.set_ylabel("Density"); ax.set_yticks([])
    save(fig, "gc_hist")


def shuffle_demo(pos):
    rng = np.random.default_rng(3)
    s = next(x for x in pos if "UGCUGCU" in x)
    k = s.index("UGCUGCU")
    sh = dinuc_shuffle(s, rng)
    pairs = [a + b for a in ALPH for b in ALPH]

    def counts(x):
        return [sum(1 for i in range(len(x) - 1) if x[i:i + 2] == p) for p in pairs]

    same = sum(a == b for a, b in zip(counts(s), counts(sh)))
    fig = newfig()
    a0 = fig.add_axes([0.02, 0.5, 0.96, 0.44]); a0.set_xlim(0, 160); a0.set_ylim(0, 30); a0.axis("off")
    st = max(0, k - 14)
    seq(a0, 30, 22, s[st:st + 32], fs=26, dx=4.0)
    seq(a0, 30, 8, sh[st:st + 32], fs=26, dx=4.0)
    label(a0, 4, 22, "Original", fs=22, color=MUTED, ha="left"); label(a0, 4, 8, "Shuffled", fs=22, color=MUTED, ha="left")
    a1 = fig.add_axes([0.07, 0.1, 0.9, 0.3])
    x = np.arange(16)
    a1.bar(x - 0.19, counts(s), 0.36, color=ACCENT, label="Original")
    a1.bar(x + 0.19, counts(sh), 0.36, color=HELD_C, label="Shuffled")
    a1.set_xticks(x); a1.set_xticklabels(pairs, family=MONO); a1.tick_params(axis="x", length=0)
    a1.set_yticks([]); a1.spines["left"].set_visible(False)
    a1.set_title(f"Letter-pair counts in the whole sequence: {same} of 16 identical")
    a1.legend(loc="upper left", ncol=2)
    save(fig, "shuffle_demo")


def shuffle_scores(model, tr, ty, ls, ly):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from transformers import AutoModel, AutoTokenizer
    rng = np.random.default_rng(0)
    bound = [s for s, l in zip(ls, ly) if l == 1]; unbound = [s for s, l in zip(ls, ly) if l == 0]
    shuf = [dinuc_shuffle(s, rng) for s in bound]
    cs = {"b": cnn_scores(model, bound), "s": cnn_scores(model, shuf), "u": cnn_scores(model, unbound)}
    d = np.load(DATA / "emb_mistral17m.npz")
    idx = np.random.default_rng(0).permutation(len(tr))[:12000]
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1)).fit(d["train"], ty[idx])
    name = "RaphaelMourad/Mistral-DNA-v1-17M-hg38"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    tok.pad_token = tok.pad_token or tok.eos_token or tok.unk_token
    fm = AutoModel.from_pretrained(name, trust_remote_code=True).to(DEV).eval()

    @torch.no_grad()
    def emb(S):
        out = []
        for i in range(0, len(S), 128):
            e = tok([x.replace("U", "T") for x in S[i:i + 128]], return_tensors="pt", padding=True, truncation=True, max_length=512)
            ids, m = e["input_ids"].to(DEV), e["attention_mask"].to(DEV)
            h = fm(input_ids=ids, attention_mask=m, output_hidden_states=True).hidden_states[-1].float()
            mk = m.unsqueeze(-1).float(); out.append(((h * mk).sum(1) / mk.sum(1)).cpu().numpy())
        return np.concatenate(out)

    ps = {k: probe.predict_proba(emb(v))[:, 1] for k, v in (("b", bound), ("s", shuf), ("u", unbound))}
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), dpi=120)
    bins = np.linspace(0, 1, 40)
    for ax, sc, t in ((axes[0], cs, "CNN"), (axes[1], ps, "Mistral-DNA + probe")):
        ax.hist(sc["u"], bins, color=UNB, alpha=.9, density=True, label="Unbound")
        ax.hist(sc["b"], bins, color=BOUND, alpha=.5, density=True, label="Bound")
        ax.hist(sc["s"], bins, color=INK, histtype="step", lw=2.5, density=True, label="Bound, shuffled")
        a = roc_auc_score([1] * len(sc["s"]) + [0] * len(sc["u"]), np.concatenate([sc["s"], sc["u"]]))
        ax.set_title(f"{t}\nshuffled vs unbound: AUROC {a:.2f}")
        ax.set_xlabel("Model score"); ax.set_ylabel("Density (per panel)"); ax.set_yticks([])
    axes[0].legend(loc="upper center")
    fig.subplots_adjust(left=0.04, right=0.98, top=0.8, bottom=0.12, wspace=0.12)
    save(fig, "shuffle_scores")


def shuffle_scoreboard():
    fig = newfig()
    ax = fig.add_axes([0.08, 0.13, 0.66, 0.75])
    x = np.array([0, 1]); real = [0.861, 0.782]; shuf = [0.661, 0.765]
    ax.bar(x - 0.17, [r - 0.5 for r in real], 0.32, bottom=0.5, color=[INK, MID])
    ax.bar(x + 0.17, [s - 0.5 for s in shuf], 0.32, bottom=0.5, color="white", edgecolor=[INK, MID], lw=2.5)
    for xi, r, s in zip(x, real, shuf):
        ax.text(xi - 0.17, r + 0.006, f"{r:.2f}", ha="center")
        ax.text(xi + 0.17, s + 0.006, f"{s:.2f}", ha="center")
        ax.text(xi - 0.17, 0.515, "Real", ha="center", fontsize=18, color="white")
        ax.text(xi + 0.17, 0.515, "Bound\nshuffled", ha="center", fontsize=18, color=INK)
    ax.axhline(0.660, ls="--", color=MID, lw=2); ax.text(1.42, 0.660, "G+C and U only  0.66", color=MUTED, va="center")
    ax.axhline(0.738, ls=":", color=MID, lw=2); ax.text(1.42, 0.738, "5-letter word counts  0.74", color=MUTED, va="center")
    ax.set_xlim(-0.45, 1.4); ax.set_xticks(x); ax.set_xticklabels(["CNN", "Mistral-DNA + probe"])
    ax.tick_params(axis="x", length=0); ax.set_ylim(0.5, 0.92); ax.set_ylabel("Held-out AUROC (axis starts at 0.5)"); ygrid(ax)
    save(fig, "shuffle_scoreboard")


# ============================================================================ deck 07
def matched_hist():
    S, y = train_set()
    idx = matched_indices(S, y)
    g = gc_u(S)[:, 0]
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), dpi=120, sharey=True)
    bins = np.linspace(0.15, 0.8, 50)
    for ax, (gg, yy, t) in zip(axes, ((g, y, f"Original ({len(S):,} sequences)"), (g[idx], y[idx], f"Matched ({len(idx):,} sequences)"))):
        ax.hist(gg[yy == 0], bins, color=UNB, alpha=.9, density=True, label="Unbound")
        ax.hist(gg[yy == 1], bins, color=BOUND, alpha=.55, density=True, label="Bound")
        ax.set_title(t); ax.set_xlabel("G+C fraction"); ax.set_yticks([]); ax.spines["left"].set_visible(False)
    axes[1].legend(loc="upper right")
    fig.subplots_adjust(left=0.03, right=0.98, top=0.86, bottom=0.12, wspace=0.08)
    save(fig, "matched_hist")


def lora():
    from transformers import AutoModel
    fm = AutoModel.from_pretrained("RaphaelMourad/Mistral-DNA-v1-17M-hg38", trust_remote_code=True)
    W = next(mod.weight for n, mod in fm.named_modules() if n.endswith("q_proj")).detach().numpy()
    r = 16
    A = np.random.default_rng(0).normal(0, 0.02, (r, W.shape[1]))
    fig = newfig()
    a1 = fig.add_axes([0.05, 0.22, 0.3, 0.55]); a2 = fig.add_axes([0.47, 0.22, 0.05, 0.55])
    a3 = fig.add_axes([0.6, 0.62, 0.36, 0.08])
    v = np.percentile(np.abs(W), 99)
    a1.imshow(W, cmap=DIVERGING, vmin=-v, vmax=v, aspect="auto")
    a1.set_title(f"W, frozen\n{W.shape[0]} × {W.shape[1]} = {W.size:,}", loc="center")
    a2.imshow(np.zeros((W.shape[0], r)), cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    a2.set_title(f"B\n{W.shape[0]} × {r}", loc="center", color=TUNED_C)
    a3.imshow(A, cmap=SEQUENTIAL, aspect="auto"); a3.set_title(f"A   {r} × {W.shape[1]}", color=TUNED_C)
    for a in (a1, a2, a3):
        a.set_xticks([]); a.set_yticks([])
    for a in (a2, a3):
        for sp in a.spines.values():
            sp.set_visible(True); sp.set_edgecolor(TUNED_C); sp.set_linewidth(2.5)
    fig.text(0.41, 0.5, "+", fontsize=52, ha="center", va="center", color=MUTED)
    fig.text(0.56, 0.66, "×", fontsize=36, ha="center", va="center", color=MUTED)
    fig.text(0.47, 0.12, "B starts at zero", ha="left", color=TUNED_C)
    fig.text(0.6, 0.45, f"Trained: 2 × {W.shape[0]} × {r} = {2 * W.shape[0] * r:,}\nRank r = {r}", ha="left", color=TUNED_C)
    save(fig, "lora")


def dumbbell():
    rows = [("G+C and U only", 0.637, 0.510, MID, "0.64", "0.51"), ("5-letter word counts", 0.762, 0.725, MID, "0.76", "0.73"),
            ("Mistral-DNA + probe", 0.806, 0.770, MID, "0.81", "0.77"), ("CNN", 0.843, 0.838, INK, "0.84", "0.84")]
    fig = newfig()
    ax = fig.add_axes([0.26, 0.16, 0.68, 0.7])
    ax.axvline(0.5, color=HAIR, lw=1.5, ls="--"); ax.text(0.503, 3.55, "Chance", color=MUTED)
    for i, (n, o, m, c, os_, ms_) in enumerate(rows):
        ax.plot([m, o], [i, i], color=c, lw=3, alpha=.4)
        ax.scatter(o, i, s=260, color="white", edgecolor=c, lw=2.5, zorder=3)
        ax.scatter(m, i, s=260, color=c, zorder=3)
        if o - m < 0.02:
            ax.text(o + 0.014, i, f"{os_} → {ms_}", va="center")
        else:
            ax.text(m, i - 0.34, ms_, ha="center")
            ax.text(o, i - 0.34, os_, ha="center", color=MUTED)
    ax.set_yticks(range(4)); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlim(0.47, 0.92); ax.set_ylim(3.6, -0.7); xgrid(ax)
    ax.set_xlabel("Held-out AUROC, new 3,000-sequence test sets")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=14, mfc="white", mec=INK, mew=2, label="Original negatives"),
                       Line2D([], [], marker="o", ls="", ms=14, color=INK, label="Matched negatives")],
              loc="lower right", ncol=2, bbox_to_anchor=(1, 1.0))
    save(fig, "dumbbell")


def headtohead():
    d = json.load(open(SPIKE / "s10_headtohead.json"))
    fig = newfig()
    ax = fig.add_axes([0.08, 0.13, 0.88, 0.7])
    series = [("cnn", "CNN from scratch (3 runs)", CNN_C), ("frozen_probe", "NT-500M frozen + probe (1 run)", FROZEN_C),
              ("lora", "NT-500M fine-tuned with LoRA (3 runs)", TUNED_C)]
    for gi, k in enumerate(("500", "20000")):
        for si, (key, name, c) in enumerate(series):
            x = gi + (si - 1) * 0.26
            r = d[k][key]
            ax.bar(x, r["mean"], 0.23, color=c, label=name if gi == 0 else None)
            ax.scatter([x] * len(r["runs"]), r["runs"], color="white" if key != "frozen_probe" else INK,
                       edgecolor=INK, s=36, zorder=3)
            ax.text(x, max(r["runs"]) + 0.012, f"{r['mean']:.2f}", ha="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["500 labelled sequences", "20,000 labelled sequences"])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0.5, 0.94); ax.set_ylabel("Held-out AUROC, matched negatives"); ygrid(ax)
    ax.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0), fontsize=17)
    save(fig, "headtohead")


def model_scan():
    d = json.load(open(SPIKE / "s8_models.json"))
    ft = {"Mistral-DNA-17M (hg38)": 0.787, "NT-500M (human ref)": 0.869, "DNABERT-2-117M (multi-species)": 0.874,
          "HyenaDNA-small-32k (hg38)": 0.723}
    nice = {"NT-500M (human ref)": "NT-500M (human ref.)", "NT-v2-500M (multi-species)": "NT-v2-500M (multi-sp.)",
            "NT-2.5B (multi-species)": "NT-2.5B (multi-sp.)"}
    fig = newfig()
    ax = fig.add_axes([0.09, 0.13, 0.87, 0.75])
    ax.axhline(0.838, color=CNN_C, lw=2, ls="--"); ax.text(2e6, 0.843, "CNN, whole matched training set", color=CNN_C)
    for n, r in d.items():
        if "matched" not in r:
            continue
        p = r["params_M"] * 1e6
        ax.scatter(p, r["matched"], s=180, color=FROZEN_C, zorder=3)
        short = nice.get(n, n.split(" (")[0].replace("-small-32k", "").replace("-medium-160k", " (medium)"))
        ax.text(p * 1.12, r["matched"] - 0.006, short, fontsize=16, color=MUTED)
        if n in ft:
            ax.annotate("", xy=(p, ft[n]), xytext=(p, r["matched"]),
                        arrowprops=dict(arrowstyle="-|>", color=TUNED_C, lw=2, mutation_scale=18))
            ax.scatter(p, ft[n], s=180, color=TUNED_C, zorder=4)
    ax.set_xscale("log"); ax.set_ylim(0.68, 0.9); ax.set_xlim(1.5e6, 9e9); ygrid(ax)
    ax.set_xlabel("Parameters"); ax.set_ylabel("Held-out AUROC, matched negatives")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=13, color=FROZEN_C, label="Frozen + probe (12,000 labels)"),
                       Line2D([], [], marker="o", ls="", ms=13, color=TUNED_C, label="Fine-tuned, LoRA (20,000 labels, one run)")],
              loc="upper left")
    save(fig, "model_scan")


def label_curve():
    d = json.load(open(SPIKE / "s7_lowdata.json")); h = json.load(open(SPIKE / "s10_headtohead.json"))
    ns = sorted(int(k) for k in d)
    fig = newfig()
    ax = fig.add_axes([0.08, 0.13, 0.68, 0.75])
    lab = {"cnn": ("CNN", CNN_C, "-"), "fm_probe": ("Mistral-DNA, frozen", FROZEN_C, "-"),
           "kmer5": ("5-letter word counts", MID, ":"), "random_probe": ("Mistral-DNA, random weights", MID, "--")}
    for k, (name, c, ls_) in lab.items():
        m = np.array([d[str(n)][k]["mean"] for n in ns]); sd = np.array([d[str(n)][k]["sd"] for n in ns])
        ax.fill_between(ns, m - sd, m + sd, color=c, alpha=.12, lw=0)
        ax.plot(ns, m, marker="o", ls=ls_, color=c, lw=2.8 if k in ("cnn", "fm_probe") else 2.2, ms=6)
        ax.text(ns[-1] * 1.12, m[-1], name, color=c if c != LIGHT else MUTED, va="center")
    for n in (500, 20000):
        c_ = h[str(n)]["cnn"]; l_ = h[str(n)]["lora"]
        ax.errorbar([n], [c_["mean"]], yerr=[c_["sd"]], fmt="s", mfc="white", mec=CNN_C, color=CNN_C, ms=11, capsize=4)
        ax.errorbar([n], [l_["mean"]], yerr=[l_["sd"]], fmt="D", color=TUNED_C, ms=12, capsize=4)

    ax.axhline(0.5, color=HAIR, lw=1.5, ls="--")
    ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels([f"{n:,}" for n in ns], fontsize=17)
    ax.minorticks_off(); ax.set_xlabel("Labelled training sequences"); ax.set_ylabel("Held-out AUROC, matched negatives")
    ax.set_ylim(0.45, 0.92); ygrid(ax)
    ax.legend(handles=[Line2D([], [], marker="D", ls="", ms=11, color=TUNED_C, label="NT-500M fine-tuned (head-to-head)"),
                       Line2D([], [], marker="s", ls="", ms=10, mfc="white", mec=CNN_C, label="CNN, other draws (head-to-head)")],
              loc="lower right", fontsize=17)
    save(fig, "label_curve2")


# ============================================================================ deck 08
def recap():
    tiles = [("seed_pairing", "The seed rule"), ("filter_dot", "A filter computes a dot product"),
             ("planting_bars", "The CNN responds to miR-16"), ("position_scan", "The CNN weighs position"),
             ("emb_pca", "Embeddings vary with G+C"), ("shuffle_scores", "The shuffle test"),
             ("dumbbell", "Matched negatives"), ("headtohead", "Fine-tuning against the CNN")]
    fig, axes = plt.subplots(2, 4, figsize=(16, 9), dpi=120)
    for ax, (n, t) in zip(axes.flat, tiles):
        ax.imshow(mpimg.imread(OUT / f"{n}.png")); ax.axis("off"); ax.set_title(t, fontsize=17)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.04, wspace=0.05, hspace=0.18)
    save(fig, "recap")


if __name__ == "__main__":
    want = set(sys.argv[1:])
    run = (lambda n: not want or n in want)
    tr, ty = split("train"); ls, ly = split("ls")
    pos = [s for s, l in zip(tr, ty) if l == 1]; neg = [s for s, l in zip(tr, ty) if l == 0]
    model = load_cnn()
    f = 7
    for n, job in [
        ("hero", lambda: print("hero score", hero(model, pos))), ("seed_pairing", lambda: seed_pairing(pos)),
        ("labels_raster", lambda: labels_raster(tr, ty)), ("lr_training", lambda: lr_training(tr, ty)),
        ("auroc", lambda: auroc(model, ls, ly)), ("auroc_ci", lambda: auroc_ci(model, tr, ty, ls, ly)),
        ("probe_trap", lambda: probe_trap(tr, ty)), ("onehot_big", onehot_big),
        ("filter_dot", lambda: filter_dot(model, pos[:6000], neg[:6000])),
        ("activation_track", lambda: activation_track(model, f, pos[:6000])), ("last_layer", lambda: last_layer(model)),
        ("learning_curve", learning_curve), ("filter_logo", lambda: filter_logo(model, pos)),
        ("logo_grid", lambda: logo_grid(model, pos)),
        ("tokens", lambda: tokens(pos)), ("pipeline_real", lambda: pipeline_real(pos)), ("next_token", lambda: next_token(pos)),
        ("landscape", landscape), ("results_part2", results_part2), ("pretrain_loss", pretrain_loss),
        ("emb_pca", lambda: print("PC1, PC2 vs G+C", emb_pca(tr, ty))), ("gc_hist", gc_hist),
        ("shuffle_demo", lambda: shuffle_demo(pos)),
        ("shuffle_scores", lambda: shuffle_scores(model, tr, ty, ls, ly)), ("shuffle_scoreboard", shuffle_scoreboard),
        ("matched_hist", matched_hist), ("lora", lora), ("dumbbell", dumbbell), ("headtohead", headtohead),
        ("model_scan", model_scan), ("label_curve2", label_curve), ("kmer_enrichment", kmer_enrichment)]:
        if run(n):
            job()
    # the random draws below must stay in this order (same numbers as the notes)
    if run("dose_response") or run("planting_bars") or run("logo_bound_unbound") or run("position_scan"):
        motif = logo_bound_unbound(model, pos, neg)
        shuf = dose_response(model, motif)
        planting_bars(model, ls, ly)
        position_scan(model, pos, shuf)
    if run("filter_learning"):
        filter_learning(tr, ty, ls, ly)
