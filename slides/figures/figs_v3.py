"""Figures added on 14.10 for the part 1 deck (same style as figs.py; frames for click-through slides).

    python figs_v3.py [name ...]      (in the hw1-00660121 env; writes ../assets/custom/<name>.png)

Frames of one animation are saved untrimmed at the same size, so they stack exactly on a slide.
"""
import sys

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch, Rectangle

from deckstyle import *  # noqa: F401,F403

ALPH = "ACGU"


def save_raw(fig, name):
    """Save without trimming, so frames of one animation keep the same size."""
    fig.savefig(OUT / f"{name}.png", dpi=120, facecolor="white")
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ about me
def timeline():
    """CV as a vertical timeline (oldest at the top)."""
    # (years, title, detail, sub-entries); a sub-entry is drawn smaller, indented, with a grey dot
    rows = [("2015 to 2020", "Unit 81", "Embedded Linux developer", []),
            ("2020 to 2024", "B.Sc. Computer Science, Technion", "Bioinformatics track",
             [("Exchange semester, National Taiwan University (2023)", "Machine learning with Prof. Hsuan-Tien Lin (General Chair, NeurIPS 2026)")]),
            ("2024 to 2026", "M.Sc. Data Science, Technion", "Research in Prof. Roee Amit's lab",
             [("Teaching assistant", "00660121 Medical Diagnostics (Intro to Deep Learning for Biotechnology)")])]
    fig, ax = canvas(10, 8.6, xlim=(0, 100), ylim=(0, 86))
    y, ys_all = 78, []
    for yr, title, detail, subs in rows:
        ax.add_patch(plt.Circle((6, y), 1.9, color=ACCENT, zorder=2))
        ax.add_patch(plt.Circle((6, y), 0.8, color="white", zorder=3))
        label(ax, 13, y + 5.2, yr, fs=19, color=ACCENT, ha="left", bold=True)
        label(ax, 13, y, title, fs=23, ha="left")
        label(ax, 13, y - 5.0, detail, fs=19, color=MUTED, ha="left")
        ys_all.append(y)
        y -= 14
        for s_title, s_detail in subs:
            ax.add_patch(plt.Circle((6, y), 1.1, color=MID, zorder=2))
            ax.plot([8, 15], [y, y], color=HAIR, lw=2, zorder=1)
            label(ax, 17, y, s_title, fs=20, ha="left")
            label(ax, 17, y - 4.6, s_detail, fs=16, color=MUTED, ha="left")
            ys_all.append(y)
            y -= 14
        y -= 3
    ax.plot([6, 6], [min(ys_all), max(ys_all)], color=HAIR, lw=4, zorder=1, solid_capstyle="round")
    save(fig, "timeline")


# ------------------------------------------------------------------ 0.5 the agenda as a timeline
def agenda():
    """10:00 to 14:00, block widths proportional to minutes; tutorial, hands-on and breaks in their own colours."""
    BREAK_C, BREAK_T = "#FEEFC3", "#B06000"
    slots = [("10:00", "10:50", "roee"), ("10:50", "11:00", "break"),
             ("11:00", "11:40", "tutorial"), ("11:40", "11:50", "break"), ("11:50", "12:15", "hands-on"),
             ("12:15", "12:45", "tutorial"), ("12:45", "13:05", "hands-on"), ("13:05", "13:15", "break"),
             ("13:15", "13:30", "tutorial"), ("13:30", "13:50", "hands-on"), ("13:50", "14:00", "break")]
    parts = [("11:00", "12:15", "Part 1", "Machine learning and CNNs"),
             ("12:15", "13:05", "Part 2", "DNA foundation\nmodels"),
             ("13:15", "13:50", "Part 3", "Shortcuts\nand controls")]
    m = lambda s: int(s[:2]) * 60 + int(s[3:])                          # noqa: E731
    x0, x1, t0, t1 = 4, 156, m("10:00"), m("14:00")
    X = lambda s: x0 + (m(s) - t0) / (t1 - t0) * (x1 - x0)              # noqa: E731
    col = {"tutorial": (ACCENT, "white"), "hands-on": (GOOD_C, "white"), "break": (BREAK_C, BREAK_T),
           "roee": (FAINT, MUTED)}
    fig, ax = canvas(16, 7.9, ylim=(-4, 75))
    y, h = 12, 36
    for a, b, kind in slots:
        fc, tc = col[kind]
        ax.add_patch(Rectangle((X(a) + 0.15, y), X(b) - X(a) - 0.3, h, fc=fc, ec="none"))
        w = X(b) - X(a)
        txt = {"roee": "Opening lecture\nProf. Roee Amit", "break": "Break", "tutorial": "Tutorial",
               "hands-on": "Hands-on"}[kind]
        if kind == "break":
            ax.text((X(a) + X(b)) / 2, y + h / 2, txt, rotation=90, ha="center", va="center", fontsize=20, color=tc)
        else:
            fs = min(23, w * 1.5)
            ax.text((X(a) + X(b)) / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=tc,
                    fontweight="bold" if kind != "roee" else "normal")
    for a, b, part, name in parts:
        ax.plot([X(a) + 0.3, X(b) - 0.3], [y + h + 2.5, y + h + 2.5], color=INK, lw=2)
        label(ax, (X(a) + X(b)) / 2, y + h + 5.5, name, fs=22, color=MUTED, va="bottom")
        label(ax, (X(a) + X(b)) / 2, y + h + 19, part, fs=27, color=INK, bold=True,
              va="bottom")
    ticks = sorted({s for a, b, _ in slots for s in (a, b)}, key=m)
    low = set()                                       # two times 10 minutes apart: the second goes one line lower
    for prev, s in zip(ticks, ticks[1:]):
        if m(s) - m(prev) <= 10 and prev not in low:
            low.add(s)
    for s in ticks:
        dy = 4.5 if s in low else 0
        ax.plot([X(s), X(s)], [y - 1.2, y - 2.6 - dy], color=MUTED, lw=1.2)
        label(ax, X(s), y - 5.5 - dy, s, fs=18, color=INK if s in ("10:00", "11:00", "14:00") else MUTED,
              bold=s in ("10:00", "11:00", "14:00"))
    save(fig, "agenda")


# ------------------------------------------------------------------ 1.6: biology in two clicks
def bio_flow(with_ago2):
    """DNA -> mRNA -> protein; on the click, an RNA-binding protein (AGO2) sitting on a site in the mRNA."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = canvas(16, 8, ylim=(0, 80))
    xs = [20, 80, 140]
    for x, n, l in zip(xs, ["Gene (DNA)", "Messenger RNA\n(mRNA)", "Protein"], ["A C G T", "A C G U", ""]):
        label(ax, x, 66, n, fs=40)
        if l:
            label(ax, x, 51, l, fs=34, color=MUTED, family=MONO)
    for (a, b), tx in zip([(37, 54), (106, 124)], ["copied", "translated"]):
        arrow(ax, a, 66, b, 66, color=MUTED, lw=3, ms=32)
        label(ax, (a + b) / 2, 73, tx, fs=28, color=MUTED)
    if with_ago2:
        rna = "UAGCUUAACAGUUUGCACAAUCGUACGGAUUCAGCU"        # a stretch of mRNA; the bound site in the middle
        x0, dx, yr = 23, 3.2, 14
        site = range(13, 20)
        # the protein: a rounded body sitting on the site
        sx0, sx1 = x0 + dx * (site[0] - 0.5), x0 + dx * (site[-1] + 0.5)
        ax.add_patch(FancyBboxPatch((sx0, yr + 3.4), sx1 - sx0, 19, boxstyle="round,pad=0,rounding_size=7",
                                    fc=ACCENT_L, ec=ACCENT, lw=2.5, zorder=2))      # the protein sits on the site
        label(ax, (sx0 + sx1) / 2, yr + 13, "AGO2", fs=30, color=ACCENT, bold=True)
        ax.plot([sx0 + 0.6, sx1 - 0.6], [yr - 3.4, yr - 3.4], color=ACCENT, lw=4, solid_capstyle="round")
        for i, ch in enumerate(rna):
            ax.text(x0 + dx * i, yr, ch, ha="center", va="center_baseline", fontsize=24, family=MONO, fontweight="bold",
                    color=BASE[ch], zorder=4)
        label(ax, (sx0 + sx1) / 2, yr - 8, "binding site", fs=24, color=ACCENT)
        label(ax, x0 - 6, yr, "mRNA", fs=24, color=MUTED, ha="right")
        label(ax, sx1 + 6, yr + 15, "an RNA-binding protein:\nit binds some mRNAs,\nat a few letters", fs=24,
              color=INK, ha="left")
        arrow(ax, 80, 45, 80, 37.6, color=MUTED, lw=2.5, ms=26)
    save_raw(fig, "bio_flow_ago2" if with_ago2 else "bio_flow_base")


# ------------------------------------------------------------------ a toy filter (4.2 animation, 5.1 worked example)
# rows A, C, G, U; columns: the 4 positions of the window. It prefers ACGU.
W_TOY = np.array([[1.0, -0.2, 0.0, -0.3],
                  [-0.2, 1.0, 0.2, 0.0],
                  [0.1, 0.0, 0.8, -0.2],
                  [0.2, 0.1, 0.0, 0.8]])
B_TOY = -1.2
TOY_SEQS = ["GGACGUAA", "CCUUAGCA", "AUCGUACC"]


def toy_out(s):
    """Per window: the picked weights, their sum + bias, and the output after ReLU."""
    res = []
    for k in range(len(s) - 3):
        picked = [W_TOY[ALPH.index(ch), j] for j, ch in enumerate(s[k:k + 4])]
        pre = round(sum(picked) + B_TOY, 2)
        res.append((picked, pre, max(0.0, pre)))
    return res


def weight_cell_color(v, vmax=1.0):
    return DIVERGING((v / vmax + 1) / 2)


def slide_filter():
    """A width-4 filter slides along GGACGUAA: one frame per position, the output row fills up."""
    s = TOY_SEQS[0]
    res = toy_out(s)
    x0, dx, cw, ch = 62, 11, 10, 8          # sequence columns
    rows_y = [64, 55, 46, 37]               # A C G U
    fx0 = 8                                 # filter columns
    for k in range(len(res)):
        fig, ax = canvas()
        # the filter
        label(ax, fx0 + 2 * 9 - 4.5, 76, "Filter: 4 × 4 weights", fs=22, color=INK)
        for r in range(4):
            label(ax, fx0 - 4, rows_y[r] + ch / 2, ALPH[r], fs=22, color=BASE[ALPH[r]], bold=True)
            for j in range(4):
                v = W_TOY[r, j]
                ax.add_patch(Rectangle((fx0 + j * 9, rows_y[r]), 8.4, ch - 0.6, fc=weight_cell_color(v), ec="white"))
                label(ax, fx0 + j * 9 + 4.2, rows_y[r] + ch / 2 - 0.3, f"{v:.1f}", fs=17,
                      color="white" if abs(v) > 0.6 else INK)
        label(ax, fx0 + 18, 30, f"bias {B_TOY:.1f}", fs=20, color=MUTED)
        # the sequence and its one-hot matrix
        for i, c in enumerate(s):
            label(ax, x0 + i * dx + cw / 2, 78, c, fs=30, color=BASE[c], family=MONO, bold=True)
            for r in range(4):
                on = ALPH[r] == c
                ax.add_patch(Rectangle((x0 + i * dx, rows_y[r]), cw, ch - 0.6, fc=BASE[c] if on else FAINT, ec="white"))
                label(ax, x0 + i * dx + cw / 2, rows_y[r] + ch / 2 - 0.3, "1" if on else "0", fs=17,
                      color="white" if on else LIGHT)
        # the window
        ax.add_patch(Rectangle((x0 + k * dx - 1.2, rows_y[-1] - 1.2), 4 * dx + 0.4, 83.5 - (rows_y[-1] - 1.2),
                               fill=False, ec=ACCENT, lw=4))
        picked, pre, out = res[k]
        for j, p in enumerate(picked):
            label(ax, x0 + (k + j) * dx + cw / 2, 31, f"{p:.1f}", fs=20, color=ACCENT)
        sign = "−" if B_TOY < 0 else "+"
        label(ax, x0 + (k + 2) * dx - 0.5, 24,
              f"sum {sum(picked):.1f} {sign} {abs(B_TOY):.1f} = {pre:.1f}" + ("" if pre > 0 else "  →  0"),
              fs=21, color=INK)
        # the output row
        label(ax, x0 - 4, 9.5, "Output", fs=22, color=INK, ha="right")
        for j in range(len(res)):
            shown = j <= k
            ax.add_patch(Rectangle((x0 + j * dx, 6), cw, 7.4, fc=ACCENT_L if j == k else FAINT, ec="white"))
            if shown:
                v = res[j][2]
                label(ax, x0 + j * dx + cw / 2, 9.7, f"{v:.1f}", fs=20, color=ACCENT if v > 0 else INK,
                      bold=v > 0)
        arrow(ax, x0 + k * dx + cw / 2, 19.5, x0 + k * dx + cw / 2, 14.2, color=ACCENT, lw=2, ms=18)
        save_raw(fig, f"slide_filter_{k}")


def toy_cutoff():
    """5.1: outputs of the toy filter on three sequences, then the cutoff and the hits (two frames)."""
    outs = [[o for _, _, o in toy_out(s)] for s in TOY_SEQS]
    allv = np.array(outs).ravel()
    mean, mx = allv.mean(), allv.max()
    cut = mean + 0.5 * (mx - mean)
    print("toy_cutoff mean", round(mean, 3), "max", mx, "cutoff", round(cut, 3))
    for frame in (0, 1):
        fig, ax = canvas()
        cx = [58, 74, 90, 106, 122]
        label(ax, 34, 80, "Sequence", fs=21, color=MUTED)
        for j, x in enumerate(cx):
            label(ax, x, 80, f"pos {j + 1}", fs=21, color=MUTED)
        for r, (s, o) in enumerate(zip(TOY_SEQS, outs)):
            y = 66 - r * 13
            label(ax, 4, y, f"seq {r + 1}", fs=21, color=MUTED, ha="left")
            for i, c in enumerate(s):
                hit_letter = frame == 1 and any(v > cut and j <= i < j + 4 for j, v in enumerate(o))
                label(ax, 21 + i * 3.7, y, c, fs=24, color=BASE[c], family=MONO, bold=True)
                if hit_letter:
                    ax.plot([21 + i * 3.7 - 1.8, 21 + i * 3.7 + 1.8], [y - 3.4, y - 3.4], color=ACCENT, lw=3)
            for j, (x, v) in enumerate(zip(cx, o)):
                hit = frame == 1 and v > cut
                if hit:
                    ax.add_patch(Rectangle((x - 6, y - 4.5), 12, 9, fc=ACCENT_L, ec="none"))
                label(ax, x, y, f"{v:.1f}", fs=24, color=ACCENT if hit else INK, bold=hit)
        ax.plot([2, 130], [74, 74], color=HAIR, lw=1.2)
        if frame == 1:
            label(ax, 66, 22, f"cutoff  =  mean + 0.5 × (max − mean)  =  {mean:.2f} + 0.5 × ({mx:.1f} − {mean:.2f})  =  {cut:.2f}",
                  fs=22, color=INK)
            hits = [(r, j) for r, o in enumerate(outs) for j, v in enumerate(o) if v > cut]
            label(ax, 66, 11, "Hits:  " + ",   ".join(f"seq {r + 1} pos {j + 1} → {TOY_SEQS[r][j:j + 4]}" for r, j in hits),
                  fs=22, color=ACCENT)
        save_raw(fig, f"toy_cutoff_{frame}")


def toy_pwm():
    """5.2: the hit windows, aligned, counted per position (PWM), drawn as a logo."""
    from intuition2 import draw_logo
    outs = [[o for _, _, o in toy_out(s)] for s in TOY_SEQS]
    allv = np.array(outs).ravel(); cut = allv.mean() + 0.5 * (allv.max() - allv.mean())
    hits = [TOY_SEQS[r][j:j + 4] for r, o in enumerate(outs) for j, v in enumerate(o) if v > cut]
    P = np.array([[sum(h[i] == b for h in hits) / len(hits) for b in ALPH] for i in range(4)])
    fig, ax = canvas(16, 7.6, ylim=(8, 84))
    label(ax, 22, 80, "Hits, aligned", fs=23, color=MUTED)
    for r, h in enumerate(hits):
        for i, c in enumerate(h):
            label(ax, 13 + i * 6, 64 - r * 10, c, fs=32, color=BASE[c], family=MONO, bold=True)
    arrow(ax, 40, 52, 52, 52, color=MUTED, lw=2, ms=24)
    label(ax, 46, 57, "count", fs=19, color=MUTED)
    label(ax, 77, 80, "PWM: letter frequency per position", fs=23, color=MUTED)
    for i in range(4):
        label(ax, 66 + i * 8, 71, str(i + 1), fs=20, color=MUTED)
    for b in range(4):
        y = 62 - b * 9
        label(ax, 58, y, ALPH[b], fs=24, color=BASE[ALPH[b]], bold=True)
        for i in range(4):
            v = P[i, b]
            label(ax, 66 + i * 8, y, "1" if v == 1 else ("0" if v == 0 else f"{v:.1f}"), fs=22,
                  color=INK if v > 0 else LIGHT, bold=v > 0)
    arrow(ax, 102, 52, 114, 52, color=MUTED, lw=2, ms=24)
    label(ax, 108, 57, "draw", fs=19, color=MUTED)
    label(ax, 136, 80, "Logo", fs=23, color=MUTED)
    lax = fig.add_axes([0.74, 0.3, 0.22, 0.56])
    draw_logo(lax, P)
    lax.set_xticks(np.arange(4) + 0.5); lax.set_xticklabels(["1", "2", "3", "4"], fontsize=18)
    for sp in ("top", "right"):
        lax.spines[sp].set_visible(False)
    label(ax, 130, 14, "letter height = frequency × information\n2 bits: one letter always; 0 bits: all four equally",
          fs=18, color=MUTED)
    save_raw(fig, "toy_pwm")


# ------------------------------------------------------------------ 2.x the learning flow, on our task
def learning_flow():
    """Six steps on a 3 x 2 grid: number centred in its circle, heading and detail aligned to it."""
    steps = [("Collect data", "inputs, each with\nits known answer"),
             ("Choose a hypothesis set", "a family of models,\nfor example one architecture"),
             ("Choose an algorithm", "for example gradient\ndescent on the loss"),
             ("Train", "the algorithm picks the\nbest model in the set"),
             ("Evaluate", "on held-out data\nthe model never saw"),
             ("Use", "predict for new inputs\nnobody has measured")]
    fig, ax = canvas(16, 7.6, ylim=(0, 76))
    col_x, row_y, r = [3, 56, 109], [62, 24], 5.2
    for k, (head, sub) in enumerate(steps):
        x, y = col_x[k % 3], row_y[k // 3]
        ax.add_patch(plt.Circle((x + r, y), r, color=ACCENT if k < 5 else GOOD_C))
        ax.text(x + r, y, str(k + 1), ha="center", va="center_baseline", fontsize=30, color="white", fontweight="bold")
        ax.text(x + 2 * r + 2.5, y, head, ha="left", va="center_baseline", fontsize=23, color=INK, fontweight="bold")
        ax.text(x + 2 * r + 2.5, y - 6.5, sub, ha="left", va="top", fontsize=21, color=MUTED, linespacing=1.3)
    save(fig, "learning_flow")


# ------------------------------------------------------------------ 3.x testing
def data_split():
    """3.1: the three roles of the data, as one bar (not to scale: the test set is 1% of the data)."""
    fig, ax = canvas(16, 6, ylim=(0, 60))
    parts = [(4, 92, TRAIN_C, "Train", "fit the weights", "92,346 sequences"),
             (98, 26, MID, "Validation", "choose settings,\nsuch as the number of epochs", ""),
             (126, 30, HELD_C, "Test", "report once", "1,000 held-out")]
    for x, w, c, name, what, n in parts:
        ax.add_patch(Rectangle((x, 30), w - 1.5, 10, fc=c, ec="none"))
        label(ax, x + (w - 1.5) / 2, 35, name, fs=26, color="white", bold=True)
        label(ax, x + (w - 1.5) / 2, 26, what, fs=22, color=INK, va="top")
        if n:
            label(ax, x + (w - 1.5) / 2, 46, n, fs=22, color=c, bold=True)
    save(fig, "data_split")


def overfit_curve(tr, ty, ls, ly):
    """A bigger CNN trained on 1,000 sequences: training loss keeps falling, held-out loss turns up."""
    import torch
    from torch import nn
    from intuition import DEV, onehot
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(tr), 1000, replace=False)
    Xtr, ytr = onehot([tr[i] for i in idx]).to(DEV), torch.tensor(ty[idx], dtype=torch.float32, device=DEV)
    Xte, yte = onehot(ls).to(DEV), torch.tensor(ly, dtype=torch.float32, device=DEV)
    model = nn.Sequential(nn.Conv2d(1, 64, (4, 10)), nn.ReLU(), nn.MaxPool2d((1, 3)),
                          nn.Conv2d(64, 64, (1, 5)), nn.ReLU(), nn.MaxPool2d((1, 3)),
                          nn.Flatten(), nn.Linear(64 * 53, 64), nn.ReLU(), nn.Linear(64, 1)).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    crit = nn.BCEWithLogitsLoss()
    Ltr, Lte = [], []
    for ep in range(81):
        model.eval()
        with torch.no_grad():
            Ltr.append(crit(model(Xtr).squeeze(1), ytr).item()); Lte.append(crit(model(Xte).squeeze(1), yte).item())
        model.train(); perm = torch.randperm(len(Xtr), device=DEV)
        for i in range(0, len(Xtr), 64):
            bi = perm[i:i + 64]
            opt.zero_grad(); crit(model(Xtr[bi]).squeeze(1), ytr[bi]).backward(); opt.step()
    best = int(np.argmin(Lte))
    print("overfit_curve best epoch", best, "held-out", round(Lte[best], 3), "end train", round(Ltr[-1], 3),
          "end held-out", round(Lte[-1], 3))
    fig = newfig()
    ax = fig.add_axes([0.08, 0.13, 0.74, 0.74])
    ax.plot(Ltr, color=TRAIN_C, lw=4); ax.plot(Lte, color=HELD_C, lw=4)
    ax.text(len(Ltr) - 0.5, Ltr[-1], "  training", color=TRAIN_C, fontsize=26, va="center")
    ax.text(len(Lte) - 0.5, Lte[-1], "  held-out", color=HELD_C, fontsize=26, va="center")
    ax.axvline(best, color=INK, lw=1.5, ls="--")
    ax.text(best + 0.8, max(Lte) * 0.98, "lowest held-out loss:\nstop here", fontsize=21, color=INK, va="top")
    top = max(max(Ltr), max(Lte)) * 1.05
    ax.text(best / 2, top, "underfitting", fontsize=24, color=MUTED, ha="center")
    ax.text((best + len(Ltr)) / 2, top, "overfitting", fontsize=24, color=MUTED, ha="center")
    ax.set_ylim(0, top * 1.08); ax.set_xlim(0, len(Ltr) - 1)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ygrid(ax)
    save(fig, "overfit_curve")


def roc_frames(ls, ly):
    """A threshold moves down the CNN's score histogram; each threshold adds one point to the ROC curve."""
    from sklearn.metrics import roc_auc_score, roc_curve
    from intuition import cnn_scores, load_cnn
    s = cnn_scores(load_cnn(), ls)
    th = [0.8, 0.5, 0.2]
    pts = [((s[ly == 0] >= t).mean(), (s[ly == 1] >= t).mean()) for t in th]
    fpr, tpr, _ = roc_curve(ly, s); auc = roc_auc_score(ly, s)
    print("roc_frames", [(t, round(f, 2), round(p, 2)) for t, (f, p) in zip(th, pts)], "AUROC", round(auc, 3))
    bins = np.linspace(0, 1, 30)
    for k in range(len(th) + 1):
        fig = newfig()
        a1 = fig.add_axes([0.06, 0.14, 0.5, 0.66]); a2 = fig.add_axes([0.64, 0.14, 0.33, 0.66])
        a1.hist(s[ly == 0], bins, color=UNBOUND, alpha=.9, label="Unbound")
        a1.hist(s[ly == 1], bins, color=BOUND, alpha=.6, label="Bound")
        a1.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.12))
        a1.set_xlabel("CNN score"); a1.set_ylabel("Sequences"); ygrid(a1)
        a1.set_ylim(0, a1.get_ylim()[1] * 1.45)
        a2.plot([0, 1], [0, 1], color=HAIR, lw=1.5, ls="--")
        if k < len(th):
            t = th[k]; f, p = pts[k]
            a1.axvspan(t, 1, color=HELD_C, alpha=0.08)
            a1.axvline(t, color=HELD_C, lw=3)
            a1.text(t, 1.02, f"threshold {t}", transform=a1.get_xaxis_transform(), color=HELD_C, fontsize=21,
                    ha="center", va="bottom")
            a1.text(0.01, 0.97, f"bound above: {p:.0%}  (true positive rate)\nunbound above: {f:.0%}  (false positive rate)",
                    transform=a1.transAxes, ha="left", va="top", fontsize=19, color=INK)
            for j in range(k + 1):
                a2.scatter(*pts[j], s=140 if j == k else 70, color=HELD_C if j == k else MUTED, zorder=3)
            a2.annotate(f"threshold {t}", pts[k], xytext=(14, -26), textcoords="offset points", fontsize=18, color=HELD_C)
        else:
            a2.plot(fpr, tpr, color=ACCENT, lw=3.5); a2.fill_between(fpr, tpr, color=ACCENT, alpha=.1)
            for j in range(len(th)):
                a2.scatter(*pts[j], s=70, color=MUTED, zorder=3)
            a2.text(0.42, 0.2, f"AUROC {auc:.2f}", fontsize=30, color=ACCENT)
        a2.set_xlabel("False positive rate"); a2.set_ylabel("True positive rate")
        a2.set_xlim(0, 1); a2.set_ylim(0, 1.01)
        if k == len(th):
            fig.savefig(OUT / "roc_auroc.png", facecolor="white")      # the AUROC slide, its own shape
            trim(OUT / "roc_auroc.png")
        save_raw(fig, f"roc_frame_{k}")


# ------------------------------------------------------------------ 4.7 the training loop as a picture
def train_loop():
    """One training step and the evaluation: flat neutral blocks, one weight of arrow, colour only for meaning."""
    fig, ax = canvas(16, 8, ylim=(0, 80))
    BLK, SUB = "#EEF0F2", MUTED

    def node(x, y, w, h, title, sub):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8", fc=BLK, ec="none"))
        label(ax, x + 2.2, y + h * 0.62, title, fs=22, color=INK, bold=True, ha="left")
        label(ax, x + 2.2, y + h * 0.28, sub, fs=17, color=SUB, ha="left")

    def row(y, items, color):
        for k, (ti, su) in enumerate(items):
            node(4 + k * 39, y, 32, 13, ti, su)
            if k < 3:
                arrow(ax, 36.6 + k * 39, y + 6.5, 42.4 + k * 39, y + 6.5, color=color, lw=2, ms=18)

    label(ax, 4, 76, "TRAINING", fs=17, color=INK, ha="left", bold=True)
    label(ax, 18, 76, "every batch, every epoch", fs=17, color=MUTED, ha="left")
    y = 55
    row(y, [("Batch", "sequences X"), ("Network", "current weights"), ("Scores", "one per sequence"),
            ("Loss", "scores vs labels y")], INK)
    label(ax, 4, y + 16.5, "forward pass", fs=16, color=INK, ha="left")
    yb = y - 6                                          # backward pass: under the blocks, back to the network
    ax.plot([137, 137, 59], [y, yb, yb], color=ACCENT, lw=2, solid_capstyle="round")
    arrow(ax, 59, yb, 59, y - 0.3, color=ACCENT, lw=2, ms=18)
    label(ax, 64, yb - 3.6, "backward pass (backpropagation): gradient of the loss for every weight", fs=16,
          color=ACCENT, ha="left")
    label(ax, 64, yb - 8.4, "optimizer step: move every weight against its gradient", fs=16, color=INK, ha="left")
    label(ax, 4, 31, "EVALUATION", fs=17, color=INK, ha="left", bold=True)
    label(ax, 21, 31, "after every epoch, forward pass only, no learning", fs=17, color=HELD_C, ha="left")
    row(10, [("Held-out", "sequences"), ("Network", "weights fixed"), ("Sigmoid", "probabilities"),
             ("AUROC", "vs the true labels")], INK)
    save(fig, "train_loop")


# ------------------------------------------------------------------ 5.x your turn: what each TODO produces
def your_turn(ls, ly):
    import matplotlib.image as mpimg
    from sklearn.metrics import roc_curve
    from intuition import cnn_scores, load_cnn
    fig = plt.figure(figsize=(16, 6.2), dpi=120)
    w, gap = 0.172, 0.03
    axs = [fig.add_axes([0.02 + k * (w + gap), 0.27, w, 0.62]) for k in range(5)]
    # 1: one-hot
    a = axs[0]; s = "GGACGU"
    for i, c in enumerate(s):
        a.text(i + 0.5, 4.5, c, ha="center", va="center", fontsize=24, color=BASE[c], family=MONO, fontweight="bold")
        for r, b in enumerate(ALPH):
            a.add_patch(Rectangle((i + 0.06, 3 - r + 0.06), 0.88, 0.88, fc=BASE[c] if b == c else FAINT, ec="none"))
    a.set_xlim(0, 6); a.set_ylim(0, 5.1)
    # 2: layers getting shorter, then one number
    a = axs[1]
    for k, (h, c) in enumerate([(4, LIGHT), (3.4, ACCENT), (2.4, ACCENT_L), (1.6, ACCENT), (1.0, ACCENT_L)]):
        a.add_patch(Rectangle((k * 1.05, 2.5 - h / 2), 0.8, h, fc=c, ec="none"))
    a.scatter([5.9], [2.5], s=500, color=INK)
    a.set_xlim(-0.2, 6.4); a.set_ylim(0, 5.1)
    # 3: ROC curve of our CNN
    a = axs[2]; fpr, tpr, _ = roc_curve(ly, cnn_scores(load_cnn(), ls))
    a.plot([0, 1], [0, 1], color=HAIR, ls="--", lw=1.5); a.plot(fpr, tpr, color=ACCENT, lw=3.5)
    a.fill_between(fpr, tpr, color=ACCENT, alpha=.1); a.set_xlim(0, 1); a.set_ylim(0, 1.02)
    a.set_xlabel("False positive rate", fontsize=16); a.set_ylabel("True positive rate", fontsize=16)
    a.set_xticks([]); a.set_yticks([])
    # 4: the logo of filter 0 (from filter_logo.png)
    a = axs[3]; im = mpimg.imread(OUT / "filter_logo.png"); h, w_ = im.shape[:2]
    a.imshow(im[int(0.12 * h):int(0.9 * h), int(0.555 * w_):]); a.set_anchor("C")
    # 5: copies of a motif written into a random sequence
    a = axs[4]
    a.plot([0, 10], [2.5, 2.5], color=HAIR, lw=10, solid_capstyle="butt")
    for x in (1.2, 4.4, 7.6):
        a.add_patch(Rectangle((x, 2.05), 1.2, 0.9, fc=ACCENT, ec="none"))
    a.set_xlim(-0.2, 10.2); a.set_ylim(0, 5.1)
    for k, a in enumerate(axs):
        if k != 2:
            a.axis("off")
    for k, (todo, what) in enumerate([("TODO-1", "encode the letters"), ("TODO-2", "build a network"),
                                      ("TODO-3", "train and test it"), ("TODO-4", "read its filters"),
                                      ("TODO-5", "test a motif")]):
        x = 0.02 + k * (w + gap) + w / 2
        fig.text(x, 0.13, todo, ha="center", fontsize=26, color=ACCENT, fontweight="bold")
        fig.text(x, 0.04, what, ha="center", fontsize=21, color=INK)
    save(fig, "your_turn")


# ------------------------------------------------------------------ 2.x features: chosen by us, or learned
def features(tr, ty):
    """One real bound sequence: two numbers we chose (classic ML), or one-hot letters and learned filters (deep learning)."""
    from intuition import load_cnn
    s = next(x for x, l in zip(tr, ty) if l == 1 and len(x) > 300)
    gcf = (s.count("G") + s.count("C")) / len(s); uf = s.count("U") / len(s)
    W = load_cnn().net[0].weight.detach().cpu().numpy()[:3, 0]          # three real first-layer filters
    frames_ = []
    for k in (0, 1):
        fig, ax = canvas()
        for row, y in ((0, 66), (1, 24)):
            if row > k:
                continue
            seq(ax, 4, y, s[:18], fs=19, dx=2.3)
            label(ax, 4 + 18 * 2.3, y, "…", fs=20, color=MUTED, ha="left")
            arrow(ax, 50, y, 58, y, color=MUTED, lw=2, ms=20)
        label(ax, 4, 82, "Classic machine learning: we choose the features", fs=23, color=INK, ha="left", bold=True)
        box(ax, 60, 70, 30, 8, f"G+C fraction  {gcf:.2f}", fc=ACCENT_L, fs=19)
        box(ax, 60, 56, 30, 8, f"U fraction  {uf:.2f}", fc=ACCENT_L, fs=19)
        label(ax, 75, 50, "2 numbers, chosen by us", fs=17, color=MUTED)
        arrow(ax, 92, 66, 100, 66, color=MUTED, lw=2, ms=20)
        label(ax, 116, 66, "logistic\nregression", fs=20, color=INK)
        arrow(ax, 132, 66, 140, 66, color=MUTED, lw=2, ms=20)
        label(ax, 147, 66, r"$\hat p$", fs=26, color=INK)
        if k == 1:
            label(ax, 4, 40, "Deep learning: the model learns its features from the letters", fs=23, color=INK,
                  ha="left", bold=True)
            for i, ch in enumerate(s[:8]):                                  # one-hot letters
                for r, b in enumerate("ACGU"):
                    ax.add_patch(Rectangle((60 + i * 2.6, 28 - r * 2.6), 2.4, 2.4, fc=BASE[ch] if b == ch else FAINT,
                                           ec="none"))
            label(ax, 71, 15, "one-hot letters", fs=17, color=MUTED)
            arrow(ax, 82, 24, 90, 24, color=MUTED, lw=2, ms=20)
            for j in range(3):                                              # real learned filters
                v = np.abs(W[j]).max()
                for r in range(4):
                    for c in range(10):
                        ax.add_patch(Rectangle((92 + c * 1.2, 30 - j * 6 - r * 1.2), 1.1, 1.1,
                                               fc=DIVERGING((W[j, r, c] / v + 1) / 2), ec="none"))
            label(ax, 98, 11, "filters learned\nin training", fs=17, color=MUTED)
            arrow(ax, 106, 24, 114, 24, color=MUTED, lw=2, ms=20)
            label(ax, 123, 24, "network", fs=20, color=INK)
            arrow(ax, 132, 24, 140, 24, color=MUTED, lw=2, ms=20)
            label(ax, 147, 24, r"$\hat p$", fs=26, color=INK)
        save_raw(fig, f"features_{k}")
        frames_.append(f"features_{k}")
    trim_group(frames_)
    print("features G+C", round(gcf, 3), "U", round(uf, 3))


def activation(kind):
    """The sigmoid (score -> probability) or ReLU (negative -> 0), each as one plot."""
    z = np.linspace(-6, 6, 400)
    fig = plt.figure(figsize=(16, 7.2), dpi=120)
    ax = fig.add_axes([0.1, 0.15, 0.55, 0.75])
    if kind == "sigmoid":
        ax.plot(z, 1 / (1 + np.exp(-z)), color=ACCENT, lw=5)
        ax.axhline(0.5, color=HAIR, ls="--", lw=1.5); ax.axhline(1, color=HAIR, lw=1); ax.axhline(0, color=HAIR, lw=1)
        for zz in (-3, 0, 3):
            pz = 1 / (1 + np.exp(-zz))
            ax.scatter(zz, pz, s=120, color=HELD_C, zorder=3)
            ax.annotate(f"z = {zz} → {pz:.2f}", (zz, pz), xytext=(14, -26 if zz > 0 else 10), textcoords="offset points",
                        fontsize=21, color=HELD_C)
        ax.set_ylim(-0.05, 1.08); ax.set_ylabel("probability of bound")
        txt = [r"$\sigma(z) = \dfrac{1}{1 + e^{-z}}$", "", "any score in,", "a number between 0 and 1 out", "",
               "score 0 → 0.5", "large positive → close to 1", "large negative → close to 0"]
    else:
        ax.plot(z, np.maximum(0, z), color=ACCENT, lw=5)
        ax.axhline(0, color=HAIR, lw=1)
        for zz, off in ((-3, (0, 16)), (2, (18, -6))):
            ax.scatter(zz, max(0, zz), s=120, color=HELD_C, zorder=3)
            ax.annotate(f"{zz} → {max(0, zz)}", (zz, max(0, zz)), xytext=off, textcoords="offset points",
                        fontsize=21, color=HELD_C, ha="left" if zz > 0 else "center")
        ax.set_ylim(-1, 6.3); ax.set_ylabel("output")
        txt = [r"$\mathrm{ReLU}(z) = \max(0, z)$", "", "negative → 0", "positive → passes unchanged", "",
               "introduces non-linearity"]
    ax.set_xlabel("score z"); ygrid(ax)
    for i, line in enumerate(txt):
        fig.text(0.7, 0.85 - i * 0.075, line, fontsize=26 if i == 0 else 21, color=INK if i == 0 else MUTED, va="top")
    save(fig, kind)


if __name__ == "__main__":
    want = set(sys.argv[1:])
    run = (lambda n: not want or n in want)
    if run("agenda"):
        agenda()
    if run("sigmoid"):
        activation("sigmoid")
    if run("relu"):
        activation("relu")
    if run("features"):
        from intuition import split
        features(*split("train"))
    if run("learning_flow"):
        learning_flow()
    if run("data_split"):
        data_split()
    if run("timeline"):
        timeline()
    if run("bio_flow"):
        bio_flow(False); bio_flow(True)
    if run("slide_filter"):
        slide_filter()
    if run("toy_cutoff"):
        toy_cutoff()
    if run("toy_pwm"):
        toy_pwm()
    if run("train_loop"):
        train_loop()
    if run("overfit_curve") or run("roc_frames") or run("your_turn"):
        from intuition import split
        tr, ty = split("train"); ls, ly = split("ls")
        if run("overfit_curve"):
            overfit_curve(tr, ty, ls, ly)
        if run("roc_frames"):
            roc_frames(ls, ly)
        if run("your_turn"):
            your_turn(ls, ly)

    # crop every click-through group to the union of its ink, so frames fill the slide and still stack exactly
    TRIM_GROUPS = [[f"slide_filter_{k}" for k in range(5)], ["toy_cutoff_0", "toy_cutoff_1"],
                   [f"roc_frame_{k}" for k in range(4)], ["bio_flow_base", "bio_flow_ago2"]]
    for g in TRIM_GROUPS:
        if all((OUT / f"{n}.png").exists() for n in g):
            trim_group(g)
