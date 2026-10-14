"""Figures added on 14.10 for the part 3 deck (LoRA step, your turn, the day in numbers).

    python figs_d3.py [name ...]      (hw1-00660121 env; writes ../assets/custom/<name>.png)
"""
import sys

import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

from deckstyle import *  # noqa: F401,F403


def block(ax, x, y, w, h, text, fc=ACCENT, tc="white", fs=22, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6", fc=fc, ec="none"))
    label(ax, x + w / 2, y + h / 2, text, fs=fs, color=tc, bold=bold)


# ------------------------------------------------------------------ 3.3 the LoRA training step
def lora_loop():
    fig, ax = canvas(16, 7.6, ylim=(0, 76))
    P = TUNED_C
    steps = [(16, 58, "a batch of\nDNA strings"), (120, 58, "loss\nagainst the labels"),
             (120, 22, "gradients: only\nfor B, A and the head"), (48, 22, "step: update B, A\nand the head")]
    for x, y, t in steps:
        label(ax, x, y, t, fs=22, color=INK)
    block(ax, 36, 48, 26, 20, "", fc=LIGHT)
    label(ax, 49, 62, "Mistral-DNA", fs=21, color=INK, bold=True)
    label(ax, 49, 55.5, "W frozen", fs=19, color=MUTED)
    block(ax, 64, 50, 9, 16, "BA", fc=P, fs=19)
    label(ax, 88, 58, "scores", fs=22, color=INK)
    for a, b in ((27, 35), (74, 82), (94, 104)):
        arrow(ax, a, 58, b, 58, color=P, lw=2.5, ms=24)
    arrow(ax, 120, 51, 120, 29, color=P, lw=2.5, ms=24)
    arrow(ax, 104, 22, 63, 22, color=P, lw=2.5, ms=24)
    arrow(ax, 48, 29, 48, 46, color=P, lw=2.5, ms=24)
    label(ax, 6, 6, "Trained: the LoRA matrices B and A and the new head.  W stays as pretrained.", fs=21, color=P,
          ha="left")
    save(fig, "lora_loop")


# ------------------------------------------------------------------ 4.1 your turn
def your_turn_d3():
    fig, ax = canvas(16, 6.4, ylim=(0, 64))
    xs = [30, 80, 130]
    # 1: two numbers per sequence
    for i, (v, c) in enumerate(((0.41, BASE["G"]), (0.29, BASE["U"]))):
        ax.add_patch(Rectangle((xs[0] - 9 + i * 10, 30), 7, v * 50, color=c))
    label(ax, xs[0] - 5.5, 27, "GC", fs=16, color=MUTED); label(ax, xs[0] + 4.5, 27, "U", fs=16, color=MUTED)
    # 2: word counts
    rng = np.random.default_rng(2)
    for i, h in enumerate(rng.integers(1, 7, 24)):
        ax.add_patch(Rectangle((xs[1] - 13 + i * 1.1, 30), 0.8, h * 3, color=ACCENT))
    # 3: original and shuffled
    from intuition import dinuc_shuffle
    orig = "UGCUGCUAACUG"
    shuf = next(x for x in (dinuc_shuffle(orig, np.random.default_rng(k)) for k in range(100)) if x != orig)
    pairs = lambda x: sorted(x[i:i + 2] for i in range(len(x) - 1))  # noqa: E731
    assert pairs(shuf) == pairs(orig)
    for r, s in enumerate((orig, shuf)):
        seq(ax, xs[2] - 11, 48 - r * 10, s, fs=20, dx=2.4)
    label(ax, xs[2], 31, "same letter pairs", fs=16, color=MUTED)
    for x, todo, what in zip(xs, ["TODO-1", "TODO-2", "TODO-3"],
                             ["two numbers per sequence:\nGC and U", "counts of all\n5-letter words",
                              "the shuffle test"]):
        label(ax, x, 18, todo, fs=26, color=ACCENT, bold=True)
        label(ax, x, 11, what, fs=19, color=INK, va="top")
    save(fig, "your_turn_d3")


# ------------------------------------------------------------------ 6.1 the day in numbers
def day_numbers():
    """Held-out AUROC of every model of the day; arrows: the same model before and after a control."""
    rows = [  # part, label, (from, to) or value, colour
        ("1", "CNN, 92,346 labels", (0.86,), CNN_C),
        ("2", "Frozen pretrained models + probe", (0.78, 0.79), FROZEN_C),
        ("3", "Two composition numbers: GC and U", (0.66,), MID),
        ("3", "Dinucleotide shuffle: CNN", (0.86, 0.66), CNN_C),
        ("3", "Dinucleotide shuffle: frozen probe", (0.78, 0.77), FROZEN_C),
        ("3", "Composition-matched data: GC and U", (0.64, 0.51), MID),
        ("3", "Composition-matched data: CNN", (0.84, 0.84), CNN_C)]
    fig = newfig()
    ax = fig.add_axes([0.42, 0.1, 0.54, 0.8])
    for i, (part, name, v, c) in enumerate(rows):
        y = len(rows) - 1 - i
        if len(v) == 1:
            ax.scatter(v[0], y, s=260, color=c, zorder=3)
            ax.text(v[0] + 0.012, y, f"{v[0]:.2f}", va="center", fontsize=20, color=c)
        elif name.startswith("Frozen pretrained"):
            ax.plot(v, [y, y], color=c, lw=6, solid_capstyle="round")
            ax.text(v[1] + 0.012, y, f"{v[0]:.2f} to {v[1]:.2f}", va="center", fontsize=20, color=c)
        else:
            ax.scatter(v[0], y, s=200, facecolor="white", edgecolor=c, lw=2.5, zorder=3)
            ax.scatter(v[1], y, s=260, color=c, zorder=4)
            if abs(v[1] - v[0]) > 0.015:
                ax.annotate("", (v[1], y), (v[0], y), arrowprops=dict(arrowstyle="-|>", color=c, lw=2.5,
                                                                        mutation_scale=20, shrinkA=8, shrinkB=9))
            ax.text(max(v) + 0.012, y, f"{v[0]:.2f} → {v[1]:.2f}", va="center", fontsize=20, color=c)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[1] for r in rows][::-1], fontsize=20)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlim(0.48, 0.95); ax.axvline(0.5, color=HAIR, ls="--", lw=1.5)
    ax.set_xlabel("Held-out AUROC (0.5 = chance)"); xgrid(ax)
    save(fig, "day_numbers")


if __name__ == "__main__":
    want = set(sys.argv[1:])
    for n, job in [("lora_loop", lora_loop), ("your_turn_d3", your_turn_d3), ("day_numbers", day_numbers)]:
        if not want or n in want:
            job()
