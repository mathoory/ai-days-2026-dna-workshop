"""Figures for 02_build_cnn and 03_look_inside (Mattan's hands-on slots)."""
import numpy as np
from matplotlib.patches import Rectangle

from deckstyle import *  # noqa: F401,F403


def you_are_here(name, active):
    fig, ax = canvas(16, 3, ylim=(70, 90))
    pipeline(ax, y=74, active=active, fs=19, h=11)
    save(fig, name)


def onehot_big():
    fig, ax = canvas()
    s = "GGACAGAAGGGUAUAG"
    x0, dx = 22, 7.6
    for i, ch in enumerate(s):
        label(ax, x0 + i * dx, 76, ch, fs=38, color=BASE[ch], bold=True)
        for r, b in enumerate("ACGU"):
            on = ch == b
            ax.add_patch(Rectangle((x0 + i * dx - 3.5, 56 - r * 9), 7, 8.4, fc=BASE[b] if on else "white",
                                   ec="#bdbdbd", lw=1.2))
            label(ax, x0 + i * dx, 60.2 - r * 9, "1" if on else "0", fs=22, color="white" if on else "#9e9e9e",
                  bold=on)
    for r, b in enumerate("ACGU"):
        label(ax, x0 - 9, 60.2 - r * 9, b, fs=28, color=BASE[b], bold=True)
    label(ax, 80, 12, "one column per letter · a 4 × L matrix · padding = 0.25", fs=26)
    save(fig, "onehot_big")


def conv_slide():
    fig, ax = canvas()
    s = "AUGCUUUGCACUAAUGGA"
    x0, dx = 16, 7.2
    motif_at = 4  # UUUGCAC... window start
    for i, ch in enumerate(s):
        label(ax, x0 + i * dx, 84, ch, fs=28, color=BASE[ch], bold=True)
        for r, b in enumerate("ACGU"):
            ax.add_patch(Rectangle((x0 + i * dx - 3.3, 70 - r * 6), 6.6, 5.6,
                                   fc=BASE[b] if ch == b else "white", ec="#cfcfcf", lw=1, alpha=.85))
    # filter window
    ax.add_patch(Rectangle((x0 + motif_at * dx - 3.8, 51.4), 7 * dx + 0.6, 24.6, fc="none", ec=RED, lw=4))
    label(ax, x0 + (motif_at + 3) * dx, 46.5, "4 × 7 filter: one dot product per position", fs=22, color=RED, bold=True)
    # activation track
    act = np.array([0.1, 0.2, 0.15, 0.3, 2.6, 0.7, 0.2, 0.1, 0.3, 0.2, 0.1, 0.2])
    xs = x0 + np.arange(len(act)) * dx + 3 * dx
    ax.bar(xs, act * 9, width=dx * 0.7, bottom=14, color=[RED if a > 2 else "#bdbdbd" for a in act])
    ax.plot([x0 - 4, x0 + 18 * dx], [14, 14], color=NAVY, lw=2)
    label(ax, x0 + 4 * dx + 3 * dx, 41, "match!", fs=24, color=RED, bold=True)
    label(ax, 150, 22, "filter\noutput", fs=22, color=GREY)
    save(fig, "conv_slide")


def network():
    fig, ax = canvas()
    blocks = [("one-hot\n4 × 501", 6, 22, 60, GREEN_L, GREEN), ("10 motif\ndetectors", 34, 18, 48, BLUE_L, BLUE),
              ("keep the\nstrongest", 58, 14, 36, BLUE_L, BLUE), ("20 motif\ncombinations", 78, 18, 30, BLUE_L, BLUE),
              ("keep the\nstrongest", 102, 14, 20, BLUE_L, BLUE), ("one\nscore", 124, 14, 12, GOLD_L, GOLD)]
    for i, (t, x, w, h, fc, ec) in enumerate(blocks):
        box(ax, x, 45 - h / 2, w, h, "", fc=fc, ec=ec, lw=3, r=1)
        label(ax, x + w / 2, 45 - h / 2 - 8, t, fs=21)
        if i < len(blocks) - 1:
            arrow(ax, x + w + 0.5, 45, blocks[i + 1][1] - 0.5, 45)
    label(ax, 150, 45, "→ P(bound)", fs=24, bold=True)
    label(ax, 80, 84, "Conv → Pool → Conv → Pool → Linear", fs=28, bold=True, color=BLUE)
    save(fig, "network")


def learning_curve():
    fig = plt.figure(figsize=(16, 9), dpi=120)
    ax = fig.add_axes([0.1, 0.16, 0.85, 0.7])
    auc = [0.7891, 0.8244, 0.8375, 0.8481, 0.852, 0.8544, 0.8539, 0.8584, 0.8602, 0.8609]
    ax.plot(range(1, 11), auc, "o-", color=BLUE, lw=4, ms=12)
    ax.axhline(0.5, ls="--", color=GREY, lw=2)
    ax.text(9.9, 0.52, "coin flip", ha="right", fontsize=22, color=GREY)
    ax.text(10, 0.875, "0.86", ha="center", fontsize=28, color=BLUE, fontweight="bold")
    ax.set_ylim(0.45, 0.95); ax.set_xticks(range(1, 11))
    ax.set_xlabel("epoch (one pass over 92k sequences, ~4 s on a T4)", fontsize=24)
    ax.set_ylabel("held-out AUROC", fontsize=24); ax.tick_params(labelsize=20); ax.grid(alpha=.3)
    save(fig, "learning_curve")


def todo_card(name, rows, title):
    fig, ax = canvas()
    label(ax, 80, 84, title, fs=30, bold=True)
    for i, (todo, what, mins) in enumerate(rows):
        y = 64 - i * 15
        box(ax, 10, y, 22, 11, todo, fc=NAVY, ec=NAVY, tc="white", fs=22, bold=True)
        box(ax, 34, y, 96, 11, what, fc=GREY_L, ec="#bdbdbd", fs=22)
        box(ax, 132, y, 18, 11, mins, fc=GOLD_L, ec=GOLD, fs=22)
    save(fig, name)


def look_vs_test():
    fig, ax = canvas()
    box(ax, 4, 46, 72, 36, "", fc=BLUE_L, ec=BLUE, lw=3)
    label(ax, 40, 76, "LOOK", fs=30, bold=True, color=BLUE)
    for i, t in enumerate(["filter", "windows it\nfires on", "logo"]):
        box(ax, 8 + i * 23, 52, 19, 16, t, fc="white", ec=BLUE, fs=19)
        if i < 2:
            arrow(ax, 27.5 + i * 23, 60, 30.5 + i * 23, 60, color=BLUE)
    label(ax, 40, 40, "→ a hypothesis", fs=24, color=BLUE, bold=True)
    box(ax, 84, 46, 72, 36, "", fc=GREEN_L, ec=GREEN, lw=3)
    label(ax, 120, 76, "TEST", fs=30, bold=True, color=GREEN)
    seq(ax, 92, 62, "ACGUAG", fs=26, dx=3.0)
    label(ax, 111.5, 62, "+", fs=26, bold=True)
    box(ax, 115, 57, 18, 10, "motif", fc=GOLD_L, ec=GOLD, fs=21, bold=True)
    label(ax, 137.5, 62, "+", fs=26, bold=True)
    seq(ax, 141, 62, "GGUC", fs=26, dx=3.0)
    label(ax, 120, 51, "plant it into random RNA · score rises?", fs=19)
    label(ax, 120, 40, "→ evidence", fs=24, color=GREEN, bold=True)
    label(ax, 80, 18, "control: plant the same letters shuffled", fs=26, color=RED, bold=True)
    label(ax, 80, 8, "pattern beats shuffled → the model reads the pattern", fs=22, color=GREY)
    save(fig, "look_vs_test")


def seed_pairing():
    fig, ax = canvas()
    mrna = "AUCAAUUGCUGCUAAGU"
    site = (5, 12)  # UGCUGCU
    x0, dx = 22, 6.4
    label(ax, x0 - 9, 62, "mRNA 5′", fs=22, ha="right")
    for i, ch in enumerate(mrna):
        label(ax, x0 + i * dx, 62, ch, fs=30, color=BASE[ch], bold=True)
    label(ax, x0 + len(mrna) * dx + 2, 62, "3′", fs=22, ha="left")
    mir_rev = "GCGGUUAUAAAUGCACGACGAU"  # miR-16-5p read 3'->5'
    # align seed (miR nt 2-8 = AGCAGCA; reversed ACGACGA) under UGCUGCU
    seed_rev_start = len(mir_rev) - 8  # index of nt 8 in reversed string
    off = site[0] - seed_rev_start
    for j, ch in enumerate(mir_rev):
        x = x0 + (j + off) * dx
        if x < 2 or x > 158:
            continue
        in_seed = seed_rev_start <= j < seed_rev_start + 7
        label(ax, x, 38, ch, fs=30, color=PURPLE if in_seed else "#9e9e9e", bold=in_seed)
        if in_seed:
            ax.plot([x, x], [43, 57], color=PURPLE, lw=3)
    label(ax, x0 + (len(mir_rev) - 1 + off) * dx + 6, 38, "5′  microRNA (miR-16)", fs=22, color=PURPLE, ha="left")
    box(ax, x0 + (site[0]) * dx - 4, 31, 7 * dx + 1.6, 38, "", fc="none", ec=PURPLE, lw=3, ls="--")
    label(ax, 80, 84, "AGO2 carries a microRNA; its 7-letter seed finds matching mRNA", fs=26, bold=True)
    label(ax, 80, 16, "target site UGCUGCU = what the model should learn", fs=24, color=PURPLE)
    save(fig, "seed_pairing")


def planting_bars():
    fig = plt.figure(figsize=(16, 9), dpi=120)
    ax = fig.add_axes([0.1, 0.14, 0.85, 0.72])
    names = ["miR-15/16\nUGCUGCU", "miR-19\nUUUGCAC", "miR-30\nUGUUUAC", "let-7\nCUACCUC"]
    site = [0.525, 0.425, 0.372, 0.386]; shuf = [0.307, 0.340, 0.383, 0.379]
    x = np.arange(4)
    ax.bar(x - 0.2, site, 0.38, color=PURPLE, label="site planted (6 copies)")
    ax.bar(x + 0.2, shuf, 0.38, color="#bdbdbd", label="same letters, shuffled")
    ax.axhline(0.292, ls="--", color=NAVY, lw=2)
    ax.text(3.62, 0.272, "nothing planted", ha="right", fontsize=20, color=NAVY)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=22)
    ax.set_ylabel("CNN binding score\non real unbound RNA", fontsize=22); ax.tick_params(labelsize=20)
    ax.set_ylim(0.2, 0.6); ax.legend(fontsize=20, frameon=False, loc="upper right")
    save(fig, "planting_bars")


if __name__ == "__main__":
    you_are_here("here_02", ("Encode", "Model", "Train"))
    you_are_here("here_03", ("Interrogate",))
    onehot_big(); conv_slide(); network(); learning_curve()
    todo_card("todo_02", [("TODO-1", "one-hot encode a sequence, check it", "3 min"),
                          ("TODO-2", "stack the layers from the table", "7 min"),
                          ("TODO-3", "measure held-out AUROC each epoch", "5 min"),
                          ("run", "train 10 epochs · expect ~0.86", "1 min")], "02_build_cnn.ipynb")
    look_vs_test(); seed_pairing(); planting_bars()
    todo_card("todo_03", [("TODO-1", "filter → windows → logo, for every filter", "8 min"),
                          ("step", "type one motif into MY_MOTIF", "2 min"),
                          ("TODO-2", "plant it 1–8×, against a shuffled copy", "7 min"),
                          ("given", "position scan · microRNA bonus", "3 min")], "03_look_inside.ipynb")
