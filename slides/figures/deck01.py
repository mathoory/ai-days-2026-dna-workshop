"""Figures for 01_opening_and_basics (opening: Mattan; biology + ML basics: Roee)."""
import numpy as np
from matplotlib.patches import Circle, Ellipse, Rectangle

from deckstyle import *  # noqa: F401,F403

SEQ = "GGACAGAAGGGUAUAGAAAAGUCCCC"


def hero():
    fig, ax = canvas()
    seq(ax, 8, 70, SEQ[:16], fs=34, dx=4.2)
    label(ax, 40, 61, "an RNA sequence", fs=22, color=GREY)
    arrow(ax, 40, 56, 40, 49)
    s = SEQ[:16]
    for i, ch in enumerate(s):
        for r, b in enumerate("ACGU"):
            ax.add_patch(Rectangle((8 + i * 4.2 - 2, 38 - r * 4.2), 4.0, 4.0,
                                   fc=BASE[b] if ch == b else "white", ec="#cfcfcf", lw=1))
    for r, b in enumerate("ACGU"):
        label(ax, 3.5, 40 - r * 4.2, b, fs=20, color=BASE[b], bold=True)
    label(ax, 40, 20, "numbers a machine can read", fs=22, color=GREY)
    arrow(ax, 77, 32, 92, 32)
    xs = [100, 113, 126]
    layers = [[22, 30, 38, 46], [26, 34, 42], [34]]
    for li, (x, ys) in enumerate(zip(xs, layers)):
        if li < 2:
            for y0 in ys:
                for y1 in layers[li + 1]:
                    ax.plot([x, xs[li + 1]], [y0, y1], color="#c5cae9", lw=1.5, zorder=1)
        for y0 in ys:
            ax.add_patch(Circle((x, y0), 2.2, fc=BLUE_L, ec=BLUE, lw=2.5, zorder=2))
    arrow(ax, 130, 34, 139, 34)
    box(ax, 140, 27, 17, 14, "bound?\n0.93", fc=GOLD_L, ec=GOLD, fs=24, bold=True)
    label(ax, 113, 64, "Applications of AI in Biotechnology and Genomics", fs=36, bold=True)
    label(ax, 113, 55, "and checking that they really did", fs=26, color=RED, style="italic")
    save(fig, "hero")


def room():
    fig, ax = canvas()
    ax.add_patch(Ellipse((58, 47), 70, 60, fc=BLUE_L, ec=BLUE, lw=3, alpha=.8))
    ax.add_patch(Ellipse((102, 47), 70, 60, fc=GREEN_L, ec=GREEN, lw=3, alpha=.8))
    label(ax, 41, 55, "I write code", fs=32, color=BLUE, bold=True)
    label(ax, 41, 43, "train/test, Python,\nmaybe a neural net", fs=22, color=BLUE)
    label(ax, 119, 55, "I work with biology", fs=32, color=GREEN, bold=True)
    label(ax, 119, 43, "genes, RNA,\ncontrols, replicates", fs=22, color=GREEN)
    label(ax, 80, 49, "sit\ntogether", fs=30, bold=True)
    label(ax, 80, 8, "2 minutes: name · field · one thing you want from today", fs=26)
    save(fig, "room")


def logistics():
    fig, ax = canvas()
    tiles = [("1", "your laptop", "charged"), ("2", "a Google account", "to open Colab"),
             ("3", "a free GPU", "Runtime → Change\nruntime type → T4"),
             ("?", "stuck?", "raise a hand, or\nload our files")]
    for i, (icon, t, sub) in enumerate(tiles):
        x = 6 + i * 38.5
        box(ax, x, 22, 34, 50, "", fc=GREY_L, ec="#bdbdbd")
        label(ax, x + 17, 59, icon, fs=54, bold=True, color=[BLUE, GREEN, PURPLE, RED][i])
        label(ax, x + 17, 44, t, fs=23, bold=True)
        label(ax, x + 17, 32, sub, fs=19, color=GREY)
    save(fig, "logistics")


def agenda():
    fig, ax = canvas()
    slots = [("opening\n+ basics", 0, 30, "R"), ("build\na CNN", 30, 30, "M"),
             ("look\ninside", 60, 30, "M"), ("foundation\nmodels", 90, 30, "R"),
             ("break", 120, 15, "B"), ("pretrained\nmodel", 135, 30, "M"),
             ("break\nit", 165, 30, "M"), ("fix\nit", 195, 25, "M"), ("wrap\nup", 220, 20, "R")]
    x0, sc = 6, 148 / 240
    for name, t, d, who in slots:
        fc, ec = {"R": (PURPLE_L, PURPLE), "M": (BLUE_L, BLUE), "B": (GREY_L, GREY)}[who]
        box(ax, x0 + t * sc, 42, d * sc - 0.8, 20, "", fc=fc, ec=ec, r=1)
        ax.text(x0 + (t + d / 2) * sc, 52, name, ha="center", va="center", fontsize=17, color=NAVY,
                linespacing=1.1)
    for h in range(10, 15):
        x = x0 + (h - 10) * 60 * sc
        ax.plot([x, x], [38, 42], color=NAVY, lw=2)
        label(ax, x, 34, f"{h}:00", fs=20)
    box(ax, 24, 12, 52, 11, "theory · Prof. Roee Amit", fc=PURPLE_L, ec=PURPLE, fs=22)
    box(ax, 84, 12, 52, 11, "hands-on · Colab", fc=BLUE_L, ec=BLUE, fs=22)
    label(ax, 80, 76, "one question all day:\nwhere does the protein AGO2 bind RNA?", fs=28, bold=True)
    save(fig, "agenda")


def spine():
    fig, ax = canvas()
    pipeline(ax, y=50, active=("Question", "Data", "Encode", "Model", "Train", "Evaluate", "Interrogate"),
             fs=18)
    subs = ["what do we\nwant to know?", "labels from an\nexperiment", "letters →\nnumbers",
            "a function with\nparameters", "fit the\nexamples", "on data it\nnever saw", "is it real\nbiology?"]
    for i, s in enumerate(subs):
        label(ax, 6 + i * 22.1 + 9.75, 37, s, fs=18, color=GREY)
    box(ax, 6 + 6 * 22.1 - 1, 47, 21.5, 15, "", fc="none", ec=RED, lw=3.5)
    save(fig, "spine")


def dogma():
    fig, ax = canvas()
    box(ax, 6, 52, 30, 16, "DNA\nthe genome", fc=GREEN_L, ec=GREEN, fs=24, bold=True)
    arrow(ax, 37, 60, 54, 60)
    box(ax, 55, 52, 34, 16, "messenger RNA\na working copy", fc=BLUE_L, ec=BLUE, fs=24, bold=True)
    arrow(ax, 90, 60, 107, 60)
    box(ax, 108, 52, 30, 16, "protein\nthe machine", fc=GOLD_L, ec=GOLD, fs=24, bold=True)
    label(ax, 98.5, 66, "✕", fs=40, color=RED, bold=True)
    box(ax, 56, 18, 32, 18, "AGO2\n+ microRNA", fc=RED_L, ec=RED, fs=24, bold=True)
    arrow(ax, 72, 37, 72, 51, color=RED, lw=4)
    label(ax, 92, 27, "binds the mRNA →\nless protein is made", fs=24, color=RED, ha="left")
    label(ax, 80, 82, "AGO2 turns genes down after they are copied", fs=28, bold=True)
    save(fig, "dogma")


def why_ago2():
    fig, ax = canvas()
    tiles = [(">60%", "of human genes are\nmicroRNA targets", GREEN),
             ("variants", "a mutation in a binding site\ncan change disease risk", RED),
             ("RNA drugs", "siRNA medicines work\nby loading into AGO2", PURPLE)]
    for i, (big, small, c) in enumerate(tiles):
        x = 6 + i * 51
        box(ax, x, 18, 46, 54, "", fc="white", ec=c, lw=4)
        label(ax, x + 23, 54, big, fs=46, bold=True, color=c)
        label(ax, x + 23, 32, small, fs=22)
    save(fig, "why_ago2")


def why_predict():
    fig, ax = canvas()
    box(ax, 6, 48, 40, 26, "experiment\n(CLIP-seq)\none cell type,\none condition", fc=GREEN_L, ec=GREEN, fs=21)
    arrow(ax, 47, 61, 60, 61)
    box(ax, 61, 48, 30, 26, "labels:\nbound /\nunbound", fc=GREEN_L, ec=GREEN, fs=21)
    arrow(ax, 92, 61, 105, 61)
    box(ax, 106, 48, 28, 26, "model", fc=BLUE_L, ec=BLUE, fs=28, bold=True)
    arrow(ax, 120, 47.5, 120, 39, color=GOLD)
    for i, t in enumerate(["transcripts never measured", "a patient's variant", "a designed RNA drug"]):
        y = 34 - i * 11
        box(ax, 94, y - 4, 62, 8.5, t, fc=GOLD_L, ec=GOLD, fs=21)
    label(ax, 30, 27, "measure once…", fs=26, color=GREEN, bold=True)
    label(ax, 30, 16, "…predict everywhere", fs=26, color=GOLD, bold=True)
    save(fig, "why_predict")


def labels_fig():
    fig, ax = canvas()
    x = np.linspace(0, 150, 1500)
    rng = np.random.default_rng(3)
    y = rng.gamma(1.2, 0.6, x.size) * 0.6
    for c in (22, 61, 98, 131):
        y += 9 * np.exp(-((x - c) / 2.2) ** 2)
    ax.fill_between(x + 5, 40, 40 + y * 2.2, color=GREEN, alpha=.85)
    ax.plot([5, 155], [40, 40], color=NAVY, lw=3)
    label(ax, 80, 33, "a stretch of the genome · AGO2 CLIP-seq signal", fs=21, color=GREY)
    for c in (22, 61, 98, 131):
        box(ax, c, 68, 10, 7, "+", fc=GREEN_L, ec=GREEN, fs=24, bold=True)
    for c in (8, 44, 80, 115, 148):
        box(ax, c, 14, 10, 7, "−", fc=GREY_L, ec=GREY, fs=24, bold=True)
    label(ax, 80, 83, "positives: where the experiment saw AGO2", fs=26, color=GREEN, bold=True)
    label(ax, 80, 4.5, "negatives: chosen by a person · how?", fs=26, color=RED, bold=True)
    save(fig, "labels")


def ml_loop():
    fig, ax = canvas()
    box(ax, 6, 52, 30, 18, "example\nsequence", fc=GREEN_L, ec=GREEN, fs=23)
    arrow(ax, 37, 61, 52, 61)
    box(ax, 53, 50, 40, 22, "model  f(x; θ)\nθ = parameters", fc=BLUE_L, ec=BLUE, fs=24, bold=True)
    arrow(ax, 94, 61, 108, 61)
    box(ax, 109, 52, 26, 18, "prediction\n0.31", fc=GOLD_L, ec=GOLD, fs=23)
    box(ax, 109, 18, 26, 16, "label\n1 (bound)", fc=GREEN_L, ec=GREEN, fs=23)
    arrow(ax, 122, 51, 122, 35, color=RED, style="<|-|>")
    label(ax, 137, 43, "loss:\nhow wrong?", fs=23, color=RED, ha="left")
    arrow(ax, 108, 26, 73, 26, color=RED)
    arrow(ax, 73, 26, 73, 49, color=RED)
    label(ax, 50, 22, "nudge θ to be\nless wrong", fs=23, color=RED)
    label(ax, 80, 84, "training = repeat this for thousands of examples", fs=26, bold=True)
    save(fig, "ml_loop")


def splits():
    fig, ax = canvas()
    box(ax, 6, 60, 96, 12, "train: the model learns from these", fc=BLUE_L, ec=BLUE, fs=23, r=0.6)
    box(ax, 103, 60, 24, 12, "validation:\ntune", fc=GOLD_L, ec=GOLD, fs=20, r=0.6)
    box(ax, 128, 60, 26, 12, "test:\nlook once", fc=RED_L, ec=RED, fs=20, r=0.6, bold=True)
    label(ax, 80, 81, "never grade a model on what it trained on", fs=26, bold=True)
    label(ax, 80, 46, "DNA twist: near-identical sequences on both sides = a leaked answer", fs=24, color=RED)
    for i, (s, side) in enumerate((("UUUGCACUAAUG", "train"), ("UUUGCACUAAUC", "test"))):
        x = 22 + i * 70
        box(ax, x - 4, 14, 52, 22, "", fc=BLUE_L if side == "train" else RED_L,
            ec=BLUE if side == "train" else RED)
        seq(ax, x + 2, 28, s, fs=26, dx=3.6)
        label(ax, x + 22, 19, side, fs=21, color=GREY)
    label(ax, 80, 25, "≈", fs=40, bold=True)
    label(ax, 80, 5, "fix: split by gene or chromosome, not by row", fs=24, color=GREEN, bold=True)
    save(fig, "splits")


def dictionary():
    fig, ax = canvas()
    rows = [("training set", "calibration samples"), ("test set", "an independent replicate"),
            ("baseline model", "negative control"), ("shuffle test", "scrambled-sequence control"),
            ("matched negatives", "matched background"), ("AUROC", "ROC of a diagnostic test")]
    box(ax, 18, 76, 56, 9, "machine learning", fc=BLUE, ec=BLUE, tc="white", fs=24, bold=True)
    box(ax, 86, 76, 56, 9, "the lab", fc=GREEN, ec=GREEN, tc="white", fs=24, bold=True)
    for i, (a, b) in enumerate(rows):
        y = 64 - i * 11
        box(ax, 18, y, 56, 8.5, a, fc=BLUE_L, ec=BLUE, fs=23)
        box(ax, 86, y, 56, 8.5, b, fc=GREEN_L, ec=GREEN, fs=23)
        label(ax, 80, y + 4.2, "=", fs=30, bold=True)
    save(fig, "dictionary")


def auroc():
    fig = plt.figure(figsize=(16, 9), dpi=120)
    a1 = fig.add_axes([0.06, 0.16, 0.42, 0.68])
    a2 = fig.add_axes([0.58, 0.16, 0.34, 0.68])
    rng = np.random.default_rng(0)
    neg, pos = rng.normal(0.38, 0.15, 4000), rng.normal(0.60, 0.15, 4000)
    bins = np.linspace(-0.1, 1.1, 50)
    a1.hist(neg, bins, color=GREY, alpha=.6, label="unbound")
    a1.hist(pos, bins, color=GREEN, alpha=.6, label="bound")
    a1.set_xlabel("model score", fontsize=22); a1.set_yticks([]); a1.legend(frameon=False, fontsize=22)
    a1.set_title("scores the model gives", fontsize=24)
    thr = np.linspace(1.2, -0.2, 200)
    tpr = [(pos > t).mean() for t in thr]; fpr = [(neg > t).mean() for t in thr]
    a2.plot(fpr, tpr, color=GREEN, lw=4); a2.fill_between(fpr, tpr, color=GREEN, alpha=.12)
    a2.plot([0, 1], [0, 1], "--", color=GREY, lw=2)
    a2.text(0.42, 0.22, f"AUROC = {np.trapz(tpr, fpr):.2f}", fontsize=26, color=GREEN, fontweight="bold")
    a2.text(0.52, 0.60, "0.5 = coin flip", fontsize=20, color=GREY, rotation=38)
    a2.set_xlabel("false positive rate", fontsize=22); a2.set_ylabel("true positive rate", fontsize=22)
    a2.set_title("ROC curve", fontsize=24)
    fig.text(0.5, 0.93, "AUROC = the chance a bound sequence outscores an unbound one", ha="center",
             fontsize=26, color=NAVY, fontweight="bold")
    save(fig, "auroc")


if __name__ == "__main__":
    for f in (hero, room, logistics, agenda, spine, dogma, why_ago2, why_predict, labels_fig,
              ml_loop, splits, dictionary, auroc):
        f()
