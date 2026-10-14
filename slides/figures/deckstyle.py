"""Shared figure style for the AI Days decks.

Light, flat, one typeface (Roboto), ink and greys plus one accent (blue) for the thing a
figure is about. Bound = accent, unbound = light grey. CNN = ink, a frozen pretrained model =
mid grey, fine-tuned = accent, shortcut baselines = grey dashed. Weight maps go grey-white-blue.
Only sequence logos use the four conventional base colours.
Labels are sentence case. Figures carry no caption sentences: the slide title and the
speaker say what a figure means. Saved to slides/assets/custom/<name>.png at 1920 x 1080.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "custom"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#202124"
MUTED = "#5F6368"
MID = "#80868B"
LIGHT = "#C4C7C5"
HAIR = "#DADCE0"
FAINT = "#F1F3F4"
ACCENT, ACCENT_L = "#1A73E8", "#D2E3FC"

# one accent; everything else is ink and greys
BOUND, UNBOUND = ACCENT, "#9AA0A6"
CNN_C, FROZEN_C, TUNED_C, SHORTCUT_C = INK, MID, ACCENT, MID

# old names, kept so older helper modules still import
GREEN, GREEN_L = ACCENT, ACCENT_L
BLUE, BLUE_L = INK, FAINT
PURPLE, PURPLE_L = MID, FAINT
ORANGE, ORANGE_L = ACCENT, ACCENT_L
RED, RED_L = ACCENT, ACCENT_L
GOLD, GOLD_L = MID, FAINT
GREY, GREY_L = LIGHT, FAINT
NAVY = INK

# sequence logos keep the conventional base colours (the only place with four colours)
BASE = {"A": "#0F9D58", "C": "#4285F4", "G": "#F4B400", "U": "#DB4437", "T": "#DB4437", "N": LIGHT}

# training vs held-out, everywhere: blue vs orange; a third line colour for "good"
TRAIN_C, HELD_C, GOOD_C = "#1A73E8", "#E8710A", "#1E8E3E"
TRAIN_L = "#AECBFA"
# models: CNN green, frozen pretrained blue, fine-tuned purple; shortcut baselines stay grey
TUNED_PURPLE = "#9334E6"
CNN_C, FROZEN_C, TUNED_C = GOOD_C, ACCENT, TUNED_PURPLE

from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
DIVERGING = LinearSegmentedColormap.from_list("orange_white_accent", [HELD_C, "#FFFFFF", ACCENT])
SEQUENTIAL = LinearSegmentedColormap.from_list("white_accent", ["#E8F0FE", ACCENT, "#174EA6"])
MONO = ["Consolas", "DejaVu Sans Mono"]

plt.rcParams.update({
    "font.family": ["Roboto", "DejaVu Sans"],
    "font.size": 22,
    "text.color": INK,
    "axes.labelcolor": INK, "axes.labelsize": 23, "axes.labelpad": 8,
    "axes.titlesize": 24, "axes.titleweight": "medium", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "axes.titlepad": 12,
    "axes.edgecolor": MUTED, "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "grid.color": HAIR, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "xtick.labelsize": 20, "ytick.labelsize": 20, "xtick.major.size": 4, "ytick.major.size": 4,
    "legend.frameon": False, "legend.fontsize": 20,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


def newfig():
    return plt.figure(figsize=(16, 9), dpi=120)


def ygrid(ax):
    ax.grid(axis="y", color=HAIR, lw=0.8); ax.set_axisbelow(True)


def xgrid(ax):
    ax.grid(axis="x", color=HAIR, lw=0.8); ax.set_axisbelow(True)


def canvas(w=16, h=9, xlim=(0, 160), ylim=(0, 90)):
    """A blank drawing area in slide units (160 x 90)."""
    fig = plt.figure(figsize=(w, h), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text="", fc=FAINT, ec="none", fs=18, tc=INK, lw=0, bold=False, r=0.4, **kw):
    """A flat filled rectangle (for tokens and matrix cells, not for diagrams)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, **kw))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
                fontweight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=1.5, ms=16, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms,
                                 color=color, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0))


def seq(ax, x, y, s, fs=24, dx=None, colored=True, weight="bold", ha="left"):
    """Draw a sequence letter by letter in base colours. Returns the x after the last letter."""
    dx = dx if dx is not None else fs * 0.105
    for i, ch in enumerate(s):
        ax.text(x + i * dx, y, ch, fontsize=fs, family=MONO, fontweight=weight,
                color=BASE.get(ch.upper(), INK) if colored else INK, ha="center", va="center")
    return x + len(s) * dx


def label(ax, x, y, text, fs=18, color=INK, ha="center", va="center", bold=False, style="normal", family=None):
    ax.text(x, y, text, fontsize=fs, color=color, ha=ha, va=va, family=family or plt.rcParams["font.family"],
            fontweight="bold" if bold else "normal", style=style, linespacing=1.25)


def trim(path, pad=24):
    """Crop a saved PNG to its ink (non-white pixels) plus a small margin, as in the Triplexis seminar."""
    import numpy as np
    from PIL import Image
    im = Image.open(path).convert("RGB")
    a = np.asarray(im)
    ys, xs = np.where((a < 246).any(axis=2))
    if len(xs):
        im.crop((max(0, xs.min() - pad), max(0, ys.min() - pad),
                 min(a.shape[1], xs.max() + 1 + pad), min(a.shape[0], ys.max() + 1 + pad))).save(path)


def save(fig, name):
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=120, facecolor="white")
    plt.close(fig)
    trim(path)
    print("wrote", name)


def trim_group(names, pad=24):
    """Crop the frames of one click-through to the union of their ink, so they still stack exactly."""
    import numpy as np
    from PIL import Image
    ims = [Image.open(OUT / f"{n}.png").convert("RGB") for n in names]
    boxes = []
    for im in ims:
        a = np.asarray(im)
        ys, xs = np.where((a < 246).any(axis=2))
        boxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
    x0 = max(0, min(b[0] for b in boxes) - pad); y0 = max(0, min(b[1] for b in boxes) - pad)
    x1 = min(ims[0].width, max(b[2] for b in boxes) + 1 + pad); y1 = min(ims[0].height, max(b[3] for b in boxes) + 1 + pad)
    for n, im in zip(names, ims):
        im.crop((x0, y0, x1, y1)).save(OUT / f"{n}.png")
