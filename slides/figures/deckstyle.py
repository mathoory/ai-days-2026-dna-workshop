"""Shared drawing helpers for the AI Days decks.

Course palette (same as the tut4 boxes): green = data, blue = CNN, purple = foundation model,
gold = head / probe, red = shortcut / warning. Every figure is drawn on a 16:9 canvas and saved
to slides/assets/custom/<name>.png at 1920x1080 (or the size given).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "custom"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#0B2052"
GREEN, GREEN_L = "#2e7d32", "#e8f5e9"
BLUE, BLUE_L = "#1565c0", "#e3f2fd"
PURPLE, PURPLE_L = "#6a0dad", "#f3e5f5"
GOLD, GOLD_L = "#c9a000", "#fffde7"
RED, RED_L = "#c62828", "#ffebee"
GREY, GREY_L = "#757575", "#f2f2f2"
BASE = {"A": "#2e7d32", "C": "#1565c0", "G": "#e6a100", "U": "#c62828", "T": "#c62828", "N": GREY}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 20, "axes.edgecolor": NAVY,
                     "axes.labelcolor": NAVY, "xtick.color": NAVY, "ytick.color": NAVY,
                     "text.color": NAVY, "axes.spines.top": False, "axes.spines.right": False})


def canvas(w=16, h=9, xlim=(0, 160), ylim=(0, 90)):
    """A blank drawing canvas in 'slide units' (160 x 90)."""
    fig = plt.figure(figsize=(w, h), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text="", fc=GREY_L, ec=NAVY, fs=22, tc=NAVY, lw=2.5, bold=False, r=1.6, **kw):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, **kw))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
                fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, color=NAVY, lw=3, ms=28, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms,
                                 color=color, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0))


def seq(ax, x, y, s, fs=26, dx=None, colored=True, weight="bold", ha="left"):
    """Draw a sequence letter by letter in base colours. Returns the x after the last letter."""
    dx = dx if dx is not None else fs * 0.105
    for i, ch in enumerate(s):
        ax.text(x + i * dx, y, ch, fontsize=fs, family="DejaVu Sans Mono", fontweight=weight,
                color=BASE.get(ch.upper(), NAVY) if colored else NAVY, ha="center", va="center")
    return x + len(s) * dx


def label(ax, x, y, text, fs=20, color=NAVY, ha="center", va="center", bold=False, style="normal"):
    ax.text(x, y, text, fontsize=fs, color=color, ha=ha, va=va,
            fontweight="bold" if bold else "normal", style=style, linespacing=1.25)


def pipeline(ax, y=78, active=(), x0=6, w=19.5, gap=2.6, h=9, fs=17):
    """The workshop spine. `active` = stage names to highlight."""
    stages = ["Question", "Data", "Encode", "Model", "Train", "Evaluate", "Interrogate"]
    for i, s in enumerate(stages):
        x = x0 + i * (w + gap)
        on = s in active
        box(ax, x, y, w, h, s, fc=NAVY if on else GREY_L, ec=NAVY if on else "#bdbdbd",
            tc="white" if on else GREY, fs=fs, bold=on)
        if i < len(stages) - 1:
            arrow(ax, x + w + 0.3, y + h / 2, x + w + gap - 0.3, y + h / 2,
                  color=NAVY if on else "#bdbdbd", lw=2, ms=16)
    return stages


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=120, facecolor="white")
    plt.close(fig)
    print("wrote", name)
