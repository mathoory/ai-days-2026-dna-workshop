"""Tile figures into one image for quick visual QA: python contact_sheet.py out.png fig1 fig2 ..."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

out, names = sys.argv[1], sys.argv[2:]
src = Path(__file__).resolve().parents[1] / "assets" / "custom"
cols = 3
rows = (len(names) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.4, rows * 3.7), dpi=100)
for ax in axes.flat:
    ax.axis("off")
for ax, n in zip(axes.flat, names):
    ax.imshow(mpimg.imread(src / f"{n}.png"))
    ax.set_title(n, fontsize=11)
plt.tight_layout()
plt.savefig(out)
