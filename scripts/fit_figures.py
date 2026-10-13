"""Size every full-width figure in the decks to the space its slide leaves.

For each `![](assets/...png){...height=...}` line in slides/0*.qmd: read the PNG's aspect ratio,
estimate the vertical space left after the title and the slide's visible text (bullets, maths,
tables), and write the height that fills that space without exceeding the usable width.
Images inside columns (width given in %) are left alone.
"""
import re
from pathlib import Path

from PIL import Image

SLIDES = Path(__file__).resolve().parents[1] / "slides"
MAX_W = 1488           # the content box is inset ~56px a side (measured in the Triplexis seminar)
BODY_H = 700           # tallest figure that clears the title, the menu bar and the bottom chrome
IMG = re.compile(r'^!\[\]\((assets/[^)]+\.png)\)\{([^}]*)\}\s*$')


def visible_cost(lines):
    """Pixels used by the visible (non-notes) text on a slide."""
    cost, in_notes, in_math = 0, False, False
    for l in lines:
        s = l.strip()
        if s.startswith("::: {.notes}"):
            in_notes = True; continue
        if in_notes:
            if s == ":::":
                in_notes = False
            continue
        if s.startswith("$$"):
            cost += 0 if in_math or s.count("$$") == 2 and len(s) > 4 else 0
            if s.count("$$") == 2 and len(s) > 4:
                cost += 95
            elif not in_math:
                in_math = True; cost += 95
            else:
                in_math = False
            continue
        if in_math:
            continue
        if s.startswith(("- ", "* ")) or re.match(r"^\d+\. ", s):
            cost += 58 if len(s) < 90 else 104
    return cost


def fit(path):
    text = path.read_text(encoding="utf-8").split("\n")
    out, slide = [], []
    starts = [i for i, l in enumerate(text) if l.startswith("## ") or l.startswith("::: {.center}")] + [len(text)]
    seg = {}
    for a, b in zip(starts[:-1], starts[1:]):
        for i in range(a, b):
            seg[i] = (a, b)
    changed = 0
    for i, l in enumerate(text):
        m = IMG.match(l)
        if m and "%" not in m.group(2) and i in seg:
            a, b = seg[i]
            others = [x for j, x in enumerate(text[a + 1:b], a + 1) if j != i]
            n_imgs = sum(1 for x in others if IMG.match(x) and "%" not in IMG.match(x).group(2))
            avail = BODY_H - visible_cost(others)
            if n_imgs:
                avail = avail // 2
            w, h = Image.open(SLIDES / m.group(1)).size
            height = int(min(avail, MAX_W * h / w))
            l = f'![]({m.group(1)}){{fig-align="center" height="{height}px"}}'
            changed += 1
        out.append(l)
    path.write_text("\n".join(out), encoding="utf-8")
    return changed


if __name__ == "__main__":
    for q in sorted(SLIDES.glob("0*.qmd")):
        print(q.name, fit(q))
