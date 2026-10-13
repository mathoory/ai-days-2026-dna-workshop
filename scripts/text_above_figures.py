"""On every slide, move visible bullets and display maths that sit below a figure to above it.

Tut style: title, then the text, then the figure. It also keeps text clear of the logo and
slide number at the bottom of the slide. Notes, columns and fenced divs are left untouched.
"""
from pathlib import Path

SLIDES = Path(__file__).resolve().parents[1] / "slides"


def fix(path):
    lines = path.read_text(encoding="utf-8").split("\n")
    out, i, moved = [], 0, 0
    while i < len(lines):
        l = lines[i]
        out.append(l)
        if l.startswith("![](assets/") and "%" not in l:
            # collect the visible block right after the figure (bullets / $$ maths / blank lines),
            # stopping at notes, a new slide or any fenced div
            j, block = i + 1, []
            while j < len(lines):
                s = lines[j]
                if s.startswith(":::") or s.startswith("## ") or s.startswith("# "):
                    break
                block.append(s); j += 1
            text = [b for b in block if b.strip()]
            if text:
                fig = out.pop()
                while out and not out[-1].strip():
                    out.pop()
                out += [""] + text + ["", fig, ""]
                moved += 1
                i = j
                continue
        i += 1
    path.write_text("\n".join(out), encoding="utf-8")
    return moved


if __name__ == "__main__":
    for q in sorted(SLIDES.glob("0*.qmd")):
        print(q.name, fix(q))
