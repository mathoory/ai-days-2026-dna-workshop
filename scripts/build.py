"""Build workshop notebooks from percent-format sources.

notebooks/src/NN_name.py  ->  notebooks/NN_name_solution.ipynb  (everything filled in)
                          ->  notebooks/NN_name.ipynb           (student version)

Source format
-------------
# %% [markdown]          markdown cell; each line starts with "# " (or is "#")
# %%                     code cell
# %include h_data.py     replaced by the contents of notebooks/src/h_data.py
# %include-body h_x.py   the same, without the file's first line when that line is a #@title

Student version: in every code cell, the lines strictly between
    # === Your code here ===
    # === End of your code ===
are replaced by a single indented `...`. Nothing else differs from the solution.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "src"
OUT = ROOT / "notebooks"
BEGIN, END = "# === Your code here ===", "# === End of your code ==="


def expand(text):
    def inc(m):
        body = (SRC / m.group(2)).read_text(encoding="utf-8").replace("\r\n", "\n").rstrip("\n")
        if m.group(1) == "-body" and body.startswith("#@title"):
            body = body.split("\n", 1)[1] if "\n" in body else ""
        return body
    return re.sub(r"^# %include(-body)? (\S+)$", inc, text, flags=re.M)


def parse(text):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        lines = buf[:]
        while lines and not lines[-1].strip():
            lines.pop()
        while lines and not lines[0].strip():
            lines.pop(0)
        if kind == "markdown":
            lines = [l[2:] if l.startswith("# ") else (l[1:] if l.startswith("#") else l) for l in lines]
        if lines:
            cells.append((kind, "\n".join(lines)))

    for line in text.splitlines():
        if line.startswith("# %% [markdown]"):
            flush()
            kind, buf = "markdown", []
        elif line.startswith("# %%"):
            flush()
            kind, buf = "code", []
        else:
            buf.append(line)
    flush()
    return cells


def studentise(src):
    out, skipping = [], False
    for line in src.split("\n"):
        s = line.strip()
        if s == BEGIN:
            out.append(line)
            out.append(line[: len(line) - len(line.lstrip())] + "...")
            skipping = True
            continue
        if s == END:
            skipping = False
        if not skipping:
            out.append(line)
    return "\n".join(out)


def to_nb(cells, student):
    nb_cells = []
    for i, (kind, src) in enumerate(cells):
        if kind == "code" and student:
            src = studentise(src)
        c = {"cell_type": kind, "id": f"c{i:03d}", "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        nb_cells.append(c)
    return {"cells": nb_cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}}}


def build(path):
    cells = parse(expand(path.read_text(encoding="utf-8")))
    stem = path.stem
    for student, name in ((False, f"{stem}_solution.ipynb"), (True, f"{stem}.ipynb")):
        (OUT / name).write_text(json.dumps(to_nb(cells, student), indent=1, ensure_ascii=False), encoding="utf-8")
    n_todo = sum(src.count(BEGIN) for k, src in cells if k == "code")
    print(f"{stem}: {len(cells)} cells, {n_todo} student regions")


if __name__ == "__main__":
    targets = sys.argv[1:] or sorted(p.name for p in SRC.glob("[0-9][0-9]_*.py"))
    for t in targets:
        build(SRC / t)
