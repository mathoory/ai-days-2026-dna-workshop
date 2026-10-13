"""Screenshot every slide of every rendered deck (headless Edge) and build one contact sheet per deck.

Needs the decks served at http://127.0.0.1:8765/ (python -m http.server 8765 in slides/).
    python scripts/shoot_decks.py [deck-stem ...]
Writes shots/<deck>/NN.png and shots/<deck>_sheet.png under the given output folder.
"""
import re
import shutil
import subprocess
import time
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
SLIDES = Path(__file__).resolve().parents[1] / "slides"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("0") else Path.cwd() / "shots"
STEMS = [a for a in sys.argv[1:] if a.startswith("0")] or [q.stem for q in sorted(SLIDES.glob("0*.qmd"))]


def n_slides(stem):
    """Title slide + contents slide + one per section divider and per ## slide."""
    lines = (SLIDES / f"{stem}.qmd").read_text(encoding="utf-8").splitlines()
    return 2 + sum(1 for l in lines if l.startswith("## ") or l.startswith("# "))


for stem in STEMS:
    d = OUT / stem
    d.mkdir(parents=True, exist_ok=True)
    n = n_slides(stem)
    for i in range(n):
        shot = d / f"{i:02d}.png"
        shot.unlink(missing_ok=True)
        # msedge.exe hands off to a browser process and returns at once, so wait for the file
        prof = OUT / "prof" / f"{stem}_{i}"
        subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--user-data-dir={prof}",
                        "--window-size=1600,900", "--virtual-time-budget=4000", f"--screenshot={shot}",
                        f"http://127.0.0.1:8765/{stem}.html#/{i}"], check=False, capture_output=True)
        for _ in range(60):
            if shot.exists() and shot.stat().st_size > 0:
                break
            time.sleep(0.5)
        time.sleep(0.5)
        # the headless browser stays alive after the screenshot; stop it and delete its profile (~100 MB each)
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        "Get-CimInstance Win32_Process -Filter \"Name='msedge.exe'\" | "
                        f"Where-Object {{ $_.CommandLine -like '*{prof.name}*' -and $_.CommandLine -like '*prof*' }} | "
                        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"],
                       check=False, capture_output=True)
        shutil.rmtree(prof, ignore_errors=True)
    shots = sorted(d.glob("*.png"))[:n]
    cols = 3
    rows = (len(shots) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 4.6), dpi=80)
    for ax in axes.flat:
        ax.axis("off")
    for ax, p in zip(axes.flat, shots):
        ax.imshow(mpimg.imread(p)); ax.set_title(p.stem, fontsize=14)
    fig.tight_layout(); fig.savefig(OUT / f"{stem}_sheet.png"); plt.close(fig)
    print(stem, n, "slides")
