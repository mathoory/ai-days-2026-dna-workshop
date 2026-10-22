"""Copy the rendered decks into the public worktree, without speaker notes, for the workshop site.

    python scripts/publish_decks.py               # copy, commit and push to origin/public
    python scripts/publish_decks.py --no-commit   # copy only (the caller commits and pushes)
    python scripts/publish_decks.py --public PATH # another checkout of the public branch

Render the decks first (quarto render slides/NN_*.qmd). For each deck it copies NN_*.html with every
<aside class="notes"> removed, the deck's NN_*_files/ folder, style.css, and only the files under
assets/ that the deck references. GitHub Actions then rebuilds the site from the public branch.
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLIDES = ROOT / "slides"
DECKS = ["01_cnn", "02_foundation_models", "03_shortcuts_and_controls",
         "agents_lab"]   # the lab tutorial: hosted, but not linked from the site
REF = re.compile(r'(?:src|href|data-src|data-background-image)="([^"#:?]+)"')
NOTES = re.compile(r'<aside class="notes">.*?</aside>', re.S)


def git(public, *args):
    return subprocess.run(["git", "-C", str(public), *args], check=True, capture_output=True, text=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", default=str(ROOT.parent / "ai-days-public"))
    ap.add_argument("--no-commit", action="store_true")
    a = ap.parse_args()
    public = Path(a.public)
    if git(public, "rev-parse", "--abbrev-ref", "HEAD").strip() != "public":
        sys.exit(f"{public} is not on branch 'public'")
    out = public / "slides"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    for d in DECKS:
        html = (SLIDES / f"{d}.html").read_text(encoding="utf-8")
        n = len(NOTES.findall(html))
        html = NOTES.sub("", html)
        if "notes" in re.findall(r'<aside class="([^"]*)"', html):
            sys.exit(f"{d}: speaker notes left after stripping")
        (out / f"{d}.html").write_text(html, encoding="utf-8")
        shutil.copytree(SLIDES / f"{d}_files", out / f"{d}_files")
        refs = {r for r in REF.findall(html) if not r.startswith(f"{d}_files/")}
        for r in sorted(refs):
            src = SLIDES / r
            if not src.is_file():
                sys.exit(f"{d}: missing {r}")
            (out / r).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out / r)
        print(f"{d}: {n} notes removed, {len(refs)} files copied")
    if a.no_commit:
        return
    git(public, "add", "slides")
    if not git(public, "status", "--porcelain", "slides").strip():
        print("decks unchanged")
        return
    git(public, "commit", "-m", "Update decks")
    git(public, "pull", "--rebase", "--autostash", "origin", "public")
    git(public, "push", "origin", "public")
    print("pushed")


if __name__ == "__main__":
    main()
