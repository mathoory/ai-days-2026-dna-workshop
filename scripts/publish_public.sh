#!/bin/bash
# Publish the student notebooks to the `public` branch on GitHub.
#
# This repo (branch main) is local only: it holds the solutions. The `public` branch is checked out
# as a git worktree (default ../ai-days-public) and is the only thing pushed. This script:
#   1. builds the notebooks (scripts/build.py) and renders the three decks (quarto)
#   2. checks the student notebooks for solution code (every "Your code here" region must be `...`)
#   3. copies ONLY notebooks/01_cnn.ipynb, 02_foundation_models.ipynb, 03_shortcuts_and_controls.ipynb
#   4. copies the rendered decks into slides/ without speaker notes (scripts/publish_decks.py --no-commit)
#   5. commits notebooks/ and slides/, rebases on origin/public (keeps other people's commits), pushes
# Nothing else in the public branch is touched (README, site/, the Pages workflow).
#
#   bash scripts/publish_public.sh "commit message"          # publish
#   bash scripts/publish_public.sh --dry-run                  # build, check and show the diff only
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PUB=${PUBLIC_DIR:-$ROOT/../ai-days-public}
STEMS="01_cnn 02_foundation_models 03_shortcuts_and_controls"
DRY=0
MSG=""
for a in "$@"; do
  if [ "$a" = "--dry-run" ]; then DRY=1; else MSG="$a"; fi
done
[ $DRY = 1 ] || [ -n "$MSG" ] || { echo "usage: $0 \"commit message\" | --dry-run"; exit 1; }

[ "$(git -C "$PUB" rev-parse --abbrev-ref HEAD)" = "public" ] || { echo "$PUB is not on branch public"; exit 1; }

echo "== build notebooks"
(cd "$ROOT" && python scripts/build.py)

echo "== render decks"
for s in $STEMS; do
  (cd "$ROOT/slides" && quarto render "$s.qmd" 2>&1 | grep -E "Output created|ERROR" ) || { echo "render failed: $s"; exit 1; }
done

echo "== solution-leak check"
PYTHONIOENCODING=utf-8 python - "$ROOT" $STEMS <<'EOF'
import json, re, sys
from pathlib import Path
root, stems = Path(sys.argv[1]), sys.argv[2:]
bad = 0
for s in stems:
    nb = json.load(open(root / "notebooks" / f"{s}.ipynb", encoding="utf-8"))
    sol = json.load(open(root / "notebooks" / f"{s}_solution.ipynb", encoding="utf-8"))
    n = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        for body in re.findall(r"# === Your code here ===(.*?)# === End of your code ===", src, re.S):
            n += 1
            if body.strip() != "...":
                bad += 1
                print(f"  LEAK in {s}: {body.strip()[:80]!r}")
    m = sum("".join(c["source"]).count("# === Your code here ===") for c in sol["cells"] if c["cell_type"] == "code")
    if n != m:
        bad += 1
        print(f"  {s}: {n} student regions but {m} in the solution")
    if any(c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code"):
        bad += 1
        print(f"  {s}: the student notebook has outputs")
    print(f"  {s}: {n} student regions, all empty")
sys.exit(1 if bad else 0)
EOF

echo "== copy into $PUB/notebooks"
mkdir -p "$PUB/notebooks"
for s in $STEMS; do cp "$ROOT/notebooks/$s.ipynb" "$PUB/notebooks/$s.ipynb"; done

echo "== copy decks into $PUB/slides (speaker notes removed)"
(cd "$ROOT" && python scripts/publish_decks.py --no-commit --public "$PUB")

FILES="$(for s in $STEMS; do printf "notebooks/%s.ipynb " "$s"; done) slides"
git -C "$PUB" --no-pager diff --stat -- $FILES

if [ $DRY = 1 ]; then
  echo "== dry run: nothing committed (restore with: git -C \"$PUB\" checkout -- notebooks slides; git -C \"$PUB\" clean -fd slides)"
  exit 0
fi
git -C "$PUB" add -N -- $FILES
if git -C "$PUB" diff --quiet -- $FILES; then
  echo "== no changes to publish"
  exit 0
fi
git -C "$PUB" add -- $FILES
git -C "$PUB" commit -q -m "$MSG" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- $FILES
git -C "$PUB" pull -q --rebase --autostash origin public
git -C "$PUB" push -q origin public
echo "== pushed: $(git -C "$PUB" log --oneline -1)"
