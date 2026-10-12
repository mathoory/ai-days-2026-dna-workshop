"""For executed student notebooks: list every cell that raised, and whether it is a TODO cell
(expected: a TODO left as `...`) or a given cell (a bug), plus cells downstream of a TODO."""
import json
import sys

for p in sys.argv[1:]:
    nb = json.load(open(p, encoding="utf-8"))
    print("#####", p)
    seen_todo = False
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        is_todo = "# === Your code here ===" in src
        errs = [o for o in c.get("outputs", []) if o.get("output_type") == "error"]
        if errs:
            kind = "TODO cell (expected)" if is_todo else ("downstream of a TODO" if seen_todo else "GIVEN CELL - BUG")
            print(f"  [{i}] {kind}: {errs[0]['ename']}: {errs[0]['evalue'][:150]}")
        seen_todo = seen_todo or is_todo
