"""Print text outputs and per-cell wall time of an executed notebook."""
import json, sys
from datetime import datetime
def ts(s): return datetime.fromisoformat(s.replace("Z", "+00:00"))
for p in sys.argv[1:]:
    nb = json.load(open(p, encoding="utf-8"))
    print("#####", p)
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code": continue
        ex = c.get("metadata", {}).get("execution", {})
        dt = ""
        if "iopub.execute_input" in ex and "shell.execute_reply" in ex:
            dt = f"{(ts(ex['shell.execute_reply']) - ts(ex['iopub.execute_input'])).total_seconds():.1f}s"
        first = "".join(c["source"]).split("\n")[0][:70]
        print(f"--- [{i}] {dt:>7} {first}")
        for o in c.get("outputs", []):
            t = "".join(o.get("text", [])) or "".join(o.get("data", {}).get("text/plain", []))
            if o.get("output_type") == "error": t = o["ename"] + ": " + o["evalue"]
            if "image/png" in o.get("data", {}): t = "[figure]"
            if t: print("    " + t.strip()[:700].replace("\n", "\n    "))
