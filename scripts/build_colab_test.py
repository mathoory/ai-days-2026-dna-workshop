"""Build notebooks/colab_timing_test.ipynb: the three solution notebooks in one, instrumented for a Colab T4 run.

The test notebook records, for every code cell: notebook, section heading, first code line, wall time,
error (type, message), peak GPU memory, and the last part of its printed output. A first cell records
the machine (GPU, bf16 support, versions, RAM, disk, download speed); a last cell writes
aidays_colab_report.json, prints a summary and downloads the file.
    python scripts/build_colab_test.py      (after scripts/build.py)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
STEMS = ["01_cnn", "02_foundation_models", "03_shortcuts_and_controls"]

SETUP = r'''#@title AI Days timing test: setup (run first) { display-mode: "form" }
import builtins, io, json, os, platform, shutil, subprocess, sys, time, traceback, urllib.request
os.environ["AIDAYS_NO_DOWNLOAD"] = "1"     # the notebooks skip their "download your file" cells in this test
T0 = time.time()
info = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "python": platform.python_version()}
try:
    info["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                        capture_output=True, text=True, timeout=30).stdout.strip()
except Exception as e:
    info["nvidia_smi"] = f"n/a ({e})"
try:
    import torch, transformers, sklearn
    info.update(torch=torch.__version__, transformers=transformers.__version__, sklearn=sklearn.__version__,
                cuda=torch.cuda.is_available(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                bf16_supported=bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()))
except Exception as e:
    info["torch_error"] = repr(e)
try:
    import psutil
    info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
except Exception:
    pass
info["disk_free_gb"] = round(shutil.disk_usage("/").free / 1e9, 1)
try:   # download speed from the workshop's data host
    t = time.time()
    with urllib.request.urlopen("https://huggingface.co/datasets/yup8/ai-days-2026-ago2/resolve/main/CLIPSEQ_AGO2.train.positives.fa", timeout=120) as r:
        n = len(r.read())
    s = time.time() - t
    info["hf_download_test"] = {"megabytes": round(n / 1e6, 1), "seconds": round(s, 2), "MB_per_s": round(n / 1e6 / s, 1)}
except Exception as e:
    info["hf_download_test"] = repr(e)

LOG = []
builtins.__AIDAYS_TEST_LOG__ = LOG


class _Tee(io.TextIOBase):
    """Forward everything to the real stdout and keep the last few thousand characters per cell."""
    def __init__(self, real):
        self.real, self.buf = real, []
    def write(self, s):
        try:
            self.buf.append(s)
        except Exception:
            pass
        return self.real.write(s)
    def flush(self):
        return self.real.flush()


_ip = get_ipython()
_state = {}


def _pre(info_=None):
    try:
        src = info_.raw_cell if info_ is not None else ""
        first = next((l for l in src.splitlines() if l.strip()), "")
        _state.update(t=time.time(), first=first[:160], tee=_Tee(sys.stdout))
        sys.stdout = _state["tee"]
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    except Exception:
        pass


def _post(result=None):
    try:
        tee = _state.get("tee")
        if tee is not None:
            sys.stdout = tee.real
        out = "".join(tee.buf) if tee is not None else ""
        rec = {"label": getattr(builtins, "__AIDAYS_LABEL__", ""), "first_line": _state.get("first", ""),
               "seconds": round(time.time() - _state.get("t", time.time()), 2), "stdout_tail": out[-3000:]}
        try:
            import torch
            if torch.cuda.is_available():
                rec["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        except Exception:
            pass
        err = getattr(result, "error_in_exec", None) or getattr(result, "error_before_exec", None)
        if err is not None:
            rec["error"] = f"{type(err).__name__}: {str(err)[:500]}"
        LOG.append(rec)
    except Exception:
        pass


_ip.events.register("pre_run_cell", _pre)
_ip.events.register("post_run_cell", _post)
print(json.dumps(info, indent=1))
'''

REPORT = r'''#@title AI Days timing test: write the report (run last) { display-mode: "form" }
import builtins, json, time
from collections import OrderedDict
log = [r for r in builtins.__AIDAYS_TEST_LOG__ if r.get("label")]
by_section, by_nb = OrderedDict(), OrderedDict()
for r in log:
    nb, sec = r["label"].split(" | ")[0], " | ".join(r["label"].split(" | ")[:2])
    by_section[sec] = round(by_section.get(sec, 0) + r["seconds"], 1)
    by_nb[nb] = round(by_nb.get(nb, 0) + r["seconds"], 1)
report = {"info": info, "total_minutes": round((time.time() - T0) / 60, 1), "per_notebook_seconds": by_nb,
          "per_section_seconds": by_section, "errors": [r for r in log if "error" in r], "cells": log}
with open("aidays_colab_report.json", "w") as f:
    json.dump(report, f, indent=1)
print("GPU:", info.get("gpu"), "| bf16:", info.get("bf16_supported"), "| total", report["total_minutes"], "min")
for k, v in by_nb.items():
    print(f"{k:32s} {v / 60:5.1f} min")
print("errors:", len(report["errors"]))
for e in report["errors"]:
    print("  ", e["label"], "->", e["error"][:200])
try:
    from google.colab import files
    files.download("aidays_colab_report.json")
except Exception:
    print("saved aidays_colab_report.json")
'''


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def main():
    cells = [md("# AI Days 2026: Colab timing test\n\n"
                "1. *Runtime → Change runtime type → T4 GPU*.\n"
                "2. *Runtime → Run all*. It takes roughly an hour; allow the file download at the end.\n"
                "3. Send the downloaded `aidays_colab_report.json` back.\n\n"
                "This runs the three workshop notebooks (solution versions) one after another and records the time, "
                "errors, GPU memory and printed results of every cell."),
             code(SETUP)]
    for stem in STEMS:
        nb = json.load(open(NB / f"{stem}_solution.ipynb", encoding="utf-8"))
        cells.append(md(f"# ===== Notebook {stem} ====="))
        heading = "(start)"
        for c in nb["cells"]:
            src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
            if c["cell_type"] == "markdown":
                for line in src.splitlines():
                    if line.startswith("#"):
                        heading = line.lstrip("#").strip()[:80]
                        break
                cells.append(md(src))
                continue
            label = f"{stem} | {heading} | {next((l for l in src.splitlines() if l.strip()), '')[:80]}"
            cells.append(code(f"import builtins; builtins.__AIDAYS_LABEL__ = {label!r}\n" + src))
    cells.append(code(REPORT))
    out = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                                        "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                        "language_info": {"name": "python"}},
           "nbformat": 4, "nbformat_minor": 5}
    json.dump(out, open(NB / "colab_timing_test.ipynb", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("wrote notebooks/colab_timing_test.ipynb with", len(cells), "cells")


if __name__ == "__main__":
    main()
