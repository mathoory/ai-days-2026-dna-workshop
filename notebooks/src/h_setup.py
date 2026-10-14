#@title ▶ Setup { display-mode: "form" }
import os, time, urllib.request
_SETUP_T0 = time.time()   # when this cell started; used only by the optional timing log at the end
from pathlib import Path
import numpy as np
import torch
from torch import nn, optim
from sklearn.metrics import roc_auc_score

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE, "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU - Runtime > Change runtime type > T4 GPU")

FORCE_FP16 = os.environ.get("AIDAYS_FORCE_FP16") == "1"
AMP_DTYPE = (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() and not FORCE_FP16
             else torch.float16)

# Plot style, shared by every notebook (the same palette as the slides).
import logging
import matplotlib.pyplot as plt
from matplotlib import cycler
from matplotlib.colors import LinearSegmentedColormap

INK, MUTED, MIDGREY, LIGHTGREY, HAIRLINE = "#202124", "#5F6368", "#80868B", "#C4C7C5", "#DADCE0"
ACCENT = "#1A73E8"
# The accent (blue) marks only bound sequences and fine-tuned models; everything else is ink or grey.
BOUND, UNBOUND = ACCENT, LIGHTGREY                            # class colours
CNN_C, FROZEN_C, FINETUNED_C, BASELINE_C = INK, MIDGREY, ACCENT, MIDGREY   # model colours (baselines dashed)
BASE_COLOURS = {"A": "#0F9D58", "C": "#4285F4", "G": "#F4B400", "U": "#DB4437", "T": "#DB4437"}  # logos only
CMAP_DIV = LinearSegmentedColormap.from_list("ws_div", [MUTED, "#FFFFFF", ACCENT])         # weight heatmaps
CMAP_SEQ = LinearSegmentedColormap.from_list("ws_seq", [HAIRLINE, MIDGREY, INK])          # sequential values: grey -> ink

plt.colormaps.register(CMAP_SEQ, force=True)
plt.colormaps.register(CMAP_DIV, force=True)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # Colab has no Roboto: fall back quietly
plt.rcParams.update({
    "font.family": ["Roboto", "DejaVu Sans"],
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlelocation": "left", "axes.titlesize": 11,
    "axes.prop_cycle": cycler(color=[INK, MIDGREY, LIGHTGREY]),
    "axes.grid": False, "grid.color": HAIRLINE, "grid.linewidth": 0.8,
    "legend.frameon": False, "legend.fontsize": 8,
    "image.cmap": "ws_seq",
})

# Where the workshop files are stored. Either one direct-download link per file (FILE_URLS, e.g. SharePoint
# "anyone with the link" + download=1), or one folder URL (DATA_URL, e.g. a Hugging Face dataset repo).
FILE_URLS = {}
DATA_URL = os.environ.get("AIDAYS_DATA_URL", "https://huggingface.co/datasets/yup8/ai-days-2026-ago2/resolve/main")
DATA_DIR = Path(os.environ.get("AIDAYS_DATA_DIR", "data"))
DATA_DIR.mkdir(exist_ok=True)

try:
    from google.colab import files as _colab_files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


def fetch(name):
    """Return the local path of a workshop file, downloading it first if needed."""
    path = DATA_DIR / name
    if not path.exists():
        url = FILE_URLS.get(name) or (f"{DATA_URL.rstrip('/')}/{name}" if DATA_URL else "")
        if not url:
            raise FileNotFoundError(f"{path} not found and no download link is set for it")
        t = time.time()
        urllib.request.urlretrieve(url, path)
        print(f"downloaded {name} ({path.stat().st_size / 1e6:.1f} MB) in {time.time() - t:.1f}s")
    return path


def _aidays_timing():
    """Optional cell timer for testing the notebooks. It does nothing unless the environment variable
    AIDAYS_TIMING=1 is set. Then every executed cell appends {first_line, seconds, raised, error_type,
    error_message} to the list builtins.__AIDAYS_LOG__, which survives reruns of this cell."""
    if os.environ.get("AIDAYS_TIMING") != "1":
        return
    try:
        import builtins
        import time as _t
        from IPython import get_ipython
        if not isinstance(getattr(builtins, "__AIDAYS_LOG__", None), list):
            builtins.__AIDAYS_LOG__ = []
        ip = get_ipython()
        if ip is None or getattr(builtins, "__AIDAYS_HOOKED__", False):
            return

        def pre(info=None):
            try:
                builtins.__AIDAYS_T0__ = _t.time()
            except Exception:
                pass

        def post(result=None):
            try:
                t0, builtins.__AIDAYS_T0__ = getattr(builtins, "__AIDAYS_T0__", None), None
                src = getattr(getattr(result, "info", None), "raw_cell", None) or ""
                err = getattr(result, "error_in_exec", None) or getattr(result, "error_before_exec", None)
                if not isinstance(getattr(builtins, "__AIDAYS_LOG__", None), list):
                    builtins.__AIDAYS_LOG__ = []
                builtins.__AIDAYS_LOG__.append({
                    "first_line": next((l.strip() for l in src.splitlines() if l.strip()), "")[:120],
                    "seconds": None if t0 is None else round(_t.time() - t0, 2),
                    "raised": err is not None,
                    "error_type": None if err is None else type(err).__name__,
                    "error_message": None if err is None else str(err)[:500]})
            except Exception:
                pass

        ip.events.register("pre_run_cell", pre)
        ip.events.register("post_run_cell", post)
        builtins.__AIDAYS_HOOKED__ = True
        builtins.__AIDAYS_T0__ = globals().get("_SETUP_T0") or _t.time()   # time this setup cell from its start
        print("timing log on: builtins.__AIDAYS_LOG__")
    except Exception:
        pass


_aidays_timing()
