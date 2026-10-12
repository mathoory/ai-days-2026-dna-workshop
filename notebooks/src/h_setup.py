#@title Setup - run this cell, nothing to change { display-mode: "form" }
import os, time, urllib.request
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

# Where the workshop files live. Either one direct-download link per file (FILE_URLS, e.g. SharePoint
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


def download_outputs(*names):
    """End of a notebook: save your results to your laptop so the next notebook can use them."""
    for n in names:
        p = DATA_DIR / n
        if not p.exists():
            print(f"{n}: not found (did the cell that makes it run?)")
        elif IN_COLAB:
            _colab_files.download(str(p))
        else:
            print(f"{n}: saved at {p}")


def upload_yours():
    """Start of a notebook: upload the files you downloaded at the end of an earlier notebook."""
    if not IN_COLAB:
        print(f"not in Colab: copy your files into {DATA_DIR}")
        return
    import re
    for n, b in _colab_files.upload().items():
        clean = re.sub(r" \(\d+\)(?=\.[^.]+$)", "", n)   # "my_cnn (1).pt" -> "my_cnn.pt" (browser renames)
        (DATA_DIR / clean).write_bytes(b)
        print("uploaded", n, "->", clean)
