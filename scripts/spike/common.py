"""Shared helpers for the AI Days spike: AGO2 data loading, splits, encoders, timing."""
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get("AGO2_DATA", Path.home() / "ai-days-2026" / "data"))
OUT = Path(os.environ.get("SPIKE_OUT", Path.home() / "ai-days-2026" / "spike"))
OUT.mkdir(parents=True, exist_ok=True)

MAX_LEN = 501
ALPHABET = "ACGU"
SEED = 42


def read_fasta_raw(path):
    """Return (headers, sequences) with original case kept (uppercase = CLIP core)."""
    heads, seqs = [], []
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                heads.append(line[1:])
            elif line:
                seqs.append(line)
    return heads, seqs


def norm(seq):
    return seq.upper().replace("T", "U")


def core_of(seq):
    """The uppercase stretch (the CLIP site) of a mixed-case record."""
    idx = [i for i, c in enumerate(seq) if c.isupper()]
    if not idx:
        return ""
    return seq[idx[0]: idx[-1] + 1]


def load_split(name):
    """name in {'train','ls'} -> (seqs_mixed_case, labels)."""
    _, pos = read_fasta_raw(DATA / f"CLIPSEQ_AGO2.{name}.positives.fa")
    _, neg = read_fasta_raw(DATA / f"CLIPSEQ_AGO2.{name}.negatives.fa")
    seqs = pos + neg
    y = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int64)
    return seqs, y


def holdout(n, frac=0.1, seed=SEED):
    """Indices for a held-out test split of the training file."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    k = int(n * frac)
    return perm[k:], perm[:k]


_LUT = np.full(256, -1, dtype=np.int64)
for _i, _b in enumerate(ALPHABET):
    _LUT[ord(_b)] = _i
_LUT[ord("T")] = 3


def encode_batch(seqs, max_len=MAX_LEN):
    """Vectorised one-hot (B,1,4,L); padding/unknown = 0.25 (same as Task 1)."""
    import torch
    B = len(seqs)
    X = np.full((B, 1, 4, max_len), 0.25, dtype=np.float32)
    for b, s in enumerate(seqs):
        s = norm(s)[:max_len]
        idx = _LUT[np.frombuffer(s.encode(), dtype=np.uint8)]
        pos = np.nonzero(idx >= 0)[0]
        X[b, 0, :, pos] = 0.0
        X[b, 0, idx[pos], pos] = 1.0
    return torch.from_numpy(X)


def kmer_counts(seqs, k, normalise=True):
    """(N, 4^k) k-mer frequency matrix over ACGU."""
    n = len(seqs)
    M = np.zeros((n, 4 ** k), dtype=np.float32)
    for r, s in enumerate(seqs):
        idx = _LUT[np.frombuffer(norm(s).encode(), dtype=np.uint8)]
        if len(idx) < k:
            continue
        codes = np.zeros(len(idx) - k + 1, dtype=np.int64)
        valid = np.ones(len(idx) - k + 1, dtype=bool)
        for j in range(k):
            part = idx[j: len(idx) - k + 1 + j]
            valid &= part >= 0
            codes = codes * 4 + np.maximum(part, 0)
        np.add.at(M[r], codes[valid], 1)
        if normalise and valid.sum():
            M[r] /= valid.sum()
    return M


def dinuc_shuffle(seq, rng):
    """Altschul-Erickson-style dinucleotide-preserving shuffle (Eulerian walk)."""
    s = norm(seq)
    if len(s) < 3:
        return s
    from collections import defaultdict
    edges = defaultdict(list)
    for a, b in zip(s[:-1], s[1:]):
        edges[a].append(b)
    last = s[-1]
    for _ in range(100):
        # choose a last-edge for every vertex except the final char -> spanning arborescence to `last`
        last_edge = {}
        for v in edges:
            if v != last:
                last_edge[v] = edges[v][rng.integers(len(edges[v]))]
        # check all vertices reach `last`
        ok = True
        for v in last_edge:
            seen, u = set(), v
            while u != last:
                if u in seen or u not in last_edge:
                    ok = False
                    break
                seen.add(u)
                u = last_edge[u]
            if not ok:
                break
        if ok:
            break
    order = {}
    for v, lst in edges.items():
        lst = list(lst)
        if v in last_edge:
            lst.remove(last_edge[v])
            rng.shuffle(lst)
            lst.append(last_edge[v])
        else:
            rng.shuffle(lst)
        order[v] = lst
    out, u = [s[0]], s[0]
    ptr = {v: 0 for v in order}
    for _ in range(len(s) - 1):
        nxt = order[u][ptr[u]]
        ptr[u] += 1
        out.append(nxt)
        u = nxt
    return "".join(out)


@contextmanager
def timer(store, key):
    t = time.time()
    yield
    store[key] = round(time.time() - t, 2)
    print(f"[time] {key}: {store[key]} s", flush=True)


def save_json(obj, name):
    p = OUT / name
    p.write_text(json.dumps(obj, indent=2, default=float))
    print("wrote", p)
