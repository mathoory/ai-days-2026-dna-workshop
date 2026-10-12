#@title Data helpers - run this cell, nothing to change { display-mode: "form" }
MAX_LEN  = 501          # fixed global length (pad / truncate)
ALPHABET = "ACGU"       # order = the 4 rows of the one-hot matrix
FILES = {split: (f"CLIPSEQ_AGO2.{split}.positives.fa", f"CLIPSEQ_AGO2.{split}.negatives.fa")
         for split in ("train", "ls")}


def read_fasta(path, label):
    """Return (sequences, labels) from a FASTA file, all with the same label.

    The files mix the DNA letter T and the RNA letter U for the same base,
    so we normalise everything to U.
    """
    seqs, labs = [], []
    with Path(path).open() as fh:
        for line in fh:
            if line.startswith(">"):
                continue
            seqs.append(line.strip().upper().replace("T", "U"))
            labs.append(label)
    return seqs, labs


def read_split(split):
    """'train' or 'ls' (the held-out validation set) -> (sequences, labels as a numpy array)."""
    pos, neg = FILES[split]
    p, lp = read_fasta(fetch(pos), 1)
    n, ln = read_fasta(fetch(neg), 0)
    return p + n, np.array(lp + ln)


_LUT = np.full(256, -1, dtype=np.int64)
for _i, _b in enumerate(ALPHABET):
    _LUT[ord(_b)] = _i


def encode_batch(seqs, max_len=MAX_LEN):
    """List of RNA strings -> one-hot FloatTensor (B, 1, 4, max_len).

    Pads/truncates to max_len; padding and unknown bases stay uniform 0.25.
    """
    X = np.full((len(seqs), 1, 4, max_len), 0.25, dtype=np.float32)
    for b, s in enumerate(seqs):
        idx = _LUT[np.frombuffer(s[:max_len].encode(), dtype=np.uint8)]
        pos = np.nonzero(idx >= 0)[0]
        X[b, 0, :, pos] = 0.0
        X[b, 0, idx[pos], pos] = 1.0
    return torch.from_numpy(X)
