#@title ▶ Shortcut toolkit { display-mode: "form" }
from collections import defaultdict


_K = {b: i for i, b in enumerate("ACGT")}
_K["U"] = 3


def kmer_counts(seqs, k=5):
    """(N, 4^k): how often each k-letter word appears in each sequence (as a fraction)."""
    M = np.zeros((len(seqs), 4 ** k), dtype=np.float32)
    for r, s in enumerate(seqs):
        code, valid = 0, 0
        for j, ch in enumerate(s):
            v = _K.get(ch, -1)
            if v < 0:
                valid = 0
                continue
            code = (code * 4 + v) % (4 ** k)
            valid += 1
            if valid >= k:
                M[r, code] += 1
        M[r] /= max(M[r].sum(), 1)
    return M


def dinuc_shuffle(seq, rng):
    """Shuffle a sequence but keep every letter-pair count (Altschul-Erickson / Eulerian walk).

    The result has exactly the same letters, the same G+C, the same letter pairs. Any longer
    pattern (a motif) is broken up.
    """
    if len(seq) < 3:
        return seq
    edges = defaultdict(list)
    for a, b in zip(seq[:-1], seq[1:]):
        edges[a].append(b)
    last = seq[-1]
    while True:  # random spanning tree of "last exits" pointing towards the final letter
        last_edge = {v: edges[v][rng.integers(len(edges[v]))] for v in edges if v != last}
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
    out, u, ptr = [seq[0]], seq[0], defaultdict(int)
    for _ in range(len(seq) - 1):
        nxt = order[u][ptr[u]]
        ptr[u] += 1
        out.append(nxt)
        u = nxt
    return "".join(out)
