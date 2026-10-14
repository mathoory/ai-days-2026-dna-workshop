"""Mechanism figures from the real data and the real models (no cartoons).

Writes to slides/assets/custom/:
  filter_dot        a learned 4x10 filter, a real one-hot window, their product and the sum
  filter_learning   one filter at epochs 0, 1, 3, 10 of a fresh training run
  activation_track  one bound sequence: a filter's output at every position
  emb_pca           13k real Mistral-DNA embeddings, PCA, coloured by GC and by label
  shuffle_scores    model scores for real bound / shuffled bound / unbound (CNN and probe)
  token_split       the real Mistral-DNA tokenizer on a sequence holding a miR-16 site
Run in an env with torch + transformers (hw1-00660121 locally).
"""
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

from deckstyle import *  # noqa: F401,F403

ROOT_REPO = Path(__file__).resolve().parents[2]
DATA = ROOT_REPO / "data"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPH = "ACGU"
LUT = {b: i for i, b in enumerate(ALPH)}


def read(path):
    return [l.strip().upper().replace("T", "U") for l in open(path) if not l.startswith(">")]


def split(name):
    p, n = read(DATA / f"CLIPSEQ_AGO2.{name}.positives.fa"), read(DATA / f"CLIPSEQ_AGO2.{name}.negatives.fa")
    return p + n, np.array([1] * len(p) + [0] * len(n))


def onehot(seqs, L=501):
    X = np.full((len(seqs), 1, 4, L), 0.25, dtype=np.float32)
    for b, s in enumerate(seqs):
        for i, ch in enumerate(s[:L]):
            if ch in LUT:
                X[b, 0, :, i] = 0
                X[b, 0, LUT[ch], i] = 1
    return torch.from_numpy(X)


class GlobalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 10, (4, 10)), nn.ReLU(), nn.MaxPool2d((1, 3)),
                                 nn.Conv2d(10, 20, (1, 5)), nn.ReLU(), nn.MaxPool2d((1, 3)),
                                 nn.Flatten(), nn.Linear(20 * 53, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


def load_cnn():
    m = GlobalCNN().to(DEV)
    m.load_state_dict(torch.load(DATA / "cnn.pt", map_location=DEV))
    return m.eval()


@torch.no_grad()
def acts(model, seqs):
    conv = model.net[0]
    out = []
    for i in range(0, len(seqs), 512):
        out.append(torch.relu(conv(onehot(seqs[i:i + 512]).to(DEV)))[:, :, 0, :].cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def cnn_scores(model, seqs):
    return np.concatenate([torch.sigmoid(model(onehot(seqs[i:i + 1024]).to(DEV))).cpu().numpy()
                           for i in range(0, len(seqs), 1024)])


def pick_filter(model, pos, neg):
    """The filter whose strongest hit differs most between bound and unbound sequences."""
    ap, an = acts(model, pos).max(2).mean(0), acts(model, neg).max(2).mean(0)
    return int(np.argmax(ap - an)), ap, an


def heat(ax, W, title, cmap="RdBu_r", vmax=None, annotate=False, fmt="{:+.1f}"):
    vmax = vmax or np.abs(W).max()
    ax.imshow(W, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(range(4)); ax.set_yticklabels(list(ALPH), fontsize=20, fontweight="bold")
    for t, b in zip(ax.get_yticklabels(), ALPH):
        t.set_color(BASE[b])
    ax.set_xticks([]); ax.set_title(title, fontsize=22, color=NAVY)
    if annotate:
        for r in range(W.shape[0]):
            for c in range(W.shape[1]):
                ax.text(c, r, fmt.format(W[r, c]), ha="center", va="center", fontsize=13,
                        color="white" if abs(W[r, c]) > 0.55 * vmax else NAVY)


# ------------------------------------------------------------------------------------------
def filter_dot(model, pos, neg):
    f, ap, an = pick_filter(model, pos, neg)
    W = model.net[0].weight[f, 0].detach().cpu().numpy()
    b = float(model.net[0].bias[f])
    A = acts(model, pos[:3000])[:, f, :]
    i, j = np.unravel_index(np.argmax(A), A.shape)
    win = pos[i][j:j + 10]
    X = onehot([win], L=10)[0, 0].numpy()
    P = W * X
    fig = plt.figure(figsize=(16, 9), dpi=120)
    a1 = fig.add_axes([0.05, 0.55, 0.27, 0.3]); a2 = fig.add_axes([0.37, 0.55, 0.27, 0.3])
    a3 = fig.add_axes([0.69, 0.55, 0.27, 0.3])
    heat(a1, W, f"filter {f:02d}: 4 × 10 learned weights", annotate=True)
    a2.imshow(X, cmap="Greys", vmin=0, vmax=1.6, aspect="auto")
    a2.set_yticks(range(4)); a2.set_yticklabels(list(ALPH), fontsize=20, fontweight="bold")
    a2.set_xticks(range(10)); a2.set_xticklabels(list(win), fontsize=20, fontweight="bold")
    for t, ch in zip(a2.get_xticklabels(), win):
        t.set_color(BASE[ch])
    a2.set_title("a real bound window", fontsize=22, color=NAVY)
    heat(a3, P, "weight × letter", annotate=True, vmax=np.abs(W).max())
    fig.text(0.345, 0.70, "×", fontsize=48, ha="center", va="center", color=NAVY)
    fig.text(0.665, 0.70, "=", fontsize=48, ha="center", va="center", color=NAVY)
    tot = P.sum() + b
    fig.text(0.5, 0.33, f"sum of the 10 picked weights + bias = {P.sum():+.2f} {b:+.2f} = {tot:+.2f}",
             ha="center", fontsize=28, color=NAVY, fontweight="bold")
    fig.text(0.5, 0.21, "the same 40 weights slide along all 501 positions; a high sum = a match",
             ha="center", fontsize=23, color=GREY)
    fig.text(0.5, 0.10, f"filter {f:02d}'s best hit is {ap[f] / max(an[f], 1e-6):.2f}× stronger on bound than on unbound RNA",
             ha="center", fontsize=21, color=BLUE)
    save(fig, "filter_dot")
    return f


def filter_learning(f_hint, tr_seqs, tr_y, val_seqs, val_y):
    torch.manual_seed(0)
    model = GlobalCNN().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    X = onehot(tr_seqs); y = torch.tensor(tr_y, dtype=torch.float32)
    snaps, aucs = {0: model.net[0].weight.detach().cpu().numpy().copy()}, {}
    from sklearn.metrics import roc_auc_score
    aucs[0] = roc_auc_score(val_y, cnn_scores(model, val_seqs))
    for ep in range(1, 11):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 256):
            bi = perm[i:i + 256]
            opt.zero_grad()
            crit(model(X[bi].to(DEV)), y[bi].to(DEV)).backward()
            opt.step()
        model.eval()
        if ep in (1, 3, 10):
            snaps[ep] = model.net[0].weight.detach().cpu().numpy().copy()
            aucs[ep] = roc_auc_score(val_y, cnn_scores(model, val_seqs))
        print("epoch", ep, flush=True)
    # the filter that changed most from its random start
    d = np.abs(snaps[10] - snaps[0]).reshape(10, -1).sum(1)
    f = int(np.argmax(d))
    fig, axes = plt.subplots(1, 4, figsize=(16, 9), dpi=120)
    vmax = max(np.abs(snaps[e][f, 0]).max() for e in snaps)
    for ax, e in zip(axes, (0, 1, 3, 10)):
        heat(ax, snaps[e][f, 0], f"epoch {e}\nAUROC {aucs[e]:.2f}", vmax=vmax)
    fig.text(0.5, 0.12, "one filter, from random numbers to a preference for particular letters at particular positions",
             ha="center", fontsize=22, color=NAVY)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.78, bottom=0.25, wspace=0.18)
    save(fig, "filter_learning")


def activation_track(model, f, pos):
    A = acts(model, pos[:2000])[:, f, :]
    i = int(np.argmax(A.max(1)))
    s, a = pos[i], A[i][:len(pos[i]) - 9]
    fig = plt.figure(figsize=(16, 9), dpi=120)
    ax = fig.add_axes([0.08, 0.3, 0.89, 0.5])
    ax.bar(range(len(a)), a, color=[RED if v > 0.6 * a.max() else "#bdbdbd" for v in a], width=1.0)
    ax.set_xlim(-1, len(a)); ax.set_ylabel(f"filter {f:02d} output", fontsize=22)
    ax.set_xlabel("position along one bound sequence", fontsize=22); ax.tick_params(labelsize=18)
    j = int(np.argmax(a))
    ax.annotate(s[j:j + 10], (j, a[j]), xytext=(j + 25 if j < 0.6 * len(a) else j - 120, a[j] * 0.95), fontsize=26, fontweight="bold",
                family="DejaVu Sans Mono", color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=2))
    fig.text(0.5, 0.88, "one filter, slid along one real sequence: mostly quiet, a few sharp hits",
             ha="center", fontsize=24, color=NAVY, fontweight="bold")
    save(fig, "activation_track")


def emb_pca(all_seqs, all_y):
    d = np.load(DATA / "emb_mistral17m.npz")
    E = d["train"]
    idx = np.random.default_rng(0).permutation(len(all_seqs))[:12000]   # workshop_data()
    S = [all_seqs[i] for i in idx]; y = all_y[idx]
    gc = np.array([(s.count("G") + s.count("C")) / len(s) for s in S])
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    P = PCA(2).fit_transform(StandardScaler().fit_transform(E))
    r = np.corrcoef(P[:, 0], gc)[0, 1]
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), dpi=120)
    sc = axes[0].scatter(P[:, 0], P[:, 1], c=gc, cmap="viridis", s=5, alpha=.7)
    cb = fig.colorbar(sc, ax=axes[0], fraction=0.046); cb.set_label("G + C fraction", fontsize=20)
    axes[0].set_title("coloured by G + C", fontsize=24, color=NAVY); axes[0].set_ylabel("direction 2", fontsize=20)
    o = np.random.default_rng(1).permutation(len(y))
    axes[1].scatter(P[o, 0], P[o, 1], c=np.where(y[o] == 1, GREEN, "#9e9e9e"), s=5, alpha=.5)
    axes[1].set_title("coloured by bound / unbound", fontsize=24, color=NAVY)
    for ax in axes:
        ax.set_xlabel("direction 1", fontsize=20)
        ax.set_xticks([]); ax.set_yticks([])
    fig.text(0.5, 0.04, f"12,000 real Mistral-DNA vectors · direction 1 vs G + C: r = {r:+.2f}",
             ha="center", fontsize=22, color=NAVY)
    fig.subplots_adjust(left=0.04, right=0.97, top=0.88, bottom=0.14, wspace=0.12)
    save(fig, "emb_pca")
    return r


def dinuc_shuffle(seq, rng):
    from collections import defaultdict
    if len(seq) < 3:
        return seq
    edges = defaultdict(list)
    for a, b in zip(seq[:-1], seq[1:]):
        edges[a].append(b)
    last = seq[-1]
    while True:
        le = {v: edges[v][rng.integers(len(edges[v]))] for v in edges if v != last}
        ok = True
        for v in le:
            seen, u = set(), v
            while u != last:
                if u in seen or u not in le:
                    ok = False; break
                seen.add(u); u = le[u]
            if not ok:
                break
        if ok:
            break
    order = {}
    for v, lst in edges.items():
        lst = list(lst)
        if v in le:
            lst.remove(le[v]); rng.shuffle(lst); lst.append(le[v])
        else:
            rng.shuffle(lst)
        order[v] = lst
    out, u, ptr = [seq[0]], seq[0], defaultdict(int)
    for _ in range(len(seq) - 1):
        nxt = order[u][ptr[u]]; ptr[u] += 1; out.append(nxt); u = nxt
    return "".join(out)


def shuffle_scores(model, all_seqs, all_y, ls_seqs, ls_y):
    rng = np.random.default_rng(0)
    bound = [s for s, l in zip(ls_seqs, ls_y) if l == 1]
    unbound = [s for s, l in zip(ls_seqs, ls_y) if l == 0]
    shuf = [dinuc_shuffle(s, rng) for s in bound]
    cs = {"bound": cnn_scores(model, bound), "bound, shuffled": cnn_scores(model, shuf),
          "unbound": cnn_scores(model, unbound)}
    # probe on Mistral-DNA embeddings: fit on the shipped train vectors, embed the test sets here
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from transformers import AutoModel, AutoTokenizer
    d = np.load(DATA / "emb_mistral17m.npz")
    idx = np.random.default_rng(0).permutation(len(all_seqs))[:12000]
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1)).fit(d["train"], all_y[idx])
    name = "RaphaelMourad/Mistral-DNA-v1-17M-hg38"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    fm = AutoModel.from_pretrained(name, trust_remote_code=True).to(DEV).eval()

    @torch.no_grad()
    def emb(S):
        out = []
        for i in range(0, len(S), 128):
            e = tok([s.replace("U", "T") for s in S[i:i + 128]], return_tensors="pt", padding=True, truncation=True, max_length=512)
            ids, m = e["input_ids"].to(DEV), e["attention_mask"].to(DEV)
            h = fm(input_ids=ids, attention_mask=m, output_hidden_states=True).hidden_states[-1].float()
            mk = m.unsqueeze(-1).float()
            out.append(((h * mk).sum(1) / mk.sum(1)).cpu().numpy())
        return np.concatenate(out)

    ps = {k: probe.predict_proba(emb(v))[:, 1] for k, v in (("bound", bound), ("bound, shuffled", shuf), ("unbound", unbound))}
    from sklearn.metrics import roc_auc_score
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), dpi=120, sharey=False)
    cols = {"bound": GREEN, "bound, shuffled": "#ef6c00", "unbound": "#9e9e9e"}
    for ax, sc, t in ((axes[0], cs, "CNN (part 1)"), (axes[1], ps, "Mistral-DNA + probe (part 2)")):
        bins = np.linspace(0, 1, 40)
        for k in ("unbound", "bound", "bound, shuffled"):
            ax.hist(sc[k], bins, color=cols[k], alpha=.5 if k != "bound, shuffled" else .65, label=k, density=True,
                    histtype="stepfilled" if k != "bound, shuffled" else "step", lw=3)
        auc = roc_auc_score([1] * len(sc["bound, shuffled"]) + [0] * len(sc["unbound"]),
                            np.concatenate([sc["bound, shuffled"], sc["unbound"]]))
        ax.set_title(f"{t}\nshuffled vs unbound: AUROC {auc:.2f}", fontsize=22, color=NAVY)
        ax.set_xlabel("model score", fontsize=20); ax.set_yticks([]); ax.tick_params(labelsize=18)
        ax.legend(fontsize=18, frameon=False)
    fig.subplots_adjust(left=0.03, right=0.98, top=0.83, bottom=0.12, wspace=0.12)
    save(fig, "shuffle_scores")


def token_split():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("RaphaelMourad/Mistral-DNA-v1-17M-hg38", trust_remote_code=True)
    s = "AUCAAUUGCUGCUAAGUCCAGGAAUGCUUA".replace("U", "T")
    site = s.index("TGCTGCT")
    toks = [t for t in tok.convert_ids_to_tokens(tok(s)["input_ids"]) if t not in ("[CLS]", "[SEP]", "<s>", "</s>")]
    fig, ax = canvas()
    x0, dx = 8, 4.6
    label(ax, 80, 84, "the miR-16 target site TGCTGCT, through Mistral-DNA's real tokenizer", fs=24, bold=True)
    # letters with the site marked
    for i, ch in enumerate(s):
        label(ax, x0 + i * dx, 66, ch, fs=26, color=BASE[ch], bold=True)
    ax.add_patch(plt.Rectangle((x0 + site * dx - 2.3, 61), 7 * dx, 10, fc="none", ec=PURPLE, lw=3.5))
    label(ax, x0 + (site + 3) * dx, 76, "microRNA site", fs=19, color=PURPLE, bold=True)
    # tokens
    pos = 0
    for k, t in enumerate(toks):
        w = len(t) * dx
        box(ax, x0 + pos * dx - 2.2, 40, w - 0.5, 10, "", fc=[PURPLE_L, BLUE_L][k % 2], ec=[PURPLE, BLUE][k % 2], lw=2, r=0.6)
        for j, ch in enumerate(t):
            label(ax, x0 + (pos + j) * dx, 45, ch, fs=22, color=BASE.get(ch, NAVY), bold=True)
        pos += len(t)
    label(ax, 80, 30, " | ".join(toks), fs=19, color=GREY)
    label(ax, 80, 14, "the model never sees the site as one unit", fs=24, color=RED, bold=True)
    save(fig, "token_split")
    print("tokens:", toks, "site at", site)


if __name__ == "__main__":
    which = sys.argv[1:] or ["dot", "learn", "track", "pca", "shuffle", "tokens"]
    tr, ty = split("train"); ls, ly = split("ls")
    pos = [s for s, l in zip(tr, ty) if l == 1][:6000]
    neg = [s for s, l in zip(tr, ty) if l == 0][:6000]
    model = load_cnn()
    f = None
    if "dot" in which or "track" in which:
        f = filter_dot(model, pos, neg)
    if "track" in which:
        activation_track(model, f, pos)
    if "learn" in which:
        filter_learning(f, tr, ty, ls, ly)
    if "pca" in which:
        print("PC1 vs GC r =", emb_pca(tr, ty))
    if "shuffle" in which:
        shuffle_scores(model, tr, ty, ls, ly)
    if "tokens" in which:
        token_split()
