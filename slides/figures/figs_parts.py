"""Figures added on 13.10 to replace text on the part decks (same style as figs.py).

    python figs_parts.py [name ...]      (in the hw1-00660121 env; writes ../assets/custom/<name>.png)
"""
import sys

import numpy as np
import torch
from matplotlib.patches import Rectangle

from deckstyle import *  # noqa: F401,F403
from intuition import split

MISTRAL = "RaphaelMourad/Mistral-DNA-v1-17M-hg38"
NT = "InstaDeepAI/nucleotide-transformer-500m-human-ref"


def bound_window(pos, site="UUUGCAC", left=12, right=18):
    s = next(x for x in pos if site in x[40:-40])
    k = s.index(site, 40)
    return s[k - left:k + right].replace("U", "T"), left


# ------------------------------------------------------------------ part 1
def bio_flow():
    """DNA -> mRNA -> protein, with AGO2 + microRNA binding the mRNA."""
    fig, ax = canvas(16, 6.2, ylim=(0, 62))
    xs = [18, 80, 142]
    names = ["Gene (DNA)", "Messenger RNA (mRNA)", "Protein"]
    letters = ["A C G T", "A C G U", ""]
    for x, n, l in zip(xs, names, letters):
        label(ax, x, 50, n, fs=30)
        if l:
            label(ax, x, 41, l, fs=24, color=MUTED, family=MONO)
    for (a, b), t in zip([(33, 54), (107, 128)], ["copied", "translated"]):
        arrow(ax, a, 50, b, 50, color=MUTED, lw=2, ms=24)
        label(ax, (a + b) / 2, 55, t, fs=22, color=MUTED)
    # AGO2 + microRNA below the mRNA
    label(ax, xs[1], 12, "AGO2 holding a microRNA (about 22 letters)", fs=26, color=ACCENT)
    arrow(ax, xs[1], 18, xs[1], 35, color=ACCENT, lw=2.5, ms=26)
    label(ax, xs[1] + 3, 26, "pairs with a site in the mRNA", fs=22, color=ACCENT, ha="left")
    label(ax, xs[2], 36, "less protein", fs=24, color=ACCENT)
    save(fig, "bio_flow")


def plant_scheme(with_control=True, name="plant_scheme"):
    """Where plant() writes copies 1 to 7 in a 330-letter sequence (and, optionally, the shuffled control)."""
    fig = newfig()
    fig.set_size_inches(16, 4.6)
    ax = fig.add_axes([0.22, 0.22, 0.75, 0.66])
    L, k = 330, 9
    ax.axvspan(150, 190, color=FAINT, zorder=0)
    ax.text(170, 2.62, "Measured site\n(letters 150 to 190)", ha="center", va="bottom", fontsize=19, color=MUTED)
    rows = [(1.6, "Motif", ACCENT)] + ([(0.5, "Same letters, shuffled", HELD_C)] if with_control else [])
    for y, row_name, c in rows:
        ax.plot([0, L], [y, y], color=HAIR, lw=8, solid_capstyle="butt", zorder=1)
        for n in range(1, 8):
            a = 40 + 45 * (n - 1)
            ax.add_patch(Rectangle((a, y - 0.22), k, 0.44, color=c, zorder=2))
            if y > 1:
                ax.text(a + k / 2, y + 0.38, str(n), ha="center", fontsize=19, color=MUTED)
        ax.text(-6, y, row_name, ha="right", va="center", fontsize=21, color=c)
    ax.set_xlim(-2, L + 2); ax.set_ylim(0 if with_control else 1.0, 3.4)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 50, 100, 150, 190, 250, 300, 330]); ax.set_xlabel("Position in a 330-letter random sequence")
    save(fig, name)


# ------------------------------------------------------------------ part 2
def _token_rows(rows, name, w, site):
    from transformers import AutoTokenizer
    h = 6 + 4.2 * len(rows)
    fig, ax = canvas(16, h, ylim=(0, h * 10))
    top = h * 10 - 10
    dx, x0 = 3.6, 36
    for i, ch in enumerate(w):
        label(ax, x0 + i * dx, top, ch, fs=24, color=BASE[ch], family=MONO, bold=True)
    label(ax, 4, top, "Sequence", fs=20, color=MUTED, ha="left")
    for r, (rname, repo, s_) in enumerate(rows):
        t = list(s_) if repo is None else None
        if repo is not None:
            tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
            t = tok.convert_ids_to_tokens(tok(s_, add_special_tokens=False)["input_ids"])
        y = top - 22 - r * 15
        label(ax, 4, y, rname, fs=19, color=MUTED, ha="left")
        p = 0 if s_ == w else 1
        for j, tk in enumerate(t):
            tk = tk.replace("▁", "").replace("Ġ", "")
            if not tk or tk[0] in "<[":
                continue
            box(ax, x0 + p * dx - 1.85, y - 4.5, len(tk) * dx - 0.45, 9, fc=[ACCENT_L, FAINT][j % 2], r=0.6)
            for q, ch in enumerate(tk):
                label(ax, x0 + (p + q) * dx, y, ch, fs=21, color=BASE.get(ch, INK), family=MONO, bold=True)
            p += len(tk)
        label(ax, x0 + len(w) * dx + 3, y, f"{sum(1 for x in t if x.replace('▁', '') and x[0] not in '<[')} tokens",
              fs=18, color=MUTED, ha="left")
    save(fig, name)


def tokens_split(pos):
    w, site = bound_window(pos)
    _token_rows([("Single letters", None, w)], "tokens_letters", w, site)
    _token_rows([("6-mers", NT, w), ("6-mers, shifted\nby one letter", NT, w[1:])], "tokens_kmer", w, site)
    _token_rows([("BPE (Mistral-DNA)", MISTRAL, w)], "tokens_bpe", w, site)


def emb_lookup(pos):
    """Real rows of Mistral-DNA's token-embedding table, next to the CNN's fixed one-hot table."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MISTRAL, trust_remote_code=True)
    fm = AutoModel.from_pretrained(MISTRAL, trust_remote_code=True).eval()
    E = fm.get_input_embeddings().weight.detach().float().numpy()
    w, _ = bound_window(pos)
    ids = tok(w, add_special_tokens=False)["input_ids"][:4]
    names = tok.convert_ids_to_tokens(ids)
    v = np.percentile(np.abs(E), 98)
    fig = newfig()
    fig.set_size_inches(16, 6.4)
    ax = fig.add_axes([0.20, 0.30, 0.56, 0.56])
    ax.imshow(E[ids], cmap=DIVERGING, vmin=-v, vmax=v, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(ids))); ax.set_yticklabels([f"{n}   row {i:,}" for n, i in zip(names, ids)], family=MONO)
    ax.set_xticks([0, 64, 128, 192, 255]); ax.set_xlabel("256 numbers per token")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"Mistral-DNA: {E.shape[0]:,} learned rows, four of them")
    a2 = fig.add_axes([0.83, 0.30, 0.13, 0.56])
    a2.imshow(np.eye(4), cmap=DIVERGING, vmin=-1, vmax=1, interpolation="nearest")
    a2.set_yticks(range(4)); a2.set_yticklabels(list("ACGU"), family=MONO); a2.set_xticks([])
    a2.tick_params(length=0)
    for sp in a2.spines.values():
        sp.set_visible(False)
    a2.set_title("The CNN: fixed")
    save(fig, "emb_lookup")


def blackbox(pos):
    """Real token vectors going in (embedding layer) and coming out (last layer) of Mistral-DNA."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MISTRAL, trust_remote_code=True)
    fm = AutoModel.from_pretrained(MISTRAL, trust_remote_code=True).eval()
    w, _ = bound_window(pos)
    enc = tok(w, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        hs = fm(**{k: enc[k] for k in ("input_ids", "attention_mask")}, output_hidden_states=True).hidden_states
    names = [t.replace("▁", "") for t in tok.convert_ids_to_tokens(enc["input_ids"][0])]
    H0, H8 = hs[0][0].numpy(), hs[-1][0].numpy()
    n = len(names)
    fig = newfig()
    fig.set_size_inches(16, 8.4)
    xs = np.linspace(0.22, 0.92, n)
    cw = min(0.08, 0.7 / n)

    def strip(x, y, vec):
        a = fig.add_axes([x - cw / 2, y, cw, 0.15])
        v = np.percentile(np.abs(vec), 97)
        a.imshow(vec.reshape(-1, 1)[:64], cmap=DIVERGING, vmin=-v, vmax=v, aspect="auto", interpolation="nearest")
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)

    for i, x in enumerate(xs):
        strip(x, 0.13, H0[i]); strip(x, 0.76, H8[i])
        fig.text(x, 0.07, names[i], ha="center", fontsize=19, family=MONO, color=INK)
    band = fig.add_axes([0.0, 0.36, 1.0, 0.32]); band.axis("off")
    band.add_patch(Rectangle((0, 0), 1, 1, color=FAINT, transform=band.transAxes))
    band.text(0.015, 0.5, "Transformer\n8 layers", ha="left", va="center", fontsize=26, color=MUTED,
              transform=band.transAxes)
    j = n // 2
    for x in xs:
        fig.add_artist(plt.Line2D([x, xs[j]], [0.30, 0.74], color=ACCENT, lw=1.2, alpha=0.7))
    fig.text(0.015, 0.20, "In", fontsize=22, color=MUTED, va="center")
    fig.text(0.015, 0.83, "Out", fontsize=22, color=MUTED, va="center")
    save(fig, "blackbox")


# ------------------------------------------------------------------ part 3
def adapt_layers():
    """Which parts are trained (accent) or frozen (grey) in four ways to adapt a pretrained model."""
    fig, ax = canvas(16, 8.4, ylim=(-8, 76))
    cols = [("Linear probe", "seconds to minutes", 0, False),
            ("Partial fine-tuning", "minutes", 3, False),
            ("Full fine-tuning", "minutes to days", 8, False),
            ("LoRA", "minutes", 0, True)]
    for c, (name, cost, top_trained, lora) in enumerate(cols):
        x = 22 + c * 35
        label(ax, x + 12, 71, name, fs=24)
        ax.add_patch(Rectangle((x, 58), 24, 4, color=ACCENT))
        for k in range(8):
            y = 50 - k * 5.4
            trained = k < top_trained
            ax.add_patch(Rectangle((x, y), 24, 3.6, color=ACCENT if trained else LIGHT))
            if lora:
                ax.add_patch(Rectangle((x + 26, y), 3, 3.6, color=ACCENT))
        label(ax, x + 12, 5, cost, fs=21, color=MUTED)
    label(ax, 4, 60, "Head", fs=20, color=MUTED, ha="left")
    label(ax, 4, 32, "8 layers", fs=20, color=MUTED, ha="left")
    ax.add_patch(Rectangle((60, -6), 4, 3, color=ACCENT)); label(ax, 66, -4.5, "trained", fs=20, color=MUTED, ha="left")
    ax.add_patch(Rectangle((84, -6), 4, 3, color=LIGHT)); label(ax, 90, -4.5, "frozen", fs=20, color=MUTED, ha="left")
    save(fig, "adapt_layers")


# ------------------------------------------------------------------ part 1, ML fundamentals
def setup_examples(model, tr, ty):
    """Four real training examples: x (start of the sequence), y, and the trained CNN's prediction."""
    from intuition import cnn_scores
    b = [i for i in range(len(ty)) if ty[i] == 1][:2]
    u = [i for i in range(len(ty)) if ty[i] == 0][:2]
    rows = [b[0], u[0], b[1], u[1]]
    p = cnn_scores(model, [tr[i] for i in rows])
    fig, ax = canvas(16, 5.6, ylim=(0, 56))
    label(ax, 4, 50, "$x$: sequence (first 32 letters)", fs=22, color=MUTED, ha="left")
    label(ax, 112, 50, "Length", fs=22, color=MUTED)
    label(ax, 130, 50, "$y$", fs=24, color=MUTED)
    label(ax, 149, 50, r"$\hat p = f_\theta(x)$", fs=24, color=MUTED)
    for r, (i, pi) in enumerate(zip(rows, p)):
        yy = 39 - r * 10
        label(ax, 4, yy, tr[i][:32] + " …", fs=22, family=MONO, ha="left")
        label(ax, 112, yy, f"{len(tr[i])}", fs=22)
        label(ax, 130, yy, f"{int(ty[i])}", fs=24, color=BOUND if ty[i] == 1 else MUTED, bold=True)
        label(ax, 149, yy, f"{pi:.2f}", fs=24)
    save(fig, "setup_examples")
    print("setup_examples", [int(ty[i]) for i in rows], [round(float(v), 3) for v in p])


def bce_curve():
    """Binary cross-entropy for one example, against the predicted probability of bound."""
    q = np.linspace(0.005, 0.995, 400)
    fig = newfig(); fig.set_size_inches(16, 7.4)
    ax = fig.add_axes([0.1, 0.15, 0.85, 0.75])
    ax.plot(q, -np.log(q), color=BOUND, lw=3.5)
    ax.plot(q, -np.log(1 - q), color=MID, lw=3.5)
    ax.text(0.10, 3.4, "True label bound ($y = 1$)\n$-\\log \\hat p$", color=BOUND, fontsize=22, va="center")
    ax.text(0.66, 3.4, "True label unbound ($y = 0$)\n$-\\log (1 - \\hat p)$", color=MUTED, fontsize=22, va="center")
    ax.plot([0.5], [np.log(2)], "o", color=INK, ms=10)
    ax.annotate("0.69 at $\\hat p = 0.5$", (0.5, np.log(2)), xytext=(0.53, 1.25), fontsize=21, color=INK)
    ax.set_xlabel(r"Predicted probability of bound, $\hat p$"); ax.set_ylabel("Loss")
    ax.set_xlim(0, 1); ax.set_ylim(0, 4.2); ygrid(ax)
    save(fig, "bce_curve")


def lr_rates(tr, ty):
    """Loss per step of the G+C/U logistic regression (same 3,000 sequences and start as lr_training), three learning rates."""
    from intuition2 import gc
    rng = np.random.default_rng(1)
    idx = rng.choice(len(tr), 3000, replace=False)
    X = np.array([[gc(tr[i]), tr[i].count("U") / len(tr[i])] for i in idx]); y = ty[idx].astype(float)
    Z = (X - X.mean(0)) / X.std(0)
    fig = newfig(); fig.set_size_inches(16, 7.4)
    ax = fig.add_axes([0.1, 0.15, 0.66, 0.75])
    for lr, c, lw in [(0.05, MID, 3.5), (2.0, GOOD_C, 3.5), (10.0, HELD_C, 2.2)]:
        w, b, L = np.array([-1.0, -1.0]), 0.8, []
        for _ in range(41):
            p = 1 / (1 + np.exp(-(Z @ w + b)))
            L.append(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))
            g = p - y; w = w - lr * Z.T @ g / len(y); b = b - lr * g.mean()
        ax.plot(range(len(L)), L, color=c, lw=lw, marker="o" if lr == 10 else None, ms=5)
        name = {0.05: "too small", 2.0: "good", 10.0: "too large"}[lr]
        ypos = {0.05: L[-1] + 0.03, 2.0: L[-1] - 0.03, 10.0: max(L) * 0.9}[lr]
        ax.text(41.5, ypos, f"$\\eta$ = {lr:g}: {name}", color=MUTED if lr == 0.05 else c, fontsize=21, va="center")
        print("lr", lr, round(L[0], 3), round(L[-1], 3), round(max(L), 3))
    ax.set_xlabel("Step"); ax.set_ylabel("Training loss"); ax.set_xlim(0, 40); ygrid(ax)
    save(fig, "lr_rates")


def sgd_noise(tr, ty):
    """The CNN's loss on every mini-batch of 256 during 10 epochs (as in the notebook), and the epoch means."""
    from torch import nn
    from intuition import DEV, GlobalCNN, onehot
    torch.manual_seed(0)
    model = GlobalCNN().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    X = onehot(tr); y = torch.tensor(ty, dtype=torch.float32)
    losses, n_per = [], len(X) // 256
    for ep in range(10):
        model.train(); perm = torch.randperm(len(X))
        for i in range(n_per):
            bi = perm[i * 256:(i + 1) * 256]
            opt.zero_grad(); loss = crit(model(X[bi].to(DEV)), y[bi].to(DEV)); loss.backward(); opt.step()
            losses.append(loss.item())
    L = np.array(losses)
    fig = newfig(); fig.set_size_inches(16, 7.4)
    ax = fig.add_axes([0.1, 0.15, 0.66, 0.75])
    ax.plot(L, color=TRAIN_L, lw=0.8)
    m = L.reshape(10, n_per).mean(1)
    ax.plot((np.arange(10) + 0.5) * n_per, m, color=TRAIN_C, lw=3, marker="o", ms=9)
    for e in range(1, 10):
        ax.axvline(e * n_per, color=HAIR, lw=1, zorder=0)
    ax.set_xticks([e * n_per for e in range(0, 11, 2)]); ax.set_xticklabels([f"{e * n_per:,}" for e in range(0, 11, 2)])
    ax.set_xlabel(f"Step (one epoch = {n_per} steps)"); ax.set_ylabel("Training loss"); ax.set_xlim(0, 10 * n_per)
    ax.text(10.15 * n_per, m[-1] - 0.02, "Mean per epoch", color=TRAIN_C, fontsize=21, va="center")
    ax.text(10.15 * n_per, np.percentile(L[-n_per:], 99), "One batch\n(256 sequences)", color=MUTED, fontsize=21, va="center")
    save(fig, "sgd_noise")
    print("sgd_noise steps/epoch", n_per, "epoch means", [round(v, 3) for v in m], "batch range last epoch",
          round(L[-n_per:].min(), 3), round(L[-n_per:].max(), 3))


def nn_diagram():
    """Logistic regression as one unit, and a network with one hidden layer of ReLU units."""
    from matplotlib.patches import Circle
    fig, ax = canvas(16, 7.4, ylim=(0, 74))

    def node(x, y, txt="", r=3.4, fc=FAINT):
        ax.add_patch(Circle((x, y), r, fc=fc, ec=MUTED, lw=1.2, zorder=3))
        if txt:
            label(ax, x, y, txt, fs=17)

    def edge(a, b):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=LIGHT, lw=1.4, zorder=1)

    label(ax, 30, 69, "Logistic regression: one unit", fs=24)
    ins = [(14, 46), (14, 26)]
    for (x, y), n in zip(ins, ["G+C", "U"]):
        node(x, y); label(ax, x - 5, y, n, fs=20, color=MUTED, ha="right")
    out = (40, 36)
    for a in ins:
        edge(a, out)
    node(*out, r=4.2, fc=ACCENT_L); label(ax, out[0], out[1], r"$\sigma$", fs=24)
    arrow(ax, 45, 36, 51, 36); label(ax, 53, 36, r"$\hat p$", fs=24, ha="left")
    label(ax, 30, 12, r"$\hat p = \sigma(w_1 x_1 + w_2 x_2 + b)$", fs=22, color=MUTED)

    x0 = 88
    label(ax, 118, 69, "A neural network: layers of units", fs=24)
    ins = [(x0, 51 - 10 * k) for k in range(4)]
    hid = [(x0 + 28, 56 - 10 * k) for k in range(5)]
    out = (x0 + 56, 36)
    for a in ins:
        for h in hid:
            edge(a, h)
    for h in hid:
        edge(h, out)
    for a in ins:
        node(*a)
    for h in hid:
        node(*h, txt="ReLU", r=3.9)
    node(*out, r=4.2, fc=ACCENT_L); label(ax, out[0], out[1], r"$\sigma$", fs=24)
    arrow(ax, out[0] + 5, 36, out[0] + 10, 36); label(ax, out[0] + 12, 36, r"$\hat p$", fs=24, ha="left")
    label(ax, x0, 9, "Input", fs=20, color=MUTED)
    label(ax, x0 + 28, 9, "Hidden layer", fs=20, color=MUTED)
    label(ax, x0 + 56, 9, "Output", fs=20, color=MUTED)
    save(fig, "nn_diagram")


def channels_pool(model, pos):
    """All 10 first-layer channels on one bound sequence after ReLU, and after max pooling over 3 positions."""
    from intuition import acts
    A = acts(model, pos[:2000])
    i = int(np.argmax(A[:, 7, :].max(1)))
    full = A[i][:, :len(pos[i]) - 9]           # positions inside the sequence; the rest is padding
    j = int(np.argmax(full[7]))
    lo = max(0, min(j - 60, full.shape[1] - 120)) // 3 * 3
    a = full[:, lo:lo + 120]                    # a 120-position window, aligned to the pooling blocks
    P = a.reshape(10, 40, 3).max(2)
    v = np.percentile(full, 99.5)
    fig = newfig(); fig.set_size_inches(16, 8.4)
    for k, (M, ttl) in enumerate([(a, "First convolution + ReLU: one value per position"),
                                  (P, "After max pooling: the largest of every 3 positions")]):
        ax = fig.add_axes([0.08, 0.56 - k * 0.45, 0.9, 0.34])
        ax.imshow(M, cmap=SEQUENTIAL, vmin=0, vmax=v, aspect="auto", interpolation="nearest",
                  extent=(lo, lo + 120, 9.5, -0.5))
        ax.set_yticks(range(10)); ax.set_yticklabels([str(f) for f in range(10)], fontsize=18)
        ax.set_ylabel("Filter"); ax.set_title(ttl)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    ax.set_xlabel("Position along one bound sequence")
    save(fig, "channels_pool")
    print("channels_pool seq len", len(pos[i]), "window", lo, lo + 120, "filter 7 peak at", j)


# ------------------------------------------------------------------ parts 2 and 3, added 14.10
def pad_mask(pos):
    """Three real sequences of different lengths, tokenized as one batch: tokens, padding and the attention mask."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MISTRAL, trust_remote_code=True)
    seqs = [pos[0][150:190], pos[1][150:172], pos[2][150:181]]
    seqs = [s.replace("U", "T") for s in seqs]
    enc = tok(seqs, padding=True, add_special_tokens=True)
    rows = [tok.convert_ids_to_tokens(ids) for ids in enc["input_ids"]]
    n = len(rows[0])
    fig, ax = canvas(16, 6.6, ylim=(0, 66))
    x0, cw = 26, 112 / n
    label(ax, 4, 60, "Tokens", fs=21, color=MUTED, ha="left")
    label(ax, x0 + n * cw + 2, 60, "Mask", fs=21, color=MUTED, ha="left")
    for r, (toks, m) in enumerate(zip(rows, enc["attention_mask"])):
        y = 50 - r * 15
        label(ax, 4, y, f"Sequence {r + 1}", fs=20, color=MUTED, ha="left")
        for j, (tk, mj) in enumerate(zip(toks, m)):
            pad = mj == 0
            box(ax, x0 + j * cw, y - 4.5, cw - 0.5, 9, fc=ACCENT_L if not pad else "#FFFFFF",
                ec=HAIR if pad else "none", lw=1 if pad else 0, r=0.5)
            t = "pad" if pad else tk.replace("▁", "")
            label(ax, x0 + j * cw + cw / 2, y + 1.6, t, fs=12 if len(t) > 5 else 14, color=LIGHT if pad else INK, family=MONO)
            label(ax, x0 + j * cw + cw / 2, y - 2.6, str(mj), fs=14, color=ACCENT if mj else LIGHT, bold=bool(mj))
        label(ax, x0 + n * cw + 2, y, f"{sum(m)} real", fs=19, color=ACCENT, ha="left")
    label(ax, x0 + n * cw / 2, 4, "Sequence embedding = mean of the last-layer vectors where the mask is 1",
          fs=20, color=MUTED)
    save(fig, "pad_mask")
    print("pad_mask", [len(s) for s in seqs], [sum(m) for m in enc["attention_mask"]], "padded to", n)


def probe_pipeline():
    """Real Mistral-DNA sequence embeddings (rows) into a standardiser and a logistic regression."""
    from pathlib import Path
    d = np.load(Path(__file__).resolve().parents[2] / "data" / "emb_mistral17m.npz")
    E = d["train"][:40]
    Z = (E - d["train"].mean(0)) / d["train"].std(0)
    fig = newfig(); fig.set_size_inches(16, 6.4)
    ax = fig.add_axes([0.04, 0.2, 0.36, 0.62])
    ax.imshow(Z, cmap=DIVERGING, vmin=-2.5, vmax=2.5, aspect="auto", interpolation="nearest")
    ax.set_xticks([0, 255]); ax.set_xticklabels(["1", "256"]); ax.set_yticks([])
    ax.set_xlabel("256 numbers per sequence"); ax.set_ylabel("Sequences (40 of 12,000)")
    ax.set_title("Sequence embeddings, standardised"); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    a2 = fig.add_axes([0.42, 0.0, 0.58, 1.0]); a2.axis("off"); a2.set_xlim(0, 108); a2.set_ylim(0, 64)
    arrow(a2, 2, 33, 12, 33, lw=2, ms=22)
    label(a2, 34, 37, r"$\hat p = \sigma(w^\top e + b)$", fs=26)
    label(a2, 34, 28, "256 weights + 1 bias", fs=20, color=MUTED)
    arrow(a2, 56, 33, 66, 33, lw=2, ms=22)
    label(a2, 82, 37, "Held-out AUROC", fs=22)
    label(a2, 82, 28, "?", fs=30, color=ACCENT, bold=True)
    label(a2, 34, 52, "Logistic regression (part 1, slide 2.12)", fs=20, color=MUTED)
    save(fig, "probe_pipeline")


def word_counts(pos):
    """One real sequence, its 5-letter words, and its 1,024 word counts."""
    from collections import Counter
    from itertools import product
    s = pos[3]
    words = [s[i:i + 5] for i in range(len(s) - 4)]
    c = Counter(words)
    vocab = ["".join(p) for p in product("ACGU", repeat=5)]
    v = np.array([c.get(w, 0) for w in vocab])
    fig = newfig(); fig.set_size_inches(16, 7.6)
    ax0 = fig.add_axes([0.0, 0.62, 1.0, 0.36]); ax0.axis("off"); ax0.set_xlim(0, 160); ax0.set_ylim(0, 36)
    show = s[:24]
    dx, x0 = 3.6, 6
    for i, ch in enumerate(show):
        label(ax0, x0 + i * dx, 30, ch, fs=22, family=MONO)
    label(ax0, x0 + 24 * dx + 2, 30, f"… ({len(s)} letters)", fs=19, color=MUTED, ha="left")
    for k in range(4):
        yk = 22 - k * 5.5
        ax0.plot([x0 + k * dx - 1.4, x0 + (k + 4) * dx + 1.4], [yk, yk], color=ACCENT if k == 0 else LIGHT, lw=3)
        label(ax0, x0 + (k + 5) * dx + 2, yk, s[k:k + 5], fs=17, family=MONO, color=ACCENT if k == 0 else MUTED, ha="left")
    label(ax0, 120, 14, f"{len(words)} overlapping words", fs=21, color=MUTED)
    ax = fig.add_axes([0.07, 0.12, 0.9, 0.42])
    ax.set_ylim(0, v.max() + 1.5)
    ax.bar(range(1024), v, width=1.0, color=INK)
    top = np.argsort(-v)[:3]
    for t in top:
        ax.annotate(vocab[t], (t, v[t]), xytext=(t - 30, v[t] + 0.2), fontsize=18, family=MONO, color=INK, ha="right",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.set_xlim(-2, 1026); ax.set_xticks([0, 1023]); ax.set_xticklabels(["AAAAA", "UUUUU"], family=MONO)
    ax.set_ylabel("Count"); ax.set_xlabel("All 1,024 five-letter words, in alphabetical order"); ygrid(ax)
    save(fig, "word_counts")
    print("word_counts len", len(s), "words", len(words), "distinct", int((v > 0).sum()), "max", int(v.max()),
          [vocab[t] for t in top])



def cnn_template():
    """The building blocks of the CNN: [convolution -> ReLU -> max pooling] (repeat) -> flatten -> linear."""
    fig, ax = canvas(16, 5.6, ylim=(0, 56))
    y0 = 20

    def grid(x, w, rows, h=2.2, gap=0.6, fill=FAINT, edge=MID):
        for r in range(rows):
            ax.add_patch(Rectangle((x, y0 + r * (h + gap)), w, h, fc=fill, ec=edge, lw=0.8))
        return y0 + rows * (h + gap) - gap

    top = grid(2, 16, 4, h=2.6)                        # one-hot input: 4 letter rows
    label(ax, 10, y0 - 5, "one-hot input\n4 letters × length", fs=17, color=MUTED)
    grid(28, 14, 6)                                     # convolution: one row per filter (channel)
    label(ax, 35, y0 - 5, "convolution\none row per filter", fs=17, color=MUTED)
    grid(50, 14, 6, fill="white")                       # ReLU: same shape
    label(ax, 57, y0 - 5, "ReLU\nnegatives to 0", fs=17, color=MUTED)
    grid(78, 6, 6)                                      # max pooling: shorter rows
    label(ax, 81, y0 - 5, "max pooling\nshorter rows", fs=17, color=MUTED)
    ax.add_patch(Rectangle((90, y0 + 6.5), 38, 2.2, fc=FAINT, ec=MID, lw=0.8))   # flatten: one long row
    label(ax, 109, y0 - 5, "flatten\nall numbers in one row", fs=17, color=MUTED)
    ax.add_patch(plt.Circle((148, y0 + 7.6), 1.8, color=INK))                     # linear: one number
    label(ax, 148, y0 - 5, "linear\none score (logit)", fs=17, color=MUTED)
    for a, b in ((19, 27), (43, 49), (65, 77), (85, 89), (129, 145)):
        arrow(ax, a, y0 + 7.6, b, y0 + 7.6, color=MUTED, lw=1.5, ms=18)
    # the repeatable block
    ax.plot([26, 26, 86, 86], [43, 46, 46, 43], color=INK, lw=1.4)
    label(ax, 56, 50, "repeat as often as you like", fs=19, color=INK)
    save(fig, "cnn_template")


if __name__ == "__main__":
    want = set(sys.argv[1:])
    run = (lambda n: not want or n in want)
    tr, ty = split("train")
    pos = [s for s, l in zip(tr, ty) if l == 1]
    model = None
    if not want or want & {"setup_examples", "channels_pool"}:
        from intuition import load_cnn
        model = load_cnn()
    for n, job in [("pad_mask", lambda: pad_mask(pos)), ("probe_pipeline", probe_pipeline),
                   ("word_counts", lambda: word_counts(pos)),
                   ("setup_examples", lambda: setup_examples(model, tr, ty)), ("bce_curve", bce_curve),
                   ("lr_rates", lambda: lr_rates(tr, ty)), ("sgd_noise", lambda: sgd_noise(tr, ty)),
                   ("nn_diagram", nn_diagram), ("channels_pool", lambda: channels_pool(model, pos)),
                   ("bio_flow", bio_flow), ("plant_scheme", plant_scheme), ("plant_scheme_motif", lambda: plant_scheme(False, "plant_scheme_motif")), ("tokens_split", lambda: tokens_split(pos)),
                   ("emb_lookup", lambda: emb_lookup(pos)), ("blackbox", lambda: blackbox(pos)),
                   ("adapt_layers", adapt_layers), ("cnn_template", cnn_template)]:
        if run(n):
            job()

