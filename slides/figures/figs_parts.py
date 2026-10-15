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
    # the 30 letters wrap into two lines of 15; each line shows the sequence, then its tokens in boxes.
    # A token that crosses the wrap is drawn as two pieces of the same colour, one at each line end.
    from transformers import AutoTokenizer
    rname, repo, s_ = rows[0]
    t = list(s_) if repo is None else None
    if repo is not None:
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        t = tok.convert_ids_to_tokens(tok(s_, add_special_tokens=False)["input_ids"])
    t = [x.replace("▁", "").replace("Ġ", "") for x in t]
    t = [x for x in t if x and x[0] not in "<["]
    assert "".join(t) == w, (t, w)
    per = (len(w) + 1) // 2
    W, H = 104, 50
    fig, ax = canvas(W / 10, H / 10, xlim=(0, W), ylim=(0, H))
    fs, dx, x0, bh = 44, 4.6, 36, 9.5          # letter size, letter pitch, first letter x, box height
    lines = [H - 4, H - 4 - 22.5]                # sequence row of each line; its token row sits 10 below
    for L, ys in enumerate(lines):
        for i, ch in enumerate(w[L * per:(L + 1) * per]):
            label(ax, x0 + i * dx, ys, ch, fs=fs, color=BASE[ch], family=MONO, bold=True)
    label(ax, 1, lines[0], "Sequence", fs=26, color=MUTED, ha="left")
    label(ax, 1, lines[0] - 10, rname, fs=26, color=MUTED, ha="left")
    p = 0
    for j, tk in enumerate(t):
        for L in range(len(lines)):
            a, b = max(p, L * per), min(p + len(tk), (L + 1) * per)
            if a >= b:
                continue
            y = lines[L] - 10
            box(ax, x0 + (a - L * per) * dx - dx / 2 + 0.25, y - bh / 2, (b - a) * dx - 0.5, bh,
                fc=[ACCENT_L, FAINT][j % 2], r=0.8)
            for q in range(a, b):
                label(ax, x0 + (q - L * per) * dx, y, w[q], fs=fs, color=BASE.get(w[q], INK), family=MONO, bold=True)
        p += len(tk)
    label(ax, x0 + (per - 0.5) * dx, lines[1] - 10 - bh / 2 - 4, f"{len(t)} tokens", fs=26, color=MUTED, ha="right")
    save(fig, name)


def tokens_split(pos):
    w, site = bound_window(pos)
    _token_rows([("Single letters", None, w)], "tokens_letters", w, site)
    _token_rows([("6-mers", NT, w)], "tokens_kmer", w, site)
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
    fig.set_size_inches(16, 16 / 2.198)                  # the save hook keeps this shape (slide's free area)
    ax = fig.add_axes([0.20, 0.14, 0.54, 0.77])
    ax.imshow(E[ids], cmap=DIVERGING, vmin=-v, vmax=v, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(ids))); ax.set_yticklabels([f"row {i:,}" for i in ids], family=MONO, fontsize=23)
    for r, n in enumerate(names):                        # the token's letters, in base colours, left of its row number
        for q, ch in enumerate(n):
            ax.annotate(ch, (0, r), xycoords=("axes fraction", "data"), textcoords="offset points",
                        xytext=(-116 - (len(n) - 1 - q) * 15, 0), ha="center", va="center_baseline", fontsize=24,
                        family=MONO, fontweight="bold", color=BASE.get(ch, INK))
    ax.set_xticks([0, 64, 128, 192, 255]); ax.set_xlabel("256 numbers per token", fontsize=25)
    ax.tick_params(length=0, labelsize=23)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"Mistral-DNA: {E.shape[0]:,} learned rows, four of them", fontsize=27)
    a2 = fig.add_axes([0.78, 0.14, 0.21, 0.77])
    a2.imshow(np.eye(4), cmap=DIVERGING, vmin=-1, vmax=1, interpolation="nearest")
    a2.set_yticks(range(4)); a2.set_yticklabels(list("ACGU"), family=MONO, fontsize=25, fontweight="bold")
    for t in a2.get_yticklabels():
        t.set_color(BASE[t.get_text()])
    a2.set_xticks([])
    a2.tick_params(length=0)
    for sp in a2.spines.values():
        sp.set_visible(False)
    a2.set_title("The CNN: fixed", fontsize=27)
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
        a = fig.add_axes([x - cw / 2, y, cw, 0.24])
        v = np.percentile(np.abs(vec), 97)
        a.imshow(vec.reshape(-1, 1)[:64], cmap=DIVERGING, vmin=-v, vmax=v, aspect="auto", interpolation="nearest")
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)

    for i, x in enumerate(xs):
        strip(x, 0.095, H0[i]); strip(x, 0.745, H8[i])
        fig.text(x, 0.025, names[i], ha="center", fontsize=25, family=MONO, color=INK)
    band = fig.add_axes([0.0, 0.375, 1.0, 0.33]); band.axis("off")
    band.add_patch(Rectangle((0, 0), 1, 1, color=FAINT, transform=band.transAxes))
    band.text(0.015, 0.5, "Transformer\n8 layers", ha="left", va="center", fontsize=30, color=MUTED,
              transform=band.transAxes)
    j = n // 2
    for x in xs:
        fig.add_artist(plt.Line2D([x, xs[j]], [0.35, 0.73], color=ACCENT, lw=1.4, alpha=0.7))
    fig.text(0.015, 0.215, "In", fontsize=28, color=MUTED, va="center")
    fig.text(0.015, 0.865, "Out", fontsize=28, color=MUTED, va="center")
    save(fig, "blackbox")


# ------------------------------------------------------------------ part 3
def adapt_layers():
    """Which parts are trained (accent) or frozen (grey) in four ways to adapt a pretrained model."""
    fig, ax = canvas(16, 8.4, ylim=(-8, 76))
    cols = [("Linear probe", "seconds to minutes", 0, False),
            ("Partial fine-tuning", "minutes", 3, False),
            ("Full fine-tuning", "minutes to days", 8, False),
            ("LoRA", "minutes", 0, True)]
    label(ax, 4, 60, "Head", fs=20, color=MUTED, ha="left")
    label(ax, 4, 32, "8 layers", fs=20, color=MUTED, ha="left")
    ax.add_patch(Rectangle((60, -6), 4, 3, color=ACCENT)); label(ax, 66, -4.5, "trained", fs=20, color=MUTED, ha="left")
    ax.add_patch(Rectangle((84, -6), 4, 3, color=LIGHT)); label(ax, 90, -4.5, "frozen", fs=20, color=MUTED, ha="left")
    for c, (name, cost, top_trained, lora) in enumerate(cols):
        if c:                                   # frames for the slide: one method per click
            fig.savefig(OUT / f"adapt_layers_{c - 1}.png", dpi=120, facecolor="white")
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
    fig.savefig(OUT / "adapt_layers_3.png", dpi=120, facecolor="white")
    trim_group([f"adapt_layers_{c}" for c in range(4)])
    save(fig, "adapt_layers")


# ------------------------------------------------------------------ part 1, ML fundamentals
def setup_examples(model, tr, ty):
    """Four real training examples: x (start of the sequence), y, and the trained CNN's prediction."""
    from intuition import cnn_scores
    b = [i for i in range(len(ty)) if ty[i] == 1][:2]
    u = [i for i in range(len(ty)) if ty[i] == 0][:2]
    rows = [b[0], u[0], b[1], u[1]]
    p = cnn_scores(model, [tr[i] for i in rows])
    names = []
    for frame in (0, 1):                                 # the loss column enters on the click
        fig, ax = canvas(16, 8, ylim=(0, 80))
        label(ax, 1, 71, "$x$: sequence (first 32 letters)", fs=28, color=MUTED, ha="left")
        label(ax, 104, 71, "Length", fs=28, color=MUTED)
        label(ax, 118.5, 71, "$y$", fs=32, color=MUTED)
        label(ax, 135, 71, r"$\hat p = g_\theta(x)$", fs=30, color=MUTED)
        if frame:
            label(ax, 153, 71, "Loss", fs=28, color=MUTED)
        for r, (i, pi) in enumerate(zip(rows, p)):
            yy = 56 - r * 15
            seq(ax, 2, yy, tr[i][:32], fs=28, dx=2.8)
            label(ax, 2 + 32 * 2.8 - 0.4, yy, "…", fs=28, color=MUTED, ha="left")
            label(ax, 104, yy, f"{len(tr[i])}", fs=30)
            label(ax, 118.5, yy, f"{int(ty[i])}", fs=32, color=BOUND if ty[i] == 1 else MUTED, bold=True)
            label(ax, 135, yy, f"{pi:.2f}", fs=32)
            if frame:
                loss = -np.log(pi) if ty[i] == 1 else -np.log(1 - pi)
                label(ax, 153, yy, f"{loss:.2f}", fs=32, color=HELD_C, bold=True)
        fig.savefig(OUT / f"setup_examples_{frame}.png", facecolor="white"); plt.close(fig)
        names.append(f"setup_examples_{frame}")
    trim_group(names)
    print("setup_examples", [int(ty[i]) for i in rows], [round(float(v), 3) for v in p])


def bce_curve():
    """Binary cross-entropy for one example, in two frames: the bound term, then the unbound term."""
    q = np.linspace(0.005, 0.995, 400)
    names = []
    for frame in (0, 1):
        fig = newfig(); fig.set_size_inches(16, 7.4)
        ax = fig.add_axes([0.1, 0.15, 0.85, 0.75])
        cb = BOUND if frame == 0 else LIGHT
        ax.plot(q, -np.log(q), color=cb, lw=3.5)
        ax.text(0.10, 3.4, "True label bound ($y = 1$)\n$-\\log \\hat p$", color=cb if frame == 0 else MID,
                fontsize=22, va="center")
        if frame == 1:
            ax.plot(q, -np.log(1 - q), color=HELD_C, lw=3.5)
            ax.text(0.66, 3.4, "True label unbound ($y = 0$)\n$-\\log (1 - \\hat p)$", color=HELD_C, fontsize=22,
                    va="center")
        ax.set_xlabel(r"Predicted probability of bound, $\hat p$"); ax.set_ylabel("Loss")
        ax.set_xlim(0, 1); ax.set_ylim(0, 4.2); ygrid(ax)
        fig.savefig(OUT / f"bce_curve_{frame}.png", dpi=120, facecolor="white"); plt.close(fig)
        names.append(f"bce_curve_{frame}")
    trim_group(names)
    print("wrote", names)


def lr_rates(tr, ty):
    """Loss per step of the GC/U logistic regression (same 3,000 sequences and start as lr_training), three learning rates."""
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
        ax.plot([a[0], b[0]], [a[1], b[1]], color=MID, lw=1.4, zorder=1)

    label(ax, 30, 69, "Logistic regression: one unit", fs=24)
    ins = [(14, 46), (14, 26)]
    for (x, y), n in zip(ins, ["GC", "U"]):
        node(x, y); label(ax, x - 5, y, n, fs=20, color=MUTED, ha="right")
    out = (40, 36)
    for a in ins:
        edge(a, out)
    node(*out, r=4.2, fc=ACCENT_L); label(ax, out[0], out[1], r"$\sigma$", fs=24)
    arrow(ax, 45, 36, 51, 36); label(ax, 53, 36, r"$\hat p$", fs=24, ha="left")
    label(ax, 30, 12, r"$\hat p = \sigma(w_1 \cdot \mathrm{GC} + w_2 \cdot \mathrm{U} + b)$", fs=22, color=MUTED)

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
    """Three frames: filter 7's track (last slide) above its heatmap row; all 10 rows; then max pooling."""
    from intuition import acts
    A = acts(model, pos[:2000])
    i = int(np.argmax(A[:, 7, :].max(1)))
    full = A[i][:, :len(pos[i]) - 9]           # positions inside the sequence; the rest is padding
    j = int(np.argmax(full[7]))
    lo = max(0, min(j - 60, full.shape[1] - 120)) // 3 * 3
    a = full[:, lo:lo + 120]                    # a 120-position window, aligned to the pooling blocks
    P = a.reshape(10, 40, 3).max(2)
    v = np.percentile(full, 99.5)
    fig = newfig(); fig.set_size_inches(16, 9)
    ext = (lo - 0.5, lo + 119.5)
    a0 = fig.add_axes([0.08, 0.75, 0.9, 0.18])                  # the track of the last slide, same positions
    a0.bar(range(lo, lo + 120), a[7], color=[ACCENT if p == j else TRAIN_L for p in range(lo, lo + 120)], width=1.0)
    a0.set_xlim(*ext); a0.set_xticks([]); a0.set_ylabel("Filter 7", fontsize=18)
    a0.set_title("Filter 7's output (slide 4.4), positions %d to %d" % (lo, lo + 119))
    a1 = fig.add_axes([0.08, 0.42, 0.9, 0.26])
    a2 = fig.add_axes([0.08, 0.13, 0.9, 0.19])
    for ax_ in (a1, a2):
        ax_.set_yticks(range(10)); ax_.set_yticklabels([str(f) for f in range(10)], fontsize=16)
        ax_.set_ylabel("Filter"); ax_.tick_params(length=0)
        for sp in ax_.spines.values():
            sp.set_visible(False)
    a2.set_xlabel("Position along one bound sequence")
    names = []
    for frame in range(3):
        a1.clear(); a2.clear()
        M = a.copy()
        if frame == 0:
            M[np.arange(10) != 7] = np.nan                    # only filter 7's row, as colour
        a1.imshow(np.ma.masked_invalid(M), cmap=SEQUENTIAL, vmin=0, vmax=v, aspect="auto", interpolation="nearest",
                  extent=(ext[0], ext[1], 9.5, -0.5))
        a1.set_title("First convolution + ReLU: one row per filter, darker = larger"
                     if frame else "The same numbers as one row of colour: darker = larger")
        if frame == 0:
            a1.add_patch(Rectangle((ext[0], 6.5), 120, 1, fill=False, ec=INK, lw=2))
        a2.set_visible(frame == 2)
        if frame == 2:
            a2.imshow(P, cmap=SEQUENTIAL, vmin=0, vmax=v, aspect="auto", interpolation="nearest",
                      extent=(ext[0], ext[1], 9.5, -0.5))
            a2.set_title("After max pooling: the largest of every 3 positions")
            a2.set_xlabel("Position along one bound sequence")
        for ax_ in (a1, a2):
            ax_.set_yticks(range(10)); ax_.set_yticklabels([str(f) for f in range(10)], fontsize=16)
            ax_.set_ylabel("Filter"); ax_.tick_params(length=0)
            for sp in ax_.spines.values():
                sp.set_visible(False)
        a1.set_xticks([])
        fig.savefig(OUT / f"channels_pool_{frame}.png", facecolor="white")
        names.append(f"channels_pool_{frame}")
    trim_group(names)
    plt.close(fig)
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
    fig, ax = canvas(10, 4.125, ylim=(0, 66))
    x0, cw = 31, 111 / n
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
            label(ax, x0 + j * cw + cw / 2, y + 1.6, t, fs=11 if len(t) > 5 else 13, color=LIGHT if pad else INK, family=MONO)
            label(ax, x0 + j * cw + cw / 2, y - 2.6, str(mj), fs=14, color=ACCENT if mj else LIGHT, bold=bool(mj))
        label(ax, x0 + n * cw + 2, y, f"{sum(m)} real", fs=19, color=ACCENT, ha="left")
    label(ax, x0 + n * cw / 2, 4, "Sequence embedding: the mean over the tokens with mask 1",
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
    label(a2, 34, 52, "Logistic regression (part 1, slide 2.11)", fs=20, color=MUTED)
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
        label(ax0, x0 + i * dx, 30, ch, fs=22, family=MONO, color=BASE[ch], bold=True)
    label(ax0, x0 + 24 * dx + 2, 30, f"… ({len(s)} letters)", fs=19, color=MUTED, ha="left")
    for k in range(4):
        yk = 22 - k * 5.5
        ax0.plot([x0 + k * dx - 1.4, x0 + (k + 4) * dx + 1.4], [yk, yk], color=ACCENT if k == 0 else LIGHT, lw=3)
        label(ax0, x0 + (k + 5) * dx + 2, yk, s[k:k + 5], fs=17, family=MONO, color=ACCENT if k == 0 else MUTED, ha="left")
    label(ax0, 120, 14, f"{len(words)} overlapping words", fs=21, color=MUTED)
    ax = fig.add_axes([0.07, 0.12, 0.9, 0.42])
    ax.set_ylim(0, v.max() + 1.5)
    ax.bar(range(1024), v, width=1.0, color=ACCENT)
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
    """The CNN's parts (top) and one sequence followed through them with numbers (bottom)."""
    fig, ax = canvas(16, 9, ylim=(0, 90))
    y0 = 47
    W_C, X_C = GOOD_C, ACCENT                          # learned weights green, numbers from the sequence blue
    rend = fig.canvas.get_renderer()

    def pieces(x, y, parts, fs=20, ha="center"):
        """Coloured text pieces on one line, placed one after the other."""
        ts, widths = [], []
        for s, c in parts:
            t = ax.text(0, y, s, fontsize=fs, color=c, ha="left", va="center")
            bb = t.get_window_extent(rend).transformed(ax.transData.inverted())
            ts.append(t); widths.append(bb.width)
        x = x - sum(widths) / 2 if ha == "center" else x - sum(widths) if ha == "right" else x
        for t, w in zip(ts, widths):
            t.set_x(x); x += w

    def grid(x, w, rows, h=3.6, gap=0.9, fill=FAINT, edge=MID):
        for r in range(rows):
            ax.add_patch(Rectangle((x, y0 + r * (h + gap)), w, h, fc=fill, ec=edge, lw=1.0))

    mid = y0 + 12.9
    cols = [10.5, 35.5, 58.5, 79.5, 110, 148]
    grid(2, 17, 4, h=5.4, gap=1.1)                      # one-hot input: 4 letter rows
    grid(28, 15, 6, fill=ACCENT_L, edge=ACCENT)         # convolution: one row per filter
    grid(51, 15, 6, fill="white", edge=ACCENT)          # ReLU: same shape
    grid(76, 7, 6, fill=ACCENT_L, edge=ACCENT)          # max pooling: shorter rows
    ax.add_patch(Rectangle((92, mid - 2.2), 36, 4.4, fc=FAINT, ec=MID, lw=1.0))     # flatten
    ax.add_patch(plt.Circle((148, mid), 2.4, color=INK))                           # linear: one number
    for a_, b_ in ((20, 27), (44, 50), (67, 75), (84, 91), (129, 144.6)):
        arrow(ax, a_, mid, b_, mid, color=MUTED, lw=2, ms=22)
    ax.plot([26, 26, 85, 85], [y0 + 28, y0 + 31, y0 + 31, y0 + 28], color=INK, lw=2)
    label(ax, 55.5, y0 + 35, "repeat as often as you like (we use 2 blocks)", fs=22, color=INK)
    names = ["one-hot input", "convolution", "ReLU", "max pooling", "flatten", "linear"]
    for x, n in zip(cols, names):
        label(ax, x, y0 - 5, n, fs=22, color=INK, bold=True)

    # one bound sequence, followed through the same columns
    ax.plot([2, 158], [34.5, 34.5], color=HAIR, lw=1.2)
    label(ax, 2, 30, "One sequence, followed through", fs=19, color=MUTED, ha="left")
    pieces(158, 30, [("numbers from the sequence", X_C), ("     ", INK), ("learned weights", W_C)], fs=19, ha="right")
    pieces(10.5, 22, [(ch + " ", BASE[ch]) for ch in "AUGGCU"] + [("…", INK)], fs=21)
    vals = {1: "1.4   −0.3   …", 2: "1.4   0.0   …", 3: "1.4   0.6   …",
            4: "x = (1.4, 0.6, …, 0.2)"}
    for k, s in vals.items():
        label(ax, cols[k], 22, s, fs=21, color=X_C)
    label(ax, cols[5], 22, "z = 1.2", fs=21, color=INK, bold=True)
    shapes = ["4 × 501", "10 filters: 10 × 492", "negatives → 0", "10 × 164, then\nblock 2: 20 × 53", "1,060 = 20 × 53", "one number"]
    for x, s in zip(cols, shapes):
        label(ax, x, 15.5, s, fs=18, color=MUTED)
    # the linear layer and the logistic function with these numbers
    pieces(80, 5, [("z = ", INK), ("0.5", W_C), (" × ", INK), ("1.4", X_C), ("  +  ", INK), ("(−0.3)", W_C),
                   (" × ", INK), ("0.6", X_C), ("  +  …  +  ", INK), ("0.1", W_C), (" × ", INK), ("0.2", X_C),
                   ("  +  ", INK), ("(−0.2)", W_C), ("  =  1.2", INK), ("          ", INK),
                   (r"$\hat p$", INK), (" = σ(z) = σ(1.2) = 0.77", INK)], fs=24)
    save_split(fig, ax, "cnn_template", 35)             # the parts first, the worked numbers on the click


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

