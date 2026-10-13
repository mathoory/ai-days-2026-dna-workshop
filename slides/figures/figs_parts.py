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


def plant_scheme():
    """Where plant() writes copies 1 to 7 in a 330-letter sequence, and the shuffled control."""
    fig = newfig()
    fig.set_size_inches(16, 4.6)
    ax = fig.add_axes([0.22, 0.22, 0.75, 0.66])
    L, k = 330, 9
    ax.axvspan(150, 190, color=FAINT, zorder=0)
    ax.text(170, 2.62, "Measured site\n(letters 150 to 190)", ha="center", va="bottom", fontsize=19, color=MUTED)
    for y, name, c in [(1.6, "Motif", INK), (0.5, "Same letters, shuffled", MID)]:
        ax.plot([0, L], [y, y], color=HAIR, lw=8, solid_capstyle="butt", zorder=1)
        for n in range(1, 8):
            a = 40 + 45 * (n - 1)
            ax.add_patch(Rectangle((a, y - 0.22), k, 0.44, color=c, zorder=2))
            if y > 1:
                ax.text(a + k / 2, y + 0.38, str(n), ha="center", fontsize=19, color=MUTED)
        ax.text(-6, y, name, ha="right", va="center", fontsize=21, color=c)
    ax.set_xlim(-2, L + 2); ax.set_ylim(0, 3.4)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 50, 100, 150, 190, 250, 300, 330]); ax.set_xlabel("Position in a 330-letter random sequence")
    save(fig, "plant_scheme")


# ------------------------------------------------------------------ part 2
def _token_rows(rows, name, w, site):
    from transformers import AutoTokenizer
    h = 6 + 4.2 * len(rows)
    fig, ax = canvas(16, h, ylim=(0, h * 10))
    top = h * 10 - 10
    dx, x0 = 3.6, 36
    for i, ch in enumerate(w):
        label(ax, x0 + i * dx, top, ch, fs=24, color=INK, family=MONO)
    ax.plot([x0 + site * dx - 1.8, x0 + (site + 6) * dx + 1.8], [top - 4.5, top - 4.5], color=ACCENT, lw=2.5)
    label(ax, x0 + (site + 3) * dx, top - 8, "miR-19 site", fs=18, color=ACCENT)
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
            box(ax, x0 + p * dx - 1.85, y - 4.5, len(tk) * dx - 0.45, 9, fc=[FAINT, HAIR][j % 2], r=0.6)
            for q, ch in enumerate(tk):
                label(ax, x0 + (p + q) * dx, y, ch, fs=21, color=INK, family=MONO)
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


if __name__ == "__main__":
    want = set(sys.argv[1:])
    run = (lambda n: not want or n in want)
    tr, ty = split("train")
    pos = [s for s, l in zip(tr, ty) if l == 1]
    for n, job in [("bio_flow", bio_flow), ("plant_scheme", plant_scheme), ("tokens_split", lambda: tokens_split(pos)),
                   ("emb_lookup", lambda: emb_lookup(pos)), ("blackbox", lambda: blackbox(pos)),
                   ("adapt_layers", adapt_layers)]:
        if run(n):
            job()
