"""Figures added on 14.10 for the part 2 deck (pretraining analogy, objectives, pipeline, heads).

    HF_HUB_OFFLINE=1 python figs_d2.py [name ...]   (hw1-00660121 env; writes ../assets/custom/<name>.png)

Frames of one click-through are saved untrimmed at the same size, so they stack exactly on a slide.
"""
import sys

import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch, Rectangle

from deckstyle import *  # noqa: F401,F403
from intuition import split

MISTRAL = "RaphaelMourad/Mistral-DNA-v1-17M-hg38"
STAGES = ["Tokenization", "Embedding", "Body", "Prediction head"]


def save_raw(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=120, facecolor="white")
    plt.close(fig)
    print("wrote", name)


def block(ax, x, y, w, h, text, fc=ACCENT, tc="white", fs=22, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6", fc=fc, ec="none"))
    label(ax, x + w / 2, y + h / 2, text, fs=fs, color=tc, bold=bold)


def token_row(ax, x, y, toks, fs=22, cw=None, colored=True, hide=None, hide_text="?"):
    """Tokens in boxes, one after the other; returns the box centres. hide: index shown as a blank."""
    xs = []
    for j, t in enumerate(toks):
        w = cw or max(6.0, 2.2 + 1.9 * len(t))
        if j == hide:
            box(ax, x, y - 4.5, w, 9, fc="white", ec=HELD_C, lw=2.5, r=0.8)
            label(ax, x + w / 2, y, hide_text, fs=fs, color=HELD_C, bold=True)
        else:
            box(ax, x, y - 4.5, w, 9, fc=[ACCENT_L, FAINT][j % 2], r=0.8)
            if colored and set(t) <= set("ACGTU"):
                for q, ch in enumerate(t):
                    label(ax, x + w / 2 + (q - (len(t) - 1) / 2) * 1.9, y, ch, fs=fs - 2, color=BASE[ch],
                          family=MONO, bold=True)
            else:
                label(ax, x + w / 2, y, t, fs=fs, color=INK)
        xs.append((x, w))
        x += w + 1.0
    return xs


# ------------------------------------------------------------------ the real example: a context and its next token
def real_context(tok, pos, cut=14, n_show=6):
    ids = tok(pos[7].replace("U", "T"), return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    toks = [t.replace("▁", "") for t in tok.convert_ids_to_tokens(ids[:cut + 1])]
    return toks[cut - n_show:cut], toks[cut]


# ------------------------------------------------------------------ 1.1 pretraining, by analogy
def pretrain_analogy(tok, pos):
    ctx, nxt = real_context(tok, pos)
    for k in (0, 1):
        fig, ax = canvas()
        label(ax, 6, 80, "Language", fs=24, color=MUTED, ha="left")
        label(ax, 6, 73, "read billions of sentences, predict each next word", fs=19, color=MUTED, ha="left")
        xs = token_row(ax, 10, 60, ["The", "cat", "sat", "on", "the", ""], hide=5, cw=13)
        x_end = xs[-1][0] + xs[-1][1]
        arrow(ax, x_end + 2, 60, x_end + 12, 60, color=MUTED, lw=2, ms=22)
        label(ax, x_end + 20, 60, "mat", fs=26, color=HELD_C, bold=True)
        if k == 1:
            label(ax, 6, 38, "Genome", fs=24, color=MUTED, ha="left")
            label(ax, 6, 31, "read billions of DNA letters, predict each next token", fs=19, color=MUTED, ha="left")
            xs = token_row(ax, 10, 18, ctx + [""], hide=len(ctx))
            x_end = xs[-1][0] + xs[-1][1]
            arrow(ax, x_end + 2, 18, x_end + 12, 18, color=MUTED, lw=2, ms=22)
            for q, ch in enumerate(nxt):
                label(ax, x_end + 16 + q * 2.4, 18, ch, fs=24, color=BASE[ch], family=MONO, bold=True)
        save_raw(fig, f"pretrain_analogy_{k}")


# ------------------------------------------------------------------ 1.2-1.4 one body, many heads
def adapt():
    def frame(rows, name, body="Model", dx=0):
        """rows: (input text, head name, output) per head; all heads share one body.
        dx: extra width for a one-line input text (everything right of it moves over)."""
        fig, ax = canvas()
        block(ax, 37 + dx, 16, 30, 65, body, fc=ACCENT, fs=32)
        n = len(rows)
        ys = [48.5] if n == 1 else list(np.linspace(68, 29, n))
        label(ax, 15 + dx / 2, 48.5, rows[0][0], fs=27, color=INK)
        arrow(ax, 29 + dx, 48.5, 35.5 + dx, 48.5, color=MUTED, lw=2.5, ms=26)
        for y, (_, head, out, c) in zip(ys, rows):
            arrow(ax, 68.5 + dx, 48.5 + (y - 48.5) * 0.4, 77.5 + dx, y, color=MUTED, lw=2.5, ms=26)
            block(ax, 79 + dx, y - 7.5, 37, 15, head, fc=c, fs=25)
            arrow(ax, 117.5 + dx, y, 123.5 + dx, y, color=MUTED, lw=2.5, ms=22)
            label(ax, 125 + dx, y, out, fs=27, color=INK, ha="left")
        label(ax, 52 + dx, 10, "learned once, kept", fs=23, color=ACCENT)
        label(ax, 97.5 + dx, 10, "one head per task", fs=23, color=MUTED)
        save_raw(fig, name)
    frame([("The cat sat on the", "next-word head", "mat", HELD_C)], "adapt_0", dx=11)
    frame([("What a\ngreat movie!", "next-word head", "and", HELD_C),
           ("What a\ngreat movie!", "sentiment head", "positive", GOOD_C),
           ("What a\ngreat movie!", "translation head", "Quel bon film !", GOOD_C)], "adapt_1")
    frame([("A sequence\nof DNA tokens", "next-token head", "the next token", HELD_C)], "adapt_2", body="Mistral-DNA")
    frame([("A sequence\nof DNA tokens", "next-token head", "the next token", HELD_C),
           ("A sequence\nof DNA tokens", "AGO2 head", "P(bound)", GOOD_C),
           ("A sequence\nof DNA tokens", "expression head", "a number", GOOD_C)], "adapt_3", body="Mistral-DNA")


# ------------------------------------------------------------------ objectives: next token and masked token
def objective_rows(tok, pos):
    ctx, nxt = real_context(tok, pos, n_show=7)
    # next token: the last one is missing
    fig, ax = canvas(16, 3.4, ylim=(0, 34))
    xs = token_row(ax, 4, 17, ctx + [""], hide=len(ctx))
    label(ax, xs[-1][0] + xs[-1][1] / 2, 5, "from the tokens before it", fs=18, color=MUTED)
    save(fig, "objective_next")
    # masked: one token in the middle is hidden; both sides are visible
    # wrapped onto two lines at the hidden token: the tokens before it run along line 1, the ones after it
    # follow it on line 2; one arrow runs under each side and turns into the hidden token. s scales it all up.
    toks = ctx + [nxt]
    m = 4
    s, W, H, cw = 1.6, 71, 36, 16.5
    fig, ax = canvas(W * s / 10, H * s / 10, xlim=(0, W), ylim=(0, H))
    y1 = H - 5; y2 = y1 - 16
    row1 = token_row(ax, 1, y1, toks[:m], fs=36, cw=cw)
    row2 = token_row(ax, 1, y2, toks[m:], hide=0, hide_text="[MASK]", fs=36, cw=cw)
    cx = row2[0][0] + row2[0][1] / 2
    for row, y, yt in ((row1, y1 - 7.6, y2 + 4.8), (row2, y2 - 7.6, y2 - 4.8)):
        ax.add_patch(FancyArrowPatch((row[-1][0] + row[-1][1] - 2, y), (cx, yt), arrowstyle="-|>", mutation_scale=26,
                                     color=ACCENT, lw=3, shrinkA=0, shrinkB=0,
                                     connectionstyle="angle,angleA=0,angleB=90,rad=12"))
    label(ax, cx + 4, y2 - 12, "from both sides", fs=30, color=ACCENT, ha="left")
    save(fig, "objective_masked")


# ------------------------------------------------------------------ the four stages, then the CNN and the DNA model
def stage_axes(ax, y, highlight=None, fs=24, h=12):
    xs = [8 + k * 38 for k in range(4)]
    for k, (x, s) in enumerate(zip(xs, STAGES)):
        on = highlight is None or highlight == k
        block(ax, x, y, 30, h, s, fc=ACCENT if on else FAINT, tc="white" if on else LIGHT, fs=fs)
        if k < 3:
            arrow(ax, x + 31, y + h / 2, x + 37, y + h / 2, color=MUTED if on or highlight is None else LIGHT, lw=2, ms=20)
    return xs


def stages():
    defs = ["cut the sequence\ninto tokens", "turn each token\ninto numbers", "mix the tokens:\nlayers of the model",
            "turn the numbers\ninto an answer"]
    cnn = ["single letters", "one-hot, 4 numbers\n(fixed)", "2 convolution\nlayers", "linear → 1 logit"]
    lm = ["BPE tokens", "a learned table,\n256 numbers", "transformer,\n8 layers", "next token, or\nyour task"]
    # the four stages run top to bottom, each with its one-line meaning to the right (fills a near 2:1 slide)
    W, H, bw, bh, gap = 136, 65, 44, 12, 5
    fig, ax = canvas(W / 10, H / 10, xlim=(0, W), ylim=(0, H))
    for k, (s, d) in enumerate(zip(STAGES, defs)):
        y = H - 1 - bh - k * (bh + gap)
        block(ax, 1, y, bw, bh, s, fs=32)
        label(ax, bw + 6, y + bh / 2, d.replace("\n", " "), fs=34, color=MUTED, ha="left")
        if k < 3:
            arrow(ax, 1 + bw / 2, y - 0.5, 1 + bw / 2, y - gap + 0.5, color=MUTED, lw=2.5, ms=24)
    save_raw(fig, "stages_0")
    for k in (1, 2):
        fig, ax = canvas()
        xs = stage_axes(ax, 72, h=10, fs=21)
        label(ax, 4, 56, "CNN (part 1)", fs=20, color=INK, ha="left", bold=True)
        for x, d in zip(xs, cnn):
            label(ax, x + 15, 48, d, fs=20, color=INK, va="top")
        if k == 2:
            label(ax, 4, 26, "DNA language model (Mistral-DNA)", fs=20, color=ACCENT, ha="left", bold=True)
            for x, d in zip(xs, lm):
                label(ax, x + 15, 18, d, fs=20, color=ACCENT, va="top")
        save_raw(fig, f"stages_{k}")
    for k in range(4):                      # section dividers: where we are
        fig, ax = canvas(16, 1.8, ylim=(0, 18))
        stage_axes(ax, 3, highlight=k, fs=21)
        save(fig, f"stage_strip_{k}")


# ------------------------------------------------------------------ 4.2 the whole pipeline on one real sequence
def walkthrough(tok, fm, pos):
    s = pos[3][:60].replace("U", "T")
    enc = tok(s, return_tensors="pt")
    with torch.no_grad():
        hs = fm(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], output_hidden_states=True).hidden_states
    toks = [t.replace("▁", "") for t in tok.convert_ids_to_tokens(enc["input_ids"][0])]
    H0, H = hs[0][0].numpy(), hs[-1][0].numpy()
    v0, v = np.percentile(np.abs(H0), 97), np.percentile(np.abs(H), 97)
    steps = ["Sequence", "1  Tokenization", "2  Embedding", "3  Body", "4  Prediction head"]
    for k in range(5):
        fig = plt.figure(figsize=(16, 9), dpi=120)
        side = fig.add_axes([0, 0, 0.2, 1]); side.set_xlim(0, 20); side.set_ylim(0, 90); side.axis("off")
        cur = {0: 1, 1: 2, 2: 3, 3: 4, 4: 4}[k]
        for j, st in enumerate(steps):
            y = 78 - j * 14
            on = j == cur
            done = j < cur
            side.add_patch(plt.Circle((2.2, y), 0.9, color=ACCENT if on else (MID if done else HAIR),
                                      transform=side.transData))
            side.text(4, y, st, fontsize=21, va="center", color=ACCENT if on else (INK if done else LIGHT),
                      fontweight="bold" if on else "normal")
            if j < 4:
                side.plot([2.2, 2.2], [y - 2, y - 12], color=HAIR, lw=2)
        main = fig.add_axes([0.22, 0, 0.78, 1]); main.set_xlim(0, 125); main.set_ylim(0, 90); main.axis("off")
        # the sequence is always on top
        for i, ch in enumerate(s[:46]):
            main.text(4 + i * 2.55, 84, ch, fontsize=17, family=MONO, color=BASE[ch], ha="center", va="center",
                      fontweight="bold")
        main.text(4 + 46 * 2.55, 84, "…", fontsize=17, color=MUTED, va="center")
        if k == 0:
            xs = token_row(main, 2, 66, toks[:9], fs=17)
            main.text(xs[-1][0] + xs[-1][1] + 2, 66, f"… {len(toks)} tokens in all, with [CLS] and [SEP]",
                      fontsize=18, color=MUTED, va="center")
        if k in (1, 2):
            M, vm = (H0, v0) if k == 1 else (H, v)
            a = fig.add_axes([0.32, 0.1, 0.62, 0.72])
            a.imshow(M, cmap=DIVERGING, vmin=-vm, vmax=vm, aspect="auto", interpolation="nearest")
            a.set_yticks(range(len(toks))); a.set_yticklabels(toks, fontsize=15, family=MONO)
            a.set_xticks([0, 64, 128, 192, 255]); a.set_xlabel("256 numbers per token"); a.tick_params(length=0)
            for sp in a.spines.values():
                sp.set_visible(False)
            a.set_title("One learned row per token, from the table" if k == 1 else
                        "After 8 transformer layers: still one row per token,\neach row now depends on all the tokens",
                        fontsize=20, loc="left")
        if k in (3, 4):
            a = fig.add_axes([0.32, 0.5, 0.62, 0.3])
            a.imshow(H, cmap=DIVERGING, vmin=-v, vmax=v, aspect="auto", interpolation="nearest")
            a.set_yticks([]); a.set_xticks([]); a.set_ylabel(f"{len(toks)} tokens", fontsize=18)
            for sp in a.spines.values():
                sp.set_visible(False)
            a.set_title("The body's output", fontsize=20, loc="left")
            b = fig.add_axes([0.32, 0.3, 0.62, 0.06])
            b.imshow(H.mean(0, keepdims=True), cmap=DIVERGING, vmin=-v, vmax=v, aspect="auto", interpolation="nearest")
            b.set_yticks([]); b.set_xticks([0, 64, 128, 192, 255]); b.tick_params(length=0)
            for sp in b.spines.values():
                sp.set_visible(False)
            fig.text(0.63, 0.43, "average over the tokens  ↓", ha="center", fontsize=20, color=MUTED)
            fig.text(0.32, 0.38, "The sequence embedding: 256 numbers", fontsize=20, color=INK)
            if k == 4:
                fig.text(0.63, 0.17, "↓", ha="center", fontsize=30, color=MUTED)
                fig.text(0.63, 0.08, "a prediction head: the probability that AGO2 binds (section 5)", ha="center",
                         fontsize=24, color=GOOD_C)
        fig.savefig(OUT / f"walk_{k}.png", dpi=120, facecolor="white"); plt.close(fig); print("wrote", f"walk_{k}")


# ------------------------------------------------------------------ 5.1 prediction heads
def heads():
    rows = [("Next token (pretraining, Mistral-DNA)", "the last token's vector", "256 → 4,096, softmax",
             "a probability for each token", HELD_C),
            ("Masked token (pretraining, Nucleotide Transformer)", "the hidden token's vector", "256 → vocabulary, softmax",
             "a probability for each token", HELD_C),
            ("Our task", "the average of all token vectors", "256 → 1, sigmoid", "P(bound)", GOOD_C)]
    for k in range(3):
        fig, ax = canvas()
        block(ax, 4, 30, 22, 30, "Body", fc=ACCENT, fs=26)
        label(ax, 15, 24, "one vector of 256\nnumbers per token", fs=18, color=MUTED, va="top")
        for j, (name, takes, head, out, c) in enumerate(rows[:k + 1]):
            y = 74 - j * 26
            arrow(ax, 27, 45, 40, y, color=MUTED, lw=2, ms=20)
            label(ax, 42, y + 7, name, fs=20, color=c, ha="left", bold=True)
            label(ax, 42, y, takes, fs=19, color=INK, ha="left")
            arrow(ax, 82, y, 88, y, color=MUTED, lw=2, ms=18)
            block(ax, 89, y - 4.5, 30, 9, head, fc=c, fs=17)
            arrow(ax, 120, y, 126, y, color=MUTED, lw=2, ms=18)
            label(ax, 128, y, out, fs=19, color=INK, ha="left")
        save_raw(fig, f"heads_{k}")


# ------------------------------------------------------------------ 6.1 loading a model, as a picture
def hf_load():
    fig, ax = canvas()
    block(ax, 2, 36, 48, 26, "Hugging Face Hub", fc=INK, fs=30)
    label(ax, 26, 31, "RaphaelMourad/\nMistral-DNA-v1-17M-hg38", fs=22, color=MUTED, va="top", family=MONO)
    for y, api, what in ((72, "AutoTokenizer.from_pretrained(name)", "the tokenizer: its 4,096-token vocabulary"),
                         (16, "AutoModel.from_pretrained(name)", "the model: 16M trained weights")):
        arrow(ax, 51, 49, 60, y, color=MUTED, lw=2.5, ms=26)
        label(ax, 62, y + 5, api, fs=27, color=ACCENT, ha="left", family=MONO)
        label(ax, 62, y - 4.5, what, fs=27, color=INK, ha="left")
    save(fig, "hf_load")


# ------------------------------------------------------------------ 6.2 your turn
def your_turn_d2():
    fig, ax = canvas()
    xs = [26, 80, 134]
    items = [(xs[0], "TODO-1", "embed: sequences in,\none vector each out"),
             (xs[1], "TODO-2", "probe_auc:\nlogistic regression\non the vectors"),
             (xs[2], "Live", "pretrain a model from\nrandom weights\nfor 3 minutes")]
    # 1: tokens -> one vector (tokens drawn here at a larger size than token_row)
    toks, dl = ["AAA", "CTG", "CAGAA"], 3.2
    ws = [3.4 + dl * len(t) for t in toks]
    x = xs[0] - (sum(ws) + 1.5 * (len(toks) - 1)) / 2
    for j, (t, w) in enumerate(zip(toks, ws)):
        box(ax, x, 72, w, 15, fc=[ACCENT_L, FAINT][j % 2], r=0.8)
        for q, ch in enumerate(t):
            label(ax, x + w / 2 + (q - (len(t) - 1) / 2) * dl, 79.5, ch, fs=30, color=BASE[ch], family=MONO, bold=True)
        x += w + 1.5
    arrow(ax, xs[0], 70, xs[0], 58, color=MUTED, lw=2.5, ms=26)
    for i in range(16):
        ax.add_patch(Rectangle((xs[0] - 21.6 + i * 2.7, 43), 2.5, 13, fc=DIVERGING((np.sin(i * 1.7) + 1) / 2), ec="none"))
    # 2: vector -> sigma -> P(bound)
    for i in range(16):
        ax.add_patch(Rectangle((xs[1] - 21.6 + i * 2.7, 74), 2.5, 13, fc=DIVERGING((np.cos(i * 1.3) + 1) / 2), ec="none"))
    arrow(ax, xs[1], 72, xs[1], 59, color=MUTED, lw=2.5, ms=26)
    label(ax, xs[1], 50, "σ  →  P(bound)", fs=34, color=GOOD_C, bold=True)
    # live: a block that learns
    block(ax, xs[2] - 20, 43, 40, 44, "105M model", fc=ACCENT, fs=32)
    for x, todo, what in items:
        label(ax, x, 35, todo, fs=38, color=ACCENT, bold=True)
        label(ax, x, 28.5, what, fs=28, color=INK, va="top")
    save(fig, "your_turn_d2")


if __name__ == "__main__":
    want = set(sys.argv[1:])
    run = (lambda n: not want or n in want)
    tr, ty = split("train")
    pos = [s for s, l in zip(tr, ty) if l == 1]
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MISTRAL, trust_remote_code=True)
    fm = None
    if run("walkthrough"):
        fm = AutoModel.from_pretrained(MISTRAL, trust_remote_code=True).eval()
    for n, job in [("pretrain_analogy", lambda: pretrain_analogy(tok, pos)), ("adapt", adapt),
                   ("objective_rows", lambda: objective_rows(tok, pos)), ("stages", stages),
                   ("walkthrough", lambda: walkthrough(tok, fm, pos)), ("heads", heads), ("hf_load", hf_load),
                   ("your_turn_d2", your_turn_d2)]:
        if run(n):
            job()
    for g in (["pretrain_analogy_0", "pretrain_analogy_1"], ["adapt_0"], ["adapt_1"], ["adapt_2", "adapt_3"], ["stages_0"],
              ["stages_1", "stages_2"], ["heads_0", "heads_1", "heads_2"], [f"walk_{k}" for k in range(5)]):
        if all((OUT / f"{n}.png").exists() for n in g) and (not want or any(n.rsplit("_", 1)[0] in want or n in want for n in g)):
            trim_group(g)
