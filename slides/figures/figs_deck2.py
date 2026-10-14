"""Part 2 deck, rework of 15.10 (Mattan's feedback): one geometry for every pipeline figure, real data throughout.

    HF_HUB_OFFLINE=1 python figs_deck2.py [name ...]   (hw1-00660121 env; writes ../assets/custom/d2_*.png)

Frames of one click-through, and figures on neighbouring slides that must not move, are cropped as one group.
"""
import json
import sys

import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch, Rectangle

from deckstyle import *  # noqa: F401,F403
from figs_d2 import MISTRAL, STAGES, block, real_context, save_raw, token_row
from figs_parts import bound_window
from intuition import split

W, H = 160, 84                        # every canvas here: 16 x 8.4 inches, about the shape a slide leaves
XS = [8 + k * 38 for k in range(4)]   # the four stage blocks, as on the section dividers
BW = 30


def canvas2():
    return canvas(16, 8.4, ylim=(0, H))


def stage_row(ax, y, h=10, fs=21, colors=None, texts=None, tcolors=None):
    """The four stage blocks with arrows, at height y. colors/texts override per block."""
    for k, x in enumerate(XS):
        fc = colors[k] if colors else ACCENT
        tc = tcolors[k] if tcolors else "white"
        block(ax, x, y, BW, h, texts[k] if texts else STAGES[k], fc=fc, tc=tc, fs=fs)
        if k < 3:
            arrow(ax, x + BW + 1, y + h / 2, x + 37, y + h / 2, color=MUTED, lw=2, ms=20)


def strip(ax, x, y, w, h, vec, vm, vertical=False):
    """A vector drawn as a row (or column) of coloured cells."""
    M = np.asarray(vec, dtype=float).reshape(-1, 1) if vertical else np.asarray(vec, dtype=float).reshape(1, -1)
    ax.imshow(M, cmap=DIVERGING, vmin=-vm, vmax=vm, extent=(x, x + w, y, y + h), aspect="auto",
              interpolation="nearest", zorder=2)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=HAIR, lw=1, zorder=3))


def heat(ax, x, y, w, h, M, vm):
    ax.imshow(M, cmap=DIVERGING, vmin=-vm, vmax=vm, extent=(x, x + w, y, y + h), aspect="auto",
              interpolation="nearest", zorder=2)


def model_and_tokenizer(need_model=True):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MISTRAL, trust_remote_code=True)
    fm = AutoModel.from_pretrained(MISTRAL, trust_remote_code=True).eval() if need_model else None
    return tok, fm


def run_model(tok, fm, s, special=True):
    enc = tok(s, return_tensors="pt", add_special_tokens=special)
    with torch.no_grad():
        hs = fm(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], output_hidden_states=True).hidden_states
    toks = [t.replace("▁", "") for t in tok.convert_ids_to_tokens(enc["input_ids"][0])]
    return toks, hs[0][0].numpy(), hs[-1][0].numpy(), enc["input_ids"][0].tolist()


# ------------------------------------------------------------------ 1.2 and 1.3: one body, an embedding, many heads
def adapt(tok, fm, pos):
    """Input tokens on top, the body, its output embedding, then one head per task. One geometry for all frames."""
    ctx, nxt = real_context(tok, pos, n_show=5)
    dna_in = " ".join([t for t in ctx])
    _, _, Hd, _ = run_model(tok, fm, "".join(ctx), special=False)
    e_dna = Hd[-1][:12]
    e_lang = np.sin(np.arange(12) * 1.9) * 1.2                 # a drawn vector: the language model is not real here
    lang_heads = [("next-word head", "mat", HELD_C), ("sentiment head", "neutral", GOOD_C),
                  ("translation head", "Le chat s'est assis sur le …", GOOD_C)]
    dna_heads = [("next-token head", nxt, HELD_C), ("AGO2 head", "P(bound)", GOOD_C),
                 ("expression head", "a number", GOOD_C)]
    frames = [("Model", ["The", "cat", "sat", "on", "the"], e_lang, lang_heads[:1]),
              ("Model", ["The", "cat", "sat", "on", "the"], e_lang, lang_heads),
              ("Mistral-DNA", ctx, e_dna, dna_heads[:1]),
              ("Mistral-DNA", ctx, e_dna, dna_heads)]
    names = []
    for k, (body, toks, e, heads) in enumerate(frames):
        fig, ax = canvas2()
        cw = 12 if body == "Model" else None
        row = token_row(ax, 4, 77, toks + [""], hide=len(toks), cw=cw, fs=22)
        rx = (row[0][0] + row[-1][0] + row[-1][1]) / 2
        label(ax, 4, 68.5, "Input", fs=19, color=MUTED, ha="left")
        arrow(ax, 33, 71.5, 33, 63.5, color=MUTED, lw=2.5, ms=24)
        block(ax, 18, 14, 30, 48, body, fc=ACCENT, fs=26)
        label(ax, 33, 8.5, "Body: learned once, kept", fs=20, color=ACCENT)
        arrow(ax, 49, 38, 56, 38, color=MUTED, lw=2.5, ms=24)
        strip(ax, 57, 20, 4.5, 36, e, np.abs(e).max(), vertical=True)
        label(ax, 59.25, 14, "Embedding", fs=20, color=INK, bold=True)
        n = len(heads)
        ys = [38] if n == 1 else list(np.linspace(56, 20, n))
        for y, (head, out, c) in zip(ys, heads):
            arrow(ax, 62.5, 38 + (y - 38) * 0.3, 71.5, y, color=MUTED, lw=2.5, ms=22)
            block(ax, 73, y - 6, 36, 12, head, fc=c, fs=22)
            arrow(ax, 110, y, 116, y, color=MUTED, lw=2.5, ms=20)
            if set(out) <= set("ACGT") and out:
                for q, ch in enumerate(out):
                    label(ax, 118 + q * 2.6, y, ch, fs=24, color=BASE[ch], family=MONO, bold=True, ha="left")
            else:
                label(ax, 118, y, out, fs=22, color=INK, ha="left")
        label(ax, 91, 8.5, "Head: one per task", fs=20, color=MUTED)
        name = f"d2_adapt_{k}"
        save_raw(fig, name); names.append(name)
    trim_group(names)
    print("adapt: DNA context", ctx, "next", nxt)


# ------------------------------------------------------------------ 1.4 two objectives on one slide
def objectives(tok, pos):
    ctx, nxt = real_context(tok, pos, n_show=7)
    names = []
    for k in (0, 1):
        fig, ax = canvas2()
        label(ax, 4, 74, "Next token", fs=24, color=INK, bold=True, ha="left")
        label(ax, 4, 67, "hide the last token, predict it from the ones before", fs=20, color=MUTED, ha="left")
        row = token_row(ax, 4, 55, ctx + [""], hide=len(ctx), fs=24)
        x0, x1 = row[0][0], row[-2][0] + row[-2][1]
        ax.annotate("", (row[-1][0] + row[-1][1] / 2, 48.5), (x0 + 2, 48.5),
                    arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2.5, mutation_scale=24))
        if k == 1:
            m = 3
            toks = ctx[:m] + ["[MASK]"] + ctx[m + 1:] + [nxt]
            label(ax, 4, 34, "Masked token", fs=24, color=INK, bold=True, ha="left")
            label(ax, 4, 27, "hide a token in the middle, predict it from both sides", fs=20, color=MUTED, ha="left")
            row = token_row(ax, 4, 15, toks, hide=m, hide_text="[MASK]", fs=24)
            cx = row[m][0] + row[m][1] / 2
            ax.annotate("", (cx - 3, 8.5), (row[0][0] + 2, 8.5),
                        arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2.5, mutation_scale=24))
            ax.annotate("", (cx + 3, 8.5), (row[-1][0] + row[-1][1] - 2, 8.5),
                        arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2.5, mutation_scale=24))
        name = f"d2_obj_{k}"
        save_raw(fig, name); names.append(name)
    trim_group(names)


# ------------------------------------------------------------------ 1.6 the four stages, with a worked example
def stage_example(tok, fm, pos):
    s = pos[3][:40].replace("U", "T")
    toks, H0, Hc, _ = run_model(tok, fm, s, special=False)
    toks, H0, Hc = toks[:6], H0[:6, :40], Hc[:6, :40]
    defs = ["cut the sequence\ninto tokens", "turn each token\ninto numbers", "mix the tokens:\nlayers of the model",
            "turn the numbers\ninto an answer"]
    names = []
    for k in range(5):
        fig, ax = canvas2()
        for i, ch in enumerate(s[:30]):
            label(ax, 8 + i * 2.6, 80, ch, fs=18, color=BASE[ch], family=MONO, bold=True)
        label(ax, 8 + 30 * 2.6, 80, "…", fs=18, color=MUTED)
        stage_row(ax, 62, h=10)
        for x, d in zip(XS, defs):
            label(ax, x + BW / 2, 55, d, fs=18, color=MUTED, va="top")
        rows_y = [38 - j * 5 for j in range(6)]                        # one row per token, same rows everywhere
        if k >= 1:
            for j, t in enumerate(toks):
                w = 2.2 + 1.9 * len(t)
                bx = XS[0] + BW / 2 - w / 2
                box(ax, bx, rows_y[j] - 2.3, w, 4.6, fc=[ACCENT_L, FAINT][j % 2], r=0.5)
                for q, ch in enumerate(t):
                    label(ax, bx + w / 2 + (q - (len(t) - 1) / 2) * 1.9, rows_y[j], ch, fs=17, color=BASE[ch],
                          family=MONO, bold=True)
            label(ax, XS[0] + BW / 2, 7, "tokens", fs=18, color=INK)
        if k >= 2:
            heat(ax, XS[1], rows_y[-1] - 2.2, BW, 30, H0, np.percentile(np.abs(H0), 97))
            label(ax, XS[1] + BW / 2, 7, "token embeddings\n(from a table)", fs=18, color=INK, va="center")
        if k >= 3:
            heat(ax, XS[2], rows_y[-1] - 2.2, BW, 30, Hc, np.percentile(np.abs(Hc), 97))
            label(ax, XS[2] + BW / 2, 7, "contextual embeddings\n(each row sees all tokens)", fs=18, color=INK)
        if k >= 4:
            label(ax, XS[3] + BW / 2, 30, "pretraining:\nthe next token", fs=19, color=HELD_C)
            label(ax, XS[3] + BW / 2, 16, "our task:\nP(bound)", fs=19, color=GOOD_C)
        name = f"d2_stages_{k}"
        save_raw(fig, name); names.append(name)
    return names


def compare():
    cnn = ["single letters", "one-hot, 4 numbers\n(fixed)", "2 convolution\nlayers", "linear → 1 logit"]
    lm = ["BPE tokens", "a learned table,\n256 numbers", "transformer,\n8 layers", "next token, or\nyour task"]
    names = []
    for k in (0, 1):
        fig, ax = canvas2()
        stage_row(ax, 62, h=10)
        label(ax, 8, 46, "CNN (part 1)", fs=21, color=GOOD_C, ha="left", bold=True)
        for x, d in zip(XS, cnn):
            label(ax, x + BW / 2, 38, d, fs=20, color=INK, va="top")
        if k == 1:
            label(ax, 8, 22, "DNA language model (Mistral-DNA)", fs=21, color=ACCENT, ha="left", bold=True)
            for x, d in zip(XS, lm):
                label(ax, x + BW / 2, 14, d, fs=20, color=INK, va="top")
        name = f"d2_compare_{k}"
        save_raw(fig, name); names.append(name)
    return names


# ------------------------------------------------------------------ 2.3 BPE, built with Mistral-DNA's own merge rules
def bpe_steps(tok, pos):
    w, _ = bound_window(pos)
    merges = json.loads(tok.backend_tokenizer.to_str())["model"]["merges"]
    rank = {(a, b): r for r, (a, b) in enumerate(merges)}
    cur, states = list(w), [(list(w), None)]
    while True:
        pairs = [(rank[(cur[i], cur[i + 1])], i) for i in range(len(cur) - 1) if (cur[i], cur[i + 1]) in rank]
        if not pairs:
            break
        r, _ = min(pairs)
        a, b = merges[r]
        out, i, hit = [], 0, []
        while i < len(cur):
            if i < len(cur) - 1 and cur[i] == a and cur[i + 1] == b:
                hit.append(len(out)); out.append(a + b); i += 2
            else:
                out.append(cur[i]); i += 1
        cur = out
        states.append((list(cur), (a, b, r, hit)))
    real = [t.replace("▁", "") for t in tok.tokenize(w)]
    assert cur == real, (cur, real)
    show = [0, 1, 2, 3, len(states) - 1]
    names = []
    for k in range(len(show)):
        fig, ax = canvas2()
        for j, si in enumerate(show[:k + 1]):
            toks, info = states[si]
            y = 76 - j * 16
            if info is None:
                head = f"Start: single letters, {len(toks)} tokens"
            elif si == len(states) - 1:
                head = f"After all {si} merges ({si - 3} not shown): {len(toks)} tokens, as Mistral-DNA's tokenizer cuts it"
            else:
                a, b, r, _ = info
                head = f"Merge {si}: {a} + {b} → {a + b}   (rule {r + 1:,} of {len(merges):,})"
            label(ax, 4, y + 6.5, head, fs=20, color=INK if si else MUTED, ha="left")
            hit = set(info[3]) if info and si != len(states) - 1 else set()
            x, dx = 4, 3.3
            for q, t in enumerate(toks):
                wq = dx * len(t)
                box(ax, x, y - 3.2, wq - 0.5, 6.4, fc=ACCENT_L if q in hit else FAINT, ec=ACCENT if q in hit else "none",
                    lw=1.5 if q in hit else 0, r=0.5)
                for c, ch in enumerate(t):
                    label(ax, x + dx / 2 + c * dx - 0.25, y, ch, fs=19, color=BASE[ch], family=MONO, bold=True)
                x += wq + 0.4
        name = f"d2_bpe_{k}"
        save_raw(fig, name); names.append(name)
    trim_group(names)
    print("bpe:", len(states) - 1, "merges apply;", [(states[i][1][0], states[i][1][1], states[i][1][2]) for i in (1, 2, 3)],
          "final", cur)


# ------------------------------------------------------------------ 3.1 the embedding as a matrix product
def emb_matmul(tok, fm, pos):
    w, _ = bound_window(pos)
    tid = tok(w, add_special_tokens=False)["input_ids"][0]
    ttxt = tok.convert_ids_to_tokens(tid).replace("▁", "")
    E = fm.get_input_embeddings().weight.detach().float().numpy()
    lo = max(0, tid - 5)
    rows = list(range(lo, lo + 11))
    Mx = E[rows, :48]
    vm = np.percentile(np.abs(E[:, :48]), 98)
    i_sel = rows.index(tid)
    names = []
    for k in (0, 1):
        fig, ax = canvas2()
        top, rh = 70, 5
        ys = [top - j * rh for j in range(len(rows))]
        # x_i: one entry beside each row of E
        label(ax, 18, 80, r"$x_i$", fs=28)
        label(ax, 18, 75.5, "one-hot", fs=17, color=MUTED)
        for j, yy in enumerate(ys):
            one = j == i_sel
            box(ax, 15, yy - rh + 0.4, 6, rh - 0.8, fc=ACCENT_L if (one and k) else "white", ec=HAIR, lw=1, r=0.3)
            label(ax, 18, yy - rh / 2, "1" if one else "0", fs=18, color=ACCENT if one else LIGHT, bold=one)
        label(ax, 18, ys[0] + 2, "⋮", fs=18, color=MUTED); label(ax, 18, ys[-1] - rh - 2.5, "⋮", fs=18, color=MUTED)
        label(ax, 27, 45, "×", fs=34, color=MUTED)
        X0 = 46
        # E: real rows
        heat(ax, X0, ys[-1] - rh, 60, rh * len(rows), Mx, vm)
        if k == 1:
            for j, yy in enumerate(ys):
                if j != i_sel:
                    ax.add_patch(Rectangle((X0, yy - rh), 60, rh, fc="white", alpha=0.7, ec="none", zorder=3))
            ax.add_patch(Rectangle((X0, ys[i_sel] - rh), 60, rh, fill=False, ec=ACCENT, lw=3, zorder=4))
        for j, (yy, r) in enumerate(zip(ys, rows)):
            if not (j == i_sel and k):
                label(ax, X0 - 1.5, yy - rh / 2, f"row {r:,}", fs=14, color=MUTED, ha="right")
        label(ax, X0 + 30, 80, r"$E$", fs=28)
        label(ax, X0 + 30, 75.5, f"learned table: {E.shape[0]:,} rows × {E.shape[1]} numbers (first 48 shown)", fs=17,
              color=MUTED)
        if k == 1:
            label(ax, 112, 45, "=", fs=34, color=MUTED)
            strip(ax, 118, 42.5, 38, rh, E[tid, :48], vm)
            label(ax, 137, 52, r"$e_i$", fs=28)
            label(ax, 137, 36, f"the token embedding of {ttxt}\n(row {tid:,} of E)", fs=17, color=INK, va="top")
            label(ax, X0 - 1.5, ys[i_sel] - rh / 2, f"row {tid:,}: {ttxt}", fs=14, color=ACCENT, ha="right", bold=True)
        name = f"d2_emb_{k}"
        save_raw(fig, name); names.append(name)
    trim_group(names)
    print("emb_matmul token", ttxt, "id", tid)


# ------------------------------------------------------------------ 4.1 the body, on one token in two contexts
def body_context(tok, fm, pos):
    """Find one token that occurs in two real bound sequences; same input row, different output rows."""
    found = None
    cache = []
    for s in pos[:400]:
        s = s[100:160].replace("U", "T")
        ids = tok(s, add_special_tokens=False)["input_ids"]
        cache.append((s, ids))
    for a in range(len(cache)):
        for t in cache[a][1][2:-2]:
            txt = tok.convert_ids_to_tokens(t).replace("▁", "")
            if len(txt) < 4:
                continue
            for b in range(a + 1, len(cache)):
                if t in cache[b][1][2:-2]:
                    found = (a, b, t, txt)
                    break
            if found:
                break
        if found:
            break
    a, b, t, txt = found
    rows = []
    for idx in (a, b):
        s, ids = cache[idx]
        toks, H0, Hc, ids2 = run_model(tok, fm, s, special=False)
        j = ids2.index(t)
        rows.append((toks[max(0, j - 2):j + 3], min(2, j), H0[j, :32], Hc[j, :32]))
    vin = np.abs(np.r_[rows[0][2], rows[1][2]]).max()
    vout = np.abs(np.r_[rows[0][3], rows[1][3]]).max()
    cos = float(np.dot(rows[0][3], rows[1][3]) / np.linalg.norm(rows[0][3]) / np.linalg.norm(rows[1][3]))
    names = []
    for k in (0, 1):
        fig, ax = canvas2()
        label(ax, 52, 80, f"token embedding of {txt}", fs=18, color=MUTED)
        label(ax, 132, 80, f"contextual embedding of {txt}", fs=18, color=MUTED)
        block(ax, 84, 12, 24, 62, "Body\n8 layers", fc=ACCENT, fs=24)
        for r_i, (ctx, j, vin_r, vout_r) in enumerate(rows[:k + 1]):
            y = 60 - r_i * 34
            label(ax, 4, y + 9, f"Sequence {r_i + 1}", fs=18, color=MUTED, ha="left")
            x = 4
            for q, tq in enumerate(ctx):
                wq = 2.4 + 2.1 * len(tq)
                hit = q == j
                box(ax, x, y - 3.5, wq, 7, fc=ACCENT_L if hit else FAINT, ec=ACCENT if hit else "none",
                    lw=2 if hit else 0, r=0.5)
                for c, ch in enumerate(tq):
                    label(ax, x + wq / 2 + (c - (len(tq) - 1) / 2) * 2.1, y, ch, fs=16, color=BASE.get(ch, INK),
                          family=MONO, bold=True)
                x += wq + 0.6
            strip(ax, 38, y - 14, 30, 6, vin_r, vin)
            arrow(ax, 69, y - 11, 83, y - 11, color=MUTED, lw=2, ms=20)
            arrow(ax, 109, y - 11, 115, y - 11, color=MUTED, lw=2, ms=20)
            strip(ax, 116, y - 14, 32, 6, vout_r, vout)
        if k == 1:
            label(ax, 53, 7, "the same row in both", fs=19, color=ACCENT, bold=True)
            label(ax, 132, 7, "different rows", fs=19, color=ACCENT, bold=True)
        name = f"d2_body_{k}"
        save_raw(fig, name); names.append(name)
    trim_group(names)
    print("body_context token", txt, "cosine of the two outputs", round(cos, 3),
          "inputs equal", bool(np.allclose(rows[0][2], rows[1][2])))


# ------------------------------------------------------------------ 5.1 heads on the walk-through sequence
def walk_tensors(tok, fm, pos):
    s = pos[3][:60].replace("U", "T")
    return run_model(tok, fm, s, special=True)


def heads(tok, fm, pos):
    toks, _, Hc, _ = walk_tensors(tok, fm, pos)
    v = np.percentile(np.abs(Hc), 97)
    n = len(toks)
    rh = 3.2
    top = 74
    names = []
    rows = [("Next token (pretraining)", n - 2, "256 → 4,096, softmax", "a probability for each\nof 4,096 tokens", HELD_C),
            ("Masked token (pretraining)", 6, "256 → 4,096, softmax", "a probability for each\nof 4,096 tokens", HELD_C),
            ("Our task", None, "256 → 1, sigmoid", "P(bound)", GOOD_C)]
    for k in range(3):
        fig, ax = canvas2()
        heat(ax, 12, top - n * rh, 32, n * rh, Hc, v)
        for j, t in enumerate(toks):
            label(ax, 11, top - (j + 0.5) * rh, t, fs=11, color=MUTED, ha="right", family=MONO)
        label(ax, 28, top + 4, "Body output: one row per token", fs=18, color=INK)
        label(ax, 28, top - n * rh - 4, "contextual embeddings", fs=17, color=MUTED)
        for j, (name, ri, head, out, c) in enumerate(rows[:k + 1]):
            y = 66 - j * 24
            if ri is not None:
                ax.add_patch(Rectangle((12, top - (ri + 1) * rh), 32, rh, fill=False, ec=c, lw=3, zorder=4))
                strip(ax, 52, y - 2, 30, 4, Hc[ri], v)
                arrow(ax, 45, top - (ri + 0.5) * rh, 51, y, color=c, lw=2, ms=18)
                label(ax, 67, y + 6, f"the row of {toks[ri]}", fs=16, color=MUTED)
            else:
                strip(ax, 52, y - 2, 30, 4, Hc.mean(0), v)
                arrow(ax, 45, top - n * rh / 2, 51, y, color=c, lw=2, ms=18)
                label(ax, 67, y + 6, "the average of all rows: the sequence embedding", fs=16, color=MUTED)
            label(ax, 52, y + 11.5, name, fs=19, color=c, ha="left", bold=True)
            arrow(ax, 83, y, 88, y, color=MUTED, lw=2, ms=18)
            block(ax, 89, y - 5, 33, 10, head, fc=c, fs=18)
            arrow(ax, 123, y, 128, y, color=MUTED, lw=2, ms=18)
            label(ax, 130, y, out, fs=18, color=INK, ha="left")
        nm = f"d2_heads_{k}"
        save_raw(fig, nm); names.append(nm)
    trim_group(names)


# ------------------------------------------------------------------ 5.2 the linear probe, written out
def probe(tok, fm, pos):
    toks, _, Hc, _ = walk_tensors(tok, fm, pos)
    v = np.percentile(np.abs(Hc), 97)
    e = Hc.mean(0)
    from pathlib import Path
    d = np.load(Path(__file__).resolve().parents[2] / "data" / "emb_mistral17m.npz")
    _, ty = split("train")
    names = []
    for k in (0, 1):
        fig, ax = canvas2()
        label(ax, 4, 80, "Body output (frozen)", fs=18, color=MUTED, ha="left")
        heat(ax, 4, 52, 30, 24, Hc, v)
        arrow(ax, 35, 64, 41, 64, color=MUTED, lw=2, ms=18)
        label(ax, 38, 69, "average", fs=16, color=MUTED)
        strip(ax, 42, 61, 48, 6, e, v)
        label(ax, 66, 73, "the sequence embedding", fs=18, color=INK)
        label(ax, 43.5, 57, r"$e_1$", fs=18); label(ax, 47, 57, r"$e_2$", fs=18)
        label(ax, 66, 57, "…", fs=18, color=MUTED); label(ax, 87.5, 57, r"$e_{256}$", fs=18)
        arrow(ax, 91, 64, 97, 64, color=MUTED, lw=2, ms=18)
        label(ax, 98, 64, "P(bound)", fs=24, color=GOOD_C, bold=True, ha="left")
        label(ax, 80, 40, r"$\hat p = \sigma\,(\,w_1 e_1 + w_2 e_2 + \dots + w_{256}\, e_{256} + b\,)$", fs=30)
        label(ax, 80, 30, r"$\sigma$: the sigmoid of part 1, turns the sum into a probability between 0 and 1",
              fs=18, color=MUTED)
        if k == 1:
            label(ax, 4, 19, "Trained on labelled sequences: only the 256 weights $w$ and the bias $b$ are learned",
                  fs=19, color=INK, ha="left")
            E = d["train"]
            y_sub = ty[np.random.default_rng(0).permutation(len(ty))[:len(E)]]   # as workshop_data() in the notebook
            b = [i for i in range(len(E)) if y_sub[i] == 1][:2]
            u = [i for i in range(len(E)) if y_sub[i] == 0][:1]
            pick = [b[0], u[0], b[1]]
            vm = np.percentile(np.abs(E[pick]), 97)
            for j, i in enumerate(pick):
                x = 4 + j * 52
                strip(ax, x, 6, 30, 5, E[i], vm)
                yy = int(y_sub[i])
                label(ax, x + 32, 8.5, "bound, y = 1" if yy == 1 else "unbound, y = 0", fs=17,
                      color=GOOD_C if yy == 1 else MUTED, ha="left")
        nm = f"d2_probe_{k}"
        save_raw(fig, nm); names.append(nm)
    trim_group(names)


# ------------------------------------------------------------------ 5.3 summary: pretraining, then adapting
def summary():
    names = []
    for k in (0, 1):
        fig, ax = canvas2()
        label(ax, 4, 80, "1  Pretraining: billions of DNA letters, no labels, every stage trained", fs=21, color=INK,
              ha="left", bold=True)
        stage_row(ax, 62, h=11, fs=20, colors=[ACCENT, ACCENT, ACCENT, HELD_C],
                  texts=["Tokenization", "Embedding", "Body", "Next-token head"])
        if k == 1:
            for x in XS[:3]:
                arrow(ax, x + BW / 2, 60.5, x + BW / 2, 51.5, color=MUTED, lw=2, ms=20)
            label(ax, XS[2] + BW / 2 + 2, 56, "kept", fs=18, color=MUTED, ha="left")
            label(ax, 4, 45, "2  Adapting: 12,000 labelled AGO2 sequences, only the new head trained", fs=21, color=INK,
                  ha="left", bold=True)
            stage_row(ax, 28, h=11, fs=20, colors=[FAINT, FAINT, FAINT, GOOD_C],
                      texts=["Tokenization", "Embedding", "Body", "AGO2 head"], tcolors=[MUTED, MUTED, MUTED, "white"])
            label(ax, XS[1] + BW / 2, 22, "frozen", fs=18, color=MUTED)
            label(ax, XS[3] + BW / 2, 22, "P(bound)", fs=18, color=GOOD_C)
            label(ax, 4, 9, "Train only the new head: a linear probe (today). Also update the body: fine-tuning (part 3).",
                  fs=19, color=INK, ha="left")
        nm = f"d2_summary_{k}"
        save_raw(fig, nm); names.append(nm)
    trim_group(names)


# ------------------------------------------------------------------ 6.1 what comes from Hugging Face
def hf():
    names = []
    for k in range(3):
        fig, ax = canvas2()
        stage_row(ax, 52, h=11, fs=20, colors=[ACCENT, ACCENT, ACCENT, GOOD_C])
        # bracket over the three pretrained stages
        x0, x1 = XS[0], XS[2] + BW
        ax.plot([x0, x0, x1, x1], [66, 69, 69, 66], color=INK, lw=2)
        label(ax, (x0 + x1) / 2, 74, "Pretrained, from the Hugging Face Hub", fs=21, color=INK, bold=True)
        label(ax, (x0 + x1) / 2, 79.5, "RaphaelMourad/Mistral-DNA-v1-17M-hg38", fs=17, color=MUTED, family=MONO)
        label(ax, XS[3] + BW / 2, 74, "Yours: TODO-2", fs=21, color=GOOD_C, bold=True)
        if k >= 1:
            ax.plot([XS[0], XS[0], XS[0] + BW, XS[0] + BW], [49, 46, 46, 49], color=ACCENT, lw=2)
            label(ax, XS[0], 39, "AutoTokenizer.from_pretrained(name)", fs=19, color=ACCENT, family=MONO, ha="left")
            label(ax, XS[0], 33, "the tokenizer: its 4,096-token vocabulary", fs=19, color=INK, ha="left")
        if k >= 2:
            ax.plot([XS[1], XS[1], XS[2] + BW, XS[2] + BW], [49, 46, 46, 49], color=ACCENT, lw=2)
            label(ax, XS[1], 22, "AutoModel.from_pretrained(name)", fs=19, color=ACCENT, family=MONO, ha="left")
            label(ax, XS[1], 16, "the model: the embedding table and the body, 16M weights", fs=19, color=INK, ha="left")
        nm = f"d2_hf_{k}"
        save_raw(fig, nm); names.append(nm)
    trim_group(names)


if __name__ == "__main__":
    want = set(sys.argv[1:])
    run = (lambda n: not want or n in want)
    tr, ty = split("train")
    pos = [s for s, l in zip(tr, ty) if l == 1]
    tok, fm = model_and_tokenizer()
    if run("adapt"):
        adapt(tok, fm, pos)
    if run("objectives"):
        objectives(tok, pos)
    if run("stages"):
        trim_group(stage_example(tok, fm, pos) + compare())      # the CNN comparison keeps the same block positions
    if run("bpe"):
        bpe_steps(tok, pos)
    if run("emb"):
        emb_matmul(tok, fm, pos)
    if run("body"):
        body_context(tok, fm, pos)
    if run("heads"):
        heads(tok, fm, pos)
    if run("probe"):
        probe(tok, fm, pos)
    if run("summary"):
        summary()
    if run("hf"):
        hf()
