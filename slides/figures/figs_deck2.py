"""Part 2 deck figures (rework of 15.10): drawn at each slide's shape, on a 10-inch canvas, so text reads at slide size.

    HF_HUB_OFFLINE=1 python figs_deck2.py [name ...]   (hw1-00660121 env; writes ../assets/custom/d2_*.png)

Scale: 160 units across 10 inches. A label of size fs is about 0.22 * fs units tall; a monospace letter is about
0.135 * fs units wide. Frames of one click-through, and figures on neighbouring slides that must not move, are
cropped as one group.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch, Rectangle

from deckstyle import *  # noqa: F401,F403
from figs_d2 import MISTRAL, STAGES, real_context, save_raw
from figs_parts import bound_window
from intuition import split

W = 160
ASPECT = json.loads((Path(__file__).resolve().parent / "aspects.json").read_text(encoding="utf-8"))


def canvas3(name, default=1.94):
    """A 10-inch-wide canvas at the shape of the slide space for `name` (from aspects.json)."""
    asp = ASPECT.get(name) or ASPECT.get(f"{name}_0") or default
    h = W / asp
    fig, ax = canvas(10, 10 / asp, ylim=(0, h))
    return fig, ax, h


def blk(ax, x, y, w, h, text, fc=ACCENT, tc="white", fs=22, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4", fc=fc, ec="none"))
    label(ax, x + w / 2, y + h / 2, text, fs=fs, color=tc, bold=bold)


def letters_w(n, fs):
    return 0.135 * fs * n


def tok_row(ax, x, y, toks, fs=22, h=None, hide=None, hide_text="?", gap=1.0, pad=2.0):
    """Tokens in boxes; returns [(x, w)] per box."""
    h = h or 0.33 * fs
    out = []
    for j, t in enumerate(toks):
        w = max(letters_w(max(len(t), 2), fs) + pad, letters_w(len(hide_text), fs) + pad if j == hide else 0)
        if j == hide:
            box(ax, x, y - h / 2, w, h, fc="white", ec=HELD_C, lw=2.5, r=0.8)
            label(ax, x + w / 2, y, hide_text, fs=fs, color=HELD_C, bold=True)
        else:
            box(ax, x, y - h / 2, w, h, fc=[ACCENT_L, FAINT][j % 2], r=0.8)
            dx = 0.135 * fs
            for q, ch in enumerate(t):
                label(ax, x + w / 2 + (q - (len(t) - 1) / 2) * dx, y, ch, fs=fs, color=BASE.get(ch, INK),
                      family=MONO, bold=True)
        out.append((x, w))
        x += w + gap
    return out


def strip(ax, x, y, w, h, vec, vm, vertical=False):
    v = np.asarray(vec, dtype=float)
    M = v.reshape(-1, 1) if vertical else v.reshape(1, -1)
    ax.imshow(M, cmap=DIVERGING, vmin=-vm, vmax=vm, extent=(x, x + w, y, y + h), aspect="auto",
              interpolation="nearest", zorder=2)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=HAIR, lw=1, zorder=3))


def heat(ax, x, y, w, h, M, vm):
    ax.imshow(M, cmap=DIVERGING, vmin=-vm, vmax=vm, extent=(x, x + w, y, y + h), aspect="auto",
              interpolation="nearest", zorder=2)


def stage_row(ax, y, xs, bw, h=13, fs=21, colors=None, texts=None, tcolors=None):
    for k, x in enumerate(xs):
        blk(ax, x, y, bw, h, (texts or STAGES_2L)[k], fc=colors[k] if colors else ACCENT,
            tc=tcolors[k] if tcolors else "white", fs=fs)
        if k < 3:
            arrow(ax, x + bw + 0.8, y + h / 2, xs[k + 1] - 0.8, y + h / 2, color=MUTED, lw=2.5, ms=22)


STAGES_2L = ["Tokenization", "Embedding", "Body", "Prediction\nhead"]
XS, BW = [2 + k * 40.5 + 0 for k in range(4)], 34      # the four stage blocks, used on every pipeline figure


def model_and_tokenizer():
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MISTRAL, trust_remote_code=True)
    fm = AutoModel.from_pretrained(MISTRAL, trust_remote_code=True).eval()
    return tok, fm


def run_model(tok, fm, s, special=True):
    enc = tok(s, return_tensors="pt", add_special_tokens=special)
    with torch.no_grad():
        hs = fm(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], output_hidden_states=True).hidden_states
    toks = [t.replace("▁", "") for t in tok.convert_ids_to_tokens(enc["input_ids"][0])]
    return toks, hs[0][0].numpy(), hs[-1][0].numpy(), enc["input_ids"][0].tolist()


# ------------------------------------------------------------------ 1.2 and 1.3: body, embedding, heads
def adapt(tok, fm, pos):
    ctx, nxt = real_context(tok, pos, n_show=4)
    _, _, Hd, _ = run_model(tok, fm, "".join(ctx), special=False)
    e_dna, e_lang = Hd[-1][:10], np.sin(np.arange(10) * 1.9) * 1.2
    lang = [("next-word head", "mat", HELD_C), ("sentiment head", "neutral", GOOD_C), ("translation head", "Le chat …", GOOD_C)]
    dna = [("next-token head", nxt, HELD_C), ("AGO2 head", "P(bound)", GOOD_C), ("expression head", "a number", GOOD_C)]
    frames = [("Model", ["The", "cat", "sat", "on", "the"], e_lang, lang[:1]), ("Model", ["The", "cat", "sat", "on", "the"], e_lang, lang),
              ("Mistral-DNA", ctx, e_dna, dna[:1]), ("Mistral-DNA", ctx, e_dna, dna)]
    names = []
    for k, (body, toks, e, heads) in enumerate(frames):
        fig, ax, H = canvas3("d2_adapt_0")
        top = H - 6
        if body == "Model":
            x = 2
            for j, t in enumerate(toks + ["?"]):
                w = 15
                if t == "?":
                    box(ax, x, top - 4.5, w, 9, fc="white", ec=HELD_C, lw=2.5, r=0.8)
                    label(ax, x + w / 2, top, "?", fs=24, color=HELD_C, bold=True)
                else:
                    box(ax, x, top - 4.5, w, 9, fc=[ACCENT_L, FAINT][j % 2], r=0.8)
                    label(ax, x + w / 2, top, t, fs=24)
                x += w + 1
        else:
            tok_row(ax, 2, top, toks + [""], fs=22, hide=len(toks))
        arrow(ax, 22, top - 6, 22, top - 13, color=MUTED, lw=2.5, ms=24)
        blk(ax, 4, 12, 36, top - 26, body, fs=28)
        label(ax, 22, 5, "Body", fs=24, color=ACCENT, bold=True)
        arrow(ax, 41, (top - 14) / 2 + 12, 47, (top - 14) / 2 + 12, color=MUTED, lw=2.5, ms=24)
        mid = (top - 14) / 2 + 12
        strip(ax, 48, mid - 16, 6, 32, e, np.abs(e).max(), vertical=True)
        label(ax, 51, 5, "Embedding", fs=22, color=INK, bold=True)
        ys = [mid] if len(heads) == 1 else [mid + 20, mid, mid - 20]
        for y, (head, out, c) in zip(ys, heads):
            arrow(ax, 55, mid + (y - mid) * 0.3, 63, y, color=MUTED, lw=2.5, ms=22)
            blk(ax, 64, y - 7, 46, 14, head, fc=c, fs=24)
            arrow(ax, 111, y, 117, y, color=MUTED, lw=2.5, ms=22)
            if out and set(out) <= set("ACGT"):
                for q, ch in enumerate(out):
                    label(ax, 119 + q * 3.4, y, ch, fs=26, color=BASE[ch], family=MONO, bold=True, ha="left")
            else:
                label(ax, 119, y, out, fs=26, ha="left")
        label(ax, 92, 5, "Head: one per task", fs=22, color=MUTED, bold=True)
        nm = f"d2_adapt_{k}"; save_raw(fig, nm); names.append(nm)
    trim_group(names)


# ------------------------------------------------------------------ 1.4 two objectives
def objectives(tok, pos):
    ctx, nxt = real_context(tok, pos, n_show=6)
    names = []
    for k in (0, 1):
        fig, ax, H = canvas3("d2_obj_0")
        y0 = H - 6
        label(ax, 2, y0, "Next token (Mistral-DNA)", fs=26, bold=True, ha="left")
        label(ax, 2, y0 - 8, r"$\mathcal{L} = -\log\, p_\theta(x_t \mid x_{<t})$", fs=26, color=MUTED, ha="left")
        row = tok_row(ax, 2, y0 - 22, ctx + [""], fs=26, hide=len(ctx))
        ax.annotate("", (row[-1][0] + row[-1][1] / 2, y0 - 30), (row[0][0] + 2, y0 - 30),
                    arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=3, mutation_scale=28))
        if k == 1:
            y1 = y0 - 42
            m = 3
            toks = ctx[:m] + ["[MASK]"] + ctx[m + 1:] + [nxt]
            label(ax, 2, y1, "Masked token (Nucleotide Transformer)", fs=26, bold=True, ha="left")
            label(ax, 2, y1 - 8, r"$\mathcal{L} = -\log\, p_\theta(x_t \mid x_{\backslash t})$", fs=26, color=MUTED, ha="left")
            row = tok_row(ax, 2, y1 - 22, toks, fs=26, hide=m, hide_text="[MASK]")
            cx = row[m][0] + row[m][1] / 2
            for a, b in ((row[0][0] + 2, cx - 4), (row[-1][0] + row[-1][1] - 2, cx + 4)):
                ax.annotate("", (b, y1 - 30), (a, y1 - 30), arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=3, mutation_scale=28))
        nm = f"d2_obj_{k}"; save_raw(fig, nm); names.append(nm)
    trim_group(names)


# ------------------------------------------------------------------ 1.5 the next-token probabilities, with the true token's rank
def next_probs(tok, pos):
    from transformers import AutoModelForCausalLM
    lm = AutoModelForCausalLM.from_pretrained(MISTRAL, trust_remote_code=True).eval()
    ids = tok(pos[7].replace("U", "T"), return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    cut = 14
    with torch.no_grad():
        p = torch.softmax(lm(input_ids=ids[None, :cut]).logits[0, -1], -1)
    true = int(ids[cut])
    rank = int((p > p[true]).sum()) + 1
    top = torch.topk(p, 6)
    items = [(tok.convert_ids_to_tokens(int(i)).replace("▁", ""), float(v), False) for v, i in zip(top.values, top.indices)]
    items.append((tok.convert_ids_to_tokens(true).replace("▁", ""), float(p[true]), True))
    fig = plt.figure(figsize=(10, 10 / (ASPECT.get("d2_next") or 1.88)), dpi=120)
    fig._canvas = True
    ax = fig.add_axes([0.2, 0.14, 0.62, 0.8])
    ys = list(range(len(items) - 1)) + [len(items)]
    ax.barh(ys, [v for _, v, _ in items], color=[HELD_C if t else TRAIN_L for *_, t in items], height=0.72)
    for y, (n, v, t) in zip(ys, items):
        txt = f"{v:.3f}" if not t else (f"{v:.4f}" if v >= 1e-4 else "below 0.0001") + f", rank {rank:,} of 4,096"
        ax.text(v + 0.0015, y, txt, va="center", fontsize=22, color=HELD_C if t else INK)
    ax.set_yticks(ys); ax.set_yticklabels([n + ("  true" if t else "") for n, _, t in items], family=MONO, fontsize=22)
    for lab, (_, _, t) in zip(ax.get_yticklabels(), items):
        lab.set_color(HELD_C if t else INK)
    ax.invert_yaxis(); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlim(0, top.values[0].item() * 1.35)
    ax.set_xlabel("Predicted probability of the next token", fontsize=22); ax.tick_params(axis="x", labelsize=20)
    ax.text(-0.02 * top.values[0].item(), len(items) - 1, "…", fontsize=26, color=MUTED, ha="right", va="center")
    fig.savefig(OUT / "d2_next.png", dpi=200, facecolor="white"); plt.close(fig); trim(OUT / "d2_next.png")
    print("next_probs true", items[-1][0], round(items[-1][1], 5), "rank", rank, "top", items[0][0], round(items[0][1], 3))


# ------------------------------------------------------------------ 1.6 four stages with a worked example, and 4.3 the comparison
def stage_frames(tok, fm, pos):
    s = pos[3][:40].replace("U", "T")
    toks, H0, Hc, _ = run_model(tok, fm, s, special=False)
    toks, H0, Hc = toks[:5], H0[:5, :32], Hc[:5, :32]
    defs = ["cut the sequence\ninto tokens", "turn each token\ninto numbers", "mix the tokens", "give the answer"]
    names = []
    for k in range(5):
        fig, ax, H = canvas3("d2_stages_0")
        for i, ch in enumerate(s[:36]):
            label(ax, 3 + i * 3.4, H - 4, ch, fs=22, color=BASE[ch], family=MONO, bold=True)
        label(ax, 3 + 36 * 3.4, H - 4, "…", fs=22, color=MUTED)
        yb = H - 24
        stage_row(ax, yb, XS, BW)
        for x, d in zip(XS, defs):
            label(ax, x + BW / 2, yb - 3, d, fs=20, color=MUTED, va="top")
        rows = [yb - 19 - j * 6 for j in range(5)]
        if k >= 1:
            for j, t in enumerate(toks):
                w = letters_w(len(t), 20) + 2
                bx = XS[0] + BW / 2 - w / 2
                box(ax, bx, rows[j] - 3, w, 6, fc=[ACCENT_L, FAINT][j % 2], r=0.6)
                for q, ch in enumerate(t):
                    label(ax, bx + w / 2 + (q - (len(t) - 1) / 2) * 2.7, rows[j], ch, fs=20, color=BASE[ch], family=MONO, bold=True)
        if k >= 2:
            heat(ax, XS[1], rows[-1] - 3, BW, rows[0] - rows[-1] + 6, H0, np.percentile(np.abs(H0), 97))
            label(ax, XS[1] + BW / 2, rows[-1] - 9, "token\nembeddings", fs=20, color=INK, bold=True)
        if k >= 3:
            heat(ax, XS[2], rows[-1] - 3, BW, rows[0] - rows[-1] + 6, Hc, np.percentile(np.abs(Hc), 97))
            label(ax, XS[2] + BW / 2, rows[-1] - 9, "contextual\nembeddings", fs=20, color=INK, bold=True)
        if k >= 4:
            label(ax, XS[3] + BW / 2, rows[1], "the next token", fs=22, color=HELD_C, bold=True)
            label(ax, XS[3] + BW / 2, rows[3], "or P(bound)", fs=22, color=GOOD_C, bold=True)
        nm = f"d2_stages_{k}"; save_raw(fig, nm); names.append(nm)
    cnn = ["single letters", "one-hot,\nfixed", "2 convolution\nlayers", "linear,\n1 number"]
    lm = ["BPE tokens", "learned table,\n256 numbers", "transformer,\n8 layers", "next token,\nor your task"]
    for k in (0, 1):
        fig, ax, H = canvas3("d2_stages_0")
        yb = H - 24
        stage_row(ax, yb, XS, BW)
        label(ax, 2, yb - 7, "CNN (part 1)", fs=24, color=GOOD_C, ha="left", bold=True)
        for x, d in zip(XS, cnn):
            label(ax, x + BW / 2, yb - 12, d, fs=22, va="top")
        if k == 1:
            label(ax, 2, yb - 33, "DNA language model (Mistral-DNA)", fs=24, color=ACCENT, ha="left", bold=True)
            for x, d in zip(XS, lm):
                label(ax, x + BW / 2, yb - 38, d, fs=22, va="top")
        nm = f"d2_compare_{k}"; save_raw(fig, nm); names.append(nm)
    trim_group(names)


# ------------------------------------------------------------------ 2.3 BPE with Mistral-DNA's merge rules
def bpe_steps(tok, pos):
    w, _ = bound_window(pos)
    merges = json.loads(tok.backend_tokenizer.to_str())["model"]["merges"]
    rank = {(a, b): r for r, (a, b) in enumerate(merges)}
    cur, states = list(w), [(list(w), None)]
    while True:
        pairs = [(rank[(cur[i], cur[i + 1])], i) for i in range(len(cur) - 1) if (cur[i], cur[i + 1]) in rank]
        if not pairs:
            break
        r = min(pairs)[0]
        a, b = merges[r]
        out, i, hit = [], 0, []
        while i < len(cur):
            if i < len(cur) - 1 and cur[i] == a and cur[i + 1] == b:
                hit.append(len(out)); out.append(a + b); i += 2
            else:
                out.append(cur[i]); i += 1
        cur = out
        states.append((list(cur), (a, b, r, hit)))
    assert cur == [t.replace("▁", "") for t in tok.tokenize(w)]
    show = [0, 1, 2, 3, len(states) - 1]
    names = []
    for k in range(len(show)):
        fig, ax, H = canvas3("d2_bpe_0")
        step = (H - 2) / 5
        for j, si in enumerate(show[:k + 1]):
            toks, info = states[si]
            y = H - 4 - j * step
            if info is None:
                head = "Start: single letters, 30 tokens"
            elif si == len(states) - 1:
                head = f"After all {si} merges: {len(toks)} tokens, as Mistral-DNA cuts it"
            else:
                a, b, r, _ = info
                head = f"Merge {si}: {a} + {b} → {a + b}   (rule {r + 1} of {len(merges):,})"
            label(ax, 2, y, head, fs=22, color=INK, ha="left", bold=si > 0)
            hit = set(info[3]) if info and si != len(states) - 1 else set()
            x, dx = 2, 4.85
            for q, t in enumerate(toks):
                wq = dx * len(t)
                yy = y - step * 0.5
                box(ax, x, yy - 3.6, wq - 0.6, 7.2, fc=ACCENT_L if q in hit else FAINT, ec=ACCENT if q in hit else "none",
                    lw=2 if q in hit else 0, r=0.6)
                for c, ch in enumerate(t):
                    label(ax, x + dx / 2 + c * dx - 0.3, yy, ch, fs=24, color=BASE[ch], family=MONO, bold=True)
                x += wq + 0.2
        nm = f"d2_bpe_{k}"; save_raw(fig, nm); names.append(nm)
    trim_group(names)


# ------------------------------------------------------------------ 3.1 token -> ID -> one-hot -> row of E
def emb_matmul(tok, fm, pos):
    w, _ = bound_window(pos)
    tid = tok(w, add_special_tokens=False)["input_ids"][0]
    ttxt = tok.convert_ids_to_tokens(tid).replace("▁", "")
    E = fm.get_input_embeddings().weight.detach().float().numpy()
    rows = list(range(tid - 3, tid + 4))
    vm = np.percentile(np.abs(E[:, :32]), 98)
    sel = rows.index(tid)
    names = []
    for k in range(3):
        fig, ax, H = canvas3("d2_emb_0")
        rh = (H - 22) / len(rows)
        ys = [H - 12 - j * rh for j in range(len(rows))]              # top of each row
        mid = (ys[0] + ys[-1] - rh) / 2
        # 1 the token and its ID
        tok_row(ax, 2, mid, [ttxt], fs=26)
        label(ax, 9, mid + 7.5, "token", fs=20, color=MUTED)
        arrow(ax, 17, mid, 23, mid, color=MUTED, lw=2.5, ms=22)
        label(ax, 33, mid + 7.5, "its ID", fs=20, color=MUTED)
        label(ax, 33, mid, f"{tid}", fs=30, color=ACCENT, bold=True)
        if k >= 1:                                                  # 2 the one-hot vector x_i
            arrow(ax, 39, mid, 45, mid, color=MUTED, lw=2.5, ms=22)
            label(ax, 50, H - 4, r"$x_i$", fs=28)
            for j, yy in enumerate(ys):
                one = j == sel
                box(ax, 47, yy - rh + 0.5, 6.5, rh - 1, fc=ACCENT_L if one else "white", ec=HAIR, lw=1, r=0.3)
                label(ax, 50.25, yy - rh / 2, "1" if one else "0", fs=22, color=ACCENT if one else LIGHT, bold=one)
            label(ax, 50.25, ys[-1] - rh - 3, "⋮", fs=20, color=MUTED)
            label(ax, 50.25, 3, "4,096 numbers", fs=18, color=MUTED)
        if k >= 2:                                                  # 3 x_i^T E selects one row
            label(ax, 58, mid, "×", fs=32, color=MUTED)
            for j, (yy, r) in enumerate(zip(ys, rows)):
                label(ax, 76, yy - rh / 2, f"{r}", fs=18, color=ACCENT if j == sel else MUTED, ha="right", bold=j == sel)
            heat(ax, 78, ys[-1] - rh, 44, rh * len(rows), E[rows, :32], vm)
            for j, yy in enumerate(ys):
                if j != sel:
                    ax.add_patch(Rectangle((78, yy - rh), 44, rh, fc="white", alpha=0.65, ec="none", zorder=3))
            ax.add_patch(Rectangle((78, ys[sel] - rh), 44, rh, fill=False, ec=ACCENT, lw=3, zorder=4))
            label(ax, 100, H - 4, r"$E$", fs=28)
            label(ax, 100, 3, "4,096 rows × 256, learned", fs=18, color=MUTED)
            label(ax, 127, mid, "=", fs=32, color=MUTED)
            strip(ax, 132, mid - rh / 2, 26, rh, E[tid, :32], vm)
            label(ax, 145, H - 4, r"$e_i$", fs=28)
            label(ax, 145, mid - 8, f"row {tid}", fs=20, color=ACCENT, bold=True)
        nm = f"d2_emb_{k}"; save_raw(fig, nm); names.append(nm)
    trim_group(names)
    print("emb token", ttxt, "id", tid)


# ------------------------------------------------------------------ 4.1 one token, two contexts
def body_context(tok, fm, pos):
    cache = []
    for s in pos[:400]:
        s = s[100:150].replace("U", "T")
        cache.append((s, tok(s, add_special_tokens=False)["input_ids"]))
    found = None
    for a in range(len(cache)):
        for t in cache[a][1][2:-2]:
            if len(tok.convert_ids_to_tokens(t)) < 5:
                continue
            for b in range(a + 1, len(cache)):
                if t in cache[b][1][2:-2]:
                    found = (a, b, t); break
            if found:
                break
        if found:
            break
    a, b, t = found
    txt = tok.convert_ids_to_tokens(t).replace("▁", "")
    rows = []
    for idx in (a, b):
        s, _ = cache[idx]
        toks, H0, Hc, ids = run_model(tok, fm, s, special=False)
        j = ids.index(t)
        lo = max(0, j - 1)
        rows.append((toks[lo:j + 2], j - lo, H0[j, :24], Hc[j, :24]))
    vin = np.abs(np.r_[rows[0][2], rows[1][2]]).max()
    vout = np.abs(np.r_[rows[0][3], rows[1][3]]).max()
    cos = float(np.dot(rows[0][3], rows[1][3]) / np.linalg.norm(rows[0][3]) / np.linalg.norm(rows[1][3]))
    names = []
    for k in (0, 1):
        fig, ax, H = canvas3("d2_body_0")
        blk(ax, 66, 12, 26, H - 16, "Body", fs=28)
        for r_i, (ctx, j, vi, vo) in enumerate(rows[:k + 1]):
            y = H - 8 - r_i * (H - 20) / 2
            label(ax, 2, y, f"Sequence {r_i + 1}", fs=20, color=MUTED, ha="left")
            boxes = tok_row(ax, 2, y - 9, ctx, fs=20)
            bx, bw = boxes[j]
            ax.add_patch(Rectangle((bx, y - 12.3), bw, 6.6, fill=False, ec=ACCENT, lw=3, zorder=4))
            strip(ax, 2, y - 22, 54, 6, vi, vin)
            arrow(ax, 57, y - 19, 65, y - 19, color=MUTED, lw=2.5, ms=22)
            arrow(ax, 93, y - 19, 100, y - 19, color=MUTED, lw=2.5, ms=22)
            strip(ax, 101, y - 22, 56, 6, vo, vout)
        left = "the same row in both" if k == 1 else f"token embedding of {txt}"
        right = "different rows" if k == 1 else f"contextual embedding"
        label(ax, 29, 4, left, fs=22, color=ACCENT, bold=True)
        label(ax, 129, 4, right, fs=22, color=ACCENT, bold=True)
        nm = f"d2_body_{k}"; save_raw(fig, nm); names.append(nm)
    trim_group(names)
    print("body token", txt, "cos", round(cos, 3), "inputs equal", bool(np.allclose(rows[0][2], rows[1][2])))


# ------------------------------------------------------------------ 5.1 heads, 5.2 probe: the walk-through sequence
def walk_tensors(tok, fm, pos):
    return run_model(tok, fm, pos[3][:60].replace("U", "T"), special=True)


def heads(tok, fm, pos):
    toks, _, Hc, _ = walk_tensors(tok, fm, pos)
    v = np.percentile(np.abs(Hc), 97)
    n = len(toks)
    rows = [("Next token", n - 2, "256 → 4,096\nsoftmax", "4,096\nprobabilities", HELD_C),
            ("Masked token", 6, "256 → 4,096\nsoftmax", "4,096\nprobabilities", HELD_C),
            ("Our task", None, "256 → 1\nsigmoid", "P(bound)", GOOD_C)]
    names = []
    for k in range(3):
        fig, ax, H = canvas3("d2_heads_0")
        top, bot = H - 9, 9
        heat(ax, 2, bot, 30, top - bot, Hc, v)
        rh = (top - bot) / n
        label(ax, 17, H - 3.5, "Body output", fs=22, bold=True)
        label(ax, 17, 3, f"{n} tokens", fs=20, color=MUTED)
        step = (H - 6) / 3
        for j, (name, ri, head, out, c) in enumerate(rows[:k + 1]):
            y = H - 10 - j * step
            if ri is not None:
                ax.add_patch(Rectangle((2, top - (ri + 1) * rh), 30, rh, fill=False, ec=c, lw=3, zorder=4))
                src = top - (ri + 0.5) * rh
                vec = Hc[ri]
            else:
                src, vec = (top + bot) / 2, Hc.mean(0)
            arrow(ax, 33, src, 40, y - 6, color=c, lw=2.5, ms=20)
            label(ax, 41, y + 1, name, fs=21, color=c, ha="left", bold=True)
            strip(ax, 41, y - 9, 34, 6, vec, v)
            arrow(ax, 76, y - 6, 82, y - 6, color=MUTED, lw=2.5, ms=20)
            blk(ax, 83, y - 12, 34, 12, head, fc=c, fs=20)
            arrow(ax, 118, y - 6, 124, y - 6, color=MUTED, lw=2.5, ms=20)
            label(ax, 125, y - 6, out, fs=22, ha="left")
        nm = f"d2_heads_{k}"; save_raw(fig, nm); names.append(nm)
    trim_group(names)


def probe(tok, fm, pos):
    toks, _, Hc, _ = walk_tensors(tok, fm, pos)
    v = np.percentile(np.abs(Hc), 97)
    e = Hc.mean(0)
    d = np.load(Path(__file__).resolve().parents[2] / "data" / "emb_mistral17m.npz")
    _, ty = split("train")
    Etr = d["train"]
    y_sub = ty[np.random.default_rng(0).permutation(len(ty))[:len(Etr)]]       # as workshop_data() in the notebook
    b = [i for i in range(len(Etr)) if y_sub[i] == 1][:2]
    u = [i for i in range(len(Etr)) if y_sub[i] == 0][:1]
    pick = [b[0], u[0], b[1]]
    names = []
    for k in (0, 1):
        fig, ax, H = canvas3("d2_probe_0")
        top = H - 3
        label(ax, 14, top, "Body output", fs=20, color=MUTED)
        heat(ax, 2, top - 24, 24, 20, Hc, v)
        arrow(ax, 27, top - 14, 34, top - 14, color=MUTED, lw=2.5, ms=20)
        strip(ax, 35, top - 18, 70, 8, e, v)
        label(ax, 70, top - 4, "the sequence embedding (average)", fs=20, color=INK)
        label(ax, 37, top - 23, r"$e_1$", fs=22); label(ax, 44, top - 23, r"$e_2$", fs=22)
        label(ax, 70, top - 23, "…", fs=22, color=MUTED); label(ax, 102, top - 23, r"$e_{256}$", fs=22)
        arrow(ax, 106, top - 14, 112, top - 14, color=MUTED, lw=2.5, ms=20)
        label(ax, 113, top - 14, "P(bound)", fs=28, color=GOOD_C, bold=True, ha="left")
        label(ax, 80, top - 33, r"$\hat p = \sigma\,(\,w_1 e_1 + w_2 e_2 + \dots + w_{256}\, e_{256} + b\,)$", fs=32)
        label(ax, 80, top - 42, r"$\sigma$: the sigmoid of part 1", fs=22, color=MUTED)
        if k == 1:
            label(ax, 2, top - 52, "Trained on 12,000 labelled sequences: only $w$ and $b$ are learned", fs=22, ha="left")
            vm = np.percentile(np.abs(Etr[pick]), 97)
            for j, i in enumerate(pick):
                x = 2 + j * 54
                strip(ax, x, top - 64, 18, 6, Etr[i], vm)
                yy = int(y_sub[i])
                label(ax, x + 20, top - 61, "bound, y = 1" if yy else "unbound, y = 0", fs=20,
                      color=GOOD_C if yy else MUTED, ha="left")
        nm = f"d2_probe_{k}"; save_raw(fig, nm); names.append(nm)
    trim_group(names)


# ------------------------------------------------------------------ 5.3 summary, 6.1 Hugging Face
def summary():
    names = []
    for k in (0, 1, 2):                                  # 2: the same adaptation, with the body trained too
        fig, ax, H = canvas3("d2_summary_0")
        label(ax, 2, H - 4, "1  Pretraining: billions of DNA letters, no labels", fs=24, ha="left", bold=True)
        yb1 = H - 22
        stage_row(ax, yb1, XS, BW, colors=[ACCENT, ACCENT, ACCENT, HELD_C],
                  texts=["Tokenization", "Embedding", "Body", "Next-token\nhead"])
        if k >= 1:
            for x in XS[:3]:
                arrow(ax, x + BW / 2, yb1 - 1.5, x + BW / 2, yb1 - 9, color=MUTED, lw=2.5, ms=22)
            label(ax, XS[2] + BW / 2 + 3, yb1 - 5.5, "kept", fs=20, color=MUTED, ha="left")
            label(ax, 2, yb1 - 15, "2  Adapting: 12,000 labelled AGO2 sequences", fs=24, ha="left", bold=True)
            yb2 = yb1 - 33
            if k == 1:                                   # linear probe: the pretrained parts stay frozen
                stage_row(ax, yb2, XS, BW, colors=[FAINT, FAINT, FAINT, GOOD_C],
                          texts=["Tokenization", "Embedding", "Body", "AGO2 head"], tcolors=[MUTED, MUTED, MUTED, "white"])
                label(ax, XS[1] + BW / 2 + 20, yb2 - 5, "frozen", fs=22, color=MUTED, bold=True)
                label(ax, XS[3] + BW / 2, yb2 - 5, "trained", fs=22, color=GOOD_C, bold=True)
                label(ax, 2, yb2 - 15, "Linear probe (today): only the new head learns", fs=22, ha="left", bold=True)
                label(ax, 2, yb2 - 22, "the pretrained model stays as it is", fs=22, ha="left", color=MUTED)
            else:                                        # fine-tuning: the pretrained parts keep learning
                stage_row(ax, yb2, XS, BW, colors=[FAINT, TUNED_C, TUNED_C, GOOD_C],
                          texts=["Tokenization", "Embedding", "Body", "AGO2 head"], tcolors=[MUTED, "white", "white", "white"])
                label(ax, XS[1] + BW / 2 + 20, yb2 - 5, "trained further", fs=22, color=TUNED_C, bold=True)
                label(ax, XS[3] + BW / 2, yb2 - 5, "trained", fs=22, color=GOOD_C, bold=True)
                label(ax, 2, yb2 - 15, "Fine-tuning (part 3): the body learns too", fs=22, ha="left", bold=True,
                      color=TUNED_C)
                label(ax, 2, yb2 - 22, "continue training the pretrained model on your own labelled data", fs=22,
                      ha="left", color=MUTED)
        nm = f"d2_summary_{k}"; save_raw(fig, nm); names.append(nm)
    trim_group(names)


def hf():
    names = []
    for k in range(3):
        fig, ax, H = canvas3("d2_hf_0")
        yb = H - 34
        stage_row(ax, yb, XS, BW, colors=[ACCENT, ACCENT, ACCENT, GOOD_C])
        x0, x1 = XS[0], XS[2] + BW
        ax.plot([x0, x0, x1, x1], [yb + 15, yb + 18, yb + 18, yb + 15], color=INK, lw=2.5)
        label(ax, (x0 + x1) / 2, yb + 23, "Pretrained, from the Hugging Face Hub", fs=24, bold=True)
        label(ax, (x0 + x1) / 2, yb + 30, "RaphaelMourad/Mistral-DNA-v1-17M-hg38", fs=20, color=MUTED, family=MONO)
        label(ax, XS[3] + BW / 2, yb + 23, "Yours: TODO-2", fs=24, color=GOOD_C, bold=True)
        if k >= 1:
            ax.plot([XS[0], XS[0], XS[0] + BW, XS[0] + BW], [yb - 2, yb - 5, yb - 5, yb - 2], color=ACCENT, lw=2.5)
            label(ax, 2, yb - 11, "AutoTokenizer.from_pretrained(name)", fs=21, color=ACCENT, family=MONO, ha="left")
            label(ax, 2, yb - 18, "the tokenizer and its 4,096-token vocabulary", fs=21, ha="left")
        if k >= 2:
            ax.plot([XS[1], XS[1], XS[2] + BW, XS[2] + BW], [yb - 2, yb - 5, yb - 5, yb - 2], color=ACCENT, lw=2.5)
            label(ax, XS[1], yb - 28, "AutoModel.from_pretrained(name)", fs=21, color=ACCENT, family=MONO, ha="left")
            label(ax, XS[1], yb - 35, "the embedding table and the body", fs=21, ha="left")
        nm = f"d2_hf_{k}"; save_raw(fig, nm); names.append(nm)
    trim_group(names)


if __name__ == "__main__":
    want = set(sys.argv[1:])
    run = (lambda n: not want or n in want)
    tr, ty = split("train")
    pos = [s for s, l in zip(tr, ty) if l == 1]
    tok, fm = model_and_tokenizer()
    for n, job in [("adapt", lambda: adapt(tok, fm, pos)), ("objectives", lambda: objectives(tok, pos)),
                   ("next", lambda: next_probs(tok, pos)), ("stages", lambda: stage_frames(tok, fm, pos)),
                   ("bpe", lambda: bpe_steps(tok, pos)), ("emb", lambda: emb_matmul(tok, fm, pos)),
                   ("body", lambda: body_context(tok, fm, pos)), ("heads", lambda: heads(tok, fm, pos)),
                   ("probe", lambda: probe(tok, fm, pos)), ("summary", summary), ("hf", hf)]:
        if run(n):
            job()
