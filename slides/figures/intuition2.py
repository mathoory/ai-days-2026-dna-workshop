"""Second batch of mechanism figures, replacing the box cartoons the audit flagged.

All from real data and real models. Writes to slides/assets/custom/.
Run locally in hw1-00660121 (torch + transformers): python intuition2.py [names...]
"""
import sys

import numpy as np
import torch
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.textpath import TextPath
import matplotlib.transforms as mtrans
from sklearn.metrics import roc_auc_score, roc_curve

from deckstyle import *  # noqa: F401,F403
from intuition import (ALPH, DATA, DEV, acts, cnn_scores, dinuc_shuffle, load_cnn, onehot, split)

MIR16 = "UAGCAGCACGUAAAUAUUGGCG"          # hsa-miR-16-5p
FP = FontProperties(family="DejaVu Sans", weight="bold")


def draw_letter(ax, ch, x, y, w, h, color):
    if h <= 0.002:
        return
    tp = TextPath((0, 0), ch, size=1, prop=FP)
    bb = tp.get_extents()
    t = mtrans.Affine2D().translate(-bb.x0, -bb.y0).scale(w / bb.width, h / bb.height).translate(x, y)
    ax.add_patch(PathPatch(t.transform_path(tp), color=color, lw=0))


def draw_logo(ax, pwm):
    """Information-content sequence logo (bits) from a (L, 4) probability table."""
    p = np.clip(pwm, 1e-9, 1)
    ic = 2 + (p * np.log2(p)).sum(1)
    for i in range(len(pwm)):
        y = 0
        for b in np.argsort(pwm[i]):
            h = pwm[i, b] * ic[i]
            draw_letter(ax, ALPH[b], i + 0.05, y, 0.9, h, BASE[ALPH[b]])
            y += h
    ax.set_xlim(0, len(pwm)); ax.set_ylim(0, 2); ax.set_ylabel("bits", fontsize=18)
    ax.set_xticks([]); ax.tick_params(labelsize=14)


def windows_pwm(model, f, seqs, K=10):
    A = acts(model, seqs)[:, f, :]
    cut = A.mean() + 0.5 * (A.max() - A.mean())
    rows, cols = np.nonzero(A > cut)
    W = [seqs[i][j:j + K] for i, j in zip(rows, cols)]
    W = [w for w in W if len(w) == K and set(w) <= set(ALPH)]
    arr = np.frombuffer("".join(W).encode(), dtype=np.uint8).reshape(-1, K)
    return np.stack([(arr == ord(b)).mean(0) for b in ALPH], 1), len(W)


def gc(s):
    return (s.count("G") + s.count("C")) / len(s)


# ------------------------------------------------------------------------------------ slot 1
def seed_pairing_real(pos):
    site = "UGCUGCU"
    s = next(x for x in pos if site in x[60:-60])
    k = s.index(site, 60)
    w0 = k - 9
    win = s[w0:w0 + 26]
    kw = k - w0
    L = len(MIR16)
    mir_rev = MIR16[::-1]
    fig, ax = canvas()
    x0, dx = 14, 5.0
    label(ax, x0 - 6, 62, "5′", fs=22, ha="right"); label(ax, x0 + len(win) * dx + 1, 62, "3′", fs=22, ha="left")
    label(ax, 4, 72, "a real bound mRNA stretch", fs=19, color=GREY, ha="left")
    for i, ch in enumerate(win):
        inside = kw <= i < kw + 7
        label(ax, x0 + i * dx, 62, ch, fs=30, color=BASE[ch], bold=True)
        if inside:
            ax.add_patch(Rectangle((x0 + i * dx - 2.4, 56.5), dx, 11, fc=PURPLE_L, ec="none", zorder=0))
    # mir_rev index m sits under window position kw + (m - (L - 8))
    for m, ch in enumerate(mir_rev):
        i = kw + (m - (L - 8))
        if i < -1 or i > len(win):
            continue
        x = x0 + i * dx
        seed = L - 8 <= m <= L - 2
        label(ax, x, 36, ch, fs=30, color=PURPLE if seed else "#9e9e9e", bold=seed)
        if seed:
            ax.plot([x, x], [41.5, 56], color=PURPLE, lw=3)
    xl = x0 + (kw + (L - 1 - (L - 8))) * dx
    label(ax, xl + 4, 36, "5′", fs=22, color=PURPLE, ha="left")
    label(ax, x0 + (kw - 14) * dx, 26, "miR-16, read 3′ → 5′", fs=20, color=PURPLE, ha="left")
    label(ax, 80, 84, "AGO2 + microRNA: letters 2–8 of the microRNA (the seed) pair with the mRNA", fs=24, bold=True)
    label(ax, 80, 12, "A·U  G·C  ·  site UGCUGCU  ·  the seed match is AGO2's main targeting rule", fs=22, color=GREY)
    save(fig, "seed_pairing")


def labels_raster(tr, ty):
    rng = np.random.default_rng(0)
    pos = [s for s, l in zip(tr, ty) if l == 1]; neg = [s for s, l in zip(tr, ty) if l == 0]
    P = [pos[i] for i in rng.choice(len(pos), 150, replace=False)]
    N = [neg[i] for i in rng.choice(len(neg), 150, replace=False)]
    cmap = {b: matplotlib.colors.to_rgb(BASE[b]) for b in ALPH}

    def img(S):
        S = sorted(S, key=gc)
        M = np.ones((len(S), 330, 3))
        for r, s in enumerate(S):
            for c, ch in enumerate(s[:330]):
                M[r, c] = cmap.get(ch, (1, 1, 1))
        return M

    fig, axes = plt.subplots(1, 2, figsize=(16, 9), dpi=120)
    for ax, S, t in ((axes[0], P, "150 bound"), (axes[1], N, "150 unbound")):
        ax.imshow(img(S), aspect="auto", interpolation="nearest")
        m = np.mean([gc(s) for s in S])
        ax.set_title(f"{t} · mean G+C {m:.2f}", fontsize=24, color=NAVY)
        ax.set_xlabel("position (letters)", fontsize=18); ax.set_yticks([]); ax.tick_params(labelsize=14)
    axes[0].set_ylabel("one row per sequence, sorted by G+C", fontsize=18)
    for b in ALPH:
        fig.text(0.40 + "ACGU".index(b) * 0.06, 0.035, b, color=BASE[b], fontsize=26, fontweight="bold")
    fig.subplots_adjust(left=0.04, right=0.98, top=0.9, bottom=0.12, wspace=0.06)
    save(fig, "labels_raster")


import matplotlib  # noqa: E402  (used by labels_raster)


def lr_training(tr, ty):
    rng = np.random.default_rng(1)
    idx = rng.choice(len(tr), 3000, replace=False)
    X = np.array([[gc(tr[i]), tr[i].count("U") / len(tr[i])] for i in idx]); y = ty[idx].astype(float)
    mu, sd = X.mean(0), X.std(0); Z = (X - mu) / sd
    w, b, lr = np.array([0.0, 0.0]), 0.0, 0.5
    w = np.array([-1.0, -1.0]); b = 0.8   # a deliberately bad start so the line visibly moves
    snaps, losses = {}, []
    for step in range(301):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        loss = -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
        losses.append(loss)
        if step in (0, 10, 300):
            snaps[step] = (w.copy(), b, loss)
        g = p - y
        w -= lr * Z.T @ g / len(y); b -= lr * g.mean()
    fig = plt.figure(figsize=(16, 9), dpi=120)
    for k, step in enumerate((0, 10, 300)):
        ax = fig.add_axes([0.065 + k * 0.225, 0.18, 0.2, 0.62])
        o = rng.permutation(len(y))
        ax.scatter(X[o, 0], X[o, 1], c=np.where(y[o] == 1, GREEN, "#9e9e9e"), s=6, alpha=.45)
        ws, bs, ls = snaps[step]
        xs = np.linspace(X[:, 0].min(), X[:, 0].max(), 50)
        zx = (xs - mu[0]) / sd[0]
        zy = -(ws[0] * zx + bs) / (ws[1] if abs(ws[1]) > 1e-6 else 1e-6)
        ax.plot(xs, zy * sd[1] + mu[1], color=RED, lw=3.5)
        ax.set_ylim(X[:, 1].min(), X[:, 1].max())
        ax.set_title(f"step {step} · loss {ls:.3f}", fontsize=21, color=NAVY)
        ax.set_xlabel("G + C fraction", fontsize=17)
        if k == 0:
            ax.set_ylabel("U fraction", fontsize=17)
        ax.tick_params(labelsize=13)
    ax = fig.add_axes([0.79, 0.18, 0.19, 0.62])
    ax.plot(losses, color=RED, lw=3); ax.set_xlabel("gradient step", fontsize=17); ax.set_ylabel("loss", fontsize=17)
    ax.set_title("loss", fontsize=21, color=NAVY); ax.tick_params(labelsize=13)
    fig.text(0.5, 0.9, "two parameters + a bias · green = bound, grey = unbound · the red line is where p = 0.5",
             ha="center", fontsize=20, color=NAVY)
    save(fig, "lr_training")


def auroc_real(model, ls, ly):
    s = cnn_scores(model, ls)
    fig = plt.figure(figsize=(16, 9), dpi=120)
    a1 = fig.add_axes([0.06, 0.16, 0.44, 0.66]); a2 = fig.add_axes([0.60, 0.16, 0.34, 0.66])
    bins = np.linspace(0, 1, 30)
    a1.hist(s[ly == 0], bins, color="#9e9e9e", alpha=.6, label="unbound (500)")
    a1.hist(s[ly == 1], bins, color=GREEN, alpha=.6, label="bound (500)")
    a1.set_xlabel("CNN score on the held-out set", fontsize=21); a1.set_ylabel("sequences", fontsize=21)
    a1.legend(fontsize=20, frameon=False); a1.tick_params(labelsize=17)
    fpr, tpr, _ = roc_curve(ly, s)
    a2.plot(fpr, tpr, color=GREEN, lw=4); a2.fill_between(fpr, tpr, color=GREEN, alpha=.12)
    a2.plot([0, 1], [0, 1], "--", color=GREY, lw=2)
    a2.text(0.38, 0.2, f"AUROC = {roc_auc_score(ly, s):.2f}", fontsize=26, color=GREEN, fontweight="bold")
    a2.set_xlabel("false positive rate", fontsize=20); a2.set_ylabel("true positive rate", fontsize=20)
    a2.tick_params(labelsize=16)
    fig.text(0.5, 0.9, "slide a threshold across the left plot → one point per threshold on the right",
             ha="center", fontsize=22, color=NAVY)
    save(fig, "auroc")


# ------------------------------------------------------------------------------------ slot 2
def network_strip(model):
    fig = plt.figure(figsize=(16, 9), dpi=120)
    ax = fig.add_axes([0, 0.42, 1, 0.55]); ax.set_xlim(0, 160); ax.set_ylim(0, 50); ax.axis("off")
    steps = [("input", "1 × 4 × 501", GREEN_L, GREEN, 501, 4), ("Conv 4×10 + ReLU", "10 × 492", BLUE_L, BLUE, 492, 10),
             ("MaxPool 3", "10 × 164", BLUE_L, BLUE, 164, 10), ("Conv 1×5 + ReLU", "20 × 160", BLUE_L, BLUE, 160, 20),
             ("MaxPool 3", "20 × 53", BLUE_L, BLUE, 53, 20), ("Flatten", "1,060", GOLD_L, GOLD, 1060, 1),
             ("Linear", "1 score", GOLD_L, GOLD, 1, 1)]
    x = 3
    for i, (name, shape, fc, ec, L, C) in enumerate(steps):
        w = 18
        h = 8 + 22 * (np.log(L + 1) / np.log(1100)) if i < 6 else 6
        box(ax, x, 22 - h / 2, w, h, "", fc=fc, ec=ec, lw=2.5, r=0.8)
        label(ax, x + w / 2, 22 + h / 2 + 4, name, fs=17, bold=True)
        label(ax, x + w / 2, 22 - h / 2 - 4, shape, fs=17, color=GREY)
        if i < len(steps) - 1:
            arrow(ax, x + w + 0.3, 22, x + w + 4.2, 22, lw=2, ms=16)
        x += w + 4.6
    a2 = fig.add_axes([0.06, 0.07, 0.86, 0.3])
    W = model.net[-1].weight.detach().cpu().numpy().reshape(20, 53)
    v = np.abs(W).max()
    a2.imshow(W, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    a2.set_ylabel("20 channels", fontsize=16); a2.set_yticks([])
    a2.set_xlabel("53 positions along the sequence (each ≈ 9 letters)", fontsize=16); a2.tick_params(labelsize=13)
    a2.set_title("the last layer: one learned weight per (channel, position) · red helps 'bound', blue hurts",
                 fontsize=18, color=NAVY)
    save(fig, "network")


def filter_logo(model, pos, f=0):
    W = model.net[0].weight[f, 0].detach().cpu().numpy()
    pwm, n = windows_pwm(model, f, pos[:5000])
    fig = plt.figure(figsize=(16, 9), dpi=120)
    a1 = fig.add_axes([0.05, 0.2, 0.42, 0.6]); a2 = fig.add_axes([0.55, 0.2, 0.42, 0.6])
    v = np.abs(W).max()
    a1.imshow(W, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    a1.set_yticks(range(4)); a1.set_yticklabels(list(ALPH), fontsize=22, fontweight="bold")
    for t, b in zip(a1.get_yticklabels(), ALPH):
        t.set_color(BASE[b])
    a1.set_xticks(range(10)); a1.set_xticklabels(range(1, 11), fontsize=14)
    for r in range(4):
        for c in range(10):
            a1.text(c, r, f"{W[r, c]:+.2f}", ha="center", va="center", fontsize=12,
                    color="white" if abs(W[r, c]) > 0.55 * v else NAVY)
    a1.set_title(f"filter {f:02d}: the 40 weights (red +, blue −)", fontsize=22, color=NAVY)
    draw_logo(a2, pwm)
    a2.set_title(f"its logo: letters in the {n:,} windows it fired on", fontsize=22, color=NAVY)
    fig.text(0.5, 0.08, "the logo keeps what the filter likes · the negative weights (what it avoids) are gone",
             ha="center", fontsize=21, color=GREY)
    save(fig, "filter_logo")


def position_scan(model, pos, f=0):
    pwm, _ = windows_pwm(model, f, pos[:5000])
    motif = "".join(ALPH[i] for i in pwm.argmax(1))
    rng = np.random.default_rng(0)
    canv = ["".join(ALPH[i] for i in row) for row in rng.integers(0, 4, (400, 330))]
    shuf = "".join(rng.permutation(list(motif)))
    base = cnn_scores(model, canv).mean()
    xs = list(range(0, 321, 4))

    def scan(m):
        out = []
        for x in xs:
            S = [c[:x] + m + c[x + len(m):] for c in canv]
            out.append(cnn_scores(model, S).mean() - base)
        return np.array(out)

    r, sh = scan(motif), scan(shuf)
    W = model.net[-1].weight.detach().cpu().numpy().reshape(20, 53)
    fig = plt.figure(figsize=(16, 9), dpi=120)
    ax = fig.add_axes([0.08, 0.16, 0.86, 0.66])
    ax.plot(xs, r, color=BLUE, lw=3.5, label=f"one copy of {motif}")
    ax.plot(xs, sh, color="#9e9e9e", lw=3, ls="--", label=f"shuffled ({shuf})")
    ax.axhline(0, color=NAVY, lw=1)
    for x in (40, 85, 130, 175, 220, 265, 310):
        ax.axvline(x, color=RED, lw=1.5, alpha=.5, ls=":")
    ax.text(42, ax.get_ylim()[1] * 0.92, "where Slot 3's plant() writes copies", color=RED, fontsize=16)
    ax.set_xlabel("position where the motif is planted", fontsize=21)
    ax.set_ylabel("change in mean score", fontsize=21); ax.tick_params(labelsize=17)
    ax.legend(fontsize=18, frameon=False, loc="lower right")
    fig.text(0.5, 0.9, "the same motif helps at some positions and not others: the CNN learned where, not only what",
             ha="center", fontsize=21, color=NAVY)
    save(fig, "position_scan")


# ------------------------------------------------------------------------------------ slot 4/5
def tokens_real(pos):
    from transformers import AutoTokenizer
    s = next(x for x in pos if "UUUGCAC" in x[40:-40])
    k = s.index("UUUGCAC", 40)
    w = s[k - 14:k + 22].replace("U", "T")
    site = 14
    toks = {}
    names = [("Mistral-DNA (BPE)", "RaphaelMourad/Mistral-DNA-v1-17M-hg38", w),
             ("DNABERT-2 (BPE)", "zhihan1996/DNABERT-2-117M", w),
             ("Nucleotide Transformer (6-mers)", "InstaDeepAI/nucleotide-transformer-500m-human-ref", w),
             ("… same sequence, one letter later", "InstaDeepAI/nucleotide-transformer-500m-human-ref", w[1:])]
    for name, repo, seq_ in names:
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        t = tok.convert_ids_to_tokens(tok(seq_, add_special_tokens=False)["input_ids"])
        toks[name] = (t, seq_)
        print(name, t)
    fig, ax = canvas()
    dx, x0 = 3.95, 10
    for i, ch in enumerate(w):
        label(ax, x0 + i * dx, 79, ch, fs=21, color=BASE[ch], bold=True)
    ax.add_patch(Rectangle((x0 + site * dx - 2, 74.5), 7 * dx, 9, fc="none", ec=PURPLE, lw=3))
    label(ax, x0 + (site + 3) * dx, 87, "miR-19 site", fs=17, color=PURPLE, bold=True)
    for r, (name, (t, seq_)) in enumerate(toks.items()):
        y = 64 - r * 17
        label(ax, 4, y + 7.5, name, fs=18, color=NAVY, ha="left", bold=True)
        pos_ = 0 if seq_ == w else 1
        for j, tk in enumerate(t):
            tk = tk.replace("▁", "").replace("Ġ", "")
            if not tk or tk.startswith("<") or tk.startswith("["):
                continue
            L = len(tk)
            box(ax, x0 + pos_ * dx - 1.9, y - 4, L * dx - 0.3, 7.5, "", fc=[PURPLE_L, BLUE_L][j % 2],
                ec=[PURPLE, BLUE][j % 2], lw=1.6, r=0.5)
            for q, ch in enumerate(tk):
                label(ax, x0 + (pos_ + q) * dx, y - 0.2, ch, fs=17, color=BASE.get(ch, NAVY), bold=True)
            pos_ += L
        ax.add_patch(Rectangle((x0 + site * dx - 2, y - 4.6), 7 * dx, 8.7, fc="none", ec=PURPLE, lw=1.5, ls="--"))
    save(fig, "tokens")


def pipeline_real(pos):
    from transformers import AutoModel, AutoTokenizer
    name = "RaphaelMourad/Mistral-DNA-v1-17M-hg38"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    fm = AutoModel.from_pretrained(name, trust_remote_code=True).eval()
    s = pos[3][:120].replace("U", "T")
    enc = tok(s, return_tensors="pt")
    enc = {k: enc[k] for k in ("input_ids", "attention_mask")}
    with torch.no_grad():
        H = fm(**enc, output_hidden_states=True).hidden_states[-1][0].numpy()
    toks = tok.convert_ids_to_tokens(enc["input_ids"][0])
    ids = enc["input_ids"][0].tolist()
    fig = plt.figure(figsize=(16, 9), dpi=120)
    a0 = fig.add_axes([0.02, 0.83, 0.96, 0.12]); a0.axis("off")
    a0.text(0.0, 0.75, "sequence  " + s[:60] + "…", fontsize=17, family="DejaVu Sans Mono", color=NAVY)
    a0.text(0.0, 0.2, "tokens    " + " | ".join(toks[:9]) + " …", fontsize=17, family="DejaVu Sans Mono", color=PURPLE)
    a1 = fig.add_axes([0.15, 0.3, 0.82, 0.46])
    v = np.percentile(np.abs(H), 98)
    a1.imshow(H, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto", interpolation="nearest")
    a1.set_yticks(range(len(toks))); a1.set_yticklabels([f"{t} ({i})" for t, i in zip(toks, ids)], fontsize=9,
                                                      family="DejaVu Sans Mono")
    a1.set_xticks([])
    a1.set_title(f"last layer: one row of 256 numbers per token ({H.shape[0]} tokens)", fontsize=20, color=NAVY)
    a2 = fig.add_axes([0.15, 0.15, 0.82, 0.05])
    a2.imshow(H.mean(0, keepdims=True), cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto", interpolation="nearest")
    a2.set_yticks([0]); a2.set_yticklabels(["average"], fontsize=15, color=NAVY); a2.tick_params(labelsize=14)
    a2.set_xlabel("256 numbers", fontsize=17)
    fig.text(0.56, 0.235, "↓ average the rows (padding excluded by the attention mask)", ha="center", fontsize=18,
             color=NAVY)
    fig.text(0.5, 0.03, "this one row is the sequence's embedding · the probe sees only this",
             ha="center", fontsize=21, color=GREY)
    save(fig, "pipeline_real")


def next_token(pos):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "RaphaelMourad/Mistral-DNA-v1-17M-hg38"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    lm = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True).eval()
    s = pos[7].replace("U", "T")
    ids = tok(s, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    cut = 14
    with torch.no_grad():
        logits = lm(input_ids=ids[None, :cut]).logits[0, -1]
    p = torch.softmax(logits, -1)
    top = torch.topk(p, 8)
    true = int(ids[cut])
    names = [tok.convert_ids_to_tokens(int(i)) for i in top.indices]
    fig = plt.figure(figsize=(16, 9), dpi=120)
    ax = fig.add_axes([0.32, 0.12, 0.62, 0.62])
    cols = [RED if int(i) == true else PURPLE for i in top.indices]
    ax.barh(range(8), top.values.numpy(), color=cols)
    ax.set_yticks(range(8)); ax.set_yticklabels(names, fontsize=20, family="DejaVu Sans Mono"); ax.invert_yaxis()
    ax.set_xlabel("probability the model gives each possible next token", fontsize=19); ax.tick_params(labelsize=16)
    prefix = "".join(t for t in tok.convert_ids_to_tokens(ids[:cut]))
    fig.text(0.02, 0.86, "after reading:  …" + prefix[-60:], fontsize=17, family="DejaVu Sans Mono", color=NAVY)
    fig.text(0.02, 0.8, f"true next token: {tok.convert_ids_to_tokens(true)}" +
             ("  (red = in the top 8)" if true in top.indices.tolist() else "  (not in the top 8)"),
             fontsize=18, color=RED)
    fig.text(0.02, 0.4, f"{len(tok)} possible\ntokens", fontsize=22, color=GREY)
    save(fig, "next_token")


def probe_trap(tr, ty):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    d = np.load(DATA / "emb_mistral17m.npz")
    idx = np.random.default_rng(0).permutation(len(tr))[:12000]
    ls, ly = split("ls")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1)).fit(d["train"], ty[idx])
    a_tr = roc_auc_score(ty[idx], clf.predict_proba(d["train"])[:, 1])
    a_te = roc_auc_score(ly, clf.predict_proba(d["test"])[:, 1])
    over = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=100)).fit(d["train"][:600], ty[idx][:600])
    o_tr = roc_auc_score(ty[idx][:600], over.predict_proba(d["train"][:600])[:, 1])
    o_te = roc_auc_score(ly, over.predict_proba(d["test"])[:, 1])
    fig = plt.figure(figsize=(16, 9), dpi=120)
    ax = fig.add_axes([0.1, 0.14, 0.85, 0.68])
    x = np.array([0, 1])
    ax.bar(x - 0.18, [a_tr, o_tr], 0.34, color="#ef9a9a", label="scored on the sequences it was fit on")
    ax.bar(x + 0.18, [a_te, o_te], 0.34, color=PURPLE, label="scored on the held-out test set")
    for xi, a, b in zip(x, (a_tr, o_tr), (a_te, o_te)):
        ax.text(xi - 0.18, a + 0.01, f"{a:.2f}", ha="center", fontsize=24, fontweight="bold")
        ax.text(xi + 0.18, b + 0.01, f"{b:.2f}", ha="center", fontsize=24, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(["probe, 12,000 sequences", "probe, 600 sequences, weak regularisation"], fontsize=20)
    ax.set_ylim(0.5, 1.08); ax.set_ylabel("AUROC", fontsize=20); ax.tick_params(labelsize=17)
    ax.legend(fontsize=18, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2)
    fig.text(0.5, 0.95, "the same probe, two answers · only the held-out number means anything", ha="center",
             fontsize=22, color=NAVY)
    save(fig, "probe_trap")
    print("probe train/test", a_tr, a_te, "overfit", o_tr, o_te)


# ------------------------------------------------------------------------------------ slot 6
def shuffle_demo_real(pos):
    rng = np.random.default_rng(3)
    s = next(x for x in pos if "UGCUGCU" in x)
    k = s.index("UGCUGCU")
    sh = dinuc_shuffle(s, rng)
    pairs = [a + b for a in ALPH for b in ALPH]

    def counts(x):
        return [sum(1 for i in range(len(x) - 1) if x[i:i + 2] == p) for p in pairs]

    fig = plt.figure(figsize=(16, 9), dpi=120)
    a0 = fig.add_axes([0.02, 0.62, 0.96, 0.3]); a0.set_xlim(0, 160); a0.set_ylim(0, 30); a0.axis("off")
    st = max(0, k - 14)
    seq(a0, 26, 22, s[st:st + 34], fs=22, dx=3.9)
    seq(a0, 26, 8, sh[st:st + 34], fs=22, dx=3.9)
    label(a0, 2, 22, "original", fs=19, color=GREY, ha="left"); label(a0, 2, 8, "shuffled", fs=19, color=GREY, ha="left")
    a0.add_patch(Rectangle((26 + (k - st) * 3.9 - 1.9, 18), 7 * 3.9, 8, fc="none", ec=PURPLE, lw=3))
    label(a0, 26 + (k - st + 3) * 3.9, 28.5, "miR-16 site", fs=16, color=PURPLE, bold=True)
    a1 = fig.add_axes([0.07, 0.1, 0.9, 0.42])
    x = np.arange(16)
    a1.bar(x - 0.2, counts(s), 0.38, color=NAVY, label="original")
    a1.bar(x + 0.2, counts(sh), 0.38, color="#ef6c00", label="shuffled")
    a1.set_xticks(x); a1.set_xticklabels(pairs, fontsize=17, family="DejaVu Sans Mono")
    a1.set_ylabel("count in the\nwhole sequence", fontsize=17); a1.legend(fontsize=17, frameon=False); a1.tick_params(labelsize=14)
    a1.set_title(f"all 16 letter pairs: identical counts · site still present after shuffle: {'UGCUGCU' in sh}",
                 fontsize=19, color=NAVY)
    save(fig, "shuffle_demo")


if __name__ == "__main__":
    which = sys.argv[1:] or ["seed", "raster", "lr", "auroc", "network", "logo", "scan", "tokens", "pipe",
                             "next", "trap", "shuffle"]
    tr, ty = split("train"); ls, ly = split("ls")
    pos = [s for s, l in zip(tr, ty) if l == 1]
    model = load_cnn()
    jobs = {"seed": lambda: seed_pairing_real(pos), "raster": lambda: labels_raster(tr, ty),
            "lr": lambda: lr_training(tr, ty), "auroc": lambda: auroc_real(model, ls, ly),
            "network": lambda: network_strip(model), "logo": lambda: filter_logo(model, pos),
            "scan": lambda: position_scan(model, pos), "tokens": lambda: tokens_real(pos),
            "pipe": lambda: pipeline_real(pos), "next": lambda: next_token(pos),
            "trap": lambda: probe_trap(tr, ty), "shuffle": lambda: shuffle_demo_real(pos)}
    for w in which:
        try:
            jobs[w]()
        except Exception as e:
            import traceback
            print("FAILED", w); traceback.print_exc()
