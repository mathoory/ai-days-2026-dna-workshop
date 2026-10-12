"""Audit leftovers: large real figures for Slot 3 and a real-weight LoRA figure for Slot 7."""
import numpy as np
import torch

from deckstyle import *  # noqa: F401,F403
from intuition import ALPH, DEV, cnn_scores, load_cnn, split
from intuition2 import draw_logo, windows_pwm

rng = np.random.default_rng(0)


def random_rna(n, L=330):
    return ["".join(ALPH[i] for i in row) for row in rng.integers(0, 4, (n, L))]


def plant(seqs, motif, copies, spacing=45, start=40):
    out = []
    for s in seqs:
        s = list(s)
        for c in range(copies):
            a = start + c * spacing
            s[a:a + len(motif)] = list(motif)
        out.append("".join(s))
    return out


def logo_bound_unbound(model, pos, neg, f=0):
    pb, nb = windows_pwm(model, f, pos[:5000])
    pu, nu = windows_pwm(model, f, neg[:5000])
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), dpi=120)
    draw_logo(axes[0], pb); axes[0].set_title(f"filter {f:02d} on BOUND RNA ({nb:,} windows)", fontsize=24, color=GREEN)
    draw_logo(axes[1], pu); axes[1].set_title(f"filter {f:02d} on UNBOUND RNA ({nu:,} windows)", fontsize=24, color=GREY)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.05, hspace=0.35)
    save(fig, "logo_bound_unbound")
    return "".join(ALPH[i] for i in pb.argmax(1))


def dose_response(model, motif):
    canvas = random_rna(1000)
    base = cnn_scores(model, canvas).mean()
    shuf = "".join(rng.permutation(list(motif)))
    copies = list(range(1, 9))
    real = [cnn_scores(model, plant(canvas, motif, c)).mean() for c in copies]
    ctrl = [cnn_scores(model, plant(canvas, shuf, c)).mean() for c in copies]
    fig = plt.figure(figsize=(16, 9), dpi=120)
    ax = fig.add_axes([0.09, 0.14, 0.86, 0.72])
    ax.plot(copies, real, "o-", color=BLUE, lw=4, ms=12, label=f"motif {motif}")
    ax.plot(copies, ctrl, "x--", color=NAVY, lw=3, ms=12, label=f"same letters shuffled ({shuf})")
    ax.axhline(base, ls=":", color=RED, lw=2.5, label="random RNA, nothing planted")
    for c in copies:
        ax.text(c, min(real + ctrl) - 0.03, f"{40 + (c - 1) * 45}", ha="center", fontsize=15, color=GREY)
    ax.text(0.6, min(real + ctrl) - 0.03, "letter:", ha="right", fontsize=15, color=GREY)
    ax.set_xlabel("copies planted (each new copy goes further along the sequence)", fontsize=21)
    ax.set_ylabel("mean CNN score", fontsize=21); ax.tick_params(labelsize=18); ax.grid(alpha=.3)
    ax.set_ylim(min(real + ctrl) - 0.06, max(real + ctrl) + 0.04)
    ax.legend(fontsize=18, frameon=False, loc="upper left")
    save(fig, "dose_response")


def planting_bars_sd(model, ls, ly):
    sites = {"miR-15/16\nUGCUGCU": "UGCUGCU", "miR-19\nUUUGCAC": "UUUGCAC", "miR-30\nUGUUUAC": "UGUUUAC",
             "let-7\nCUACCUC": "CUACCUC"}
    canvas = [s for s, y in zip(ls, ly) if y == 0]
    base = cnn_scores(model, canvas).mean()
    r, m, sd = [], [], []
    for name, site in sites.items():
        r.append(cnn_scores(model, plant(canvas, site, 6)).mean())
        sh = [cnn_scores(model, plant(canvas, "".join(rng.permutation(list(site))), 6)).mean() for _ in range(10)]
        m.append(np.mean(sh)); sd.append(np.std(sh))
    fig = plt.figure(figsize=(16, 9), dpi=120)
    ax = fig.add_axes([0.1, 0.14, 0.85, 0.72])
    x = np.arange(4)
    ax.bar(x - 0.2, r, 0.38, color=PURPLE, label="site planted (6 copies)")
    ax.bar(x + 0.2, m, 0.38, yerr=sd, capsize=8, color="#bdbdbd", label="same letters shuffled (mean ± sd of 10)")
    ax.axhline(base, ls="--", color=NAVY, lw=2, label="nothing planted")
    for xi, a in zip(x, r):
        ax.text(xi - 0.2, a + 0.01, f"{a:.2f}", ha="center", fontsize=20, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(list(sites), fontsize=21)
    ax.set_ylabel("CNN score on real unbound RNA", fontsize=21); ax.tick_params(labelsize=18)
    ax.set_ylim(0.2, 0.62); ax.legend(fontsize=19, frameon=False, loc="upper right")
    save(fig, "planting_bars")
    print("planting", [round(v, 3) for v in r], [round(v, 3) for v in m], [round(v, 3) for v in sd], round(base, 3))


def lora_real():
    from transformers import AutoModel
    fm = AutoModel.from_pretrained("RaphaelMourad/Mistral-DNA-v1-17M-hg38", trust_remote_code=True)
    W = next(mod.weight for n, mod in fm.named_modules() if n.endswith("q_proj")).detach().numpy()
    torch.manual_seed(0)
    r = 16
    A = np.random.default_rng(0).normal(0, 0.02, (r, W.shape[1]))
    fig = plt.figure(figsize=(16, 9), dpi=120)
    a1 = fig.add_axes([0.04, 0.16, 0.42, 0.68]); a2 = fig.add_axes([0.56, 0.16, 0.025, 0.68])
    a3 = fig.add_axes([0.62, 0.62, 0.34, 0.04])
    v = np.percentile(np.abs(W), 99)
    a1.imshow(W, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto"); a1.set_xticks([]); a1.set_yticks([])
    a1.set_title(f"W: one real weight matrix of Mistral-DNA\n{W.shape[0]} × {W.shape[1]} = {W.size:,} numbers · frozen",
                 fontsize=20, color=NAVY)
    a2.imshow(np.zeros((W.shape[0], r)), cmap="Oranges", vmin=0, vmax=1, aspect="auto")
    a2.set_xticks([]); a2.set_yticks([]); a2.set_title(f"B\n{W.shape[0]}×{r}", fontsize=18, color="#ef6c00")
    for s in a2.spines.values():
        s.set_edgecolor("#ef6c00"); s.set_linewidth(3)
    a3.imshow(A, cmap="Oranges", aspect="auto"); a3.set_xticks([]); a3.set_yticks([])
    a3.set_title(f"A  ({r}×{W.shape[1]})", fontsize=18, color="#ef6c00")
    for s in a3.spines.values():
        s.set_edgecolor("#ef6c00"); s.set_linewidth(3)
    fig.text(0.505, 0.5, "+", fontsize=60, ha="center", va="center", color=NAVY)
    fig.text(0.6, 0.64, "×", fontsize=40, ha="center", va="center", color=NAVY)
    fig.text(0.79, 0.4, f"trained: {2 * W.shape[0] * r:,} numbers\n(B starts at zero, so training\nbegins from the pretrained model)",
             ha="center", fontsize=20, color="#ef6c00")
    fig.text(0.5, 0.06, "the layer computes (W + B·A)·x : W stays as pretraining left it, B·A is the learned correction",
             ha="center", fontsize=20, color=NAVY)
    save(fig, "lora")


if __name__ == "__main__":
    tr, ty = split("train"); ls, ly = split("ls")
    pos = [s for s, l in zip(tr, ty) if l == 1]; neg = [s for s, l in zip(tr, ty) if l == 0]
    model = load_cnn()
    motif = logo_bound_unbound(model, pos, neg)
    dose_response(model, motif)
    planting_bars_sd(model, ls, ly)
    lora_real()
