"""Figures for 04_foundation_models (Roee's theory slot; Mattan drafts)."""
import numpy as np
from matplotlib.patches import Rectangle

from deckstyle import *  # noqa: F401,F403
from deck02_03 import you_are_here

DNA = "TGATTTATCAAGAGTATTTTCATTTTTCAATCCATAAATGTGG"


def trad_vs_fm():
    fig, ax = canvas()
    label(ax, 40, 84, "traditional ML", fs=30, bold=True, color=BLUE)
    label(ax, 120, 84, "foundation model", fs=30, bold=True, color=PURPLE)
    ax.plot([80, 80], [8, 80], color="#bdbdbd", lw=2)
    for i, t in enumerate(["AGO2 binding", "splice sites", "promoters"]):
        y = 62 - i * 20
        box(ax, 6, y, 26, 12, "labels:\n" + t, fc=GREEN_L, ec=GREEN, fs=17)
        arrow(ax, 33, y + 6, 43, y + 6)
        box(ax, 44, y, 30, 12, "model from\nscratch", fc=BLUE_L, ec=BLUE, fs=18)
    box(ax, 86, 56, 26, 18, "the whole\ngenome\n(no labels)", fc=GREEN_L, ec=GREEN, fs=18)
    arrow(ax, 113, 65, 120, 65)
    box(ax, 121, 54, 34, 22, "pretrained\nmodel", fc=PURPLE_L, ec=PURPLE, fs=24, bold=True)
    for i, t in enumerate(["AGO2", "splice", "promoter"]):
        x = 112 + i * 15
        arrow(ax, 138, 53, x + 6, 38, color=PURPLE, lw=2, ms=18)
        box(ax, x, 26, 13, 11, t, fc=GOLD_L, ec=GOLD, fs=16)
    save(fig, "trad_vs_fm")


def dna_not_english():
    fig, ax = canvas()
    tiles = [("no spaces", "thecatsatonthemat\nwhere are the words?"),
             ("two strands", "5′ ATGCC 3′\n3′ TACGG 5′"),
             ("long range", "a switch 100,000 letters\naway can control a gene"),
             ("mostly not 'text'", "~2% of the genome\ncodes for protein")]
    for i, (big, small) in enumerate(tiles):
        x, y = 6 + (i % 2) * 76, 46 - (i // 2) * 38
        box(ax, x, y, 72, 32, "", fc="white", ec=PURPLE, lw=3)
        label(ax, x + 36, y + 23, big, fs=30, bold=True, color=PURPLE)
        label(ax, x + 36, y + 10, small, fs=22, color=NAVY)
    label(ax, 80, 85, "DNA is a language with no dictionary and no grammar book", fs=26, bold=True)
    save(fig, "dna_not_english")


def tokens():
    fig, ax = canvas()
    s = DNA[:42]
    rows = [("single letters", "HyenaDNA", list(s)),
            ("6-letter words", "Nucleotide Transformer", [s[i:i + 6] for i in range(0, 42, 6)]),
            ("learned pieces (BPE)", "Mistral-DNA, DNABERT-2",
             ["TGATTTA", "TCAA", "GAGTA", "TTTT", "CATTTT", "TCAA", "TCCA", "TAAATGTG", "G"])]
    cols = [GREEN, BLUE, PURPLE]
    for r, (name, who, toks) in enumerate(rows):
        y = 68 - r * 26
        label(ax, 4, y + 9, name, fs=23, bold=True, color=cols[r], ha="left")
        label(ax, 156, y + 9, who, fs=19, color=GREY, ha="right")
        x = 4
        per = 3.62
        for t in toks:
            w = len(t) * per
            box(ax, x, y - 4, w - 0.4, 8.5, "", fc=[GREEN_L, BLUE_L, PURPLE_L][r], ec=cols[r], lw=1.8, r=0.6)
            for j, ch in enumerate(t):
                label(ax, x + (j + 0.5) * per - 0.2, y + 0.2, ch, fs=17, color=BASE[ch], bold=True)
            x += w
    label(ax, 80, 5, "same 42 letters · 42 tokens, 7 tokens, or 9 tokens", fs=24, bold=True)
    save(fig, "tokens")


def embedding():
    fig, ax = canvas()
    toks = ["TGATTTA", "TCAA", "GAGTA"]
    for i, t in enumerate(toks):
        y = 66 - i * 16
        box(ax, 4, y, 20, 10, t, fc=PURPLE_L, ec=PURPLE, fs=19, bold=True)
        arrow(ax, 25, y + 5, 33, y + 5)
        box(ax, 34, y, 14, 10, f"id {[1532, 87, 640][i]}", fc="white", ec=NAVY, fs=18)
        arrow(ax, 49, y + 5, 57, y + 5)
        vals = np.random.default_rng(i).normal(0, 1, 10)
        for j, v in enumerate(vals):
            c = plt.cm.RdBu_r((v + 2.2) / 4.4)
            ax.add_patch(Rectangle((58 + j * 4.2, y + 1), 4, 8, fc=c, ec="white"))
    label(ax, 79, 82, "lookup table: one learned row per token", fs=22, color=GREY)
    arrow(ax, 101, 50, 112, 50, lw=3)
    label(ax, 106.5, 60, "average", fs=20, color=GREY)
    vals = np.random.default_rng(9).normal(0, 0.6, 10)
    for j, v in enumerate(vals):
        c = plt.cm.RdBu_r((v + 2.2) / 4.4)
        ax.add_patch(Rectangle((113 + j * 4.2, 45), 4, 10, fc=c, ec=PURPLE, lw=1.5))
    label(ax, 134, 38, "one vector per sequence\n(256 numbers)", fs=22, color=PURPLE, bold=True)
    label(ax, 80, 12, "similar sequences → nearby vectors", fs=26, bold=True)
    save(fig, "embedding")


def anatomy():
    fig, ax = canvas()
    parts = [("DNA", GREEN_L, GREEN, 10), ("tokenizer", PURPLE_L, PURPLE, 18), ("embedding", PURPLE_L, PURPLE, 20),
             ("backbone", NAVY, NAVY, 34), ("head", GOLD_L, GOLD, 16)]
    x = 4
    for i, (t, fc, ec, w) in enumerate(parts):
        box(ax, x, 38, w, 22, t, fc=fc, ec=ec, fs=22 if t != "backbone" else 26, bold=True,
            tc="white" if t == "backbone" else NAVY)
        if i < len(parts) - 1:
            arrow(ax, x + w + 0.5, 49, x + w + 6.5, 49)
        x += w + 7
    label(ax, 101, 30, "transformer · Hyena · Mamba\nwe won't open it today", fs=19, color=GREY)
    label(ax, 150, 49, "answer", fs=22, bold=True)
    label(ax, 80, 80, "letters → tokens → vectors → better vectors → answer", fs=26, bold=True)
    label(ax, 80, 12, "pretraining builds the first four · your task adds the head", fs=24, color=PURPLE)
    save(fig, "anatomy")


def heads():
    fig, ax = canvas()
    box(ax, 6, 36, 34, 22, "pretrained\nbackbone", fc=NAVY, ec=NAVY, tc="white", fs=24, bold=True)
    hs = [("classify", "is this bound by AGO2?", "yes / no"), ("regress", "how strongly is this gene expressed?", "a number"),
          ("per position", "where are the splice sites?", "a label per letter"), ("generate", "write a new promoter", "next letters")]
    for i, (h, q, out) in enumerate(hs):
        y = 70 - i * 18
        arrow(ax, 41, 47, 54, y + 5, color=GOLD, lw=2.5)
        box(ax, 55, y, 26, 10, h, fc=GOLD_L, ec=GOLD, fs=21, bold=True)
        label(ax, 85, y + 5, q, fs=21, ha="left")
        label(ax, 156, y + 5, out, fs=18, color=GREY, ha="right")
    save(fig, "heads")


def pretraining():
    fig, ax = canvas()
    s = ["TGATTT", "ATCAAG", "AGTATT", "TTCATT", "TTTCAA"]
    for r in range(2):
        y = 58 - r * 34
        name = ["fill in the blank (masked)", "predict what comes next"][r]
        who = ["Nucleotide Transformer, DNABERT-2", "Mistral-DNA, HyenaDNA, Evo"][r]
        label(ax, 4, y + 17, name, fs=25, bold=True, color=PURPLE, ha="left")
        label(ax, 156, y + 17, who, fs=19, color=GREY, ha="right")
        for i, t in enumerate(s):
            x = 4 + i * 26
            hidden = (r == 0 and i == 2) or (r == 1 and i >= 3)
            box(ax, x, y, 23, 10, "?" if hidden else t, fc=RED_L if hidden else PURPLE_L,
                ec=RED if hidden else PURPLE, fs=21, bold=hidden, tc=RED if hidden else NAVY)
        if r == 0:
            arrow(ax, 64, y - 1, 64, y - 7, color=RED)
            label(ax, 64, y - 10, "AGTATT", fs=20, color=RED, bold=True)
        else:
            label(ax, 140, y + 5, "→ TTCATT ?", fs=21, color=RED, bold=True)
    save(fig, "pretraining")


def landscape():
    fig = plt.figure(figsize=(16, 9), dpi=120)
    ax = fig.add_axes([0.1, 0.13, 0.86, 0.75])
    pts = [("DNABERT", 2021.0, 110e6, "M"), ("Nucleotide Transformer", 2023.0, 2.5e9, "M"),
           ("NT-50M", 2023.0, 50e6, "M"), ("DNABERT-2", 2023.4, 117e6, "M"), ("HyenaDNA", 2023.5, 6.6e6, "A"),
           ("Mistral-DNA", 2024.0, 17e6, "A"), ("Evo", 2024.2, 7e9, "A"), ("Evo 2", 2025.1, 40e9, "A")]
    for n, x, p, kind in pts:
        c = "#00897b" if kind == "M" else "#8d6e63"
        ax.scatter(x, p, s=380, color=c, zorder=3, edgecolor="white", lw=2)
        ax.text(x + 0.06, p * 1.35, n, fontsize=19, color=c)
    ax.plot([2023.0, 2023.0], [50e6, 2.5e9], color="#00897b", lw=3, alpha=.4)
    ax.set_yscale("log"); ax.set_xlim(2020.6, 2025.8)
    ax.set_xticks([2021, 2022, 2023, 2024, 2025]); ax.tick_params(labelsize=20)
    ax.set_ylabel("parameters", fontsize=22); ax.grid(alpha=.3)
    ax.scatter([], [], s=200, color="#00897b", label="fill in the blank"); ax.scatter([], [], s=200, color="#8d6e63", label="predict next")
    ax.legend(fontsize=20, frameon=False, loc="upper left")
    fig.text(0.5, 0.93, "genome language models, 2021–2025 (parameters, log scale)", ha="center", fontsize=26,
             color=NAVY, fontweight="bold")
    save(fig, "landscape")


def adapt():
    fig, ax = canvas()
    cols = [("linear probe", [0, 0, 0, 0], "head only", "seconds"),
            ("partial fine-tune", [1, 1, 0, 0], "top layers", "minutes"),
            ("full fine-tune", [1, 1, 1, 1], "everything", "hours"),
            ("LoRA", [2, 2, 2, 2], "~1-10% add-ons", "minutes")]
    for i, (name, layers, what, cost) in enumerate(cols):
        x = 6 + i * 38.5
        label(ax, x + 16, 84, name, fs=24, bold=True)
        for j, st in enumerate(layers):
            y = 60 - j * 11
            fc, ec = (("#eceff1", "#90a4ae"), ("#ffe0b2", "#ef6c00"), ("#eceff1", "#90a4ae"))[st]
            box(ax, x + 4, y, 24, 9, "frozen" if st != 1 else "trained", fc=fc, ec=ec, fs=17, r=0.8)
            if st == 2:
                box(ax, x + 26, y + 1.5, 7, 6, "+", fc="#ffe0b2", ec="#ef6c00", fs=17, bold=True, r=0.6)
        box(ax, x + 4, 72, 24, 7, "new head", fc="#ffe0b2", ec="#ef6c00", fs=17, bold=True, r=0.8)
        label(ax, x + 16, 16, what, fs=20)
        label(ax, x + 16, 9, cost, fs=20, color=GREY)
    label(ax, 80, 2.5, "grey = frozen (kept from pretraining) · orange = trained on your labels", fs=19, color=GREY)
    save(fig, "adapt")


if __name__ == "__main__":
    you_are_here("here_04", ("Encode", "Model"))
    for f in (trad_vs_fm, dna_not_english, tokens, embedding, anatomy, heads, pretraining, landscape, adapt):
        f()
