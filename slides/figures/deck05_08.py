"""Figures for 05_pretrained_model, 06_break_it, 07_fix_it and 08_question_to_model.

Every number is measured: executed notebooks (notebooks/executed/) and scripts/spike/out/*.json.
"""
import json

import numpy as np
from matplotlib.patches import Rectangle

from deckstyle import *  # noqa: F401,F403
from deck02_03 import todo_card, you_are_here

DATA = ROOT.parent / "data"
SPIKE = ROOT.parent / "scripts" / "spike" / "out"


def read(path):
    return [l.strip().upper().replace("T", "U") for l in open(path) if not l.startswith(">")]


def train_set():
    pos = read(DATA / "CLIPSEQ_AGO2.train.positives.fa")
    neg = read(DATA / "CLIPSEQ_AGO2.train.negatives.fa")
    return pos + neg, np.array([1] * len(pos) + [0] * len(neg))


def gc_u(S):
    return np.array([((s.count("G") + s.count("C")) / len(s), s.count("U") / len(s)) for s in S])


def matched_indices(S, y, seed=0):  # identical to the part 3 notebook
    rng = np.random.default_rng(seed)
    C = gc_u(S)
    L = np.array([len(s) for s in S])
    key = (np.clip((C[:, 0] * 12 / 0.8).astype(int), 0, 11) * 100
           + np.clip((C[:, 1] * 12 / 0.6).astype(int), 0, 11) * 10 + np.clip((L - 130) // 60, 0, 4))
    keep = []
    for k in np.unique(key[y == 1]):
        p = np.nonzero((key == k) & (y == 1))[0]
        n = np.nonzero((key == k) & (y == 0))[0]
        m = min(len(p), len(n))
        keep += list(rng.choice(p, m, replace=False)) + list(rng.choice(n, m, replace=False))
    return np.array(keep)


# ---------------------------------------------------------------- slot 5
def hf_code():
    fig, ax = canvas()
    box(ax, 4, 22, 100, 54, "", fc="#263238", ec="#263238", r=1.5)
    lines = [("from transformers import AutoTokenizer, AutoModel", "#b0bec5"),
             ('name = "RaphaelMourad/Mistral-DNA-v1-17M-hg38"', "#ffcc80"),
             ("tok   = AutoTokenizer.from_pretrained(name)", "white"),
             ("model = AutoModel.from_pretrained(name)", "white"),
             ('ids   = tok("TGATTTATCAAG...", return_tensors="pt")', "white"),
             ("out   = model(**ids, output_hidden_states=True)", "white"),
             ("vecs  = out.hidden_states[-1]", "white")]
    for i, (t, c) in enumerate(lines):
        ax.text(7, 71 - i * 7.6, t, fontsize=19, family="DejaVu Sans Mono", color=c, va="center")
    notes = [(2, "the model's name on the Hub"), (3, "download its word list"), (4, "download its weights"),
             (5, "sequence → token ids"), (6, "run the model"), (7, "last layer: one vector per token")]
    for i, t in notes:
        y = 71 - (i - 1) * 7.6
        arrow(ax, 105, y, 111, y, color=PURPLE, lw=2, ms=16)
        label(ax, 112, y, t, fs=20, color=PURPLE, ha="left")
    label(ax, 80, 86, "Hugging Face: a GitHub for models · the same 3 lines load any of them", fs=25, bold=True)
    label(ax, 80, 10, "the vectors are what we feed the linear probe", fs=24, color=GREY)
    save(fig, "hf_code")


def agent_loop():
    fig, ax = canvas()
    steps = [("prompt", "paste the 🤖 prompt", PURPLE_L, PURPLE), ("code", "the assistant writes it", PURPLE_L, PURPLE),
             ("run", "Shift+Enter", BLUE_L, BLUE), ("check", "shape? which data?\nis the test set\nreally unseen?", RED_L, RED)]
    xy = [(18, 56), (80, 70), (142, 56), (80, 18)]
    xy = [(10, 50), (60, 64), (110, 50), (60, 14)]
    for (t, sub, fc, ec), (x, y) in zip(steps, xy):
        box(ax, x, y, 40, 18, "", fc=fc, ec=ec, lw=3)
        label(ax, x + 20, y + 13, t, fs=26, bold=True, color=ec)
        label(ax, x + 20, y + 5.5, sub.replace("🤖 ", ""), fs=17)
    arrow(ax, 50, 62, 59, 70); arrow(ax, 100, 70, 112, 66); arrow(ax, 128, 49, 101, 27); arrow(ax, 59, 22, 30, 49)
    label(ax, 148, 84, "fast", fs=26, color=PURPLE, bold=True, ha="right")
    label(ax, 148, 10, "yours", fs=26, color=RED, bold=True, ha="right")
    save(fig, "agent_loop")


def results_05():
    fig = plt.figure(figsize=(16, 9), dpi=120)
    ax = fig.add_axes([0.36, 0.12, 0.6, 0.76])
    rows = [("our model, random weights", 0.554, "#bdbdbd"), ("our model, 3 min of pretraining", 0.717, "#b39ddb"),
            ("Mistral-DNA (16M), pretrained", 0.782, PURPLE), ("Nucleotide Transformer (486M)", 0.788, PURPLE),
            ("your CNN from part 1", 0.861, BLUE)]
    for i, (n, v, c) in enumerate(rows):
        ax.barh(i, v, color=c, height=0.66)
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=24, color=NAVY, fontweight="bold")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=22)
    ax.set_xlim(0.5, 0.92); ax.axvline(0.5, color=NAVY); ax.invert_yaxis(); ax.tick_params(labelsize=20)
    ax.set_xlabel("held-out test AUROC (language models: frozen + linear probe)", fontsize=22)
    save(fig, "results_05")


# ---------------------------------------------------------------- slot 6
def three_attacks():
    fig, ax = canvas()
    tiles = [("2 numbers", "how much G+C?\nhow much U?", "composition"),
             ("word counts", "how often is each\n5-letter word there?", "no order, no position"),
             ("shuffle", "same letters,\norder destroyed", "does the model notice?")]
    for i, (big, mid, small) in enumerate(tiles):
        x = 6 + i * 51
        box(ax, x, 16, 46, 60, "", fc="white", ec=RED, lw=3.5)
        label(ax, x + 23, 64, big, fs=34, bold=True, color=RED)
        label(ax, x + 23, 44, mid, fs=23)
        label(ax, x + 23, 25, small, fs=20, color=GREY)
    label(ax, 80, 86, "each is a model that should lose", fs=28, bold=True)
    save(fig, "three_attacks")


def gc_hist():
    S, y = train_set()
    g = gc_u(S)[:, 0]
    fig = plt.figure(figsize=(16, 9), dpi=120)
    ax = fig.add_axes([0.08, 0.15, 0.88, 0.7])
    bins = np.linspace(0.15, 0.8, 60)
    ax.hist(g[y == 0], bins, color="#616161", histtype="step", lw=3, density=True, label=f"unbound · mean {g[y == 0].mean():.2f}")
    ax.hist(g[y == 1], bins, color=GREEN, histtype="step", lw=3, density=True, label=f"bound · mean {g[y == 1].mean():.2f}")
    ax.set_xlabel("fraction of G + C letters", fontsize=24); ax.set_yticks([]); ax.tick_params(labelsize=20)
    ax.legend(fontsize=22, frameon=False)
    fig.text(0.5, 0.92, "bound and unbound RNA are made of different letters", ha="center", fontsize=26,
             fontweight="bold", color=NAVY)
    save(fig, "gc_hist")


def shuffle_demo():
    fig, ax = canvas()
    orig = "GCAUUUGCACUAAUGUUUACAG"
    shuf = "GCAUUAUGUUUAGCACUCAUAG"
    label(ax, 6, 66, "original", fs=22, color=GREY, ha="left")
    seq(ax, 34, 66, orig, fs=30, dx=4.6)
    ax.add_patch(Rectangle((34 + 3 * 4.6 - 2.4, 60.5), 7 * 4.6, 11, fc="none", ec=PURPLE, lw=3.5))
    label(ax, 34 + 6.5 * 4.6, 77, "motif", fs=20, color=PURPLE, bold=True)
    label(ax, 6, 42, "shuffled", fs=22, color=GREY, ha="left")
    seq(ax, 34, 42, shuf, fs=30, dx=4.6)
    label(ax, 80, 26, "same letters · same letter pairs · same G+C", fs=25, color=GREEN, bold=True)
    label(ax, 80, 15, "motif gone", fs=25, color=RED, bold=True)
    label(ax, 80, 86, "can the model tell a real bound site from its scrambled copy?", fs=26, bold=True)
    save(fig, "shuffle_demo")


def scoreboard_06():
    fig = plt.figure(figsize=(16, 9), dpi=120)
    ax = fig.add_axes([0.08, 0.14, 0.88, 0.72])
    x = np.array([0, 1])
    real = [0.861, 0.782]; shuf = [0.661, 0.765]
    ax.bar(x - 0.18, real, 0.34, color=[BLUE, PURPLE], label="real sequences")
    ax.bar(x + 0.18, shuf, 0.34, color=[BLUE, PURPLE], alpha=.35, hatch="//", label="bound sites shuffled")
    for xi, r, s in zip(x, real, shuf):
        ax.text(xi - 0.18, r + 0.008, f"{r:.2f}", ha="center", fontsize=26, fontweight="bold")
        ax.text(xi + 0.18, s + 0.008, f"{s:.2f}", ha="center", fontsize=26, fontweight="bold")
    ax.axhline(0.660, ls="--", color=RED, lw=2.5); ax.text(1.42, 0.668, "2 numbers: 0.66", fontsize=20, color=RED, ha="left")
    ax.axhline(0.738, ls=":", color=GREY, lw=2.5); ax.text(1.42, 0.746, "word counts: 0.74", fontsize=20, color=GREY, ha="left")
    ax.set_xlim(-0.5, 1.95)
    ax.set_xticks(x); ax.set_xticklabels(["CNN (part 1)", "Mistral-DNA + probe (part 2)"], fontsize=24)
    ax.set_ylim(0.5, 0.92); ax.set_ylabel("held-out AUROC", fontsize=22); ax.tick_params(labelsize=20)
    ax.legend(fontsize=20, frameon=False, loc="upper center")
    save(fig, "scoreboard_06")


# ---------------------------------------------------------------- slot 7
def matched_hist():
    S, y = train_set()
    idx = matched_indices(S, y)
    g = gc_u(S)[:, 0]
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), dpi=120, sharey=True)
    bins = np.linspace(0.15, 0.8, 50)
    for ax, (gg, yy, t) in zip(axes, ((g, y, "original negatives"), (g[idx], y[idx], "matched negatives"))):
        ax.hist(gg[yy == 0], bins, color=GREY, alpha=.55, density=True, label="unbound")
        ax.hist(gg[yy == 1], bins, color=GREEN, alpha=.65, density=True, label="bound")
        ax.set_title(t, fontsize=26, color=NAVY); ax.set_xlabel("fraction of G + C", fontsize=22)
        ax.set_yticks([]); ax.tick_params(labelsize=18); ax.legend(fontsize=20, frameon=False)
    fig.text(0.5, 0.02, f"keep, in every G+C × U × length bin, as many unbound as bound · {len(idx):,} of {len(S):,} kept",
             ha="center", fontsize=21, color=NAVY)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.18, wspace=0.08)
    save(fig, "matched_hist")


def lora_fig():
    fig, ax = canvas()
    ax.add_patch(Rectangle((10, 20), 50, 50, fc=BLUE_L, ec=BLUE, lw=3))
    label(ax, 35, 45, "W\npretrained\nfrozen", fs=28, color=BLUE, bold=True)
    label(ax, 35, 13, "256 × 256 = 65,536 numbers", fs=19, color=GREY)
    label(ax, 70, 45, "+", fs=50, bold=True)
    ax.add_patch(Rectangle((82, 20), 7, 50, fc="#ffe0b2", ec="#ef6c00", lw=3))
    label(ax, 85.5, 45, "B", fs=26, color="#ef6c00", bold=True)
    label(ax, 94, 45, "×", fs=36, bold=True)
    ax.add_patch(Rectangle((99, 41.5), 50, 7, fc="#ffe0b2", ec="#ef6c00", lw=3))
    label(ax, 124, 45, "A", fs=26, color="#ef6c00", bold=True)
    label(ax, 117, 13, "2 × 256 × 16 = 8,192 numbers · trained", fs=19, color="#ef6c00")
    label(ax, 80, 84, "LoRA: keep the big matrix, learn a thin correction", fs=28, bold=True)
    save(fig, "lora")


def dumbbell():
    fig = plt.figure(figsize=(16, 9), dpi=120)
    ax = fig.add_axes([0.32, 0.14, 0.64, 0.74])
    rows = [("G+C and U only", 0.637, 0.510, RED), ("5-letter word counts", 0.762, 0.725, GREY),
            ("Mistral-DNA + probe (frozen)", 0.806, 0.770, PURPLE), ("CNN", 0.843, 0.838, BLUE)]
    for i, (n, o, m, c) in enumerate(rows):
        ax.plot([m, o], [i, i], color=c, lw=5, alpha=.5)
        ax.scatter(o, i, s=420, color="white", edgecolor=c, lw=4, zorder=3)
        ax.scatter(m, i, s=420, color=c, zorder=3)
        if o - m < 0.02:
            ax.text(o + 0.012, i, f"{o:.2f} → {m:.2f}", va="center", fontsize=22, color=c, fontweight="bold")
        else:
            ax.text(m - 0.012 if m > 0.55 else m - 0.032, i + 0.32, f"{m:.2f}", ha="center", fontsize=22, color=c,
                    fontweight="bold")
            ax.text(o + 0.012, i + 0.32, f"{o:.2f}", ha="center", fontsize=22, color=c)
    ax.set_yticks(range(4)); ax.set_yticklabels([r[0] for r in rows], fontsize=22); ax.invert_yaxis()
    ax.set_xlim(0.44, 0.95); ax.axvline(0.5, color=NAVY, lw=1.5); ax.tick_params(labelsize=20)
    ax.set_xlabel("test AUROC   ○ original negatives   ● matched negatives", fontsize=22)
    save(fig, "dumbbell")


def model_scan():
    d = json.load(open(SPIKE / "s8_models.json"))
    fig = plt.figure(figsize=(16, 9), dpi=120)
    ax = fig.add_axes([0.1, 0.14, 0.86, 0.72])
    for n, r in d.items():
        if "matched" not in r:
            continue
        p = r["params_M"] * 1e6
        c = BLUE if "Hyena" in n else (PURPLE if "NT" in n else GOLD)
        ax.scatter(p, r["matched"], s=360, color=c, edgecolor="white", lw=2, zorder=3)
        ax.text(p * 1.12, r["matched"] + 0.002, n.split(" (")[0], fontsize=16, color=c)
    ax.axhline(0.838, color=BLUE, lw=3, ls="--")
    ax.text(2e6, 0.843, "small CNN trained on the task: 0.84", fontsize=22, color=BLUE, fontweight="bold")
    ax.axhline(0.510, color=RED, lw=2, ls=":")
    ax.set_xscale("log"); ax.set_ylim(0.68, 0.87); ax.set_xlim(1.5e6, 9e9)
    ax.set_xlabel("model size (parameters)", fontsize=22); ax.set_ylabel("AUROC, matched negatives", fontsize=22)
    ax.tick_params(labelsize=20); ax.grid(alpha=.3)
    fig.text(0.5, 0.92, "it isn't the model: 9 pretrained models, frozen + probe", ha="center", fontsize=26,
             fontweight="bold", color=NAVY)
    save(fig, "model_scan")


def when_fm():
    fig, ax = canvas()
    tiles = [("few labels", "a new cell type,\na rare disease, a pilot", GREEN),
             ("no labels", "is this patient's\nmutation harmful?", RED),
             ("new organism", "no experiment\ndone there yet", BLUE),
             ("design", "write a new\nsequence", PURPLE)]
    for i, (big, small, c) in enumerate(tiles):
        x = 5 + i * 38.5
        box(ax, x, 26, 35, 46, "", fc="white", ec=c, lw=3.5)
        label(ax, x + 17.5, 60, big, fs=28, bold=True, color=c)
        label(ax, x + 17.5, 41, small, fs=20)
    label(ax, 80, 86, "where pretraining earns its place", fs=28, bold=True)
    label(ax, 80, 12, "plenty of labels? start with a small model trained on the task", fs=24, color=GREY)
    save(fig, "when_fm")


def yours():
    fig, ax = canvas()
    box(ax, 6, 18, 66, 60, "", fc=PURPLE_L, ec=PURPLE, lw=3)
    label(ax, 39, 68, "the assistant", fs=30, bold=True, color=PURPLE)
    for i, t in enumerate(["embed()", "probe_auc()", "baselines", "reruns"]):
        label(ax, 39, 58 - i * 8, t, fs=24, color=NAVY)
    label(ax, 39, 24, "seconds", fs=24, color=PURPLE, bold=True)
    box(ax, 88, 18, 66, 60, "", fc=RED_L, ec=RED, lw=3)
    label(ax, 121, 68, "you", fs=30, bold=True, color=RED)
    for i, t in enumerate(["which negatives?", "the shuffle test", "the matched control", "how many labels?"]):
        label(ax, 121, 58 - i * 8, t, fs=24, color=NAVY)
    label(ax, 121, 24, "what counts as evidence", fs=24, color=RED, bold=True)
    save(fig, "yours")


# ---------------------------------------------------------------- slot 8
def traps():
    fig, ax = canvas()
    stages = pipeline(ax, y=66, active=("Question", "Data", "Encode", "Model", "Train", "Evaluate", "Interrogate"), fs=17)
    t = ["which cell\ntype? which\ncondition?", "who chose\nthe negatives?", "two strands?\nhow wide a\nwindow?",
         "a simple\nbaseline\nfirst", "similar\nsequences\nacross split", "AUROC on\nwhich test\nset?", "plant ·\nshuffle ·\nmatch"]
    for i, s in enumerate(t):
        x = 6 + i * 22.1
        box(ax, x, 22, 19.5, 34, "", fc=RED_L, ec=RED, lw=2)
        label(ax, x + 9.75, 39, s, fs=17, color=RED)
        arrow(ax, x + 9.75, 65, x + 9.75, 57, color=RED, lw=2, ms=16)
    label(ax, 80, 86, "the DNA trap at every step", fs=28, bold=True)
    label(ax, 80, 10, "today you met all seven", fs=24, color=GREY)
    save(fig, "traps")


def five_ideas():
    fig, ax = canvas()
    ideas = [("1", "an experiment is the label,\na person chose the negatives", GREEN),
             ("2", "DNA has no words:\nencoding is a choice", PURPLE),
             ("3", "models take shortcuts:\ncomposition first", RED),
             ("4", "test a model like a sample:\nplant · shuffle · match", BLUE),
             ("5", "pretrained: frozen ≈ composition,\nfine-tuned can win · test both", GOLD)]
    for i, (n, t, c) in enumerate(ideas):
        y = 74 - i * 15.5
        box(ax, 14, y, 11, 12, n, fc=c, ec=c, tc="white", fs=30, bold=True)
        label(ax, 30, y + 6, t.replace("\n", " "), fs=25, ha="left")
    save(fig, "five_ideas")


def survey():
    fig, ax = canvas()
    box(ax, 55, 22, 50, 50, "QR\n(form link\ngoes here)", fc=GREY_L, ec=NAVY, fs=26, lw=3)
    label(ax, 80, 84, "2 minutes · what worked, what didn't", fs=28, bold=True)
    label(ax, 80, 12, "notebooks + slides stay online after today", fs=24, color=GREY)
    save(fig, "survey")


def headtohead():
    d = json.load(open(SPIKE / "s10_headtohead.json"))
    fig = plt.figure(figsize=(16, 9), dpi=120)
    ax = fig.add_axes([0.08, 0.14, 0.88, 0.7])
    groups = [("500", "500 labels"), ("20000", "20,000 labels")]
    series = [("cnn", "CNN from scratch", BLUE), ("frozen_probe", "NT-500M frozen + probe", "#b39ddb"),
              ("lora", "NT-500M fine-tuned (LoRA)", "#ef6c00")]
    for gi, (k, gname) in enumerate(groups):
        for si, (key, name, c) in enumerate(series):
            x = gi * 1.0 + (si - 1) * 0.27
            r = d[k][key]
            ax.bar(x, r["mean"], 0.24, color=c, label=name if gi == 0 else None)
            ax.scatter([x] * len(r["runs"]), r["runs"], color=NAVY, s=40, zorder=3)
            ax.text(x, r["mean"] + 0.012, f"{r['mean']:.2f}", ha="center", fontsize=22, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels([g[1] for g in groups], fontsize=24)
    ax.set_ylim(0.5, 0.92); ax.set_ylabel("test AUROC (matched negatives)", fontsize=21); ax.tick_params(labelsize=18)
    ax.legend(fontsize=19, frameon=False, loc="upper left")
    fig.text(0.5, 0.9, "same labelled sequences, same 3,000-sequence test set · dots = 3 seeds · run offline on an A30",
             ha="center", fontsize=19, color=GREY)
    save(fig, "headtohead")


def model_scan2():
    d = json.load(open(SPIKE / "s8_models.json"))
    ft = {"Mistral-DNA-17M (hg38)": 0.787, "NT-500M (human ref)": 0.869, "DNABERT-2-117M (multi-species)": 0.874,
          "HyenaDNA-small-32k (hg38)": 0.723}
    fig = plt.figure(figsize=(16, 9), dpi=120)
    ax = fig.add_axes([0.1, 0.14, 0.86, 0.72])
    for n, r in d.items():
        if "matched" not in r:
            continue
        p = r["params_M"] * 1e6
        ax.scatter(p, r["matched"], s=300, color="#b39ddb", edgecolor="white", lw=2, zorder=3)
        short = n.split(" (")[0].replace("-small-32k", "").replace("-medium-160k", " (medium)")
        ax.text(p * 1.12, r["matched"] - 0.007, short, fontsize=15, color=NAVY)
        if n in ft:
            ax.annotate("", xy=(p, ft[n]), xytext=(p, r["matched"]),
                        arrowprops=dict(arrowstyle="-|>", color="#ef6c00", lw=2.5, mutation_scale=22))
            ax.scatter(p, ft[n], s=300, color="#ef6c00", zorder=4)
    ax.axhline(0.838, color=BLUE, lw=3, ls="--")
    ax.text(2e6, 0.843, "CNN trained on the task: 0.84", fontsize=20, color=BLUE, fontweight="bold")
    ax.set_xscale("log"); ax.set_ylim(0.68, 0.9); ax.set_xlim(1.5e6, 9e9)
    ax.set_xlabel("model size (parameters)", fontsize=21); ax.set_ylabel("test AUROC, matched negatives", fontsize=21)
    ax.tick_params(labelsize=18); ax.grid(alpha=.3)
    ax.scatter([], [], s=200, color="#b39ddb", label="frozen + linear probe (12k labels)")
    ax.scatter([], [], s=200, color="#ef6c00", label="fine-tuned with LoRA (20k labels, one run each)")
    ax.legend(fontsize=17, frameon=False, loc="upper left")
    save(fig, "model_scan")


def label_curve2():
    d = json.load(open(SPIKE / "s7_lowdata.json"))
    h = json.load(open(SPIKE / "s10_headtohead.json"))
    ns = sorted(int(k) for k in d)
    fig = plt.figure(figsize=(16, 9), dpi=120)
    ax = fig.add_axes([0.08, 0.13, 0.88, 0.75])
    lab = {"cnn": ("CNN from scratch", BLUE), "fm_probe": ("Mistral-DNA frozen + probe", "#b39ddb"),
           "kmer5": ("5-letter word counts", "#9e9d24"), "random_probe": ("same model, random weights", "#9e9e9e")}
    for k, (name, c) in lab.items():
        m = [d[str(n)][k]["mean"] for n in ns]; sd = [d[str(n)][k]["sd"] for n in ns]
        ax.errorbar(ns, m, yerr=sd, fmt="o-", color=c, lw=3 if k in ("cnn", "fm_probe") else 2, ms=9,
                    capsize=3, label=name)
    for n in (500, 20000):
        r = h[str(n)]["lora"]
        ax.errorbar([n], [r["mean"]], yerr=[r["sd"]], fmt="D", color="#ef6c00", ms=16, capsize=4,
                    label="NT-500M fine-tuned (offline)" if n == 500 else None)
    ax.axhline(0.5, ls="--", color=NAVY, lw=1)
    ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels([f"{n:,}" for n in ns], fontsize=16)
    ax.set_xlabel("labelled training sequences", fontsize=21); ax.set_ylabel("test AUROC (matched negatives)", fontsize=21)
    ax.tick_params(labelsize=16); ax.set_ylim(0.45, 0.9); ax.grid(alpha=.3); ax.legend(fontsize=16, frameon=False, loc="upper left")
    save(fig, "label_curve2")


if __name__ == "__main__":
    you_are_here("here_05", ("Model", "Train", "Evaluate"))
    you_are_here("here_06", ("Interrogate",))
    you_are_here("here_07", ("Data", "Evaluate"))
    you_are_here("here_08", ("Question", "Data", "Encode", "Model", "Train", "Evaluate", "Interrogate"))
    for f in (hf_code, results_05, gc_hist, scoreboard_06, matched_hist, lora_fig, dumbbell, headtohead,
              model_scan2, label_curve2, five_ideas, survey):
        f()
    todo_card("todo_05", [("TODO-1", "embed(): sequence → one vector", "7 min"),
                          ("TODO-2", "probe_auc(): the linear probe", "5 min"),
                          ("given", "the 30× bigger model, precomputed", "1 min"),
                          ("live", "watch pretraining start", "4 min")], "05_pretrained_model.ipynb")
    todo_card("todo_06", [("TODO-1", "AUROC from 2 numbers (G+C, U)", "5 min"),
                          ("TODO-2", "AUROC from 5-letter word counts", "5 min"),
                          ("TODO-3", "shuffle test, CNN and language model", "10 min")], "06_break_it.ipynb")
    todo_card("todo_07", [("TODO-1", "three AUROCs, on both benchmarks", "8 min"),
                          ("given", "retrain the CNN on both", "1.5 min"),
                          ("TODO-2", "the LoRA step · then it trains", "3 + 4 min"),
                          ("given", "who needs fewer labels?", "3 min")], "07_fix_it.ipynb")
