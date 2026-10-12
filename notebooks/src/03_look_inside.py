# %% [markdown]
# # Slot 3 · Look inside: what did the CNN learn?
#
# **AI Days 2026 · Teaching Machines to Read DNA** — Technion
#
# Your CNN scores well, so it learned *something* about which RNA AGO2 binds. This notebook asks
# **what**, and then does something more interesting: it **tests** whether the answer is real.
#
# 1. **Read a motif off the model.** Each first-layer filter is a motif detector. We collect the
#    10-letter windows it fires on and draw them as a *sequence logo*.
# 2. **Test it.** A logo is a hypothesis. We plant the motif into random RNA and see whether the
#    model's score goes up — an experiment on the model, not just a look at it.
#
# | cell | what you do |
# |---|---|
# | `#@title ... run this cell, nothing to change` | just run it |
# | `# TODO-N` with `# === Your code here ===` | **write the code yourself** |

# %%
# %include h_setup.py
!pip -q install logomaker

# %%
# %include h_data.py

# %%
# %include h_cnn.py

# %%
#@title Rejoin here: your CNN from Slot 2, or ours - run this cell { display-mode: "form" }
USE_MY_CNN = False   # True = upload the my_cnn.pt you downloaded at the end of Slot 2
if USE_MY_CNN:
    upload_yours()
train_seqs, train_y = read_split("train")
val_seqs, val_y = read_split("ls")
model = load_cnn()
print(f"held-out test AUROC = {roc_auc_score(val_y, score_sequences(model, val_seqs)):.3f}")

# %% [markdown]
# ## The toolkit (given — run it, nothing to change)
#
# Read the names and one-line descriptions: you will assemble them yourself in a moment.
#
# | function | what it does |
# |---|---|
# | `filter_activations(model, seqs)` | run only the first conv layer → `(n_seqs, n_filters, n_positions)` |
# | `cutoff_for(acts_f)` | one filter's firing cutoff: `mean + 0.5·(max − mean)` |
# | `windows_above(acts_f, seqs, cutoff)` | the 10-letter windows where that filter fired |
# | `pwm_of(windows)` | how often each letter appears at each position → a `(10, 4)` table |
# | `consensus_of(pwm)` | the most common letter at each position, as a string |
# | `draw_logo(pwm, title)` | draw the table as a sequence logo |
# | `random_rna(n)`, `plant(seqs, motif, copies)` | blank canvases, and writing a motif into them |

# %%
#@title Motif toolkit - run this cell, nothing to change { display-mode: "form" }
import logomaker
import matplotlib.pyplot as plt
import pandas as pd

CONV1 = next(l for l in model.net if isinstance(l, nn.Conv2d))
KERNEL = CONV1.kernel_size[1]          # read from the model, so a different width still works
_rng = np.random.default_rng(0)


@torch.no_grad()
def filter_activations(model, seqs, batch=512):
    """Run ONLY the first conv layer (+ReLU). Returns (n_seqs, n_filters, n_positions)."""
    conv1 = next(l for l in model.net if isinstance(l, nn.Conv2d))
    model.eval()
    out = []
    for i in range(0, len(seqs), batch):
        a = torch.relu(conv1(encode_batch(seqs[i:i + batch]).to(DEVICE)))
        out.append((a[:, :, 0, :] if a.shape[2] == 1 else a.amax(2)).cpu())  # (batch, filters, positions)
    return torch.cat(out).numpy()


def cutoff_for(acts_f):
    """The firing cutoff for one filter: halfway between its typical and its strongest response."""
    return acts_f.mean() + 0.5 * (acts_f.max() - acts_f.mean())


def windows_above(acts_f, seqs, cutoff):
    """Every KERNEL-letter window where this filter's output exceeded the cutoff (padding dropped)."""
    rows, cols = np.nonzero(acts_f > cutoff)
    out = [seqs[i][j:j + KERNEL] for i, j in zip(rows, cols)]
    return [w for w in out if len(w) == KERNEL and set(w) <= set(ALPHABET)]


def pwm_of(windows):
    """(KERNEL, 4) table: how often each letter appears at each position."""
    arr = np.frombuffer("".join(windows).encode(), dtype=np.uint8).reshape(-1, KERNEL)
    counts = np.stack([(arr == ord(b)).sum(0) for b in ALPHABET], 1).astype(float)
    return counts / counts.sum(1, keepdims=True)


def consensus_of(pwm):
    return "".join(ALPHABET[i] for i in pwm.argmax(1))


def draw_logo(pwm, title="", ax=None):
    """Sequence logo in bits: tall letter = strongly preferred; four short letters = don't care."""
    df = pd.DataFrame(pwm, columns=list(ALPHABET))
    info = logomaker.transform_matrix(df, from_type="probability", to_type="information")
    ax = ax or plt.subplots(figsize=(4, 1.6))[1]
    logomaker.Logo(info, ax=ax, color_scheme="classic")
    ax.set_title(title, fontsize=9); ax.set_ylim(0, 2); ax.set_xticks([])
    return ax


def random_rna(n, length=330):
    """n random RNA sequences - a blank canvas to plant motifs into."""
    idx = _rng.integers(0, 4, size=(n, length))
    return ["".join(ALPHABET[i] for i in row) for row in idx]


def plant(seqs, motif, copies=1, spacing=45, start=40):
    """Write `copies` copies of `motif` into each sequence, evenly spaced."""
    out = []
    for s in seqs:
        s = list(s)
        for c in range(copies):
            at = start + c * spacing
            s[at:at + len(motif)] = list(motif)
        out.append("".join(s))
    return out


print(f"toolkit ready (filter width = {KERNEL})")

# %% [markdown]
# ## 1. Read a motif off the model (TODO-1)
#
# For **every** filter: compute its cutoff, collect the windows above it, turn them into a letter
# table, and record its consensus. Keep the tables — we draw them right after.

# %%
pos_seqs = [s for s, y in zip(val_seqs, val_y) if y == 1] + \
           [s for s, y in zip(train_seqs, train_y) if y == 1][:5000]
acts = filter_activations(model, pos_seqs)
n_filters = acts.shape[1]
print("activations:", acts.shape, "-> (sequences, filters, positions)")

motifs, pwms, n_hits = {}, {}, {}
for f in range(n_filters):
    # TODO-1: for filter f, use cutoff_for -> windows_above -> pwm_of -> consensus_of and fill
    #         motifs[f] (consensus string), pwms[f] (the table) and n_hits[f] (number of windows).
    # === Your code here ===
    a = acts[:, f, :]
    windows = windows_above(a, pos_seqs, cutoff_for(a))
    pwms[f] = pwm_of(windows)
    motifs[f] = consensus_of(pwms[f])
    n_hits[f] = len(windows)
    # === End of your code ===
    print(f"filter {f:02d}: {n_hits[f]:>7,} windows | consensus {motifs[f]}")

# %%
#@title Stuck? Run this only if TODO-1 failed - it builds the tables for you { display-mode: "form" }
motifs, pwms, n_hits = {}, {}, {}
for f in range(n_filters):
    a = acts[:, f, :]
    windows = windows_above(a, pos_seqs, cutoff_for(a))
    pwms[f], n_hits[f] = pwm_of(windows), len(windows)
    motifs[f] = consensus_of(pwms[f])
print("reference tables ready:", {f: motifs[f] for f in range(3)}, "...")

# %%
#@title Draw the logos - run this cell, nothing to change { display-mode: "form" }
cols = 5
rows = int(np.ceil(n_filters / cols))
fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 1.7 * rows))
for f, ax in zip(range(n_filters), axes.flat):
    draw_logo(pwms[f], f"filter {f:02d} · {n_hits[f]:,} hits", ax)
for ax in list(axes.flat)[n_filters:]:
    ax.axis("off")
plt.tight_layout(); plt.show()

# %% [markdown]
# **Reading a logo.** The height of a column is its information in **bits**: 2 bits = the filter
# always wants one letter there; ~0 bits = it doesn't care. Letter height within a column = how often
# that letter appeared in the windows the filter fired on.
#
# **Pick one (step, not a TODO).** Choose the filter whose logo looks most like a *specific* pattern
# (tall, clean columns), read its motif off the logo (the tallest letter in each column, left to
# right) and **type it into `MY_MOTIF` below**. That string is your hypothesis: *"the model responds
# to this pattern."* If you skip this, you test filter 00's motif.

# %%
MY_MOTIF = motifs[0]   # <- replace with the motif you read off your chosen logo, e.g. "UUUGCACUUA"
print("testing:", MY_MOTIF)

# %% [markdown]
# **The control for looking.** Does your filter draw the same logo on **unbound** RNA? If so, it
# measures something about RNA in general (e.g. how U-rich it is), not something AGO2-specific.

# %%
#@title Does the same filter fire on the NEGATIVES? - run this cell { display-mode: "form" }
neg_seqs = [s for s, y in zip(val_seqs, val_y) if y == 0] + \
           [s for s, y in zip(train_seqs, train_y) if y == 0][:5000]
acts_neg = filter_activations(model, neg_seqs)
f = next(k for k, v in motifs.items() if v == MY_MOTIF) if MY_MOTIF in motifs.values() else 0
a_n = acts_neg[:, f, :]
pwm_neg = pwm_of(windows_above(a_n, neg_seqs, cutoff_for(a_n)))
fig, ax = plt.subplots(1, 2, figsize=(8, 1.8))
draw_logo(pwms[f], f"filter {f:02d} on BOUND", ax[0]); draw_logo(pwm_neg, f"filter {f:02d} on UNBOUND", ax[1])
plt.tight_layout(); plt.show()


# %% [markdown]
# ## 2. Now test it (TODO-2)
#
# A logo tells you what a *part* of the network looks at. Whether that drives the *output* is a
# separate question — and the only way to answer it is to intervene:
#
# 1. Score a batch of `random_rna` sequences: the **baseline**.
# 2. `plant` your motif into the same sequences (1, 2, … 8 copies) and score them again.
# 3. **Control:** do the same with a *shuffled* copy of your motif — same letters, different order.
#    If the real motif pushes the score up and the shuffled one doesn't, the model responds to the
#    **pattern**, not just to the letters it is made of.
#
# *(This shuffle just reorders the motif's 10 letters. In Slot 6 you'll meet a stricter shuffle that
# keeps every letter **pair** of a whole sequence.)*
#
# **Hint for TODO-2 (and for your AI assistant):** `plant(seqs, motif, copies)` returns new strings;
# `score_sequences(model, seqs)` returns one probability per string; you need the **mean** for each
# number of copies, as two lists of 8 numbers.

# %%
canvas = random_rna(1000)
baseline = score_sequences(model, canvas).mean()
shuffled = "".join(_rng.permutation(list(MY_MOTIF)))
copies = list(range(1, 9))

# TODO-2: fill real_scores and control_scores with the MEAN model score after planting
#         c copies (for every c in `copies`) of MY_MOTIF and of `shuffled`, respectively.
# === Your code here ===
real_scores = [score_sequences(model, plant(canvas, MY_MOTIF, c)).mean() for c in copies]
control_scores = [score_sequences(model, plant(canvas, shuffled, c)).mean() for c in copies]
# === End of your code ===

plt.figure(figsize=(5.5, 3.5))
plt.plot(copies, real_scores, "o-", label=f"motif {MY_MOTIF}")
plt.plot(copies, control_scores, "x--", c="k", label=f"shuffled {shuffled}")
plt.axhline(baseline, ls=":", c="r", label="random RNA")
plt.xlabel("copies planted"); plt.ylabel("mean binding score"); plt.legend(fontsize=8); plt.grid(alpha=.3)
plt.title("Dose-response: does the model react to the motif?"); plt.tight_layout(); plt.show()

# %% [markdown]
# **Reading your result.** Look at the gap between the two curves, not at either curve alone: the
# shuffled copy can move the score too (same letters!). If the curve jumps at one particular copy
# number, suspect **position**: `plant()` writes copy 1 at letter 40, copy 2 at 85, copy 3 at 130,
# copy 4 at 175… The next cell checks that.

# %%
#@title Where in the sequence does the motif count? - run this cell, nothing to change { display-mode: "form" }
# One copy of the motif at every position, 400 random sequences each; change in mean score vs no motif.
positions = list(range(0, 321, 8))
scan_canvas = random_rna(400)
base_scan = score_sequences(model, scan_canvas).mean()
real_scan = [score_sequences(model, [c[:x] + MY_MOTIF + c[x + len(MY_MOTIF):] for c in scan_canvas]).mean() - base_scan
             for x in positions]
shuf_scan = [score_sequences(model, [c[:x] + shuffled + c[x + len(shuffled):] for c in scan_canvas]).mean() - base_scan
             for x in positions]
plt.figure(figsize=(7, 3.2))
plt.plot(positions, real_scan, "-", lw=2, label=f"motif {MY_MOTIF}")
plt.plot(positions, shuf_scan, "--", c="k", label="shuffled")
plt.axhline(0, c="grey", lw=1)
for x in (40, 85, 130, 175, 220, 265, 310):
    plt.axvline(x, c="r", ls=":", lw=1)
plt.xlabel("position of the single planted copy"); plt.ylabel("change in mean score")
plt.legend(fontsize=8); plt.title("The CNN learned WHERE as well as WHAT (red: where plant() writes)")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### What to take away
#
# * A logo is a **hypothesis**; planting is the **experiment**. Looking at a model and doing an
#   experiment on it are different things.
# * A filter firing tells you nothing about the **sign** of its effect: it reaches the output
#   through every layer above it. Your dose-response curve can go *down*.
# * If the logo looks the same on bound and unbound RNA, that filter is measuring something about
#   RNA in general (e.g. how U-rich it is), not about AGO2.
# * The CNN's last layer has one weight per (channel, **position**). Every sequence in this data set
#   (bound and unbound alike) was cut so the measured ~40-letter site sits at letters 150–190, with
#   context on both sides. The CNN learned to look **there**. That's sensible, but it means "how much
#   does this motif matter?" has no single answer: it depends on where you put it.
#
# **Discuss with your neighbour:** did your motif beat its shuffled control? Did theirs?

# %%
#@title Bonus (optional): plant known microRNA target sites - run this cell { display-mode: "form" }
# AGO2 is guided to its targets by a microRNA; it binds RNA that pairs with the microRNA's
# "seed" (letters 2-8). These are the target sites of a few abundant microRNAs, plus controls.
SITES = {"miR-15/16": "UGCUGCU", "miR-19": "UUUGCAC", "miR-30": "UGUUUAC", "let-7": "CUACCUC"}
neg_canvas = [s for s, y in zip(val_seqs, val_y) if y == 0]   # real unbound RNA as the canvas
base_neg = score_sequences(model, neg_canvas).mean()
print(f"unbound RNA, nothing planted: {base_neg:.3f}   (6 copies planted below; control = mean of 10 shuffles)")
for name, site in SITES.items():
    r = score_sequences(model, plant(neg_canvas, site, 6)).mean()
    c = np.mean([score_sequences(model, plant(neg_canvas, "".join(_rng.permutation(list(site))), 6)).mean()
                 for _ in range(10)])
    print(f"{name:10s} site {site}: {r:.3f}   | shuffled: {c:.3f}   | difference {r - c:+.3f}")
