# %% [markdown]
# # Slot 3: Look inside the CNN
#
# **Technion AI Days 2026: Teaching Machines to Read DNA**
#
# Your CNN scores well, so it has picked up some signal that separates bound from unbound RNA. This
# notebook asks what that signal is, and then tests the answer.
#
# 1. **Read a motif off the model.** Each first-layer filter scores every 10-letter window. We
#    collect the windows where its output (its *activation*) is high and summarise them as a
#    *sequence logo*.
# 2. **Test it.** The logo suggests what a filter responds to. We insert that motif into random RNA
#    and check whether the model's score goes up.
#
# | cell | what you do |
# |---|---|
# | `#@title ... run this cell, nothing to change` | run it |
# | `# TODO-N` with `# === Your code here ===` | write the code yourself |
# | `#@title Stuck? ...` | run it only if your TODO fails: it does the step for you |

# %%
# %include h_setup.py
!pip -q install logomaker==0.8.7

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
test_seqs, test_y = read_split("ls")
model = load_cnn()
print(f"held-out test AUROC = {roc_auc_score(test_y, score_sequences(model, test_seqs)):.3f}")

# %% [markdown]
# ## The toolkit (given: run it, nothing to change)
#
# Read the names and descriptions. You will combine these functions in the TODOs below.
#
# | function | what it does |
# |---|---|
# | `filter_activations(model, seqs)` | run only the first conv layer → `(n_seqs, n_filters, n_positions)` |
# | `cutoff_for(acts_f)` | one filter's activation threshold: `mean + 0.5 × (max − mean)` |
# | `windows_above(acts_f, seqs, cutoff)` | the 10-letter windows where that filter's activation exceeds the threshold |
# | `pwm_of(windows)` | the frequency of each letter at each position → a `(10, 4)` table |
# | `consensus_of(pwm)` | the most common letter at each position, as a string |
# | `draw_logo(pwm, title)` | draw the table as a sequence logo |
# | `random_rna(n, rng)` | `n` random RNA sequences of 330 letters to insert motifs into |
# | `plant(seqs, motif, copies)` | insert `copies` copies of `motif` into each sequence (they overwrite letters) |

# %%
#@title Motif toolkit - run this cell, nothing to change { display-mode: "form" }
import logomaker
import matplotlib.pyplot as plt
import pandas as pd

CONV1 = next(l for l in model.net if isinstance(l, nn.Conv2d))
KERNEL = CONV1.kernel_size[1]          # read from the model, so a different width still works


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
    """The activation threshold for one filter: halfway between its mean and its maximum activation."""
    return acts_f.mean() + 0.5 * (acts_f.max() - acts_f.mean())


def windows_above(acts_f, seqs, cutoff):
    """Every KERNEL-letter window where this filter's activation exceeds the threshold (padding dropped)."""
    rows, cols = np.nonzero(acts_f > cutoff)
    out = [seqs[i][j:j + KERNEL] for i, j in zip(rows, cols)]
    return [w for w in out if len(w) == KERNEL and set(w) <= set(ALPHABET)]


def pwm_of(windows):
    """(KERNEL, 4) table: the frequency of each letter at each position."""
    arr = np.frombuffer("".join(windows).encode(), dtype=np.uint8).reshape(-1, KERNEL)
    counts = np.stack([(arr == ord(b)).sum(0) for b in ALPHABET], 1).astype(float)
    return counts / counts.sum(1, keepdims=True)


def consensus_of(pwm):
    return "".join(ALPHABET[i] for i in pwm.argmax(1))


def draw_logo(pwm, title="", ax=None):
    """Sequence logo in bits: one tall letter = conserved position; four short letters = no preference."""
    df = pd.DataFrame(pwm, columns=list(ALPHABET))
    info = logomaker.transform_matrix(df, from_type="probability", to_type="information")
    ax = ax or plt.subplots(figsize=(4, 1.6))[1]
    logomaker.Logo(info, ax=ax, color_scheme={b: BASE_COLOURS[b] for b in ALPHABET})
    ax.set_title(title, fontsize=9); ax.set_ylim(0, 2); ax.set_xticks([]); ax.set_ylabel("Bits", fontsize=8)
    return ax


def random_rna(n, rng, length=330):
    """n random RNA sequences (each letter equally likely), drawn with the random generator `rng`."""
    idx = rng.integers(0, 4, size=(n, length))
    return ["".join(ALPHABET[i] for i in row) for row in idx]


def plant(seqs, motif, copies=1, spacing=45, start=40):
    """Insert `copies` copies of `motif` into each sequence (overwriting letters), at letters
    40, 85, 130, ... A copy that would not fit inside a sequence is skipped."""
    out = []
    for s in seqs:
        s = list(s)
        for c in range(copies):
            at = start + c * spacing
            if at + len(motif) <= len(s):
                s[at:at + len(motif)] = list(motif)
        out.append("".join(s))
    return out


print(f"toolkit ready (filter width = {KERNEL})")

# %% [markdown]
# ## 1. Read a motif off the model (TODO-1)
#
# For **every** filter: compute its threshold, collect the windows above it, turn them into a letter
# frequency table, and record its consensus (the most frequent letter at each position). Keep the
# tables; the next cells draw them.

# %%
pos_seqs = [s for s, y in zip(test_seqs, test_y) if y == 1] + \
           [s for s, y in zip(train_seqs, train_y) if y == 1][:5000]
acts = filter_activations(model, pos_seqs)
n_filters = acts.shape[1]
print("activations:", acts.shape, "-> (sequences, filters, positions) for", len(pos_seqs), "bound sequences")

motifs, pwms, n_hits = {}, {}, {}
for f in range(n_filters):
    # TODO-1: for filter f, take its activations with  acts_f = acts[:, f, :]  (sequences x positions).
    #         Then cutoff_for -> windows_above (the sequences are pos_seqs) -> pwm_of -> consensus_of,
    #         and fill motifs[f] (consensus string), pwms[f] (the table) and n_hits[f] (number of windows).
    # === Your code here ===
    acts_f = acts[:, f, :]
    windows = windows_above(acts_f, pos_seqs, cutoff_for(acts_f))
    pwms[f] = pwm_of(windows)
    motifs[f] = consensus_of(pwms[f])
    n_hits[f] = len(windows)
    # === End of your code ===
    print(f"filter {f:02d}: {n_hits[f]:>7,} windows | consensus {motifs[f]}")

# %%
#@title Stuck? Run this only if TODO-1 failed: it builds the tables for you { display-mode: "form" }
motifs, pwms, n_hits = {}, {}, {}
for f in range(n_filters):
    acts_f = acts[:, f, :]
    windows = windows_above(acts_f, pos_seqs, cutoff_for(acts_f))
    pwms[f], n_hits[f] = pwm_of(windows), len(windows)
    motifs[f] = consensus_of(pwms[f])
print("reference tables ready:", {f: motifs[f] for f in range(3)}, "...")

# %%
#@title Draw the logos - run this cell, nothing to change { display-mode: "form" }
cols = 5
rows = int(np.ceil(n_filters / cols))
fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 1.7 * rows))
for f, ax in zip(range(n_filters), axes.flat):
    draw_logo(pwms[f], f"Filter {f:02d}, {n_hits[f]:,} windows", ax)
for ax in list(axes.flat)[n_filters:]:
    ax.axis("off")
plt.tight_layout(); plt.show()

# %% [markdown]
# **Reading a logo.** The height of a column is its information content in **bits**. 2 bits means
# that every collected window has the same letter at that position; about 0 bits means all four
# letters are about equally frequent there. Within a column, each letter's height is proportional to
# its frequency in the collected windows.
#
# **Choose a filter.** This is not a TODO, but the rest of the notebook uses it. Pick the filter whose
# logo looks most like a specific pattern (tall, clean columns) and set `MY_FILTER` to its number.
# `MY_MOTIF` is then its consensus, the tallest letter in each column from left to right. You can also
# type the motif you read off the logo yourself. That string is your hypothesis for what the model
# responds to. If you change nothing, you test filter 00.

# %%
MY_FILTER = 0                  # <- the number of the filter you chose
MY_MOTIF = motifs[MY_FILTER]   # its consensus; or type the motif you read off its logo (10 letters)
print(f"filter {MY_FILTER:02d} | testing the motif {MY_MOTIF}")

# %% [markdown]
# **A first control.** Run the same filter on **unbound** RNA. If it gives the same logo there, it picks
# up a property of RNA in general (for example, how U-rich it is) and tells you little about AGO2. In
# that case choose another filter, set `MY_FILTER` above, and rerun that cell and this one.

# %%
#@title The same filter on bound and on unbound RNA - run this cell { display-mode: "form" }
neg_seqs = [s for s, y in zip(test_seqs, test_y) if y == 0] + \
           [s for s, y in zip(train_seqs, train_y) if y == 0][:5000]
acts_neg = filter_activations(model, neg_seqs)
f = MY_FILTER
a_n = acts_neg[:, f, :]
pwm_neg = pwm_of(windows_above(a_n, neg_seqs, cutoff_for(a_n)))
fig, ax = plt.subplots(1, 2, figsize=(8, 1.8))
draw_logo(pwms[f], f"Filter {f:02d}, bound RNA", ax[0]); draw_logo(pwm_neg, f"Filter {f:02d}, unbound RNA", ax[1])
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 2. Now test it (TODO-2)
#
# A logo shows which windows activate one filter. Whether that filter changes the model's *output* is a
# separate question, and you answer it by intervening:
#
# 1. Score a batch of `random_rna` sequences: the **baseline**.
# 2. Insert your motif into the same sequences with `plant` (1, 2, … 7 copies) and score them again.
# 3. **Control:** do the same with a *shuffled* copy of your motif (same letters, different order).
#    If the real motif raises the score and the shuffled one does not, the model is responding to the
#    order of the letters as well as to which letters are present.
#
# *(This shuffle only reorders the motif's 10 letters. Slot 6 uses a stricter shuffle that keeps
# every letter pair of a whole sequence.)*
#
# **Hint for TODO-2 (and for your AI assistant):** `plant(seqs, motif, copies)` returns new strings;
# `score_sequences(model, seqs)` returns one probability per string; you need the **mean** for each
# number of copies, as two lists of 7 numbers.

# %%
rng = np.random.default_rng(2)          # this cell's own random numbers, so a rerun gives the same result
canvas = random_rna(1000, rng)
baseline = score_sequences(model, canvas).mean()
shuffled = "".join(rng.permutation(list(MY_MOTIF)))
copies = list(range(1, 8))              # 7 copies fit into a 330-letter sequence

# TODO-2: fill real_scores and control_scores with the MEAN model score after inserting
#         c copies (for every c in `copies`) of MY_MOTIF and of `shuffled`, respectively.
# === Your code here ===
real_scores = [score_sequences(model, plant(canvas, MY_MOTIF, c)).mean() for c in copies]
control_scores = [score_sequences(model, plant(canvas, shuffled, c)).mean() for c in copies]
# === End of your code ===

# %%
#@title Stuck? Run this only if TODO-2 failed: it computes both lists for you { display-mode: "form" }
real_scores = [score_sequences(model, plant(canvas, MY_MOTIF, c)).mean() for c in copies]
control_scores = [score_sequences(model, plant(canvas, shuffled, c)).mean() for c in copies]
print("reference scores ready")

# %%
#@title Plot score vs number of copies - run this cell, nothing to change { display-mode: "form" }
plt.figure(figsize=(5.5, 3.5))
plt.plot(copies, real_scores, "o-", c=INK, label=f"Motif {MY_MOTIF}")
plt.plot(copies, control_scores, "x--", c=BASELINE_C, label=f"Shuffled {shuffled}")
plt.axhline(baseline, ls=":", c=MUTED, lw=1, label="Random RNA, nothing inserted")
plt.xlabel("Copies inserted"); plt.ylabel("Mean predicted probability"); plt.legend(); plt.grid(axis="y")
plt.title("Score vs number of copies"); plt.tight_layout(); plt.show()

# %% [markdown]
# **Reading your result.** Compare the two curves with each other. The shuffled copy can move the
# score too, because it contains the same letters. If the curve jumps at one particular copy number,
# suspect **position**: `plant()` writes copy 1 at letter 40, copy 2 at 85, copy 3 at 130, copy 4 at
# 175, and so on. The next cell checks this.

# %%
#@title Effect of one copy at each position - run this cell, nothing to change { display-mode: "form" }
# One copy of the motif at every position, 400 random sequences each; change in mean score vs no motif.
rng = np.random.default_rng(3)
positions = list(range(0, 321, 8))
scan_canvas = random_rna(400, rng)
base_scan = score_sequences(model, scan_canvas).mean()
real_scan = [score_sequences(model, [c[:x] + MY_MOTIF + c[x + len(MY_MOTIF):] for c in scan_canvas]).mean() - base_scan
             for x in positions]
shuf_scan = [score_sequences(model, [c[:x] + shuffled + c[x + len(shuffled):] for c in scan_canvas]).mean() - base_scan
             for x in positions]
plt.figure(figsize=(7, 3.2))
plt.plot(positions, real_scan, "-", lw=2, c=INK, label=f"Motif {MY_MOTIF}")
plt.plot(positions, shuf_scan, "--", c=BASELINE_C, label="Shuffled")
plt.axhline(0, c=MUTED, lw=1)
for x in (40, 85, 130, 175, 220, 265, 310):
    plt.axvline(x, c=LIGHTGREY, ls=":", lw=1, label="Where plant() inserts copies" if x == 40 else None)
plt.xlabel("Position of the single inserted copy"); plt.ylabel("Change in mean predicted probability")
plt.legend(); plt.title("Effect of one copy by position")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### What to take away
#
# * The logo suggests a pattern. Inserting the motif, next to a shuffled control, tests whether the
#   model's output responds to it.
# * A strongly activated filter can still push the score down. Its effect reaches the output
#   through every layer above it, so your curve can go down.
# * If the logo looks the same on bound and unbound RNA, that filter picks up a property of RNA in
#   general (for example, how U-rich it is) and tells you little about AGO2.
# * The CNN's last layer has one weight per (filter, **position**). Almost every sequence in this data
#   set, bound or unbound, was cut so that the measured ~40-letter site sits at letters 150–190, with
#   context on both sides (the sequences are 137 to 375 letters long), and the CNN learned to weight
#   that region. In our runs the shuffled copy gets much of the same boost there, so part of the effect
#   comes from which letters sit in letters 150–190, whatever their order. How much a motif matters
#   depends on where it is placed.
#
# **Discuss with your neighbour:** did your motif beat its shuffled control? Did theirs?

# %%
#@title Bonus (optional): insert known microRNA target sites - run this cell { display-mode: "form" }
# AGO2 carries a microRNA and binds RNA that base-pairs with the microRNA's "seed" (nucleotides
# 2-8). These sites are complementary to the seeds of a few abundant microRNAs. Each one is
# compared with 10 shuffled copies of itself.
rng = np.random.default_rng(4)
SITES = {"miR-15/16": "UGCUGCU", "miR-19": "UUUGCAC", "miR-30": "UGUUUAC", "let-7": "CUACCUC"}
neg_canvas = [s for s, y in zip(test_seqs, test_y) if y == 0]   # real unbound RNA as the canvas
base_neg = score_sequences(model, neg_canvas).mean()
print(f"unbound RNA, nothing inserted: {base_neg:.3f}   (6 copies inserted below; "
      f"shuffled = mean ± standard deviation over 10 shuffles)")
for name, site in SITES.items():
    r = score_sequences(model, plant(neg_canvas, site, 6)).mean()
    c = [score_sequences(model, plant(neg_canvas, "".join(rng.permutation(list(site))), 6)).mean() for _ in range(10)]
    print(f"{name:10s} site {site}: {r:.3f}   | shuffled: {np.mean(c):.3f} ± {np.std(c):.3f}   | difference {r - np.mean(c):+.3f}")

# %% [markdown]
# **Reading the bonus result.** Compare each site with its own shuffled copies; the ± value is the
# standard deviation over the 10 shuffles. With our reference CNN, only the miR-15/16 site (UGCUGCU)
# raises the score well beyond the spread of its shuffled copies. The other three differ from their
# shuffled copies by less than one standard deviation. The exact numbers depend on the random draw,
# and a CNN trained with another seed (yours, for example) may respond differently. Of the 6 copies,
# only copy 4 (letter 175) lands in letters 150–190, the region the position scan showed the CNN
# weights most. A copy that does not fit into a shorter sequence is skipped.
