# %% [markdown]
# # Slot 6: Break it. Look for shortcuts
#
# **Technion AI Days 2026: Teaching Machines to Read DNA**
#
# Two models now separate bound from unbound RNA:
#
# | model | what it is | held-out test AUROC |
# |---|---|---|
# | your CNN (Slot 2) | 10 first-layer filters, trained on about 92,000 labelled sequences | ~0.86 |
# | Mistral-DNA (16M) + probe (Slot 5) | frozen genome-pretrained language model; probe trained on 12,000 labelled sequences | ~0.78 |
#
# A score does not tell you what a model uses. A **shortcut** is a simple property of the sequences
# that differs between bound and unbound ones in this data set, so that a model can score well by
# using it without detecting binding sites. Before concluding that either model detects binding
# sites, we check three simpler explanations:
#
# 1. **Composition:** the fractions of G+C and of U letters in a sequence (2 numbers).
# 2. **Word counts:** how often each 5-letter word appears, ignoring order and position.
# 3. **The shuffle test:** shuffle each bound sequence so that it keeps its letters and letter pairs,
#    and see how much of each model's score survives.
#
# | cell | what you do |
# |---|---|
# | `#@title ... run this cell, nothing to change` | run it |
# | `# TODO-N` with `# === Your code here ===` | write it (yourself or with an assistant), then **check** it |
# | `#@title Stuck? ...` | run it only if your TODO doesn't pass its check: it computes our version |

# %%
# %include h_setup.py

# %%
# %include h_data.py

# %%
# %include h_fm.py

# %%
# %include h_fm_embed.py

# %%
# %include h_fm_probe.py

# %%
# %include h_cnn.py

# %%
# %include h_shortcuts.py

# %%
#@title Rejoin here: the CNN (Slot 2) and the vectors (Slot 5), yours or ours - run this cell { display-mode: "form" }
USE_MY_FILES = False   # True = upload my_cnn.pt and/or my_emb_mistral17m.npz from Slots 2 and 5
if USE_MY_FILES:
    upload_yours()
S_train, y_train, S_test, y_test = workshop_data()
emb = DATA_DIR / "my_emb_mistral17m.npz"
d = np.load(emb if emb.exists() else fetch("emb_mistral17m.npz"))
print("vectors:", "YOURS from Slot 5" if emb.exists() else "the reference ones")
E_train, E_test = d["train"], d["test"]
cnn = load_cnn()
tok, fm = load_fm(FM_REPO)

table = {}
table["CNN (Slot 2)"] = roc_auc_score(y_test, score_sequences(cnn, S_test))
table["Mistral-DNA + probe (Slot 5)"], probe = probe_auc(E_train, y_train, E_test, y_test)
# The shuffle test (TODO-3) embeds sequences with the reference embed() of this notebook. If you
# uploaded your own vectors, they may come from a slightly different embed(), so for the shuffle test
# we refit the probe on the reference vectors, which come from the same function.
probe_real = table["Mistral-DNA + probe (Slot 5)"]
if emb.exists():
    d_ref = np.load(fetch("emb_mistral17m.npz"))
    probe_real, probe = probe_auc(d_ref["train"], y_train, d_ref["test"], y_test)
    print(f"(shuffle test: probe refitted on the reference vectors, test AUROC {probe_real:.3f})")
for k, v in table.items():
    print(f"{k:32s} test AUROC {v:.3f}")

# %% [markdown]
# ## 1. Composition
#
# Look first: are bound and unbound sequences made of different letters?

# %%
#@title Plot G+C and U content - run this cell { display-mode: "form" }
import matplotlib.pyplot as plt
C = gc_u(S_train)
fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
for j, name in enumerate(("G+C fraction", "U fraction")):
    for lab, c, lbl in ((0, UNBOUND, "Unbound"), (1, BOUND, "Bound")):
        ax[j].hist(C[y_train == lab, j], bins=40, alpha=.7, color=c, label=lbl, density=True)
    ax[j].set_xlabel(name); ax[j].set_ylabel("Density"); ax[j].legend()
plt.tight_layout(); plt.show()

# %% [markdown]
# Recall from Slot 2 that each unbound sequence is a binding site moved to a random position in the
# same gene. So this comparison sets binding sites against random positions in the same genes. Part of
# a difference in composition may be real biology of where AGO2 binds (mostly in 3′UTRs, the stretch
# of a messenger RNA after the protein-coding part, which is rich in A and U), and part may come from
# how the unbound positions were drawn. Sequence length is also a weak shortcut (alone it gives an AUROC
# of about 0.54); Slot 7 controls for it together with composition.
#
# ### TODO-1: A two-number model
#
# Fit a logistic regression on only these **two numbers** per sequence and report its AUROC on the
# test set. The tools are already here:
#
# - `gc_u(seqs)` → an array of shape `(len(seqs), 2)`: G+C fraction and U fraction per sequence;
# - `probe_auc(F_train, y_train, F_test, y_test)` → **returns `(auc, probe)`**; it works on any
#   feature matrix, not only embeddings.
#
# **Check:** `auc_gc` ≈ **0.66**.

# %%
# TODO-1: AUROC of a model that only sees G+C and U content
# === Your code here ===
auc_gc, _ = probe_auc(gc_u(S_train), y_train, gc_u(S_test), y_test)
# === End of your code ===
table["G+C and U content only (2 numbers)"] = auc_gc
print(f"2 numbers: {auc_gc:.3f}")

# %%
#@title Stuck? Run this only if TODO-1 failed its check: it computes auc_gc for you { display-mode: "form" }
auc_gc, _ = probe_auc(gc_u(S_train), y_train, gc_u(S_test), y_test)
table["G+C and U content only (2 numbers)"] = auc_gc
print(f"2 numbers (reference): {auc_gc:.3f}")

# %% [markdown]
# ## 2. Word counts
#
# `kmer_counts(seqs, k=5)` turns each sequence into 1,024 numbers: how often each 5-letter word
# occurs. A linear model on these counts sees which words appear, but not where or in what order.
# Word counts can pick up short motifs too (a 7-letter site contains three 5-letter words); they lose
# where the motif sits.
#
# **Tools:** `kmer_counts(seqs, k=5)` → array `(len(seqs), 1024)`, then `probe_auc` as above.
#
# **Check:** `auc_k5` ≈ **0.74**. (Counting 12,000 sequences takes ~5 s.)

# %%
# TODO-2: AUROC of a linear model on 5-letter word counts (use kmer_counts + probe_auc)
# === Your code here ===
K_train, K_test = kmer_counts(S_train, 5), kmer_counts(S_test, 5)
auc_k5, _ = probe_auc(K_train, y_train, K_test, y_test)
# === End of your code ===
table["5-letter word counts (1,024 numbers)"] = auc_k5
print(f"word counts: {auc_k5:.3f}")

# %%
#@title Stuck? Run this only if TODO-2 failed its check: it computes auc_k5 for you { display-mode: "form" }
auc_k5, _ = probe_auc(kmer_counts(S_train, 5), y_train, kmer_counts(S_test, 5), y_test)
table["5-letter word counts (1,024 numbers)"] = auc_k5
print(f"word counts (reference): {auc_k5:.3f}")

# %% [markdown]
# ## 3. The shuffle test
#
# Take every **bound** test sequence and shuffle it with `dinuc_shuffle`. The result has the same
# letters, the same letter pairs and the same G+C content, but any longer motif is broken up. Then
# score these shuffled "bound" sequences against the real unbound ones.
#
# The shuffled sequences keep their composition, so they still differ from the unbound ones in the
# way TODO-1 measured. A model can therefore score about 0.66 on them from composition alone, and the
# AUROC has no reason to fall all the way to 0.5.
#
# * A model that uses only composition (letters and letter pairs) scores about the same before and
#   after the shuffle.
# * A model that also uses letter order beyond neighbouring pairs, such as motifs, loses that part,
#   and its AUROC drops towards the composition score (≈ 0.66).
# * The shuffle also moves a binding site out of letters 150–190. The CNN weights that region (Slot
#   3), so for the CNN the drop mixes letter order and position.

# %%
#@title Make the shuffled sequences - run this cell { display-mode: "form" }
rng = np.random.default_rng(0)
bound = [s for s, y in zip(S_test, y_test) if y == 1]
unbound = [s for s, y in zip(S_test, y_test) if y == 0]
shuffled = [dinuc_shuffle(s, rng) for s in bound]
print("original:", bound[0][:60])
print("shuffled:", shuffled[0][:60])
print("same G+C and U?", np.allclose(gc_u(bound), gc_u(shuffled)))
y_shuf = np.array([1] * len(shuffled) + [0] * len(unbound))

# %% [markdown]
# ### TODO-3: Run the shuffle test on both models
#
# Compute two AUROCs, both against `y_shuf` (1 for the shuffled bound sequences, 0 for the real
# unbound ones), for the list `shuffled + unbound`:
#
# * **CNN** → `auc_shuf_cnn`: `score_sequences(cnn, seqs)` gives one probability per sequence.
# * **Language model** → `auc_shuf_fm`: `embed(tok, fm, dna_seqs)` gives the vectors (the model
#   expects T, not U: convert each sequence with `to_dna` first), then `probe.predict_proba(vectors)[:, 1]`.
#
# **Gotchas to tell your assistant:** use the `probe` that was already fitted on the real training
# vectors; do **not** fit a new one on shuffled data (that would answer a different question).
#
# **Check:** CNN ≈ **0.67**, language model ≈ **0.77**.

# %%
# TODO-3: shuffle-test AUROC for the CNN and for Mistral-DNA + probe
# === Your code here ===
auc_shuf_cnn = roc_auc_score(y_shuf, score_sequences(cnn, shuffled + unbound))
E_shuf = embed(tok, fm, [to_dna(s) for s in shuffled + unbound])
auc_shuf_fm = roc_auc_score(y_shuf, probe.predict_proba(E_shuf)[:, 1])
# === End of your code ===
print(f"CNN         : real {table['CNN (Slot 2)']:.3f}  ->  shuffled {auc_shuf_cnn:.3f}")
print(f"Mistral-DNA : real {probe_real:.3f}  ->  shuffled {auc_shuf_fm:.3f}")

# %%
#@title Stuck? Run this only if TODO-3 failed its check: it runs the shuffle test for you { display-mode: "form" }
auc_shuf_cnn = roc_auc_score(y_shuf, score_sequences(cnn, shuffled + unbound))
auc_shuf_fm = roc_auc_score(y_shuf, probe.predict_proba(embed(tok, fm, [to_dna(s) for s in shuffled + unbound]))[:, 1])
print(f"CNN         : real {table['CNN (Slot 2)']:.3f}  ->  shuffled {auc_shuf_cnn:.3f}")
print(f"Mistral-DNA : real {probe_real:.3f}  ->  shuffled {auc_shuf_fm:.3f}")

# %%
#@title The scoreboard - run this cell { display-mode: "form" }
import json
results = dict(table)
results["CNN on shuffled bound sequences"] = auc_shuf_cnn
results["Mistral-DNA + probe on shuffled bound sequences"] = auc_shuf_fm
# CNN = ink, frozen language model = mid grey, shortcut baselines = dashed grey outline, shuffled = hatched
styles = {"CNN (Slot 2)": dict(color=CNN_C),
          "Mistral-DNA + probe (Slot 5)": dict(color=FROZEN_C),
          "G+C and U content only (2 numbers)": dict(color="white", edgecolor=BASELINE_C, linestyle="--"),
          "5-letter word counts (1,024 numbers)": dict(color="white", edgecolor=BASELINE_C, linestyle="--"),
          "CNN on shuffled bound sequences": dict(color="white", edgecolor=CNN_C, hatch="///"),
          "Mistral-DNA + probe on shuffled bound sequences": dict(color="white", edgecolor=FROZEN_C, hatch="///")}
names = [k for k in styles if k in results]
fig, ax = plt.subplots(figsize=(7, 3))
for i, k in enumerate(names):
    ax.barh(i, results[k] - 0.5, left=0.5, linewidth=1, **styles[k])
    ax.text(results[k] + 0.005, i, f"{results[k]:.2f}", va="center", fontsize=8, color=INK)
ax.set_yticks(range(len(names)), names)
ax.set_xlim(0.5, 0.95); ax.invert_yaxis()
ax.set_xlabel("Test AUROC (bars start at 0.5, a coin flip)"); ax.set_title("Scoreboard")
plt.tight_layout(); plt.show()
_ = (DATA_DIR / "my_slot6_results.json").write_text(json.dumps(results, indent=1))

# %% [markdown]
# ### Discuss
#
# 1. How much of each model's score can two numbers reproduce?
# 2. Which model was least affected by the shuffle? Compare its shuffled score with the composition
#    score (≈ 0.66). What does that suggest it relies on?
# 3. Two numbers (G+C and U) already reach about 0.66. The unbound sequences are random positions in
#    the same genes as the binding sites. Which part of the composition difference could be biology
#    (AGO2 binds mostly in A/U-rich 3′UTRs), and which part could come from how the unbound positions
#    were drawn?
#
# In Slot 7 we change the data and leave the models as they are.
