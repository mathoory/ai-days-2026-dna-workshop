# %% [markdown]
# # Slot 6 · Break it: is it biology, or something that looks like it?
#
# **AI Days 2026 · Teaching Machines to Read DNA** — Technion
#
# Two models now separate bound from unbound AGO2 sites:
#
# | model | what it is | held-out AUROC |
# |---|---|---|
# | your CNN (Slot 2) | 10 motif detectors, trained on 92k sequences | ~0.86 |
# | Mistral-DNA + probe (Slot 5) | genome-pretrained language model, 12k sequences | ~0.78 |
#
# A score is not an explanation. Before we believe either model reads *binding sites*, we try three
# cheap ways it could be cheating — each one a model that **should lose**:
#
# 1. **Composition:** just *how many* G/C and U letters a sequence has (2 numbers).
# 2. **Word counts:** how often each 5-letter word appears — no order, no position.
# 3. **The shuffle test:** keep a sequence's letters (and letter pairs) but destroy its order.
#    A model that reads motifs should be fooled; a model that reads composition won't notice.
#
# | cell | what you do |
# |---|---|
# | `#@title ... run this cell, nothing to change` | just run it |
# | `# TODO-N` with `# === Your code here ===` | write it (yourself or with an assistant), then **check** it |

# %%
# %include h_setup.py

# %%
# %include h_data.py

# %%
# %include h_fm.py

# %%
# %include h_fm_embed.py

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
for k, v in table.items():
    print(f"{k:32s} test AUROC {v:.3f}")

# %% [markdown]
# ## 1. Shortcut #1: composition
#
# Look first: are bound and unbound sequences made of different letters?

# %%
#@title Plot GC and U content - run this cell { display-mode: "form" }
import matplotlib.pyplot as plt
C = gc_u(S_train)
fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
for j, name in enumerate(("G+C fraction", "U fraction")):
    for lab, c, lbl in ((0, "tab:grey", "unbound"), (1, "tab:red", "bound")):
        ax[j].hist(C[y_train == lab, j], bins=40, alpha=.55, color=c, label=lbl, density=True)
    ax[j].set_xlabel(name); ax[j].legend()
plt.tight_layout(); plt.show()

# %% [markdown]
# ### TODO-1 · A two-number model
#
# Fit a logistic regression on just these **two numbers** per sequence and report its AUROC on the
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

# %% [markdown]
# ## 2. Shortcut #2: counting words
#
# `kmer_counts(seqs, k=5)` turns each sequence into 1,024 numbers: how often each 5-letter word
# occurs. A linear model on those counts knows *which* words appear — but nothing about where or
# in what order.

# **Tools:** `kmer_counts(seqs, k=5)` → array `(len(seqs), 1024)`, then `probe_auc` as above.
# **Check:** `auc_k5` ≈ **0.74**. (Counting 12,000 sequences takes ~5 s.)

# %%
# TODO-2: AUROC of a linear model on 5-letter word counts (use kmer_counts + probe_auc)
# === Your code here ===
K_train, K_test = kmer_counts(S_train, 5), kmer_counts(S_test, 5)
auc_k5, _ = probe_auc(K_train, y_train, K_test, y_test)
# === End of your code ===
table["5-letter word counts (1,024 numbers)"] = auc_k5
print(f"word counts: {auc_k5:.3f}")

# %% [markdown]
# ## 3. The shuffle test
#
# Take every **bound** test sequence and shuffle it with `dinuc_shuffle`: same letters, same
# letter pairs, same GC — but motifs destroyed. Then ask each model: *can you still tell these
# fake "bound" sequences from the real unbound ones?*
#
# * AUROC stays **high** → the model is reading composition.
# * AUROC drops towards **0.5** → the model was reading the order of letters: motifs.

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
# ### TODO-3 · Run the shuffle test on both models
#
# Compute two AUROCs, both against `y_shuf` (1 for the shuffled bound sequences, 0 for the real
# unbound ones), for the list `shuffled + unbound`:
#
# * **CNN** → `auc_shuf_cnn`: `score_sequences(cnn, seqs)` gives one probability per sequence.
# * **Language model** → `auc_shuf_fm`: `embed(tok, fm, dna_seqs)` gives the vectors (the model reads
#   T, not U: convert each sequence with `to_dna` first), then `probe.predict_proba(vectors)[:, 1]`.
#
# **Gotchas to tell your assistant:** use the `probe` that was already fitted on the real training
# vectors; do **not** fit a new one on shuffled data (that would answer a different question).
#
# **Check:** CNN ≈ **0.66**, language model ≈ **0.77**.

# %%
# TODO-3: shuffle-test AUROC for the CNN and for Mistral-DNA + probe
# === Your code here ===
auc_shuf_cnn = roc_auc_score(y_shuf, score_sequences(cnn, shuffled + unbound))
E_shuf = embed(tok, fm, [to_dna(s) for s in shuffled + unbound])
auc_shuf_fm = roc_auc_score(y_shuf, probe.predict_proba(E_shuf)[:, 1])
# === End of your code ===
print(f"CNN         : real {table['CNN (Slot 2)']:.3f}  ->  shuffled {auc_shuf_cnn:.3f}")
print(f"Mistral-DNA : real {table['Mistral-DNA + probe (Slot 5)']:.3f}  ->  shuffled {auc_shuf_fm:.3f}")

# %%
#@title The scoreboard - run this cell { display-mode: "form" }
import json
rows = list(table.items()) + [("CNN on SHUFFLED bound sites", auc_shuf_cnn),
                              ("Mistral-DNA on SHUFFLED bound sites", auc_shuf_fm)]
fig, ax = plt.subplots(figsize=(7, 3))
names, vals = zip(*rows)
colors = ["tab:blue", "tab:purple", "tab:grey", "tab:grey", "tab:blue", "tab:purple"]
ax.barh(names, vals, color=colors, alpha=.8)
ax.axvline(0.5, ls="--", c="k", lw=1); ax.set_xlim(0.45, 0.95); ax.invert_yaxis()
for i, v in enumerate(vals):
    ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=8)
ax.set_xlabel("test AUROC"); plt.tight_layout(); plt.show()
_ = (DATA_DIR / "my_slot6_results.json").write_text(json.dumps(dict(rows), indent=1))

# %% [markdown]
# ### Discuss
#
# 1. How much of each model's score is available from **two numbers**?
# 2. Which model barely noticed that its input was shuffled? What does that say it was reading?
# 3. Two numbers (G+C, U) already work. Is that the model's fault — or the **data's**? Where do the
#    negatives come from, and why might they be made of different letters?
#
# Slot 7: fix the data, not the model.
