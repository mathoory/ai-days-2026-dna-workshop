# %% [markdown]
# # Part 3: Shortcuts, controls and fine-tuning
#
# **Technion AI Days 2026: Applications of AI in Biotechnology and Genomics**
#
# Two models now separate bound from unbound RNA:
#
# | model | what it is | held-out test AUROC |
# |---|---|---|
# | your CNN (part 1) | 10 first-layer filters, trained on about 92,000 labelled sequences | ~0.86 |
# | Mistral-DNA (16M) + probe (part 2) | frozen genome-pretrained language model; probe trained on 12,000 labelled sequences | ~0.78 |
#
# In this notebook you will:
#
# 1. measure what two simple sets of features reach on their own: the G+C and U fractions (TODO-1)
#    and 5-letter word counts (TODO-2);
# 2. run the shuffle test on the CNN and on the probe (TODO-3);
# 3. select bound and unbound sequences with the same composition, then train and test every model
#    again on them.
#
#
# | cell | what you do |
# |---|---|
# | ▶ a cell with a title (its code is hidden) | run it (Shift+Enter) |
# | `# TODO-N` with `# === Your code here ===` | write it (yourself or with an assistant) |
# | ▶ Test TODO-N | run it: it checks your code and says what is wrong |
#
# Before you start: *Runtime → Change runtime type → T4 GPU*.

# %% [markdown]
# ## 0. Setup
#
# Run the setup cell. It also loads our CNN from part 1 (the network of the part 1 slides) and our
# Mistral-DNA embeddings from part 2, so everyone starts from the same models.

# %%
#@title ▶ Setup: helpers, the CNN (part 1) and the embeddings (part 2) { display-mode: "form" }
# %include-body h_setup.py
# %include-body h_data.py
# %include-body h_encode.py
# %include-body h_fm.py
# %include-body h_fm_embed.py
# %include-body h_fm_probe.py
# %include-body h_cnn.py
# %include-body h_shortcuts.py

S_train, y_train, S_test, y_test = workshop_data()
d = np.load(fetch("emb_mistral17m.npz"))
E_train, E_test = d["train"], d["test"]
cnn = load_cnn()
tok, fm = load_fm(FM_REPO)

table = {}
table["CNN (part 1)"] = roc_auc_score(y_test, score_sequences(cnn, S_test))
table["Mistral-DNA + probe (part 2)"], probe = probe_auc(E_train, y_train, E_test, y_test)
probe_real = table["Mistral-DNA + probe (part 2)"]
for k, v in table.items():
    print(f"{k:32s} test AUROC {v:.3f}")

# %% [markdown]
# # Look for shortcuts (sections 1 to 3)
#
# A score does not tell you what a model uses. A **shortcut** is a simple property of the sequences
# that differs between bound and unbound ones in this data set, so that a model can score well by
# using it without detecting binding sites. Before concluding that either model detects binding
# sites, we check three simpler explanations:
#
# 1. **Composition:** the fractions of G+C and of U letters in a sequence (2 numbers).
# 2. **Word counts:** how often each 5-letter word appears, ignoring order and position.
# 3. **The shuffle test:** shuffle each bound sequence so that it keeps its letters and letter pairs,
#    and see how much of each model's score remains.

# %% [markdown]
# ## 1. Composition (TODO-1)
#
# Two numbers per sequence, the fraction of its letters that are G or C and the fraction that are U,
# show whether bound and unbound sequences are made of different letters. You write the function
# that computes them, first for one sequence (TODO-1a), then for a list of sequences (TODO-1b). Then
# you fit a logistic regression on only these **two numbers** per sequence and report its AUROC on
# the test set (TODO-1c), with the probe function of part 2. For counting letters, Python's
# [`str.count`](https://docs.python.org/3/library/stdtypes.html#str.count) is enough.
#
# The probe function of part 2:
#
# - `probe_auc(F_train, y_train, F_test, y_test)` → **returns `(auc, probe)`**; it works on any
#   table of numbers with one row per sequence, not only embeddings.

# %%
#@title ▶ One sequence { display-mode: "form" }
s = S_train[0]
print("first 60 letters:", s[:60], "| length:", len(s), "| label:", y_train[0])
print("letter counts:", {b: s.count(b) for b in "ACGU"})

# %%
# TODO-1a: write gc_u_one(seq): for ONE sequence, two numbers, in this order: the fraction of its
#          letters that are G or C, and the fraction that are U.
def gc_u_one(seq):
    # === Your code here ===
    gc = (seq.count("G") + seq.count("C")) / len(seq)
    u = seq.count("U") / len(seq)
    return gc, u
    # === End of your code ===


gc, u = gc_u_one(s)
print("first 60 letters:", s[:60], "...")
print(f"G+C fraction {gc:.3f} | U fraction {u:.3f}")

# %%
# TODO-1b: write gc_u(seqs): the two numbers for every sequence of a list, as a numpy array with one
#          row per sequence and two columns (G+C fraction, U fraction).
def gc_u(seqs):
    # === Your code here ===
    return np.array([gc_u_one(s) for s in seqs])
    # === End of your code ===


C3 = gc_u(S_train[:3])
print("shape:", C3.shape, "(sequences, numbers)")
for seq, row in zip(S_train[:3], C3):
    print(seq[:40], "... ->", np.round(row, 3))

# %%
# TODO-1c: the test AUROC of a model that sees only these two numbers per sequence (trained on the
#          training sequences, measured on the test sequences). Keep it as auc_gc.
# === Your code here ===
auc_gc, _ = probe_auc(gc_u(S_train), y_train, gc_u(S_test), y_test)
# === End of your code ===
table["G+C and U content only (2 numbers)"] = auc_gc
print(f"2 numbers: {auc_gc:.3f}")

# %%
#@title ▶ Test TODO-1 { display-mode: "form" }
def test_gc_u():
    rng = np.random.default_rng(0)
    seqs = [S_train[i] for i in rng.choice(len(S_train), 500, replace=False)]
    C = gc_u(seqs)
    assert isinstance(C, np.ndarray), f"❌ gc_u returned a {type(C).__name__}, want a numpy array"
    assert C.shape == (500, 2), f"❌ gc_u of 500 sequences has shape {C.shape}, want (500, 2)"
    for i, s in enumerate(seqs):
        gc, u = (s.count("G") + s.count("C")) / len(s), s.count("U") / len(s)
        if not np.isclose(C[i, 0], gc):
            hint = (" (that is the U fraction, so the two columns are swapped)" if np.isclose(C[i, 0], u) else
                    " (a fraction is between 0 and 1: divide by the length)" if C[i, 0] > 1 else "")
            raise AssertionError(f"❌ sequence {i}: column 0 is {C[i, 0]:.4f}, want the G+C fraction {gc:.4f}{hint}")
        assert np.isclose(C[i, 1], u), f"❌ sequence {i}: column 1 is {C[i, 1]:.4f}, want the U fraction {u:.4f}"
    assert 0.63 <= auc_gc <= 0.69, \
        f"❌ auc_gc = {auc_gc:.3f}, want 0.63 to 0.69: train on the two numbers of the training sequences, measure on the test sequences"
    print(f"✅ TODO-1 test passed: gc_u matches the reference on 500 sequences; auc_gc = {auc_gc:.3f}")


test_gc_u()

# %%
#@title ▶ Plot G+C and U content { display-mode: "form" }
import matplotlib.pyplot as plt
C = gc_u(S_train)
fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
for j, name in enumerate(("G+C fraction", "U fraction")):
    for lab, c, lbl in ((0, UNBOUND, "Unbound"), (1, BOUND, "Bound")):
        ax[j].hist(C[y_train == lab, j], bins=40, alpha=.7, color=c, label=lbl, density=True)
    ax[j].set_xlabel(name); ax[j].set_ylabel("Density"); ax[j].legend()
plt.tight_layout(); plt.show()

# %% [markdown]
# Recall from part 1 that each unbound sequence is a binding site moved to a random position in the
# same gene. So this comparison sets binding sites against random positions in the same genes. Part of
# a difference in composition may be real biology of where AGO2 binds (mostly in 3′UTRs, the stretch
# of a messenger RNA after the protein-coding part, which is rich in A and U), and part may come from
# how the unbound positions were drawn. Sequence length is also a weak shortcut (alone it gives an AUROC
# of about 0.54); section 4 controls for it together with composition.

# %% [markdown]
# ## 2. Word counts (TODO-2)
#
# `kmer_counts(seqs, k=5)` turns each sequence into 1,024 numbers: how often each 5-letter word
# occurs. A linear model on these counts sees which words appear, but not where or in what order.
# Word counts can pick up short motifs too (a 7-letter site contains three 5-letter words); they lose
# where the motif sits.
#
# You first count the words of one short sequence yourself and compare your counts with
# `kmer_counts` (TODO-2a). Then you fit the model on the counts of all sequences (TODO-2b).
# For counting, Python's [`Counter`](https://docs.python.org/3/library/collections.html#collections.Counter)
# turns a list of words into word → count.
#
# **Tools:** `kmer_counts(seqs, k=5)` → a table with one row of 1,024 numbers per sequence, and
# `probe_auc` as above.

# %%
#@title ▶ The 1,024 words and one short sequence { display-mode: "form" }
import itertools
WORDS5 = ["".join(w) for w in itertools.product("ACGU", repeat=5)]   # column j of kmer_counts(seqs, 5) counts WORDS5[j]
s = min(S_train, key=len)
print(len(WORDS5), "words:", WORDS5[:3], "...", WORDS5[-2:])
print(f"the shortest training sequence, {len(s)} letters:\n{s}")

# %%
# TODO-2a: write count_words(seq, k=5): how often each k-letter word occurs in seq, counting every
#          window of k neighbouring letters (the windows overlap), as a Counter: word -> count.
from collections import Counter


def count_words(seq, k=5):
    # === Your code here ===
    return Counter(seq[i:i + k] for i in range(len(seq) - k + 1))
    # === End of your code ===


counts = count_words(s)
total = sum(counts.values())
row = kmer_counts([s], 5)[0]            # the same sequence as 1,024 numbers
print(f"{total} windows, {len(counts)} different words")
print("kmer_counts gives each count as a fraction of all windows. The 6 most frequent words:")
print("word    your count   / windows   kmer_counts")
for w, n in counts.most_common(6):
    print(f"{w}   {n:10d}   {n / total:9.4f}   {row[WORDS5.index(w)]:11.4f}")

# %%
# TODO-2b: the test AUROC of a linear model on the 5-letter word counts of the sequences (the tools
#          above, trained and measured as in TODO-1c). Keep it as auc_k5.
# === Your code here ===
K_train, K_test = kmer_counts(S_train, 5), kmer_counts(S_test, 5)
auc_k5, _ = probe_auc(K_train, y_train, K_test, y_test)
# === End of your code ===
table["5-letter word counts (1,024 numbers)"] = auc_k5
print(f"word counts: {auc_k5:.3f}")

# %%
#@title ▶ Test TODO-2 { display-mode: "form" }
def test_words():
    rng = np.random.default_rng(0)
    seqs = [S_train[i] for i in rng.choice(len(S_train), 50, replace=False)]
    K = kmer_counts(seqs, 5)
    for n, seq in enumerate(seqs):
        c = count_words(seq)
        total = len(seq) - 5 + 1
        assert sum(c.values()) == total, \
            f"❌ sequence {n} ({len(seq)} letters): {sum(c.values())} windows counted, want {total} (length - 5 + 1)"
        for j, w in enumerate(WORDS5):
            assert np.isclose(c.get(w, 0) / total, K[n, j], atol=1e-6), \
                (f"❌ sequence {n}, word {w}: your count is {c.get(w, 0)}, kmer_counts gives "
                 f"{K[n, j]:.4f} of {total} windows = {K[n, j] * total:.0f}")
    assert 0.71 <= auc_k5 <= 0.77, \
        f"❌ auc_k5 = {auc_k5:.3f}, want 0.71 to 0.77: train on the word counts of the training sequences, measure on the test sequences"
    print(f"✅ TODO-2 test passed: your word counts match kmer_counts on 50 sequences; auc_k5 = {auc_k5:.3f}")


test_words()

# %% [markdown]
# ## 3. The shuffle test (TODO-3)
#
# Take every **bound** test sequence and shuffle it with `dinuc_shuffle`. The result has the same
# letters, the same letter pairs and the same G+C content, but any longer motif is broken up. Then
# score these shuffled "bound" sequences against the real unbound ones.
#
# The shuffled sequences keep their composition, so they still differ from the unbound ones in the
# way TODO-1 measured. A model can therefore still separate them from composition alone, about as well
# as your two-number model (`auc_gc`), and the AUROC has no reason to fall all the way to 0.5.
#
# * A model that uses only composition (letters and letter pairs) scores about the same before and
#   after the shuffle.
# * A model that also uses letter order beyond neighbouring pairs, such as motifs, loses that part,
#   and its AUROC drops towards the composition score `auc_gc`.
# * The shuffle also moves a binding site out of letters 150–190. The CNN weights that region (part
#   1), so for the CNN the drop mixes letter order and position.
#
# First shuffle one sequence and check what the shuffle keeps (TODO-3a): letters, letter pairs, longer words.

# %%
s = next(q for q, label in zip(S_test, y_test) if label == 1)     # one bound test sequence
s_shuf = dinuc_shuffle(s, np.random.default_rng(1))
print("original:", s[:60], "...")
print("shuffled:", s_shuf[:60], "...")

# TODO-3a: check what the shuffle kept, with your count_words from TODO-2a. Compare s and s_shuf:
#          `same_letters`  True if they have the same counts of single letters,
#          `same_pairs`    True if they have the same counts of letter pairs,
#          `same_words5`   True if they have the same counts of 5-letter words,
#          `same_sequence` True if they are the same sequence.
# === Your code here ===
same_letters = count_words(s, 1) == count_words(s_shuf, 1)
same_pairs = count_words(s, 2) == count_words(s_shuf, 2)
same_words5 = count_words(s, 5) == count_words(s_shuf, 5)
same_sequence = s == s_shuf
# === End of your code ===
pc, ps = count_words(s, 2), count_words(s_shuf, 2)
print("\nthe 16 letter-pair counts, original / shuffled:")
for a in "ACGU":
    print("   ".join(f"{a + b}: {pc[a + b]:3d} / {ps[a + b]:3d}" for b in "ACGU"))
print(f"\nsame letters: {same_letters} | same letter pairs: {same_pairs} | "
      f"same 5-letter words: {same_words5} | same sequence: {same_sequence}")

# %%
#@title ▶ Make the shuffled sequences { display-mode: "form" }
rng = np.random.default_rng(0)
bound = [s for s, y in zip(S_test, y_test) if y == 1]
unbound = [s for s, y in zip(S_test, y_test) if y == 0]
shuffled = [dinuc_shuffle(s, rng) for s in bound]
print("original:", bound[0][:60])
print("shuffled:", shuffled[0][:60])
print("same G+C and U?", np.allclose(gc_u(bound), gc_u(shuffled)))
y_shuf = np.array([1] * len(shuffled) + [0] * len(unbound))

# %% [markdown]
# ### TODO-3b: Run the shuffle test on both models
#
# Compute two AUROCs for the list `shuffled + unbound`, both against `y_shuf` (1 for the shuffled bound
# sequences, 0 for the real unbound ones):
#
# * **CNN** → `auc_shuf_cnn`, from the CNN's probability of "bound" for each sequence.
# * **Language model** → `auc_shuf_fm`, from the probability of "bound" of the probe of part 2 on the
#   embeddings of the sequences. The language model reads DNA, so U becomes T first (`to_dna`).
#
# The models are the ones already trained on real sequences: `cnn`, and `probe` on top of the frozen
# language model. Do **not** train anything on the shuffled sequences; the question is how the trained
# models respond to them.
#
# **API:** `score_sequences(cnn, seqs)` gives the CNN's probability of "bound" for each RNA sequence;
# `embed(tok, fm, dna_seqs)` gives the embeddings (it needs DNA letters: `to_dna(s)` turns U into T);
# `probe.predict_proba(embeddings)[:, 1]` gives the probe's probability of "bound"
# (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.predict_proba);
# `roc_auc_score(labels, probabilities)` gives the AUROC (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).

# %%
# TODO-3b: shuffle-test AUROC for the CNN and for Mistral-DNA + probe
# === Your code here ===
auc_shuf_cnn = roc_auc_score(y_shuf, score_sequences(cnn, shuffled + unbound))
E_shuf = embed(tok, fm, [to_dna(s) for s in shuffled + unbound])
auc_shuf_fm = roc_auc_score(y_shuf, probe.predict_proba(E_shuf)[:, 1])
# === End of your code ===
print(f"CNN         : real {table['CNN (part 1)']:.3f}  ->  shuffled {auc_shuf_cnn:.3f}")
print(f"Mistral-DNA : real {probe_real:.3f}  ->  shuffled {auc_shuf_fm:.3f}")

# %%
#@title ▶ Test TODO-3 { display-mode: "form" }
def test_shuffle_test():
    for name, got, want in (("same_letters", same_letters, count_words(s, 1) == count_words(s_shuf, 1)),
                            ("same_pairs", same_pairs, count_words(s, 2) == count_words(s_shuf, 2)),
                            ("same_words5", same_words5, count_words(s, 5) == count_words(s_shuf, 5)),
                            ("same_sequence", same_sequence, s == s_shuf)):
        assert got == want, f"❌ {name} = {got}, but comparing the counts of s and s_shuf gives {want}"
    assert unbound == [q for q, label in zip(S_test, y_test) if label == 0], \
        "❌ unbound is no longer the list of real unbound test sequences: the test shuffles only the bound ones"
    assert len(shuffled) == len(bound), f"❌ shuffled has {len(shuffled)} sequences, want {len(bound)}, one per bound sequence"
    ref = roc_auc_score(y_shuf, score_sequences(cnn, shuffled + unbound))
    assert abs(auc_shuf_cnn - ref) < 1e-6, \
        (f"❌ auc_shuf_cnn = {auc_shuf_cnn:.3f}, but the CNN scores {ref:.3f} on shuffled + unbound against y_shuf. "
         "Score the list shuffled + unbound in this order, and keep the unbound sequences unshuffled")
    ref_fm = roc_auc_score(y_shuf, probe.predict_proba(embed(tok, fm, [to_dna(q) for q in shuffled + unbound]))[:, 1])
    assert abs(auc_shuf_fm - ref_fm) < 2e-3, \
        (f"❌ auc_shuf_fm = {auc_shuf_fm:.3f}, but the probe of part 2 scores {ref_fm:.3f} on shuffled + unbound "
         "against y_shuf. Use the embeddings of these sequences as DNA and the probe already trained on real sequences")
    print(f"✅ TODO-3 test passed: the shuffle keeps the letter pairs; CNN {auc_shuf_cnn:.3f}, "
          f"Mistral-DNA + probe {auc_shuf_fm:.3f}, with the unbound sequences unshuffled")


test_shuffle_test()

# %%
#@title ▶ The scoreboard { display-mode: "form" }
import json
results = dict(table)
results["CNN on shuffled bound sequences"] = auc_shuf_cnn
results["Mistral-DNA + probe on shuffled bound sequences"] = auc_shuf_fm
# CNN = ink, frozen language model = mid grey, shortcut baselines = dashed grey outline, shuffled = hatched
styles = {"CNN (part 1)": dict(color=CNN_C),
          "Mistral-DNA + probe (part 2)": dict(color=FROZEN_C),
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
_ = (DATA_DIR / "my_part3_shuffle_results.json").write_text(json.dumps(results, indent=1))

# %% [markdown]
# ### Discuss
#
# 1. How much of each model's score can two numbers reproduce?
# 2. Which model was least affected by the shuffle? Compare its shuffled score with the composition
#    score `auc_gc`. What does that suggest it relies on?
# 3. Two numbers (G+C and U) already separate the classes. The unbound sequences are random positions in
#    the same genes as the binding sites. Which part of the composition difference could be biology
#    (AGO2 binds mostly in A/U-rich 3′UTRs), and which part could come from how the unbound positions
#    were drawn?
#
# From section 4 on, we change the data and leave the models as they are.

# %% [markdown]
# # Fix the data, then test again (sections 4 to 7)
#
# In sections 1 to 3, two numbers (the G+C and U fractions of a sequence) already separated bound from
# unbound RNA (AUROC ≈ 0.66). Each unbound sequence is a binding site moved to a random position in
# the same gene (part 1), so this compares binding sites with random positions in the same genes. Part
# of the difference may be real biology of where AGO2 binds (mostly in A/U-rich 3′UTRs), and part may
# come from how the unbound positions were drawn. Either way, a model trained on these data can score
# well by reading single-letter composition, and the score alone would not show it.
#
# A **control** removes that signal. We select bound and unbound sequences so that both classes have
# the same G+C fraction, U fraction and length, bin by bin. On the selected data, single-letter
# composition carries no information about the label. Letter-pair and 5-letter-word composition still
# differ between the classes, so a score above 0.5 can still come from them.

# %% [markdown]
# ## 4. A composition-matched benchmark
#
# We sort every training sequence into a bin by its G+C fraction, U fraction and length. In each bin
# we keep the **same number of bound and unbound sequences** (the smaller of the two counts) and drop
# the rest. This selects a subset of **both** classes: about 30% of the bound sequences have no
# unbound match in their bin and are dropped. Afterwards the two classes have the same single-letter
# composition and length, bin by bin.
#
# For comparison we also draw an equally large random subset of the same file, with the bound and
# unbound sequences as they were. Each benchmark keeps 3,000 sequences aside for testing: the
# **original-negatives test set** and the **matched test set**. Both come from the training file, so
# they differ from the 1,000-sequence held-out test set used in parts 1 and 2 and in sections 1 to 3. The
# models also train differently here (the CNN for 6 epochs on about 65,000 sequences), so the
# "original" numbers below differ slightly from those of sections 1 to 3.

# %%
#@title ▶ Build the two benchmarks { display-mode: "form" }
import matplotlib.pyplot as plt

all_seqs, all_y = read_split("train")


def matched_indices(seqs, y, seed=0):
    rng = np.random.default_rng(seed)
    C = gc_u(seqs)
    L = np.array([len(s) for s in seqs])
    key = (np.clip((C[:, 0] * 12 / 0.8).astype(int), 0, 11) * 100
           + np.clip((C[:, 1] * 12 / 0.6).astype(int), 0, 11) * 10
           + np.clip((L - 130) // 60, 0, 4))
    keep = []
    for k in np.unique(key[y == 1]):
        p = np.nonzero((key == k) & (y == 1))[0]
        n = np.nonzero((key == k) & (y == 0))[0]
        m = min(len(p), len(n))
        keep += list(rng.choice(p, m, replace=False)) + list(rng.choice(n, m, replace=False))
    keep = np.array(keep)
    rng.shuffle(keep)
    return keep


N_TEST = 3000
m_idx = matched_indices(all_seqs, all_y)
o_idx = np.random.default_rng(1).permutation(len(all_seqs))[:len(m_idx)]   # same size, original negatives
print(f"matching kept {int(all_y[m_idx].sum()):,} of {int(all_y.sum()):,} bound sequences")
BENCH = {}
for name, idx in (("original", o_idx), ("matched", m_idx)):
    te, tr = idx[:N_TEST], idx[N_TEST:]
    BENCH[name] = ([all_seqs[i] for i in tr], all_y[tr], [all_seqs[i] for i in te], all_y[te])
    print(f"{name:8s}: train {len(tr):,} | test {len(te):,} | bound {all_y[tr].mean():.0%}")

fig, ax = plt.subplots(1, 2, figsize=(8, 2.6), sharex=True, sharey=True)
for a, name in zip(ax, BENCH):
    S, y = BENCH[name][0], BENCH[name][1]
    g = gc_u(S)[:, 0]
    for lab, c, lbl in ((0, UNBOUND, "Unbound"), (1, BOUND, "Bound")):
        a.hist(g[y == lab], bins=40, alpha=.7, color=c, density=True, label=lbl)
    a.set_title("Original negatives" if name == "original" else "Matched"); a.set_xlabel("G+C fraction"); a.legend()
ax[0].set_ylabel("Density")
plt.tight_layout(); plt.show()

# %% [markdown]
# Two benchmarks of **equal size**, drawn from the same file: `original` keeps the sequences as they
# were, `matched` uses the composition-matched subset. Every model below is trained and tested on both,
# so the two columns of results differ only in how the sequences were selected.
#
# ### One function, two benchmarks
#
# `shortcut_scores(train_seqs, train_y, test_seqs, test_y)` repeats TODO-1, TODO-2 and the frozen
# probe of part 2 on one benchmark. It takes the first 12,000 training sequences as `tr`, `ytr` and
# computes three test AUROCs:
#
# | variable | features for `probe_auc` |
# |---|---|
# | `auc_gc` | `gc_u(tr)` and `gc_u(test_seqs)` |
# | `auc_k5` | `kmer_counts(tr, 5)` and `kmer_counts(test_seqs, 5)` |
# | `auc_fm` | `embed(tok, fm, [to_dna(s) for s in tr])`, same for `test_seqs` |
#
# The function is called twice, once per benchmark, so it uses only its arguments and `tok`, `fm`.

# %%
#@title ▶ Three simple models on both benchmarks { display-mode: "form" }
def shortcut_scores(train_seqs, train_y, test_seqs, test_y, n=12000):
    tr, ytr = train_seqs[:n], train_y[:n]
    auc_gc, _ = probe_auc(gc_u(tr), ytr, gc_u(test_seqs), test_y)
    auc_k5, _ = probe_auc(kmer_counts(tr, 5), ytr, kmer_counts(test_seqs, 5), test_y)
    auc_fm, _ = probe_auc(embed(tok, fm, [to_dna(s) for s in tr]), ytr,
                          embed(tok, fm, [to_dna(s) for s in test_seqs]), test_y)
    return {"G+C and U content only (2 numbers)": auc_gc,
            "5-letter word counts (1,024 numbers)": auc_k5,
            "Mistral-DNA + probe (frozen)": auc_fm}


scores = {name: shortcut_scores(*BENCH[name]) for name in BENCH}
for k in scores["original"]:
    print(f"{k:38s} original {scores['original'][k]:.3f} | matched {scores['matched'][k]:.3f}")

# %% [markdown]
# ## 5. Retrain the CNN on both benchmarks
#
# Same architecture as part 1, 6 epochs on each training set of about 65,000 sequences.
# The three models above used 12,000 of them.

# %%
#@title ▶ Retrain the CNN twice { display-mode: "form" }
from torch.utils.data import DataLoader, TensorDataset
for name in BENCH:
    S_tr, y_tr, S_te, y_te = BENCH[name]
    t0 = time.time()
    torch.manual_seed(SEED)
    cnn_b = GlobalCNN().to(DEVICE)
    cnn_opt = optim.Adam(cnn_b.parameters(), lr=1e-3, weight_decay=1e-4)
    cnn_crit = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(encode_batch(S_tr), torch.tensor(y_tr, dtype=torch.float32)),
                        batch_size=256, shuffle=True, drop_last=True)
    for ep in range(6):
        cnn_b.train()
        for X, y in loader:
            cnn_opt.zero_grad()
            cnn_crit(cnn_b(X.to(DEVICE)), y.to(DEVICE)).backward()
            cnn_opt.step()
    scores[name]["CNN (part 1 architecture)"] = roc_auc_score(y_te, score_sequences(cnn_b, S_te))
    print(f"CNN, {name:8s}: {scores[name]['CNN (part 1 architecture)']:.3f}  ({time.time() - t0:.0f}s)")

# %% [markdown]
# ## 6. Before and after the control

# %%
#@title ▶ The before/after table { display-mode: "form" }
import json
print(f"{'model':40s} {'original':>9s} {'matched':>9s}   lost to the control")
for k in scores["original"]:
    if k not in scores["matched"]:
        continue
    o, m = scores["original"][k], scores["matched"][k]
    print(f"{k:40s} {o:9.3f} {m:9.3f}   {'█' * int(round(max(o - m, 0) * 200))} {o - m:+.3f}")
print()
print("(each █ = 0.005 AUROC lost once single-letter composition is matched)")
print("(labelled training sequences: CNN about 65,000; the other three 12,000)")
_ = (DATA_DIR / "my_part3_matched_results.json").write_text(json.dumps(scores, indent=1))

# %% [markdown]
# ## 7. Pretraining and the number of labels (optional)
#
# The table above compares models trained on different numbers of labels: about 65,000 for the CNN
# and 12,000 for the others. A lab studying a new question often has fewer labels.
#
# The next cell compares both approaches trained on 100 to 12,000 labelled sequences (matched
# benchmark, matched test set). The light grey dashed line is the same language model with **random
# weights**. A random network still produces features, so this line shows how much of the probe's
# score depends on pretraining.
#
# **This cell is optional and precomputed.** By default it plots our offline results (3 random draws
# of the labels per size: points are means, bars are standard deviations). Set `RUN_LIVE = True` to
# rerun one draw at 4 sizes yourself.

# %%
#@title ▶ Learn from fewer labels (optional; precomputed by default) { display-mode: "form" }
RUN_LIVE = False   # True = rerun the label curve yourself (one draw per size)

# Our offline runs on this matched benchmark and test set: mean and standard deviation over 3 draws.
PRECOMPUTED = {
    "NS": [100, 250, 500, 1000, 2500, 5000, 12000],
    "CNN from scratch": {"mean": [0.537, 0.552, 0.583, 0.664, 0.720, 0.766, 0.822],
                         "sd":   [0.029, 0.025, 0.057, 0.039, 0.003, 0.016, 0.005]},
    "Pretrained Mistral-DNA (16M) + probe": {"mean": [0.684, 0.706, 0.731, 0.739, 0.744, 0.752, 0.765],
                                             "sd":   [0.008, 0.018, 0.006, 0.001, 0.001, 0.004, 0.001]},
    "Random-weight Mistral-DNA (16M) + probe": {"mean": [0.497, 0.498, 0.501, 0.528, 0.531, 0.538, 0.554],
                                                "sd":   [0.009, 0.008, 0.008, 0.006, 0.003, 0.009, 0.004]},
}

if RUN_LIVE:
    from transformers import AutoConfig, AutoModel
    M_tr, my_tr, M_te, my_te = BENCH["matched"]
    POOL = 12000
    t0 = time.time()
    D_pool, D_test = [to_dna(s) for s in M_tr[:POOL]], [to_dna(s) for s in M_te]
    E_pool, E_mtest = embed(tok, fm, D_pool), embed(tok, fm, D_test)   # E_mtest: not the rejoin cell's E_test
    torch.manual_seed(SEED)
    fm_rand = AutoModel.from_config(AutoConfig.from_pretrained(FM_REPO)).to(DEVICE).eval()
    R_pool, R_test = embed(tok, fm_rand, D_pool), embed(tok, fm_rand, D_test)
    del fm_rand
    X_pool, y_pool = encode_batch(M_tr[:POOL]), my_tr[:POOL]

    def cnn_from_scratch(sel, seed=0):
        """Part 1's CNN trained on `sel` only; stops early using 10% of `sel` (never the test set)."""
        sel = np.random.default_rng(seed).permutation(sel)
        nv = max(len(sel) // 10, 10)
        va, tr = sel[:nv], sel[nv:]
        torch.manual_seed(seed)
        model = GlobalCNN().to(DEVICE)
        opt_c = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit_c = nn.BCEWithLogitsLoss()
        bs = min(256, max(16, len(tr) // 8))
        epochs = int(np.clip(1500 / max(len(tr) // bs, 1), 6, 200))
        best, best_state, bad = -1, None, 0
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(len(tr))
            for i in range(0, len(tr), bs):
                b = tr[perm[i:i + bs].numpy()]
                opt_c.zero_grad()
                crit_c(model(X_pool[b].to(DEVICE)), torch.tensor(y_pool[b], dtype=torch.float32, device=DEVICE)).backward()
                opt_c.step()
            v = roc_auc_score(y_pool[va], score_sequences(model, [M_tr[i] for i in va]))
            if v > best:
                best, bad, best_state = v, 0, {k: t.clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= 8:
                    break
        model.load_state_dict(best_state)
        return roc_auc_score(my_te, score_sequences(model, M_te))

    NS = [100, 500, 2500, 12000]
    curve = {k: {"mean": [], "sd": None} for k in list(PRECOMPUTED)[1:]}
    for n in NS:
        sel = np.random.default_rng(n).choice(POOL, n, replace=False)
        C = 0.01 if n < 2500 else 0.1          # stronger regularisation (smaller C) for small training sets
        curve["Pretrained Mistral-DNA (16M) + probe"]["mean"].append(probe_auc(E_pool[sel], y_pool[sel], E_mtest, my_te, C)[0])
        curve["Random-weight Mistral-DNA (16M) + probe"]["mean"].append(probe_auc(R_pool[sel], y_pool[sel], R_test, my_te, C)[0])
        curve["CNN from scratch"]["mean"].append(cnn_from_scratch(sel))
        print(f"{n:6,} labels | " + " | ".join(f"{k} {v['mean'][-1]:.3f}" for k, v in curve.items()))
    print(f"({time.time() - t0:.0f}s)")
else:
    NS = PRECOMPUTED["NS"]
    curve = {k: v for k, v in PRECOMPUTED.items() if k != "NS"}

curve_styles = {"CNN from scratch": dict(c=CNN_C, ls="-", marker="o"),
          "Pretrained Mistral-DNA (16M) + probe": dict(c=FROZEN_C, ls="-", marker="o"),
          "Random-weight Mistral-DNA (16M) + probe": dict(c=LIGHTGREY, ls="--", marker="s")}
plt.figure(figsize=(6, 3.8))
for k, v in curve.items():
    plt.errorbar(NS, v["mean"], yerr=v["sd"], capsize=2, label=k, **curve_styles[k])
plt.axhline(0.5, ls=":", c=MUTED, lw=1)
plt.xscale("log"); plt.xticks(NS, [f"{n:,}" for n in NS])
plt.xlabel("Labelled training sequences"); plt.ylabel("Test AUROC (matched data)")
plt.title("Test AUROC vs number of labels" + ("" if RUN_LIVE else " (precomputed, 3 draws)"))
plt.legend(); plt.grid(axis="y"); plt.tight_layout(); plt.show()

# %% [markdown]
# ### What to take away
#
# * **Frozen probe.** It still scores about 0.77 after matching, and in section 3 it barely changed under
#   the shuffle. So it relies mostly on short-word composition (letter pairs, 5-letter words), which
#   the matching leaves in place, and little on single letters.
# * **CNN.** It lost almost nothing to the matched control (0.84 → 0.84), so it uses more than
#   single-letter composition.
# * **Labels.** With few labels the frozen probe does better than the CNN, and with 100 labels the CNN
#   is close to a coin flip. The random-weight line stays close to 0.5 (at most about 0.55), so the
#   probe's advantage comes from pretraining on the genome.
