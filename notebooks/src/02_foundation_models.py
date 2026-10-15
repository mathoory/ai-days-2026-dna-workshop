# %% [markdown]
# # Part 2: DNA foundation models
#
# **Use a pretrained DNA model**
#
# **Technion AI Days 2026: Applications of AI in Biotechnology and Genomics**
#
# In part 1 you trained a CNN from scratch on our data only. Now we use a **foundation model**: a
# language model trained on the human genome before it ever sees our task. This general-purpose
# training is called *pretraining*. The two models in this notebook were pretrained in different ways:
#
# * **Mistral-DNA** learns to predict each token (a short stretch of DNA) from the tokens before it
#   (*next-token prediction*);
# * **Nucleotide Transformer** learns to predict hidden (masked) tokens from the tokens on both sides
#   (*masked-token prediction*).
#
# Neither has seen an AGO2 label. The method (see the talk):
#
# 1. turn each sequence into one vector, its **embedding**: the model's last-layer output, averaged
#    over all non-padding tokens (this average includes the special tokens `[CLS]` and `[SEP]`);
# 2. train a **linear probe** to predict bound vs unbound: a logistic regression on the embeddings.
#    The language model's weights are not updated; we say the model is *frozen*.
#
# In this notebook you will:
#
# 1. turn each sequence into one vector with Mistral-DNA: its tokens, the last layer, and the average
#    over the real tokens (TODO-1);
# 2. write `probe_auc()`, which trains a linear probe on these vectors (TODO-2), and compare the
#    result with a 30× larger model;
# 3. pretrain a model from random weights for 3 minutes, and probe it before and after.
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
# Run the setup cell and the data cell below.

# %%
#@title ▶ Setup and helpers { display-mode: "form" }
# %include-body h_setup.py
# %include-body h_data.py
# %include-body h_encode.py
# %include-body h_fm.py

# %% [markdown]
# The language models were trained on DNA, so we write the RNA letter U as T before giving them a
# sequence (`to_dna`). The probe trains on a fixed subset of 12,000 training sequences and is tested
# on the same 1,000-sequence held-out test set as the CNN.

# %%
#@title ▶ The data: a fixed subset of AGO2 sequences { display-mode: "form" }
S_train, y_train, S_test, y_test = workshop_data()
S_train, S_test = [to_dna(s) for s in S_train], [to_dna(s) for s in S_test]
print(f"train {len(S_train):,} | test {len(S_test):,} | bound: {y_train.mean():.0%}")
print("example:", S_train[0][:70], "...")

# %% [markdown]
# ## 1. Load a pretrained DNA language model
#
# **Mistral-DNA (16M)** (Mourad, 2024): 16 million parameters, pretrained on the human genome (hg38).
# Its repository name says 17M because it also counts the output layer that predicts the next token,
# which we do not use. Its tokenizer uses **byte-pair encoding (BPE)**: starting from single letters,
# it repeatedly merges the most frequent neighbouring pair into a new token, which gives a vocabulary
# of 4,096 DNA substrings of varying length, built from the genome. `[CLS]` and `[SEP]` are special
# tokens the tokenizer adds at the start and end; they are not DNA.

# %%
#@title ▶ Download + load the model, look at its tokens { display-mode: "form" }
t0 = time.time()
tok, fm = load_fm(FM_REPO)
print(f"loaded {FM_REPO} in {time.time() - t0:.1f}s | "
      f"{sum(p.numel() for p in fm.parameters()) / 1e6:.0f}M parameters | {fm.config.hidden_size} numbers per token")
ids = tok(S_train[0][:60])["input_ids"]
print("\nfirst 60 letters :", S_train[0][:60])
print("as tokens        :", " | ".join(tok.convert_ids_to_tokens(ids)))

# %% [markdown]
# ### TODO-1: Turn sequences into vectors
#
# Each sequence becomes one vector of 256 numbers, its **embedding**. First look at what the first steps
# do to one sequence, and to the same sequence next to a longer one.

# %%
#@title ▶ Tokens, padding and the mask { display-mode: "form" }
s = S_train[0][:60]      # the first 60 letters of one sequence, so the printout is short
enc = tok([s, S_train[1][:120]], return_tensors="pt", padding=True)
for i in range(2):
    print(f"sequence {i + 1}, {len(enc['input_ids'][i])} positions:")
    print("  tokens        :", tok.convert_ids_to_tokens(enc["input_ids"][i]))
    print("  attention_mask:", enc["attention_mask"][i].tolist())
with torch.no_grad():
    h = fm(input_ids=enc["input_ids"].to(DEVICE), attention_mask=enc["attention_mask"].to(DEVICE),
           output_hidden_states=True).hidden_states[-1]
print("\nthe last layer for these 2 sequences:", tuple(h.shape), "= (sequences, tokens, numbers per token)")

# %% [markdown]
# **Look at the printout.** How many letters does one token hold? The sequences of our data have about
# 330 letters: do they fit the model's limit of 512 tokens? The first sequence was **padded** to the
# length of the second: the extra tokens are not DNA, and the **attention mask** marks which tokens are
# real (1) and which are padding (0).
#
# **Build** `embed(tok, model, seqs, batch=128)`: a list of DNA sequences in, a numpy array out, with one
# row of 256 numbers per sequence. It handles `batch` sequences at a time:
#
# | step | what it means | API |
# |---|---|---|
# | tokenize | the batch's token numbers and attention masks; short sequences padded, long ones cut to 512 tokens | call the tokenizer on the list: `tok(seqs, ...)` with `padding`, `truncation`, `max_length`, `return_tensors="pt"` ([tokenizer](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__)) |
# | run the model | without learning, on the GPU | [`torch.no_grad`](https://pytorch.org/docs/stable/generated/torch.no_grad.html), `.to(DEVICE)`, `model(input_ids=..., attention_mask=..., output_hidden_states=True)` |
# | last layer | one vector of 256 numbers per token | the output's `.hidden_states`, one entry per layer ([model outputs](https://huggingface.co/docs/transformers/main_classes/output)); `.float()` for 32-bit numbers |
# | average | per sequence, the mean of its token vectors over the **real** tokens only | `*`, `/` and [`.sum(dim)`](https://pytorch.org/docs/stable/generated/torch.sum.html); the mask has one number per token and needs an extra axis of size 1 to multiply each token's 256 numbers ([`unsqueeze`](https://pytorch.org/docs/stable/generated/torch.unsqueeze.html)) |
# | collect | one row per sequence, in the order of `seqs`, as a numpy array | `.cpu().numpy()`, [`np.concatenate`](https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html) |
#
# **The rule that matters:** a sequence must get the same embedding whether it is alone or in a batch
# with longer ones. If the padding tokens enter the average, it does not. The test checks this.

# %%
# TODO-1: write embed(tok, model, seqs, batch=128), following the table above.
def embed(tok, model, seqs, batch=128):
    # === Your code here ===
    model.eval()
    out = []
    for i in range(0, len(seqs), batch):
        enc = tok(seqs[i:i + batch], return_tensors="pt", padding=True, truncation=True, max_length=512)
        ids, mask = enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)
        with torch.no_grad():
            h = model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1].float()
        m = mask.unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
    return np.concatenate(out)
    # === End of your code ===


E = embed(tok, fm, [s, S_train[1][:120]])
print("2 sequences ->", E.shape)
print("the first sequence alone and next to the longer one, largest difference:",
      f"{np.abs(embed(tok, fm, [s])[0] - E[0]).max():.4f}")

# %%
#@title ▶ Test TODO-1 { display-mode: "form" }
def test_embed():
    E = embed(tok, fm, S_train[:3])
    assert isinstance(E, np.ndarray), f"❌ embed returned a {type(E).__name__}, want a numpy array"
    assert E.shape == (3, 256), f"❌ embed of 3 sequences has shape {E.shape}, want (3, 256)"
    assert np.isfinite(E).all(), "❌ the vectors contain NaN or infinite values"
    short, long_ = min(S_train[:50], key=len), max(S_train[:50], key=len)
    alone, padded = embed(tok, fm, [short]), embed(tok, fm, [short, long_])[:1]
    assert np.allclose(alone, padded, atol=2e-2), \
        (f"❌ a sequence gives a different vector alone and next to a longer one (largest difference "
         f"{np.abs(alone - padded).max():.3f}), so the padding changes the average: average over the real tokens only")
    E5, E5_in_2s = embed(tok, fm, S_train[:5]), embed(tok, fm, S_train[:5], batch=2)
    assert E5_in_2s.shape == (5, 256), \
        f"❌ embed of 5 sequences with batch=2 has shape {E5_in_2s.shape}, want (5, 256): keep the vectors of every batch"
    assert np.allclose(E5, E5_in_2s, atol=5e-2), \
        f"❌ embed gives different vectors with batch=2 and with batch=128 (largest difference {np.abs(E5 - E5_in_2s).max():.3f})"
    print("✅ TODO-1 test passed: embed gives shape (3, 256), the same vector alone "
          "or next to longer sequences, and the same vectors in batches of 2")


test_embed()

# %%
#@title ▶ Embed all sequences and draw a 2-D map of the vectors { display-mode: "form" }
t0 = time.time()
E_train, E_test = embed(tok, fm, S_train), embed(tok, fm, S_test)
print(f"embedded {len(S_train) + len(S_test):,} sequences in {time.time() - t0:.1f}s -> {E_train.shape[1]} numbers each")

# PCA finds the 2 directions along which the 12,000 vectors vary most, so we can look at them.
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
P = PCA(2).fit_transform(StandardScaler().fit_transform(E_train))
gc_frac = np.array([(s.count("G") + s.count("C")) / len(s) for s in S_train])
order = np.random.default_rng(0).permutation(len(P))    # random drawing order: neither class hides the other
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
sc = ax[0].scatter(*P[order].T, c=gc_frac[order], s=3, cmap="viridis")
plt.colorbar(sc, ax=ax[0], label="GC fraction")
ax[1].scatter(*P[order].T, c=[BOUND if y else UNBOUND for y in y_train[order]], s=3, alpha=.6)
for col, name in ((UNBOUND, "Unbound"), (BOUND, "Bound")):
    ax[1].scatter([], [], color=col, s=12, label=name)
ax[1].legend()
ax[0].set_title("Coloured by GC fraction"); ax[1].set_title("Coloured by label")
for a in ax:
    a.set_xticks([]); a.set_yticks([]); a.set_xlabel("PCA direction 1"); a.set_ylabel("PCA direction 2")
plt.tight_layout(); plt.show()
for j in (0, 1):
    print(f"correlation of PCA direction {j + 1} with GC: {np.corrcoef(P[:, j], gc_frac)[0, 1]:+.2f}")

# %% [markdown]
# **Reading the map.** If one colouring changes smoothly across the cloud, that property accounts for
# much of the variation between the vectors. Here it is the GC fraction. It changes mostly along PCA
# direction 2, the vertical axis (correlation about 0.81), and less along direction 1 (about 0.52).
# Bound sequences concentrate at the low-GC end of the map. Part 3 tests whether the probe uses this.
#
# ### TODO-2: A linear probe
#
# **Build** `probe_auc(E_train, y_train, E_test, y_test)`: train a linear probe on the training
# embeddings and measure it on the test embeddings. It returns two values, `(auc, probe)`: the AUROC on
# the **test** set and the fitted probe. The probe has two steps:
#
# 1. **standardise** every one of the 256 numbers: shift and scale it to mean 0 and standard deviation 1,
#    with the mean and standard deviation measured on the **training** set and reused unchanged on the
#    test set;
# 2. a **logistic regression** on the standardised numbers, with regularisation strength `C = 0.1` and
#    up to 3,000 iterations. Regularisation keeps the weights small, which guards against fitting noise
#    in 12,000 sequences with 256 numbers each (a smaller `C` means stronger regularisation).
#
# Fit on the training set only. The AUROC compares the probe's probability of "bound" for every test
# sequence with the true test labels. Return the two steps as **one** object (`make_pipeline`), so that
# `probe.predict_proba` on raw vectors standardises them itself: later cells, and part 3, call it that way.
#
# **Tools:** [`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html),
# [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html),
# [`make_pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html),
# [`roc_auc_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).

# %%
# TODO-2: the linear probe, as described above. Return (auc, probe).
def probe_auc(E_train, y_train, E_test, y_test):
    # === Your code here ===
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))
    probe.fit(E_train, y_train)
    return roc_auc_score(y_test, probe.predict_proba(E_test)[:, 1]), probe
    # === End of your code ===


auc_fm, _ = probe_auc(E_train, y_train, E_test, y_test)
results = {"Mistral-DNA (16M), pretrained": auc_fm}
print(f"Mistral-DNA + linear probe: test AUROC = {auc_fm:.3f}")

# %%
#@title ▶ Test TODO-2 { display-mode: "form" }
def test_probe():
    res = probe_auc(E_train, y_train, E_test, y_test)
    assert isinstance(res, tuple) and len(res) == 2, \
        f"❌ probe_auc returned a {type(res).__name__}, want two values: (auc, probe)"
    auc, probe = res
    assert hasattr(probe, "predict_proba"), \
        f"❌ the second value is a {type(probe).__name__}, want the fitted probe (the pipeline), which has predict_proba"
    test_auc = roc_auc_score(y_test, probe.predict_proba(E_test)[:, 1])
    train_auc = roc_auc_score(y_train, probe.predict_proba(E_train)[:, 1])
    assert abs(auc - train_auc) > 1e-6, \
        (f"❌ auc = {auc:.3f} is the AUROC on the training vectors, the ones the probe was fitted on. "
         "Measure it on the test vectors")
    assert abs(auc - test_auc) < 1e-6, \
        (f"❌ auc = {auc:.3f}, but the returned probe scores {test_auc:.3f} on the test vectors: auc must be the AUROC "
         "of the probe's probability of 'bound' on the test vectors against y_test")
    assert 0.75 <= auc <= 0.81, \
        f"❌ test AUROC {auc:.3f}, want 0.75 to 0.81: check the standardisation and the regularisation strength C = 0.1"
    print(f"✅ TODO-2 test passed: test AUROC {auc:.3f} (on the training vectors the same probe scores {train_auc:.3f})")


test_probe()

# %% [markdown]
# ## 2. A 30× larger model: Nucleotide Transformer
#
# **Nucleotide Transformer 500M** (NT-500M, InstaDeep): 486 million parameters, pretrained on the human
# reference genome by masked-token prediction. The download is about 2 GB, so by default we load its
# embeddings, precomputed on these same sequences with the same `embed`. Set `RUN_NT500M = True` to
# compute them yourself.

# %%
#@title ▶ Nucleotide Transformer 500M { display-mode: "form" }
RUN_NT500M = False   # True = download the 2 GB model and compute the vectors yourself
if RUN_NT500M or os.environ.get("AIDAYS_RUN_NT500M") == "1":
    t0 = time.time()
    tok_nt, nt = load_fm(NT_REPO)
    N_train, N_test = embed(tok_nt, nt, S_train, batch=64), embed(tok_nt, nt, S_test, batch=64)
    print(f"computed in {time.time() - t0:.0f}s")
    np.savez_compressed(DATA_DIR / "emb_nt500m.npz", train=N_train, test=N_test)
    del nt
    torch.cuda.empty_cache()
else:
    d = np.load(fetch("emb_nt500m.npz"))
    N_train, N_test = d["train"], d["test"]
auc_nt, _ = probe_auc(N_train, y_train, N_test, y_test)
results["Nucleotide Transformer (486M), pretrained"] = auc_nt
print(f"NT-500M + probe: {auc_nt:.3f}   vs   Mistral-DNA (16M): {auc_fm:.3f}")

# %% [markdown]
# ## 3. Pretraining from scratch, live
#
# Real pretraining takes days of GPU time. Here we run the first 3 minutes of it, following the Galaxy
# Training Network tutorial (Mourad & Batut, CC-BY). The new model has the same design as Mistral-DNA
# but is larger: 105M parameters (the tutorial's configuration file calls it Mixtral, a variant of
# Mistral). It starts from random weights and is trained to predict the next token on 200-letter pieces
# of the human genome. We probe it on AGO2 before and after.
#
# **Think first.** The probe on Mistral-DNA's embeddings separates bound from unbound RNA. Is that
# because of what the model learned in pretraining, or would any network of the same design give
# embeddings that work as well? Which comparison would answer this? Hint: what would the same network
# give before it has learned anything? Discuss with your neighbour, then run the cell and compare its
# result with your answer.
#
# Start the cell, then read **What to take away** below it while the loss goes down.

# %%
#@title ▶ Pretrain from scratch for a few minutes, then the summary { display-mode: "form" }
import pandas as pd
from transformers import AutoConfig, AutoModelForCausalLM

PRETRAIN_MINUTES = 3
GALAXY = "https://raw.githubusercontent.com/raphaelmourad/Mistral-DNA/e8fc28660dee60b580c1f1b1f33934153804ae56/data"
for name, sub in (("pretrain_config.json", "models/Mixtral-8x7B-v0.1/config.json"),
                  ("hg38_200b_verysmall.csv.gz", "genome_sequences/hg38/sequences_hg38_200b_verysmall.csv.gz")):
    if not (DATA_DIR / name).exists():
        urllib.request.urlretrieve(f"{GALAXY}/{sub}", DATA_DIR / name)

cfg = AutoConfig.from_pretrained(DATA_DIR / "pretrain_config.json")
torch.manual_seed(SEED)
scratch = AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa").float().to(DEVICE)
print(f"fresh model: {sum(p.numel() for p in scratch.parameters()) / 1e6:.0f}M parameters, random weights")

genome = pd.read_csv(DATA_DIR / "hg38_200b_verysmall.csv.gz")["text"].astype(str).tolist()
tok.padding_side = "left"
enc = tok(genome, padding="longest", truncation=True, max_length=128, return_tensors="pt")
tok.padding_side = "right"
ids_all, att_all = enc["input_ids"], enc["attention_mask"]
print(f"{len(genome):,} pieces of genome, tokenized")


def embed_checked(model, seqs):
    """embed(), but replace any NaN or infinite numbers by 0 and say so."""
    E = embed(tok, model, seqs)
    bad = int((~np.isfinite(E)).any(1).sum())
    if bad:
        print(f"WARNING: {bad:,} of {len(E):,} vectors contained NaN or infinite values; "
              "they were set to 0, so this probe score is less reliable")
    return np.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0)


auc_before, _ = probe_auc(embed_checked(scratch, S_train), y_train, embed_checked(scratch, S_test), y_test)
print(f"AGO2 probe BEFORE pretraining: {auc_before:.3f}\n")

scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda" and AMP_DTYPE == torch.float16)
opt = torch.optim.AdamW(scratch.parameters(), lr=5e-4, weight_decay=0.01)
losses, step, t0 = [], 0, time.time()
scratch.train()
while time.time() - t0 < PRETRAIN_MINUTES * 60:
    for i in torch.randperm(len(ids_all)).split(32):
        x, a = ids_all[i].to(DEVICE), att_all[i].to(DEVICE)
        with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=DEVICE.type == "cuda"):
            loss = scratch(input_ids=x, attention_mask=a, labels=x.masked_fill(a == 0, -100)).loss
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())
        step += 1
        if step % 100 == 0:
            print(f"step {step:5d} | {(time.time() - t0) / 60:4.1f} min | loss {np.mean(losses[-100:]):.3f}")
        if time.time() - t0 > PRETRAIN_MINUTES * 60:
            break

auc_after, _ = probe_auc(embed_checked(scratch, S_train), y_train, embed_checked(scratch, S_test), y_test)
results["105M model, random weights"] = auc_before
results[f"105M model, after {PRETRAIN_MINUTES} min of pretraining"] = auc_after
print(f"\nAGO2 probe AFTER {step} steps ({PRETRAIN_MINUTES} min): {auc_after:.3f}   (before: {auc_before:.3f})")
plt.figure(figsize=(5, 3)); plt.plot(pd.Series(losses).rolling(50).mean(), c=INK, label="Loss (50-step average)")
plt.axhline(np.log(len(tok)), ls="--", c=MIDGREY, lw=1, label=f"Uniform guess among {len(tok):,} tokens")
plt.xlabel("Step"); plt.ylabel("Next-token loss"); plt.title("Pretraining loss"); plt.legend()
plt.grid(axis="y"); plt.tight_layout(); plt.show()
del scratch
torch.cuda.empty_cache()

import json
print("test AUROC, bars start at 0.5 (a coin flip)")
for k, v in sorted(results.items(), key=lambda kv: kv[1]):
    print(f"{v:.3f}  {'█' * int((v - 0.5) * 80):40s} {k}")
_ = (DATA_DIR / "my_part2_results.json").write_text(json.dumps(results, indent=1))

# %% [markdown]
# ### What to take away
#
# **Reading the loss.** The loss is minus the log of the probability the model gave the true next token. A
# uniform guess among 4,096 tokens gives ln 4096 ≈ 8.3 (the dashed line). A model that knows only how
# often each token occurs, with no context, gets about 7.1. Most of the early drop is learning those
# frequencies; a loss below about 7.1 means the model uses the tokens before it to predict the next one.
#
# * On the embeddings of the random-weight model, the probe scores only slightly above a coin flip
#   (≈ 0.55). Three minutes of pretraining already raise it. Compare your before and after numbers.
#   The 105M model stays below Mistral-DNA (16M) because it was pretrained for 3 minutes, against days
#   for Mistral-DNA.
# * A 30× larger model scores about the same here: 0.78 vs 0.79 is within the ±0.025 noise of a
#   1,000-sequence test set.
# * With the language model frozen, both probes (trained on 12,000 labelled sequences) score below
#   the small CNN from part 1 (~0.86, trained on about 92,000). Part 3 compares the two at equal
#   numbers of labels.
#
# An AUROC does not tell you what signal a model uses. Part 3 tests that.
