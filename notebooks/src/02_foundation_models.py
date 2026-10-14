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
# 1. write `embed()`, which turns each sequence into one vector with Mistral-DNA (TODO-1);
# 2. write `probe_auc()`, which trains a linear probe on these vectors (TODO-2), and compare the
#    result with a 30× larger model;
# 3. pretrain a model from random weights for 3 minutes, and probe it before and after.
#
#
# Use an AI assistant (Gemini in Colab, or any chat assistant) as much as you like. Each TODO states
# what to build, the variable names to use, the gotchas to tell your assistant, and a check with the
# number to expect. Run the check before you move on.
#
# | cell | what you do |
# |---|---|
# | `#@title ... run this cell, nothing to change` | run it |
# | `# TODO-N` with `# === Your code here ===` | write it (yourself or with an assistant), then check it |
# | `#@title Stuck? ...` | run it only if your TODO doesn't pass its check: it defines our version |
#
# Before you start: *Runtime → Change runtime type → T4 GPU*.

# %% [markdown]
# ## 0. Setup
#
# Run the three setup cells and the data cell below.

# %%
# %include h_setup.py

# %%
# %include h_data.py
# %include h_encode.py

# %%
# %include h_fm.py

# %% [markdown]
# The language models were trained on DNA, so we write the RNA letter U as T before giving them a
# sequence (`to_dna`). The probe trains on a fixed subset of 12,000 training sequences and is tested
# on the same 1,000-sequence held-out test set as the CNN.

# %%
#@title The data: a fixed subset of AGO2 sequences - run this cell { display-mode: "form" }
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
#@title Download + load the model, look at its tokens - run this cell { display-mode: "form" }
t0 = time.time()
tok, fm = load_fm(FM_REPO)
print(f"loaded {FM_REPO} in {time.time() - t0:.1f}s | "
      f"{sum(p.numel() for p in fm.parameters()) / 1e6:.0f}M parameters | hidden size {fm.config.hidden_size}")
ids = tok(S_train[0][:60])["input_ids"]
print("\nfirst 60 letters :", S_train[0][:60])
print("as tokens        :", " | ".join(tok.convert_ids_to_tokens(ids)))

# %% [markdown]
# ### TODO-1: Turn sequences into vectors
#
# **Build** `embed(tok, model, seqs, batch=128)`: a list of DNA strings in, a numpy array of shape
# `(len(seqs), 256)` out, one vector per sequence. For each batch of sequences:
#
# 1. tokenize with `padding=True, truncation=True, max_length=512, return_tensors="pt"`;
# 2. run the model with `output_hidden_states=True` under `torch.no_grad()`;
# 3. take the **last** layer, `outputs.hidden_states[-1]`: shape (batch, tokens, 256);
# 4. average over the non-padding tokens only (`[CLS]` and `[SEP]` count as non-padding).
#
# **Gotchas to tell your assistant:**
# - sequences in a batch have different lengths, so the tokenizer **pads** the short ones; the
#   `attention_mask` is 1 for real tokens and 0 for padding. Average with the mask, not with `.mean(1)`.
# - move `input_ids` and `attention_mask` to `DEVICE`, and the result back with `.cpu().numpy()`.
# - the tokenizer returns more keys than the model accepts; pass only `input_ids` and `attention_mask`.
# - optional: our version runs the model under `torch.autocast(..., dtype=torch.float16)`, which only
#   makes it faster.
#
# **Check** (the next cell runs it): shape `(3, 256)`, and the same sequence must give the **same**
# vector whether it is embedded alone or next to a longer one (padding must not change it).

# %%
# TODO-1: the embed function
# === Your code here ===
@torch.no_grad()
def embed(tok, model, seqs, batch=128):
    model.eval()
    out = []
    for i in range(0, len(seqs), batch):
        enc = tok(seqs[i:i + batch], return_tensors="pt", padding=True, truncation=True, max_length=512)
        ids, mask = enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)
        with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            h = model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[-1].float()
        m = mask.unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
    return np.concatenate(out)
# === End of your code ===

E = embed(tok, fm, S_train[:3])
short = min(S_train[:50], key=len); long_ = max(S_train[:50], key=len)
alone, padded = embed(tok, fm, [short]), embed(tok, fm, [short, long_])[:1]
print("check 1, shape:", E.shape, "(want (3, 256))")
print("check 2, padding ignored:", np.allclose(alone, padded, atol=2e-2), "(want True)")

# %%
#@title Stuck? Run this only if TODO-1 failed its checks: it defines our embed() { display-mode: "form" }
# %include h_fm_embed.py
print("using the reference embed()")

# %%
#@title Embed all sequences - run this cell, nothing to change { display-mode: "form" }
t0 = time.time()
E_train, E_test = embed(tok, fm, S_train), embed(tok, fm, S_test)
print(f"embedded {len(S_train) + len(S_test):,} sequences in {time.time() - t0:.1f}s -> {E_train.shape[1]} numbers each")
np.savez_compressed(DATA_DIR / "my_emb_mistral17m.npz", train=E_train, test=E_test)

# %% [markdown]
# ### A 2-D map of the vectors

# %%
#@title A 2-D map of the vectors - run this cell { display-mode: "form" }
# PCA finds the 2 directions along which the 12,000 vectors vary most, so we can look at them.
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
P = PCA(2).fit_transform(StandardScaler().fit_transform(E_train))
gc_frac = np.array([(s.count("G") + s.count("C")) / len(s) for s in S_train])
order = np.random.default_rng(0).permutation(len(P))    # random drawing order: neither class hides the other
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
sc = ax[0].scatter(*P[order].T, c=gc_frac[order], s=3, cmap=CMAP_SEQ)
plt.colorbar(sc, ax=ax[0], label="G+C fraction")
ax[1].scatter(*P[order].T, c=[BOUND if y else UNBOUND for y in y_train[order]], s=3, alpha=.6)
for col, name in ((UNBOUND, "Unbound"), (BOUND, "Bound")):
    ax[1].scatter([], [], color=col, s=12, label=name)
ax[1].legend()
ax[0].set_title("Coloured by G+C fraction"); ax[1].set_title("Coloured by label")
for a in ax:
    a.set_xticks([]); a.set_yticks([]); a.set_xlabel("PCA direction 1"); a.set_ylabel("PCA direction 2")
plt.tight_layout(); plt.show()
for j in (0, 1):
    print(f"correlation of PCA direction {j + 1} with G+C: {np.corrcoef(P[:, j], gc_frac)[0, 1]:+.2f}")

# %% [markdown]
# **Reading the map.** If one colouring changes smoothly across the cloud, that property accounts for
# much of the variation between the vectors. Here it is the G+C fraction. It changes mostly along PCA
# direction 2, the vertical axis (correlation about 0.81), and less along direction 1 (about 0.52);
# the signs can flip between runs. Bound sequences concentrate at the low-G+C end of the map. Part 3
# tests whether the probe uses this.
#
# ### TODO-2: A linear probe
#
# **Build** `probe_auc(E_train, y_train, E_test, y_test)`. It must **return a tuple `(auc, probe)`**:
# the AUROC on the **test** set, and the fitted model. Inside:
#
# 1. standardise the features (`StandardScaler`: each of the 256 numbers is shifted and scaled to
#    mean 0 and standard deviation 1 on the training set);
# 2. fit `LogisticRegression(max_iter=3000, C=0.1)` on the **training** vectors. `C` sets the
#    regularisation: a smaller `C` keeps the weights smaller (stronger regularisation), which guards
#    against fitting noise in 12,000 sequences with 256 numbers each;
# 3. `probe.predict_proba(E_test)[:, 1]` = probability of "bound" → `roc_auc_score(y_test, ...)`.
#
# **Gotchas to tell your assistant:** fit on train, score on test (never score the sequences you fit
# on); use `make_pipeline(StandardScaler(), LogisticRegression(...))` so the scaling learned on train
# is reused on test; return both values.
#
# **Check:** the test AUROC should be **≈ 0.78**. If you get about 0.82, you probably scored the
# training vectors: on the sequences it was fitted on, this probe scores about 0.82.

# %%
# TODO-2: the linear probe
# === Your code here ===
def probe_auc(E_train, y_train, E_test, y_test):
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))
    probe.fit(E_train, y_train)
    return roc_auc_score(y_test, probe.predict_proba(E_test)[:, 1]), probe
# === End of your code ===

auc_fm, _ = probe_auc(E_train, y_train, E_test, y_test)
results = {"Mistral-DNA (16M), pretrained": auc_fm}
print(f"Mistral-DNA + linear probe: test AUROC = {auc_fm:.3f}   (want ≈ 0.78)")

# %%
#@title Stuck? Run this only if TODO-2 failed its check: it defines our probe_auc() { display-mode: "form" }
# %include h_fm_probe.py
auc_fm, _ = probe_auc(E_train, y_train, E_test, y_test)
results = {"Mistral-DNA (16M), pretrained": auc_fm}
print(f"reference probe: test AUROC = {auc_fm:.3f}")

# %% [markdown]
# ## 2. A 30× larger model: Nucleotide Transformer
#
# **Nucleotide Transformer 500M** (NT-500M, InstaDeep): 486 million parameters, pretrained on the human
# reference genome by masked-token prediction. The download is about 2 GB, so by default we load its
# embeddings, precomputed on these same sequences with the same `embed`. Set `RUN_NT500M = True` to
# compute them yourself.

# %%
#@title Nucleotide Transformer 500M - run this cell { display-mode: "form" }
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
# Start the cell, then read **What to take away** below the Summary cell while the loss goes down.

# %%
#@title Pretrain from scratch for a few minutes - run this cell { display-mode: "form" }
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
# float32 weights; autocast does the half-precision maths
scratch = AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa").float().to(DEVICE)
print(f"fresh model: {sum(p.numel() for p in scratch.parameters()) / 1e6:.0f}M parameters, random weights")

genome = pd.read_csv(DATA_DIR / "hg38_200b_verysmall.csv.gz")["text"].astype(str).tolist()
tok.padding_side = "left"
enc = tok(genome, padding="longest", truncation=True, max_length=128, return_tensors="pt")
tok.padding_side = "right"
ids_all, att_all = enc["input_ids"], enc["attention_mask"]
print(f"{len(genome):,} pieces of genome, tokenized")


def embed_checked(model, seqs):
    """embed(), but replace any NaN or infinite numbers by 0 (float16 can overflow) and say so."""
    E = embed(tok, model, seqs)
    bad = int((~np.isfinite(E)).any(1).sum())
    if bad:
        print(f"WARNING: {bad:,} of {len(E):,} vectors contained NaN or infinite values (float16 overflow); "
              "they were set to 0, so this probe score is less reliable")
    return np.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0)


auc_before, _ = probe_auc(embed_checked(scratch, S_train), y_train, embed_checked(scratch, S_test), y_test)
print(f"AGO2 probe BEFORE pretraining: {auc_before:.3f}\n")

# bfloat16 where the GPU supports it; otherwise float16, with a GradScaler so that small gradients do not round to 0
print("half precision:", "float16 + GradScaler" if AMP_DTYPE == torch.float16 else "bfloat16")
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

# %%
#@title Summary - run this cell { display-mode: "form" }
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

# %%
#@title Download your vectors (optional: part 3 can use them, or ours) - run this cell { display-mode: "form" }
download_outputs("my_emb_mistral17m.npz")
