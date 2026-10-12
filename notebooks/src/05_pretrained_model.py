# %% [markdown]
# # Slot 5 · Borrow a pretrained DNA model: probe it, and watch pretraining start
#
# **AI Days 2026 · Teaching Machines to Read DNA** — Technion
#
# In Slot 2 you built a CNN from scratch, on our data only. Now we borrow a **foundation model**:
# a language model that was *pretrained* on the whole human genome (it read billions of letters
# and learned to **predict the next chunk**) and was **never told anything about AGO2**.
#
# The recipe (Roee's Slot 4):
#
# 1. the model turns each sequence into one vector: an **embedding**;
# 2. a **linear probe** (logistic regression: a dot product + sigmoid) learns to separate bound from
#    unbound vectors.
#
# Use an AI assistant (Gemini in Colab, ✨, or any chat) as much as you like. Each TODO says exactly
# what to build, with the variable names it must use, the gotchas worth telling your assistant, and a
# check with the number to expect. **Run the check before moving on.**
#
# | cell | what you do |
# |---|---|
# | `#@title ... run this cell, nothing to change` | just run it |
# | `# TODO-N` with `# === Your code here ===` | write it (yourself or with an assistant), then check it |
# | `#@title Stuck? ...` | run it only if your TODO doesn't pass its check: it defines our version |

# %%
# %include h_setup.py

# %%
# %include h_data.py

# %%
# %include h_fm.py

# %%
#@title The data: a fixed subset of AGO2 sequences - run this cell { display-mode: "form" }
S_train, y_train, S_test, y_test = workshop_data()
S_train, S_test = [to_dna(s) for s in S_train], [to_dna(s) for s in S_test]
print(f"train {len(S_train):,} | test {len(S_test):,} | bound: {y_train.mean():.0%}")
print("example:", S_train[0][:70], "...")

# %% [markdown]
# ## 1. Load a pretrained DNA language model
#
# **Mistral-DNA** (Mourad, 2024): 16 million parameters (the "17M" in its name counts the output layer
# we don't use), pretrained on the human genome (hg38). Its tokenizer chops DNA into short "words"
# (tokens) learned from the genome. `[CLS]` and `[SEP]` are special tokens the tokenizer adds at the
# start and end; they are not DNA.

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
# ### TODO-1 · Turn sequences into vectors
#
# **Build** `embed(tok, model, seqs, batch=128)`: a list of DNA strings in, a numpy array of shape
# `(len(seqs), 256)` out, one vector per sequence. For each batch of sequences:
#
# 1. tokenize with `padding=True, truncation=True, max_length=512, return_tensors="pt"`;
# 2. run the model with `output_hidden_states=True` under `torch.no_grad()`;
# 3. take the **last** layer, `outputs.hidden_states[-1]`: shape (batch, tokens, 256);
# 4. average over the **real** tokens only.
#
# **Gotchas to tell your assistant:**
# - sequences in a batch have different lengths, so the tokenizer **pads** the short ones; the
#   `attention_mask` is 1 for real tokens and 0 for padding. Average with the mask, not with `.mean(1)`.
# - move `input_ids` and `attention_mask` to `DEVICE`, and the result back with `.cpu().numpy()`.
# - the tokenizer returns more keys than the model accepts; pass only `input_ids` and `attention_mask`.
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
print("check 1 · shape:", E.shape, "(want (3, 256))")
print("check 2 · padding ignored:", np.allclose(alone, padded, atol=2e-2), "(want True)")

# %%
#@title Stuck? Run this only if TODO-1 failed its checks - it defines our embed() { display-mode: "form" }
# %include h_fm_embed.py
print("using the reference embed() and probe_auc()")

# %%
#@title Embed all sequences - run this cell, nothing to change { display-mode: "form" }
t0 = time.time()
E_train, E_test = embed(tok, fm, S_train), embed(tok, fm, S_test)
print(f"embedded {len(S_train) + len(S_test):,} sequences in {time.time() - t0:.1f}s -> {E_train.shape[1]} numbers each")
np.savez_compressed(DATA_DIR / "my_emb_mistral17m.npz", train=E_train, test=E_test)

# %%
#@title A 2-D map of the vectors - run this cell { display-mode: "form" }
# PCA finds the 2 directions along which the 12,000 vectors vary most, so we can look at them.
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
P = PCA(2).fit_transform(StandardScaler().fit_transform(E_train))
gc_frac = np.array([(s.count("G") + s.count("C")) / len(s) for s in S_train])
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
sc = ax[0].scatter(*P.T, c=gc_frac, s=3, cmap="viridis"); plt.colorbar(sc, ax=ax[0], label="G+C fraction")
ax[1].scatter(*P.T, c=np.where(y_train == 1, "tab:green", "tab:grey"), s=3, alpha=.4)
ax[0].set_title("coloured by G+C"); ax[1].set_title("coloured by bound (green) / unbound")
for a in ax:
    a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.show()
print(f"correlation of direction 1 with G+C: {np.corrcoef(P[:, 0], gc_frac)[0, 1]:+.2f}")

# %% [markdown]
# **Reading the map.** If one colouring follows the main direction of the cloud, that property is what
# the vectors mostly encode. Remember which one it is: Slot 6 tests it.
#
# ### TODO-2 · A linear probe
#
# **Build** `probe_auc(E_train, y_train, E_test, y_test)`. It must **return a tuple `(auc, probe)`**:
# the AUROC on the **test** set, and the fitted model. Inside:
#
# 1. standardise the features (`StandardScaler`: each of the 256 numbers rescaled to mean 0, sd 1);
# 2. fit `LogisticRegression(max_iter=3000, C=0.1)` on the **training** vectors;
# 3. `probe.predict_proba(E_test)[:, 1]` = probability of "bound" → `roc_auc_score(y_test, ...)`.
#
# **Gotchas to tell your assistant:** fit on train, score on test (never score the sequences you fit
# on); use `make_pipeline(StandardScaler(), LogisticRegression(...))` so the scaling learned on train
# is reused on test; return both values.
#
# **Check:** the test AUROC should be **≈ 0.78**. Above 0.95? You probably scored the training set.

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
print(f"Mistral-DNA + linear probe: test AUROC = {auc_fm:.3f}   (want ≈ 0.78; the Slot 2 CNN: ~0.86)")

# %%
#@title Stuck? Run this only if TODO-2 failed its check - it defines our probe_auc() { display-mode: "form" }
# %include h_fm_embed.py
auc_fm, _ = probe_auc(E_train, y_train, E_test, y_test)
results = {"Mistral-DNA (16M), pretrained": auc_fm}
print(f"reference probe: test AUROC = {auc_fm:.3f}")

# %% [markdown]
# ## 2. Bigger model? Nucleotide Transformer, 30× larger
#
# **Nucleotide Transformer** (InstaDeep): 486 million parameters, also pretrained on the human genome.
# Downloading it costs ~2 GB, so by default we load its embeddings, precomputed on exactly these
# sequences. Set `RUN_NT500M = True` to compute them yourself (~1 min download + ~1 min on a T4).

# %%
#@title Nucleotide Transformer 500M - run this cell { display-mode: "form" }
RUN_NT500M = False   # True = download the 2 GB model and compute the vectors yourself
have_precomputed = (DATA_DIR / "emb_nt500m.npz").exists() or "emb_nt500m.npz" in FILE_URLS or bool(DATA_URL)
if RUN_NT500M or os.environ.get("AIDAYS_RUN_NT500M") == "1" or not have_precomputed:
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
print(f"NT-500M + probe: {auc_nt:.3f}   vs   Mistral-DNA 16M: {auc_fm:.3f}")

# %% [markdown]
# ## 3. Does pretraining matter? Watch it start (live, ~4 minutes)
#
# Real pretraining takes GPU-days. We run the *first few minutes* of it, following the Galaxy
# Training Network tutorial (Mourad & Batut, CC-BY): a fresh Mixtral-style model, random weights,
# reading 200-letter pieces of the human genome and learning to **predict the next letter-chunk**.
# Then we probe it on AGO2 — before and after.
#
# Start it, then listen to the explanation while the loss goes down.

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
scratch = AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa").to(DEVICE)
print(f"fresh model: {sum(p.numel() for p in scratch.parameters()) / 1e6:.0f}M parameters, random weights")

genome = pd.read_csv(DATA_DIR / "hg38_200b_verysmall.csv.gz")["text"].astype(str).tolist()
tok.padding_side = "left"
enc = tok(genome, padding="longest", truncation=True, max_length=128, return_tensors="pt")
tok.padding_side = "right"
ids_all, att_all = enc["input_ids"], enc["attention_mask"]
print(f"{len(genome):,} pieces of genome, tokenized")

auc_before, _ = probe_auc(embed(tok, scratch, S_train), y_train, embed(tok, scratch, S_test), y_test)
print(f"AGO2 probe BEFORE pretraining: {auc_before:.3f}\n")

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda" and not use_bf16)
opt = torch.optim.AdamW(scratch.parameters(), lr=5e-4, weight_decay=0.01)
losses, step, t0 = [], 0, time.time()
scratch.train()
while time.time() - t0 < PRETRAIN_MINUTES * 60:
    for i in torch.randperm(len(ids_all)).split(32):
        x, a = ids_all[i].to(DEVICE), att_all[i].to(DEVICE)
        with torch.autocast(DEVICE.type, dtype=amp_dtype, enabled=DEVICE.type == "cuda"):
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

auc_after, _ = probe_auc(embed(tok, scratch, S_train), y_train, embed(tok, scratch, S_test), y_test)
results[f"our model, random weights"] = auc_before
results[f"our model, after {PRETRAIN_MINUTES} min of pretraining"] = auc_after
print(f"\nAGO2 probe AFTER {step} steps ({PRETRAIN_MINUTES} min): {auc_after:.3f}   (before: {auc_before:.3f})")
plt.figure(figsize=(5, 3)); plt.plot(pd.Series(losses).rolling(50).mean(), label="loss (50-step average)")
plt.axhline(np.log(len(tok)), ls="--", c="grey", label=f"blind guess among {len(tok):,} tokens")
plt.xlabel("step"); plt.ylabel("next-chunk loss"); plt.title("Pretraining, live"); plt.legend(fontsize=8)
plt.grid(alpha=.3); plt.tight_layout(); plt.show()
del scratch
torch.cuda.empty_cache()

# %%
#@title Summary - run this cell { display-mode: "form" }
import json
for k, v in sorted(results.items(), key=lambda kv: kv[1]):
    print(f"{v:.3f}  {'█' * int((v - 0.5) * 80):40s} {k}")
_ = (DATA_DIR / "my_slot5_results.json").write_text(json.dumps(results, indent=1))

# %% [markdown]
# ### Before Slot 6
#
# **Reading the loss.** The loss is how surprised the model is by the true next token. Guessing blindly
# among 4,096 tokens gives ln 4096 ≈ 8.3 (the dashed line); every step below it is something learned.
#
# * Pretraining matters: the same network with random weights is barely better than a coin flip,
#   and even 3 minutes of reading the genome moves it a long way.
# * A 30× bigger model scores about the same here (0.78 vs 0.79 is within the ±0.025 noise of a
#   1,000-sequence test set).
# * Frozen, both probes are **below the little CNN** you trained in Slot 2.
#
# **But what did the model actually learn?** A high AUROC is not evidence that a model reads the
# biology. Slot 6 tries to break it.

# %%
#@title Download your vectors (optional: Slot 6 can use them, or ours) - run this cell { display-mode: "form" }
download_outputs("my_emb_mistral17m.npz")
