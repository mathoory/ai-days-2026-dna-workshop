# %% [markdown]
# # Slot 7: Fix the data, then test again
#
# **Technion AI Days 2026: Teaching Machines to Read DNA**
#
# In Slot 6, two numbers (the G+C and U fractions of a sequence) already separated bound from
# unbound RNA (AUROC ≈ 0.66). Each unbound sequence is a binding site moved to a random position in
# the same gene (Slot 2), so this compares binding sites with random positions in the same genes. Part
# of the difference may be real biology of where AGO2 binds (mostly in A/U-rich 3′UTRs), and part may
# come from how the unbound positions were drawn. Either way, a model trained on these data can score
# well by reading single-letter composition, and the score alone would not show it.
#
# A **control** removes that signal. We select bound and unbound sequences so that both classes have
# the same G+C fraction, U fraction and length, bin by bin. On the selected data, single-letter
# composition carries no information about the label. Letter-pair and 5-letter-word composition still
# differ between the classes, so a score above 0.5 can still come from them.
#
# | cell | what you do |
# |---|---|
# | `#@title ... run this cell, nothing to change` | run it |
# | `# TODO-N` with `# === Your code here ===` | write it (yourself or with an assistant), then **check** it |
# | `#@title Stuck? ...` | run it only if your TODO doesn't pass its check: it defines our version |

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

# %% [markdown]
# ## 1. A composition-matched benchmark
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
# they differ from the 1,000-sequence held-out test set of Slots 2 to 6. The models also train
# differently here (the CNN for 6 epochs on about 65,000 sequences), so the "original" numbers below
# differ slightly from Slot 6 (for example the CNN 0.84 here, 0.86 to 0.87 there).

# %%
#@title Build the two benchmarks - run this cell, nothing to change { display-mode: "form" }
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
# ### TODO-1: One function, two benchmarks
#
# **Build** the body of `shortcut_scores(train_seqs, train_y, test_seqs, test_y)`: it gets the first
# 12,000 training sequences as `tr`, `ytr` and must compute three test AUROCs (numbers), as in Slot 6.
# The `return` line is already written.
#
# | variable | features for `probe_auc` |
# |---|---|
# | `auc_gc` | `gc_u(tr)` and `gc_u(test_seqs)` |
# | `auc_k5` | `kmer_counts(tr, 5)` and `kmer_counts(test_seqs, 5)` |
# | `auc_fm` | `embed(tok, fm, [to_dna(s) for s in tr])`, same for `test_seqs` |
#
# **Gotchas to tell your assistant:** `probe_auc(F_train, y_train, F_test, y_test)` returns a
# **tuple `(auc, probe)`**: store only the `auc`. `embed(tok, model, seqs)` needs DNA letters, hence
# `to_dna`. The function is called twice (once per benchmark), so it must use only its arguments and
# `tok`, `fm`, not variables like `S_train`.
#
# **Check:** on `matched`, the 2-number model must be close to **0.5** (≈ 0.51). If it isn't, stop and
# find out why. Expected, in table order: 0.64, 0.76, 0.81 on `original` and 0.51, 0.73, 0.77 on
# `matched`.

# %%
tok, fm = load_fm(FM_REPO)


def shortcut_scores(train_seqs, train_y, test_seqs, test_y, n=12000):
    tr, ytr = train_seqs[:n], train_y[:n]
    # TODO-1: compute auc_gc, auc_k5 and auc_fm (see the table above)
    # === Your code here ===
    auc_gc, _ = probe_auc(gc_u(tr), ytr, gc_u(test_seqs), test_y)
    auc_k5, _ = probe_auc(kmer_counts(tr, 5), ytr, kmer_counts(test_seqs, 5), test_y)
    auc_fm, _ = probe_auc(embed(tok, fm, [to_dna(s) for s in tr]), ytr,
                          embed(tok, fm, [to_dna(s) for s in test_seqs]), test_y)
    # === End of your code ===
    return {"G+C and U content only (2 numbers)": auc_gc,
            "5-letter word counts (1,024 numbers)": auc_k5,
            "Mistral-DNA + probe (frozen)": auc_fm}


scores = {name: shortcut_scores(*BENCH[name]) for name in BENCH}
for k in scores["original"]:
    print(f"{k:38s} original {scores['original'][k]:.3f} | matched {scores['matched'][k]:.3f}")

# %%
#@title Stuck? Run this only if TODO-1 failed its check: it computes the three scores for you { display-mode: "form" }
def shortcut_scores(train_seqs, train_y, test_seqs, test_y, n=12000):
    tr, ytr = train_seqs[:n], train_y[:n]
    return {"G+C and U content only (2 numbers)": probe_auc(gc_u(tr), ytr, gc_u(test_seqs), test_y)[0],
            "5-letter word counts (1,024 numbers)": probe_auc(kmer_counts(tr, 5), ytr, kmer_counts(test_seqs, 5), test_y)[0],
            "Mistral-DNA + probe (frozen)": probe_auc(embed(tok, fm, [to_dna(s) for s in tr]), ytr,
                                                      embed(tok, fm, [to_dna(s) for s in test_seqs]), test_y)[0]}
scores = {name: shortcut_scores(*BENCH[name]) for name in BENCH}
for k in scores["original"]:
    print(f"{k:38s} original {scores['original'][k]:.3f} | matched {scores['matched'][k]:.3f}")

# %% [markdown]
# ## 2. Retrain the CNN on both benchmarks
#
# Same architecture as Slot 2, 6 epochs on each training set of about 65,000 sequences (~1.5 minutes
# on a T4 for both). The three models above used 12,000 of them.

# %%
#@title Retrain the CNN twice - run this cell, nothing to change { display-mode: "form" }
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
    scores[name]["CNN (Slot 2 architecture)"] = roc_auc_score(y_te, score_sequences(cnn_b, S_te))
    print(f"CNN, {name:8s}: {scores[name]['CNN (Slot 2 architecture)']:.3f}  ({time.time() - t0:.0f}s)")

# %% [markdown]
# ## 3. Fine-tuning with LoRA
#
# So far the language model was **frozen**: we only trained a linear probe on top. **Fine-tuning**
# also updates the model's own weights for the AGO2 task. A 16M-parameter model like Mistral-DNA could
# be fully fine-tuned on a T4. Models of 500M parameters and more are usually adapted with **LoRA**
# (low-rank adaptation), so we practise it here. The original weights stay fixed, and each attention
# projection matrix gets a trainable low-rank correction. These corrections, plus a new output layer,
# are about 1.6% of the parameters.
#
# We train on the **matched** benchmark, 2 epochs over 20,000 labelled sequences (the frozen probe used
# 12,000), and test on the matched test set. This takes **about 6 to 8 minutes on a T4**. Start the
# TODO-2 cell as soon as you have written it, and read on while it trains.
#
# The setup cell defines what TODO-2 needs: `make_lora()`, which returns a fresh model with LoRA
# add-ons (`clf`), its optimizer (`opt`) and a gradient scaler (`scaler`); `crit` (the loss); and
# `batch_logits(list_of_dna_strings)` → one raw score per sequence. On a T4 the model runs in float16,
# and the `scaler` multiplies the loss before `backward()` so that small gradients do not round to 0.
# On a GPU with bfloat16 the scaler does nothing.

# %%
#@title Set up LoRA - run this cell, nothing to change { display-mode: "form" }
!pip -q install peft==0.21.2   # tested with peft 0.21.2 and Colab's preinstalled transformers (5.19 in our tests)
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification


def make_lora():
    """A fresh Mistral-DNA (16M) with LoRA add-ons, its optimizer, and a gradient scaler."""
    torch.manual_seed(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(FM_REPO, num_labels=1, trust_remote_code=True)
    model = model.float()   # the checkpoint is stored in bfloat16; float32 weights + autocast work with the GradScaler
    model.config.pad_token_id = tok.pad_token_id
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="SEQ_CLS",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])).to(DEVICE)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4)
    scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda" and AMP_DTYPE == torch.float16)
    return model, optimizer, scaler


clf, opt, scaler = make_lora()
n_train = sum(p.numel() for p in clf.parameters() if p.requires_grad)
n_all = sum(p.numel() for p in clf.parameters())
print(f"training {n_train / 1e3:.0f}k of {n_all / 1e6:.1f}M parameters ({n_train / n_all:.1%})")
print("half precision:", "float16 + GradScaler" if AMP_DTYPE == torch.float16 else "bfloat16")
crit = nn.BCEWithLogitsLoss()
LORA_EPOCHS, LORA_N = 2, 20000
M_tr, my_tr, M_te, my_te = BENCH["matched"]
D_tr, ytr, D_te = [to_dna(s) for s in M_tr[:LORA_N]], my_tr[:LORA_N], [to_dna(s) for s in M_te]


def batch_logits(seqs):
    enc = tok(seqs, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.autocast(DEVICE.type, dtype=AMP_DTYPE, enabled=DEVICE.type == "cuda"):
        return clf(input_ids=enc["input_ids"].to(DEVICE), attention_mask=enc["attention_mask"].to(DEVICE)).logits.float().squeeze(-1)


@torch.no_grad()
def lora_auc():
    clf.eval()
    s = np.concatenate([batch_logits(D_te[i:i + 256]).cpu().numpy() for i in range(0, len(D_te), 256)])
    clf.train()
    return roc_auc_score(my_te, s)

# %%
clf, opt, scaler = make_lora()   # a fresh model every time this cell runs
t0 = time.time()
clf.train()
for epoch in range(LORA_EPOCHS):
    order = np.random.default_rng(epoch).permutation(len(D_tr))
    for step, i in enumerate(range(0, len(order), 32)):
        b = order[i:i + 32]
        y = torch.tensor(ytr[b], dtype=torch.float32, device=DEVICE)
        # TODO-2: one training step, the four lines of Slot 2's train(), with backward and step going
        #         through `scaler` (needed for float16 on a T4):
        #             loss = crit(batch_logits(<the DNA strings of this batch>), y)
        #             opt.zero_grad()
        #             scaler.scale(loss).backward()
        #             scaler.step(opt); scaler.update()
        #         The batch's strings: [D_tr[j] for j in b]
        # === Your code here ===
        loss = crit(batch_logits([D_tr[j] for j in b]), y)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        # === End of your code ===
        if step % 200 == 0:
            print(f"epoch {epoch + 1} step {step:4d} | loss {loss.item():.3f} | {time.time() - t0:.0f}s")
    print(f"--> after epoch {epoch + 1}: matched test AUROC {lora_auc():.3f}")
scores["matched"]["Mistral-DNA, LoRA fine-tuned"] = lora_auc()
print()
print(f"LoRA fine-tuned: {scores['matched']['Mistral-DNA, LoRA fine-tuned']:.3f}  vs frozen + probe: "
      f"{scores['matched']['Mistral-DNA + probe (frozen)']:.3f}   ({time.time() - t0:.0f}s)   (want ≈ 0.80)")

# %%
#@title Stuck? Run this only if TODO-2 failed: it trains a fresh LoRA model for you { display-mode: "form" }
clf, opt, scaler = make_lora()
t0 = time.time()
clf.train()
for epoch in range(LORA_EPOCHS):
    order = np.random.default_rng(epoch).permutation(len(D_tr))
    for i in range(0, len(order), 32):
        b = order[i:i + 32]
        loss = crit(batch_logits([D_tr[j] for j in b]), torch.tensor(ytr[b], dtype=torch.float32, device=DEVICE))
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    print(f"--> after epoch {epoch + 1}: matched test AUROC {lora_auc():.3f}")
scores["matched"]["Mistral-DNA, LoRA fine-tuned"] = lora_auc()
print(f"LoRA fine-tuned: {scores['matched']['Mistral-DNA, LoRA fine-tuned']:.3f}  ({time.time() - t0:.0f}s)")

# %% [markdown]
# ## 4. Before and after the control

# %%
#@title The before/after table - run this cell { display-mode: "form" }
import json
print(f"{'model':40s} {'original':>9s} {'matched':>9s}   lost to the control")
for k in scores["original"]:
    if k not in scores["matched"]:
        continue
    o, m = scores["original"][k], scores["matched"][k]
    print(f"{k:40s} {o:9.3f} {m:9.3f}   {'█' * int(round(max(o - m, 0) * 200))} {o - m:+.3f}")
k = "Mistral-DNA, LoRA fine-tuned"
if k in scores["matched"]:
    print(f"{k:40s} {'':9s} {scores['matched'][k]:9.3f}")
print()
print("(each █ = 0.005 AUROC lost once single-letter composition is matched)")
print("(labelled training sequences: CNN about 65,000; LoRA 20,000; the other three 12,000)")
_ = (DATA_DIR / "my_slot7_results.json").write_text(json.dumps(scores, indent=1))

# %% [markdown]
# ## 5. Pretraining and the number of labels
#
# The table above compares models trained on different numbers of labels. At equal numbers the CNN
# still scores higher. With 12,000 labels on matched data it reaches about 0.82, against 0.77 for the
# frozen probe. A lab studying a new question rarely has that many labels.
#
# The next cell compares both approaches trained on 100 to 12,000 labelled sequences (matched
# benchmark, matched test set). The light grey dashed line is the same language model with **random
# weights**. A random network still produces features, so this line shows how much of the probe's
# score depends on pretraining.
#
# **This cell is optional and precomputed.** By default it plots our offline results (3 random draws
# of the labels per size: points are means, bars are standard deviations) and prints our offline
# NT-500M head-to-head. Set `RUN_LIVE = True` to rerun one draw at 4 sizes yourself (~3 minutes on a T4).

# %%
#@title Learn from fewer labels (optional; precomputed by default) { display-mode: "form" }
RUN_LIVE = False   # True = rerun the label curve yourself (one draw per size, ~3 min on a T4)

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
# Our offline head-to-head with NT-500M on the same matched test set (CNN and LoRA: mean of 3 seeds;
# frozen probe: one run).
HEAD_TO_HEAD = {500: {"CNN from scratch": 0.655, "NT-500M + probe (frozen)": 0.733, "NT-500M, LoRA fine-tuned": 0.750},
                20000: {"CNN from scratch": 0.835, "NT-500M + probe (frozen)": 0.783, "NT-500M, LoRA fine-tuned": 0.869}}

if RUN_LIVE:
    from transformers import AutoConfig, AutoModel
    M_tr, my_tr, M_te, my_te = BENCH["matched"]
    POOL = 12000
    t0 = time.time()
    D_pool, D_test = [to_dna(s) for s in M_tr[:POOL]], [to_dna(s) for s in M_te]
    E_pool, E_test = embed(tok, fm, D_pool), embed(tok, fm, D_test)
    torch.manual_seed(SEED)
    fm_rand = AutoModel.from_config(AutoConfig.from_pretrained(FM_REPO)).to(DEVICE).eval()
    R_pool, R_test = embed(tok, fm_rand, D_pool), embed(tok, fm_rand, D_test)
    del fm_rand
    X_pool, y_pool = encode_batch(M_tr[:POOL]), my_tr[:POOL]

    def cnn_from_scratch(sel, seed=0):
        """Slot 2's CNN trained on `sel` only; stops early using 10% of `sel` (never the test set)."""
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
        curve["Pretrained Mistral-DNA (16M) + probe"]["mean"].append(probe_auc(E_pool[sel], y_pool[sel], E_test, my_te, C)[0])
        curve["Random-weight Mistral-DNA (16M) + probe"]["mean"].append(probe_auc(R_pool[sel], y_pool[sel], R_test, my_te, C)[0])
        curve["CNN from scratch"]["mean"].append(cnn_from_scratch(sel))
        print(f"{n:6,} labels | " + " | ".join(f"{k} {v['mean'][-1]:.3f}" for k, v in curve.items()))
    print(f"({time.time() - t0:.0f}s)")
else:
    NS = PRECOMPUTED["NS"]
    curve = {k: v for k, v in PRECOMPUTED.items() if k != "NS"}

styles = {"CNN from scratch": dict(c=CNN_C, ls="-", marker="o"),
          "Pretrained Mistral-DNA (16M) + probe": dict(c=FROZEN_C, ls="-", marker="o"),
          "Random-weight Mistral-DNA (16M) + probe": dict(c=LIGHTGREY, ls="--", marker="s")}
plt.figure(figsize=(6, 3.8))
for k, v in curve.items():
    plt.errorbar(NS, v["mean"], yerr=v["sd"], capsize=2, label=k, **styles[k])
plt.axhline(0.5, ls=":", c=MUTED, lw=1)
plt.xscale("log"); plt.xticks(NS, [f"{n:,}" for n in NS])
plt.xlabel("Labelled training sequences"); plt.ylabel("Test AUROC (matched data)")
plt.title("Test AUROC vs number of labels" + ("" if RUN_LIVE else " (precomputed, 3 draws)"))
plt.legend(); plt.grid(axis="y"); plt.tight_layout(); plt.show()

print("Offline head-to-head with NT-500M on the matched test set:")
for n, row in HEAD_TO_HEAD.items():
    print(f"{n:6,} labels | " + " | ".join(f"{k} {v:.3f}" for k, v in row.items()))

# %% [markdown]
# ### What to take away
#
# * **Frozen probe.** It still scores about 0.77 after matching, and in Slot 6 it barely changed under
#   the shuffle. So it relies mostly on short-word composition (letter pairs, 5-letter words), which
#   the matching leaves in place, and little on single letters.
# * **CNN.** It lost almost nothing to the matched control (0.84 → 0.84), so it uses more than
#   single-letter composition.
# * **Fine-tuning.** Your fine-tuned Mistral-DNA (≈ 0.81) improved on the frozen probe (≈ 0.77) but
#   stayed below the CNN (≈ 0.84). In our offline runs a 30× larger model, NT-500M with LoRA, reached
#   0.87 against the CNN's 0.84 with 20,000 labels, and 0.75 against 0.66 with 500 labels (3 seeds each
#   for the CNN and LoRA). Those fine-tuned models have not yet been put through the shuffle test.
# * **Labels.** With few labels the frozen probe does better than the CNN, and with 100 labels the CNN
#   is close to a coin flip. The random-weight line stays close to 0.5 (at most about 0.55), so the
#   probe's advantage comes from pretraining on the genome.
