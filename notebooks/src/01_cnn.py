# %% [markdown]
# # Part 1: Machine learning and CNNs
#
# **Build a CNN that predicts AGO2 binding, and look inside it**
#
# **Technion AI Days 2026: Applications of AI in Biotechnology and Genomics**
#
# **AGO2** is a protein that carries a small RNA, a microRNA, and binds RNA molecules that pair with
# that microRNA. **CLIP-seq** measures where it binds: UV light crosslinks proteins to the RNA they
# touch, an antibody pulls down AGO2, and the RNA fragments attached to it are sequenced.
#
# **The data.** We use the AGO2 set of the GraphProt CLIP benchmark (Maticzka et al. 2014), as used by
# Pan & Shen (2018). Each **bound** sequence contains a CLIP-seq binding site of about 40 letters, with
# flanking context on both sides (in the files the site is in capital letters and the context in small
# letters; we convert everything to capitals). Each **unbound** sequence was made by moving a binding
# site to a random position in the same gene. The files call the two classes *positives* and
# *negatives*.
#
# In this notebook you will:
#
# 1. build and train a model that takes an RNA sequence and estimates the probability that AGO2
#    binds it, $f(\text{sequence}) = P(\text{bound} \mid \text{sequence})$;
# 2. read off the letter patterns that its first-layer filters respond to, as sequence logos;
# 3. test one of these patterns by inserting it into random RNA, next to a shuffled control.
#
# Each section heading gives a projected time on a T4 GPU, including reading and typing.
#
# **How this notebook works**
#
# | cell | what you do |
# |---|---|
# | `#@title ... run this cell, nothing to change` | run it (Shift+Enter) |
# | `# TODO-N` with `# === Your code here ===` | write the code yourself (replace the `...`) |
# | `#@title Stuck? ...` | run it only if your TODO fails: it defines our version |
#
# Before you start: *Runtime → Change runtime type → T4 GPU*.

# %% [markdown]
# ## 0. Setup (about 3 min)
#
# Run the next two cells. The first one also installs `logomaker`, which draws the sequence logos in
# section 7.

# %%
# %include h_setup.py
!pip -q install logomaker==0.8.7

# %%
# %include h_data.py

# %% [markdown]
# # Build a CNN (sections 1 to 5)
#
# ## 1. The data (about 2 min)
#
# Each sequence is a string over the 4-letter RNA alphabet `A C G U`. The *train* files are used to
# fit the model. The *ls* files hold 1,000 other sequences, the **held-out test set**, which we use only
# for evaluation.
#
# The split is random by sequence, so some test sequences lie near a training sequence in the same
# gene. We checked whether this inflates the score. On the 698 test sequences with no training
# sequence centred within 150 letters, our CNN scores 0.85, against 0.86 on all 1,000, so the score
# barely changes. Real projects split by gene or by chromosome.

# %%
#@title Load the sequences - run this cell, nothing to change { display-mode: "form" }
train_seqs, train_y = read_split("train")
test_seqs, test_y = read_split("ls")
print(f"train: {len(train_seqs):,} sequences ({train_y.mean():.0%} bound)")
print(f"test : {len(test_seqs):,} sequences ({test_y.mean():.0%} bound)")
print(f"length: {min(map(len, train_seqs))}-{max(map(len, train_seqs))} letters")
print("\na bound sequence:\n", train_seqs[0][:90], "...")

# %% [markdown]
# ## 2. One-hot encoding: a sequence becomes a matrix (TODO-1, about 3 min)
#
# The model takes numbers as input. We encode each letter as a column of 4 numbers: 1 in the row of
# that letter and 0 elsewhere (rows = `A, C, G, U`). A sequence of length $L$ becomes a $4 \times L$ matrix,
# which we treat as a one-channel image of height 4.
#
# Every input has the same width, 501 columns, the input size used by Pan & Shen. Our sequences are
# 137 to 375 letters long, so the remaining columns are padding: columns of 0.25 (equal weight on all
# four letters).

# %%
# TODO-1: one-hot encode the FIRST training sequence with `encode_batch` and print
#         its first 8 columns. Check them against the first 8 letters printed above.
#         `encode_batch` takes a list of strings and returns shape (N, 1, 4, 501).
#         Indexing hint: X[0, 0, :, :8] = sequence 0, its only channel, all 4 rows, columns 0-7.
#         Expected: shape (1, 1, 4, 501), and in each of the 8 columns a single 1, in the row
#         of that column's letter.
# === Your code here ===
X_one = encode_batch([train_seqs[0]])
print("shape:", tuple(X_one.shape))
print("letters:", "  ".join(train_seqs[0][:8]))
print(X_one[0, 0, :, :8].numpy())
# === End of your code ===

# %%
#@title Stuck? Run this only if TODO-1 failed: it prints the first 8 columns for you { display-mode: "form" }
X_one = encode_batch([train_seqs[0]])
print("shape:", tuple(X_one.shape))
print("letters:", "  ".join(train_seqs[0][:8]))
print(X_one[0, 0, :, :8].numpy())

# %% [markdown]
# ## 3. The model (TODO-2, about 7 min)
#
# A **convolution** layer computes the dot product of a small weight matrix (a *filter*) with every
# window of the input. Our first-layer filters are $4 \times 10$, so each one gives a score to every
# 10-letter window. The score is high when the window matches the letter pattern stored in the
# filter's weights, so each filter responds most strongly to one letter pattern (a candidate **motif**,
# a short sequence pattern). Each filter produces one row of output, called a **channel**, so 10
# filters give 10 channels.
#
# **ReLU** (`max(0, x)`) sets negative values to 0. Max pooling keeps the largest value in each block
# of 3 positions. A second convolution combines the first-layer channels, and a final linear layer
# maps them to one number per sequence, the **logit**. A logit can be any real number: large and
# positive means "bound", large and negative means "unbound". Section 4 turns it into a probability.
#
# Shapes, for one sequence of width 501:
#
# | layer | output shape |
# |---|---|
# | input | `1 × 4 × 501` |
# | `Conv2d(1, 10, (4, 10))` + ReLU | `10 × 1 × 492` |
# | `MaxPool2d((1, 3))` | `10 × 1 × 164` |
# | `Conv2d(10, 20, (1, 5))` + ReLU | `20 × 1 × 160` |
# | `MaxPool2d((1, 3))` | `20 × 1 × 53` |
# | `Flatten` → `Linear(20*53, 1)` | `1` |

# %%
class GlobalCNN(nn.Module):
    """One-hot batch (N, 1, 4, 501) -> one raw score (logit) per sequence, shape (N,)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # TODO-2: build the network from the table above, in order, separated by commas.
            #         The last layer is nn.Linear(20 * 53, 1).
            # === Your code here ===
            nn.Conv2d(1, 10, (4, 10)), nn.ReLU(),
            nn.MaxPool2d((1, 3)),
            nn.Conv2d(10, 20, (1, 5)), nn.ReLU(),
            nn.MaxPool2d((1, 3)),
            nn.Flatten(),
            nn.Linear(20 * 53, 1),
            # === End of your code ===
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


# sanity check: two sequences in, two scores out
print(GlobalCNN()(encode_batch(train_seqs[:2])).shape, "<- should be torch.Size([2])")

# %%
#@title Stuck? Run this only if TODO-2 failed: it defines our GlobalCNN { display-mode: "form" }
class GlobalCNN(nn.Module):
    """One-hot batch (N, 1, 4, 501) -> one raw score (logit) per sequence, shape (N,)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 10, (4, 10)), nn.ReLU(),
            nn.MaxPool2d((1, 3)),
            nn.Conv2d(10, 20, (1, 5)), nn.ReLU(),
            nn.MaxPool2d((1, 3)),
            nn.Flatten(),
            nn.Linear(20 * 53, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


print("reference GlobalCNN:", GlobalCNN()(encode_batch(train_seqs[:2])).shape, "<- should be torch.Size([2])")

# %% [markdown]
# ## 4. Training (TODO-3, about 7 min)
#
# Training minimises a **loss** by gradient descent (here the Adam variant): each step computes the
# gradient of the loss on one batch of 256 sequences and moves every weight a small amount against
# its gradient. Our loss is binary cross-entropy, the negative log of the probability the model
# assigns to the correct label, averaged over the batch. The model outputs a logit $z$; the sigmoid
# $\sigma(z) = 1 / (1 + e^{-z})$ (`torch.sigmoid`) turns it into a probability between 0 and 1.
#
# One training step is always the same four lines (find them in the cell below):
#
# | line | what it does |
# |---|---|
# | `opt.zero_grad()` | reset the gradients from the previous step to zero |
# | `loss = criterion(model(X), y)` | compute the loss on one batch |
# | `loss.backward()` | compute the gradient of the loss for every weight (backpropagation) |
# | `opt.step()` | update every weight using its gradient |
#
# After every pass over the training data (an *epoch*) we evaluate the model on the held-out test
# set: the 1,000 `ls` sequences it never trains on. `model.eval()` puts the model in evaluation mode
# and `torch.no_grad()` turns off gradient tracking, so scoring these sequences does not change the
# weights. The number we report is **AUROC**: the probability that a randomly chosen bound sequence
# gets a higher score than a randomly chosen unbound one. 0.5 is a coin flip, 1.0 is perfect.

# %%
def train(model, train_loader, test_seqs, test_y, epochs=10, lr=1e-3, wd=1e-4):
    model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.BCEWithLogitsLoss()
    X_test = encode_batch(test_seqs)
    history = []
    for ep in range(1, epochs + 1):
        t = time.time()
        model.train()
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            opt.step()

        # TODO-3: measure the model on the held-out test set.
        #         1. call model.eval(), then do steps 2 and 3 inside a `with torch.no_grad():` block
        #         2. probs = the sigmoid of the model's output on X_test. Move X_test to DEVICE first,
        #            and bring the result back with .cpu().numpy() (roc_auc_score needs a numpy array)
        #         3. auc = roc_auc_score(true labels, probs)
        #         Expected (the training cell below prints it): about 0.79 after epoch 1, about
        #         0.86 after epoch 10.
        # === Your code here ===
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(X_test.to(DEVICE))).cpu().numpy()
        auc = roc_auc_score(test_y, probs)
        # === End of your code ===
        history.append(auc)
        print(f"epoch {ep:02d} | held-out test AUROC = {auc:.4f} | {time.time() - t:.1f}s")
    return history

# %%
#@title Stuck? Run this only if TODO-3 failed: it defines our train() { display-mode: "form" }
def train(model, train_loader, test_seqs, test_y, epochs=10, lr=1e-3, wd=1e-4):
    model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.BCEWithLogitsLoss()
    X_test = encode_batch(test_seqs)
    history = []
    for ep in range(1, epochs + 1):
        t = time.time()
        model.train()
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(X_test.to(DEVICE))).cpu().numpy()
        auc = roc_auc_score(test_y, probs)
        history.append(auc)
        print(f"epoch {ep:02d} | held-out test AUROC = {auc:.4f} | {time.time() - t:.1f}s")
    return history


print("using the reference train()")

# %%
#@title Encode everything and train - run this cell, nothing to change { display-mode: "form" }
from torch.utils.data import DataLoader, TensorDataset

t0 = time.time()
X_train = encode_batch(train_seqs)
y_train = torch.tensor(train_y, dtype=torch.float32)
print(f"encoded {len(train_seqs):,} sequences in {time.time() - t0:.1f}s")
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True, drop_last=True)

EPOCHS = 10
torch.manual_seed(SEED)
model = GlobalCNN()
history = train(model, train_loader, test_seqs, test_y, epochs=EPOCHS)
torch.save(model.state_dict(), DATA_DIR / "my_cnn.pt")
print(f"\nfinal held-out test AUROC: {history[-1]:.3f} - saved to {DATA_DIR / 'my_cnn.pt'}")

# %%
#@title Plot the learning curve - run this cell, nothing to change { display-mode: "form" }
import matplotlib.pyplot as plt
plt.figure(figsize=(5, 3))
plt.plot(range(1, len(history) + 1), history, marker="o", c=CNN_C, label="Your CNN")
plt.axhline(0.5, ls="--", c=MIDGREY, lw=1, label="Coin flip")
plt.xlabel("Epoch"); plt.ylabel("Held-out test AUROC"); plt.ylim(0.45, 1); plt.legend(); plt.grid(axis="y")
plt.title("Learning curve"); plt.tight_layout(); plt.show()

# %% [markdown]
# **Reading your result.** The curve should climb quickly and level off at about **0.86** (0.86 to
# 0.87). In our tests, three seeds finished within 0.01 of each other. The test set has only 1,000
# sequences, so any AUROC on it has a sampling uncertainty of about ±0.025 (95% range, part 1 talk).
# A difference of 0.01 (0.85 against 0.86) is within this noise; a difference of 0.06 (0.80 against 0.86) is not.

# %% [markdown]
# ## 5. Take your model with you (about 1 min)
#
# Each Colab notebook runs on its own machine. Download your trained CNN now. At the start of part 3
# (`03_shortcuts_and_controls`) you can upload it, or use ours.

# %%
#@title Download my_cnn.pt - run this cell { display-mode: "form" }
download_outputs("my_cnn.pt")

# %% [markdown]
# # Look inside the CNN (sections 6 to 10)
#
# Your CNN scores well, so it has picked up some signal that separates bound from unbound RNA. This
# half asks what that signal is, and then tests the answer. The cells below use `model`, the CNN
# you trained in section 4.
#
# 1. **Read a motif off the model.** Each first-layer filter scores every 10-letter window. We
#    collect the windows where its output (its *activation*) is high and summarise them as a
#    *sequence logo*.
# 2. **Test it.** The logo suggests what a filter responds to. We insert that motif into random RNA
#    and check whether the model's score goes up.

# %% [markdown]
# ## 6. The toolkit (given, about 3 min)
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
# | `score_sequences(model, seqs)` | the model's probability of "bound" for each sequence (a numpy array) |

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


@torch.no_grad()
def score_sequences(model, seqs, batch=1024):
    """The model's binding probability for each sequence (list of RNA strings)."""
    model.eval()
    out = []
    for i in range(0, len(seqs), batch):
        X = encode_batch(seqs[i:i + batch])
        out.append(torch.sigmoid(model(X.to(DEVICE))).cpu().numpy())
    return np.concatenate(out)


print(f"toolkit ready (filter width = {KERNEL})")

# %% [markdown]
# ## 7. Read a motif off the model (TODO-4, about 9 min)
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
    # TODO-4: for filter f, take its activations with  acts_f = acts[:, f, :]  (sequences x positions).
    #         Then cutoff_for -> windows_above (the sequences are pos_seqs) -> pwm_of -> consensus_of,
    #         and fill motifs[f] (consensus string), pwms[f] (the table) and n_hits[f] (number of windows).
    #         Expected: 10 lines, one per filter, each with its window count and a 10-letter
    #         consensus; pwms[f] is a numpy array of shape (10, 4) whose rows sum to 1.
    # === Your code here ===
    acts_f = acts[:, f, :]
    windows = windows_above(acts_f, pos_seqs, cutoff_for(acts_f))
    pwms[f] = pwm_of(windows)
    motifs[f] = consensus_of(pwms[f])
    n_hits[f] = len(windows)
    # === End of your code ===
    print(f"filter {f:02d}: {n_hits[f]:>7,} windows | consensus {motifs[f]}")

# %%
#@title Stuck? Run this only if TODO-4 failed: it builds the tables for you { display-mode: "form" }
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
# ## 8. Now test it (TODO-5, about 8 min)
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
# *(This shuffle only reorders the motif's 10 letters. Part 3 uses a stricter shuffle that keeps
# every letter pair of a whole sequence.)*
#
# **Hint for TODO-5 (and for your AI assistant):** `plant(seqs, motif, copies)` returns new strings;
# `score_sequences(model, seqs)` returns one probability per string; you need the **mean** for each
# number of copies, as two lists of 7 numbers.

# %%
rng = np.random.default_rng(2)          # this cell's own random numbers, so a rerun gives the same result
canvas = random_rna(1000, rng)
baseline = score_sequences(model, canvas).mean()
shuffled = "".join(rng.permutation(list(MY_MOTIF)))
copies = list(range(1, 8))              # 7 copies fit into a 330-letter sequence

# TODO-5: fill real_scores and control_scores with the MEAN model score after inserting
#         c copies (for every c in `copies`) of MY_MOTIF and of `shuffled`, respectively.
#         Expected: two lists of 7 numbers between 0 and 1. In our runs the motif's score jumps
#         at copy 4 (letter 175); the plot below compares the two lists.
# === Your code here ===
real_scores = [score_sequences(model, plant(canvas, MY_MOTIF, c)).mean() for c in copies]
control_scores = [score_sequences(model, plant(canvas, shuffled, c)).mean() for c in copies]
# === End of your code ===

# %%
#@title Stuck? Run this only if TODO-5 failed: it computes both lists for you { display-mode: "form" }
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
# score too, because it contains the same letters. If the curve jumps at one copy number,
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
# ### What to take away (about 3 min)
#
# * The logo suggests a pattern. Inserting the motif, next to a shuffled control, tests whether the
#   model's output responds to it.
# * A strongly activated filter can still lower the score. Its effect on the output passes
#   through every layer above it, so your curve can go down.
# * If the logo looks the same on bound and unbound RNA, that filter picks up a property of RNA in
#   general (for example, how U-rich it is) and tells you little about AGO2.
# * The CNN's last layer has one weight per (filter, **position**). Almost every sequence in this data
#   set, bound or unbound, was cut so that the measured ~40-letter site sits at letters 150–190, with
#   context on both sides (the sequences are 137 to 375 letters long), and the CNN learned to weight
#   that region. In our runs the shuffled copy gets much of the same increase there, so part of the effect
#   comes from which letters occupy positions 150–190, whatever their order. A motif's effect on the score
#   depends on where it is placed.
#
# **Discuss with your neighbour:** did your motif score higher than its shuffled control? Did theirs?

# %% [markdown]
# ## 9. Bonus (optional): known microRNA target sites (about 2 min)

# %%
#@title Bonus (optional): insert known microRNA target sites - run this cell { display-mode: "form" }
# AGO2 carries a microRNA and binds RNA that base-pairs with the microRNA's "seed" (letters 2-7;
# these sites also pair letter 8). These sites are complementary to the seeds of a few abundant microRNAs. Each one is
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
# and a CNN trained from other random starting weights (yours, for example) may respond differently. Of the 6 copies,
# only copy 4 (letter 175) lands in letters 150–190, the region the position scan showed the CNN
# weights most. A copy that does not fit into a shorter sequence is skipped.

# %% [markdown]
# ## 10. If you finish early (about 3 min)
#
# Change `EPOCHS` (e.g. 3 or 20) and rerun the training cell of section 4. Does more training still
# help after epoch 10? Compare with your neighbour. (Keep the architecture as it is: part 3 loads this
# exact shape.) The training cell saves each new model as `my_cnn.pt`; to take the new one to part 3,
# run the download cell of section 5 again.
