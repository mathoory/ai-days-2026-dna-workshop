# %% [markdown]
# # Slot 2: Build a CNN that predicts AGO2 binding
#
# **Technion AI Days 2026: Teaching Machines to Read DNA**
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
# In the next 30 minutes you will build a model that takes an RNA sequence and estimates the
# probability that AGO2 binds it:
#
# $$f(\text{sequence}) = P(\text{bound} \mid \text{sequence})$$
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

# %%
# %include h_setup.py

# %%
# %include h_data.py

# %% [markdown]
# ## 1. The data
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
# ## 2. One-hot encoding: a sequence becomes a matrix
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
# === Your code here ===
X_one = encode_batch([train_seqs[0]])
print("shape:", tuple(X_one.shape))
print("letters:", "  ".join(train_seqs[0][:8]))
print(X_one[0, 0, :, :8].numpy())
# === End of your code ===

# %% [markdown]
# ## 3. The model
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
# ## 4. Training
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
# sequences, so any AUROC on it has a sampling uncertainty of about ±0.025 (95% range, Slot 1). A 0.85
# and a 0.86 are the same result; a 0.80 and a 0.86 are not.
#
# ## 5. If you finish early
#
# Change `EPOCHS` (e.g. 3 or 20) and rerun the training cell. Does more training still help after
# epoch 10? Compare with your neighbour. (Keep the architecture as it is: Slot 3 loads this exact shape.)
#
# ## 6. Take your model with you
#
# Each Colab notebook runs on its own machine. Download your trained CNN now. At the start of Slot 3
# you can upload it, or use ours.

# %%
#@title Download my_cnn.pt - run this cell { display-mode: "form" }
download_outputs("my_cnn.pt")
