# %% [markdown]
# # Slot 2 · Build: a CNN that reads RNA
#
# **AI Days 2026 · Teaching Machines to Read DNA** — Technion
#
# A protein called **AGO2** sticks to RNA. A lab experiment (CLIP-seq) recorded thousands of RNA
# stretches it was stuck to (**positives**) and stretches it was not (**negatives**).
# In the next 30 minutes you will build a model that reads an RNA sequence and predicts whether
# AGO2 binds it:
#
# $$f(\text{sequence}) = P(\text{bound} \mid \text{sequence})$$
#
# **How this notebook works**
#
# | cell | what you do |
# |---|---|
# | `#@title ... run this cell, nothing to change` | just run it (Shift+Enter) |
# | `# TODO-N` with `# === Your code here ===` | **write the code yourself** (replace the `...`) |
#
# Before you start: *Runtime → Change runtime type → T4 GPU*.

# %%
# %include h_setup.py

# %%
# %include h_data.py

# %% [markdown]
# ## 1. The data
#
# Each sequence is a string over the 4-letter RNA alphabet `A C G U`. The *train* files are used
# to fit the model; the *ls* files are a separate held-out set we only use to measure it.

# %%
#@title Load the sequences - run this cell, nothing to change { display-mode: "form" }
train_seqs, train_y = read_split("train")
val_seqs, val_y = read_split("ls")
print(f"train: {len(train_seqs):,} sequences ({train_y.mean():.0%} bound)")
print(f"ls   : {len(val_seqs):,} sequences ({val_y.mean():.0%} bound)")
print(f"length: {min(map(len, train_seqs))}-{max(map(len, train_seqs))} letters")
print("\na bound sequence:\n", train_seqs[0][:90], "...")

# %% [markdown]
# ## 2. One-hot encoding: a sequence becomes a matrix
#
# A model needs numbers. We write each letter as a column of 4 numbers with a single 1 in the row
# of that letter (rows = `A, C, G, U`). A sequence of length $L$ becomes a $4 \times L$ matrix —
# a one-channel "image" of height 4. Sequences shorter than 501 are padded with columns of 0.25.

# %%
# TODO-1: one-hot encode the FIRST training sequence with `encode_batch` and print
#         its first 8 columns. Check them against the first 8 letters printed above.
#         `encode_batch` takes a list of strings and returns shape (N, 1, 4, 501).
# === Your code here ===
X_one = encode_batch([train_seqs[0]])
print("shape:", tuple(X_one.shape))
print("letters:", "  ".join(train_seqs[0][:8]))
print(X_one[0, 0, :, :8].numpy())
# === End of your code ===

# %% [markdown]
# ## 3. The model
#
# A **convolution** slides a small weight matrix along the sequence and takes a dot product at
# every position. Our first-layer filters are $4 \times 10$: each one scores every 10-letter window,
# so each filter is a **motif detector**. Pooling and a second convolution combine those detections;
# a final linear layer turns them into one number (a *logit*; the sigmoid turns it into a probability).
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

# %% [markdown]
# ## 4. Training
#
# Training = repeatedly nudge the weights downhill on a **loss** (binary cross-entropy: how wrong
# the predicted probabilities are). The model outputs a raw score (a *logit*); `torch.sigmoid`
# squashes it into a probability between 0 and 1.
#
# One training step is always the same four lines (read them in the cell below):
#
# | line | what it does |
# |---|---|
# | `opt.zero_grad()` | forget the previous step's gradients |
# | `loss = criterion(model(X), y)` | predict a batch, measure how wrong |
# | `loss.backward()` | compute how each weight should move |
# | `opt.step()` | move every weight a little |
#
# After every pass over the training data (*epoch*) we **measure** the model on the held-out test
# set (the 1,000 `ls` sequences it never trains on). Measuring = `model.eval()` + `torch.no_grad()`:
# "don't learn from this, just score it". The number we report is **AUROC**: the chance that a random
# bound sequence gets a higher score than a random unbound one. 0.5 = coin flip, 1.0 = perfect.

# %%
def train(model, train_loader, val_seqs, val_y, epochs=10, lr=1e-3, wd=1e-4):
    model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.BCEWithLogitsLoss()
    X_val = encode_batch(val_seqs)
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

        # TODO-3: measure the model on the held-out set.
        #         1. switch to eval mode and turn off gradients (torch.no_grad())
        #         2. probabilities = sigmoid of the model's output on X_val (move X_val to DEVICE)
        #         3. auc = roc_auc_score(true labels, probabilities)
        # === Your code here ===
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(X_val.to(DEVICE))).cpu().numpy()
        auc = roc_auc_score(val_y, probs)
        # === End of your code ===
        history.append(auc)
        print(f"epoch {ep:02d} | held-out AUROC = {auc:.4f} | {time.time() - t:.1f}s")
    return history

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
history = train(model, train_loader, val_seqs, val_y, epochs=EPOCHS)
torch.save(model.state_dict(), DATA_DIR / "my_cnn.pt")
print(f"\nfinal held-out AUROC: {history[-1]:.3f} - saved to {DATA_DIR / 'my_cnn.pt'}")

# %%
#@title Plot the learning curve - run this cell, nothing to change { display-mode: "form" }
import matplotlib.pyplot as plt
plt.figure(figsize=(5, 3))
plt.plot(range(1, len(history) + 1), history, marker="o")
plt.axhline(0.5, ls="--", c="grey", label="coin flip")
plt.xlabel("epoch"); plt.ylabel("held-out AUROC"); plt.ylim(0.45, 1); plt.legend(); plt.grid(alpha=.3)
plt.title("Your CNN"); plt.tight_layout(); plt.show()

# %% [markdown]
# **Reading your result.** The curve should climb fast and flatten around **0.86**. Two training runs
# differ by less than 0.01; and because the test set has only 1,000 sequences, any AUROC on it carries
# about ±0.025 of sampling uncertainty (Slot 1). So 0.85 vs 0.86 is the same result; 0.80 vs 0.86 is not.
#
# ## 5. If you finish early
#
# Change `EPOCHS` (e.g. 3 or 20) and rerun the training cell. Does more training still help after
# epoch 10? Compare with your neighbour. (Keep the architecture as it is: Slot 3 loads this exact shape.)
#
# ## 6. Take your model with you
#
# Each Colab notebook runs on its own machine. Download your trained CNN now; at the start of Slot 3
# you can upload it (or simply use ours).

# %%
#@title Download my_cnn.pt - run this cell { display-mode: "form" }
download_outputs("my_cnn.pt")
