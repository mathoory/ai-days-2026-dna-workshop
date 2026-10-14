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
# 3. test one of these patterns by inserting it into random RNA, next to a control you design.
#
#
# **How this notebook works**
#
# | cell | what you do |
# |---|---|
# | ▶ a cell with a title (its code is hidden) | run it (Shift+Enter) |
# | `# TODO-N` with `# === Your code here ===` | write the code yourself (replace the `...`) |
# | ▶ Test TODO-N | run it: it checks your code and says what is wrong |
#
# Before you start: *Runtime → Change runtime type → T4 GPU*.

# %% [markdown]
# ## 0. Setup
#
# Run the next cell. It also installs `logomaker`, which draws the sequence logos in section 6.

# %%
#@title ▶ Setup and data helpers { display-mode: "form" }
# %include-body h_setup.py
!pip -q install logomaker==0.8.7
# %include-body h_data.py

# %% [markdown]
# # Build a CNN (sections 1 to 4)
#
# ## 1. The data
#
# Each sequence is a string over the 4-letter RNA alphabet `A C G U`. The *train* files are used to
# fit the model. The *ls* files hold 1,000 other sequences, the **held-out test set**, which we use only
# for evaluation.

# %%
#@title ▶ Load the sequences { display-mode: "form" }
train_seqs, train_y = read_split("train")
test_seqs, test_y = read_split("ls")
print(f"train: {len(train_seqs):,} sequences ({train_y.mean():.0%} bound)")
print(f"test : {len(test_seqs):,} sequences ({test_y.mean():.0%} bound)")
print(f"length: {min(map(len, train_seqs))}-{max(map(len, train_seqs))} letters")
print("\na bound sequence:\n", train_seqs[0][:90], "...")

# %% [markdown]
# ## 2. One-hot encoding: a sequence becomes a matrix (TODO-1)
#
# The model takes numbers as input. We encode each letter as a column of 4 numbers: 1 in the row of
# that letter and 0 elsewhere (rows = `A, C, G, U`). A sequence of length $L$ becomes a $4 \times L$ matrix,
# which we treat as a one-channel image of height 4.
#
# Every input has the same width, 501 columns, the input size used by Pan & Shen. Our sequences are
# 137 to 375 letters long, so the remaining columns are padding: columns of 0.25 (equal weight on all
# four letters).
#
# You will write this conversion in two steps: first for one sequence, then for a list of sequences.
#
# **Tools:** [`np.full`](https://numpy.org/doc/stable/reference/generated/numpy.full.html), [`np.stack`](https://numpy.org/doc/stable/reference/generated/numpy.stack.html) and
# [`torch.from_numpy`](https://pytorch.org/docs/stable/generated/torch.from_numpy.html).

# %%
#@title ▶ One sequence and the alphabet { display-mode: "form" }
s = train_seqs[0]
print("first 10 letters:", s[:10], "| length:", len(s))
print("ALPHABET =", repr(ALPHABET), "| the row of each letter:", {b: ALPHABET.index(b) for b in ALPHABET})
print("MAX_LEN  =", MAX_LEN, "columns per sequence")

# %%
# TODO-1a: write one_hot(seq), which turns ONE sequence into its one-hot matrix, with 4 rows
#          (A, C, G, U, in the order of ALPHABET) and MAX_LEN columns, one column per letter.
#          - the column of a letter has a 1 in that letter's row and 0 in the other three rows
#          - every column after the end of the sequence is padding: 0.25 in all four rows
#          - a letter other than A, C, G or U also gets a column of 0.25
def one_hot(seq):
    # === Your code here ===
    M = np.full((4, MAX_LEN), 0.25, dtype=np.float32)
    for i, b in enumerate(seq[:MAX_LEN]):
        if b in ALPHABET:
            M[:, i] = 0.0
            M[ALPHABET.index(b), i] = 1.0
    return M
    # === End of your code ===


M = one_hot(train_seqs[0])
L = len(train_seqs[0])
print("shape:", M.shape, "(rows, columns)")
print("letters:", "  ".join(train_seqs[0][:8]))
print(M[:, :8])
print(f"columns {L - 1} and {L} (the last letter, then the first padding column):")
print(M[:, L - 1:L + 1])

# %%
# TODO-1b: write encode_batch(seqs), which turns a list of N sequences into one input for the CNN:
#          the one-hot matrices of all N sequences, stacked into one PyTorch tensor of 32-bit decimal
#          numbers (float32), with the shape (N, 1, 4, MAX_LEN). The CNN reads each matrix like a
#          one-colour image, and the 1 is that single colour channel.
def encode_batch(seqs):
    # === Your code here ===
    X = np.stack([one_hot(s) for s in seqs])[:, None]
    return torch.from_numpy(X)
    # === End of your code ===


X3 = encode_batch(train_seqs[:3])
print("shape:", tuple(X3.shape), "(sequences, channel, rows, columns) | numbers stored as:", X3.dtype)

# %%
#@title ▶ Test TODO-1 { display-mode: "form" }
def test_encoding():
    rng = np.random.default_rng(0)
    seqs = [train_seqs[i] for i in rng.choice(len(train_seqs), 200, replace=False)] + ["ACGUN"]
    X = encode_batch(seqs)
    assert tuple(X.shape) == (len(seqs), 1, 4, MAX_LEN), f"❌ shape {tuple(X.shape)}, want {(len(seqs), 1, 4, MAX_LEN)}"
    assert X.dtype == torch.float32, f"❌ the numbers are stored as {X.dtype}, want torch.float32 (32-bit decimal numbers)"
    X = X.numpy()
    for n, s in enumerate(seqs):
        L = min(len(s), MAX_LEN)
        for i, b in enumerate(s[:L]):
            want = np.eye(4)[ALPHABET.index(b)] if b in ALPHABET else np.full(4, 0.25)
            assert np.allclose(X[n, 0, :, i], want), f"❌ sequence {n!r}, column {i} (letter {b}): {X[n, 0, :, i]}, want {want}"
        assert np.allclose(X[n, 0, :, L:], 0.25), f"❌ sequence {n}: the padding columns after letter {L} should all be 0.25"
    print(f"✅ Nice work! TODO-1 passed: {len(seqs)} sequences encoded correctly (including a letter N, which stays 0.25)")


test_encoding()

# %% [markdown]
# ## 3. The model (TODO-2)
#
# You build the network yourself, from a few kinds of layers, like building blocks:
#
# | part | in PyTorch | what it does | what you choose |
# |---|---|---|---|
# | **convolution** | [`nn.Conv2d`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) | slides each of its filters (small weight matrices) over the input; at every position it multiplies the window by the filter and adds up the products: a high score where the window matches the filter's pattern. Each filter gives one row of output, a **channel** | how many filters, and how wide they are |
# | **ReLU** | [`nn.ReLU`](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html) | sets negative values to 0 | nothing |
# | **max pooling** | [`nn.MaxPool2d`](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html) | keeps the largest value in each block of neighbouring positions, so the output gets shorter | how many positions per block |
# | **flatten** | [`nn.Flatten`](https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html) | lays all remaining numbers out in one row | nothing |
# | **linear** | [`nn.Linear`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) | a weighted sum of all its inputs plus a bias | how many inputs (the length of the flattened row) |
#
# **The template.** One or more blocks of convolution → ReLU → max pooling, then flatten → linear:
#
# ```
# [convolution → ReLU → max pooling]  (repeat as often as you like)
# → flatten → linear → one number per sequence
# ```
#
# **The rules.** The input is one sequence as a one-hot matrix with one channel, `1 × 4 × 501`. The
# first convolution's filters must cover all 4 letter rows, so each filter reads a window of whole
# letters. The linear layer ends with **one** number per sequence, the **logit**: any number,
# large and positive for "bound", large and negative for "unbound" (section 4 turns it into a
# probability). Everything else is your choice: how many blocks, how many filters, how wide.
#
# The cell below builds a small example network for a different input (8 × 8 pictures) from the same
# parts, and `show_shapes` prints what each layer does to the shape. Use `show_shapes` on your own
# network to build it layer by layer: the shape after flatten tells you how many inputs the linear
# layer needs.

# %%
#@title ▶ An example network, and show_shapes { display-mode: "form" }
def show_shapes(layers, x=None):
    """Run one input through `layers` one layer at a time and print the shape after each layer."""
    h = encode_batch(train_seqs[:1]) if x is None else x
    print(f"{'input':38s} {tuple(h.shape[1:])}")
    for i, layer in enumerate(layers):
        try:
            h = layer(h)
        except Exception as e:
            print(f"layer {i}, {layer}: fails on an input of shape {tuple(h.shape[1:])}: {e}")
            return
        print(f"{str(layer)[:38]:38s} {tuple(h.shape[1:])}")


example = nn.Sequential(
    nn.Conv2d(1, 4, (3, 3)), nn.ReLU(), nn.MaxPool2d((2, 2)),   # 4 filters of 3 x 3, then blocks of 2 x 2
    nn.Flatten(),
    nn.Linear(4 * 3 * 3, 1),                                    # 4 channels x 3 x 3 numbers after the pooling
)
print("an example network for 8 x 8 pictures with one channel:")
show_shapes(example, torch.zeros(1, 1, 8, 8))

# %%
class GlobalCNN(nn.Module):
    """One-hot matrices of N sequences, (N, 1, 4, 501) -> one score (logit) per sequence."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # TODO-2: your network, following the template and the rules above.
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


show_shapes(GlobalCNN().net)
print(f"\n{sum(p.numel() for p in GlobalCNN().parameters()):,} trainable parameters")

# %%
#@title ▶ Test TODO-2 { display-mode: "form" }
def test_network():
    model = GlobalCNN()
    layers = list(model.net)
    kinds = [type(l).__name__ for l in layers]
    assert kinds and kinds[0] == "Conv2d", f"❌ the network starts with {kinds[:1]}, want a convolution first"
    first = layers[0]
    assert first.in_channels == 1, \
        f"❌ the first convolution takes {first.in_channels} input channels, want 1: the one-hot matrix has one channel"
    assert first.kernel_size[0] == 4, \
        f"❌ the first convolution's filters are {first.kernel_size[0]} rows tall, want 4: one row per letter"
    for i, k in enumerate(kinds):
        if k == "Conv2d":
            assert kinds[i + 1:i + 2] == ["ReLU"], \
                f"❌ layer {i} is a convolution, and the template puts a ReLU right after it; layer {i + 1} is {kinds[i + 1:i + 2]}"
    assert "MaxPool2d" in kinds, "❌ the network has no max pooling layer (see the template)"
    assert kinds[-2:] == ["Flatten", "Linear"], f"❌ the network ends with {kinds[-2:]}, want flatten, then linear"
    assert layers[-1].out_features == 1, \
        f"❌ the linear layer gives {layers[-1].out_features} numbers, want 1: one logit per sequence"
    X = encode_batch(train_seqs[:5])
    try:
        out = model(X)
    except Exception as e:
        raise AssertionError(f"🤔 the network fails on 5 sequences: {e}. Run show_shapes(GlobalCNN().net) to "
                             "see where; the linear layer needs as many inputs as flatten gives") from None
    assert tuple(out.shape) == (5,), \
        f"❌ the model's output for 5 sequences has shape {tuple(out.shape)}, want (5,): one logit per sequence"
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧬 Your network is built! TODO-2 passed: {kinds.count('Conv2d')} convolution block(s), one logit per sequence, "
          f"{n:,} trainable parameters")


test_network()

# %% [markdown]
# ## 4. Training (TODO-3)
#
# Training minimises a **loss** by gradient descent (here the Adam variant): each step computes the
# gradient of the loss on one batch of 256 sequences and moves every weight a small amount against
# its gradient. Our loss is binary cross-entropy, the negative log of the probability the model
# assigns to the correct label, averaged over the batch. The model outputs a logit $z$; the sigmoid
# $\sigma(z) = 1 / (1 + e^{-z})$ turns it into a probability between 0 and 1.
#
# After every pass over the training data (an *epoch*) we evaluate the model on the held-out test
# set: the 1,000 `ls` sequences it never trains on. During evaluation the model only scores: no
# gradients are computed and the weights do not change. The number we report is **AUROC**: the probability that a randomly chosen bound sequence
# gets a higher score than a randomly chosen unbound one. 0.5 is a coin flip, 1.0 is perfect.
#
# The whole training loop, in pseudocode:
#
# ```
# for each epoch:
#     for each batch (X, y) of the training set:
#         clear the gradients
#         predict: one score (logit) per sequence of X
#         loss: compare the scores with the labels y
#         gradients: the gradient of the loss for every weight
#         step: move every weight against its gradient
#     evaluate on the held-out test set:
#         no gradients
#         probabilities: the sigmoid of the scores
#         AUROC of the probabilities against the true labels
# ```
#
# What each step means, and the function that does it:
#
# | step | what it means | API |
# |---|---|---|
# | clear the gradients | forget the gradients of the previous batch | [`opt.zero_grad`](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html) |
# | predict | the model gives one score (logit) per sequence in the batch | call the model on the batch, `model(X)` |
# | loss | one number for the batch: how far the scores are from the labels | `criterion`, a [`nn.BCEWithLogitsLoss`](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) |
# | gradients | for every weight, how much the loss changes when that weight changes | [`loss.backward`](https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html) |
# | step | the optimizer (`opt`) moves every weight a little in the direction that lowers the loss | [`opt.step`](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.step.html) |
# | evaluate | score the held-out test set without learning: no gradients, the weights stay as they are | [`model.eval`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval), [`torch.no_grad`](https://pytorch.org/docs/stable/generated/torch.no_grad.html) |
# | probabilities | the sigmoid turns each score into a probability between 0 and 1 | [`torch.sigmoid`](https://pytorch.org/docs/stable/generated/torch.sigmoid.html) |
# | AUROC | how well the probabilities separate the bound from the unbound test sequences | [`roc_auc_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) |
#
# You write the two pieces of this loop as functions: one training step on one batch (TODO-3a) and
# the evaluation (TODO-3b). The function `train()` below then repeats them for every batch and epoch.

# %%
#@title ▶ Encode the training set and cut it into batches { display-mode: "form" }
from torch.utils.data import DataLoader, TensorDataset

t0 = time.time()
X_train = encode_batch(train_seqs)
y_train = torch.tensor(train_y, dtype=torch.float32)
print(f"encoded {len(train_seqs):,} sequences in {time.time() - t0:.1f}s")
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True, drop_last=True)
print(f"{len(train_loader):,} batches of 256 sequences per epoch")

# %%
# TODO-3a: write train_step(model, opt, criterion, X, y): one training step on one batch (the five
#          steps of the pseudocode, from clearing the gradients to the optimizer step). It returns the
#          loss of the batch, before the step, as a number.
def train_step(model, opt, criterion, X, y):
    # === Your code here ===
    opt.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    opt.step()
    return loss.item()
    # === End of your code ===


# try it on one batch, with a fresh model
X, y = next(iter(train_loader))          # one batch: 256 one-hot sequences and their 256 labels
X, y = X.to(DEVICE), y.to(DEVICE)
torch.manual_seed(SEED)
model = GlobalCNN().to(DEVICE)           # random weights
opt = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()
for step in range(1, 6):
    print(f"step {step}: loss on this batch {train_step(model, opt, criterion, X, y):.4f}")

# %%
# TODO-3b: write evaluate(model, X_test, test_y): the held-out test AUROC, without learning. The model
#          scores every test sequence, the scores become probabilities of "bound", and the AUROC compares
#          them with the true labels test_y. It returns the AUROC as a number.
def evaluate(model, X_test, test_y):
    # === Your code here ===
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_test.to(DEVICE))).cpu().numpy()
    return roc_auc_score(test_y, probs)
    # === End of your code ===


print(f"held-out test AUROC of the model above, after 5 steps on one batch: {evaluate(model, encode_batch(test_seqs), test_y):.3f}")

# %%
#@title ▶ Test TODO-3 (also defines train()) { display-mode: "form" }
def train(model, train_loader, test_seqs, test_y, epochs=10, lr=1e-3, wd=1e-4):
    """Trains with your train_step. Measures with your evaluate before training, four times during the
    first epoch and after every epoch: on the held-out test set and on 1,000 training sequences."""
    model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.BCEWithLogitsLoss()
    X_test = encode_batch(test_seqs)
    ds = train_loader.dataset
    pick = torch.from_numpy(np.random.default_rng(0).choice(len(ds), min(1000, len(ds)), replace=False))
    X_fit, y_fit = ds.tensors[0][pick], ds.tensors[1][pick].numpy()
    n_batches = len(train_loader)
    checks = {max(1, round(n_batches * q)) for q in (0.25, 0.5, 0.75)}
    curve = {"epoch": [], "train_auc": [], "test_auc": [], "loss": []}
    losses = []

    def measure(at):
        curve["epoch"].append(at)
        curve["train_auc"].append(evaluate(model, X_fit, y_fit))
        curve["test_auc"].append(evaluate(model, X_test, test_y))
        curve["loss"].append(float(np.mean(losses)) if losses else float("nan"))
        losses.clear()
        model.train()

    measure(0)
    print(f"before training | held-out test AUROC = {curve['test_auc'][-1]:.4f}")
    for ep in range(1, epochs + 1):
        t = time.time()
        model.train()
        for i, (X, y) in enumerate(train_loader, 1):
            losses.append(train_step(model, opt, criterion, X.to(DEVICE), y.to(DEVICE)))
            if ep == 1 and i in checks:
                measure(i / n_batches)
        measure(ep)
        print(f"epoch {ep:02d} | training AUROC = {curve['train_auc'][-1]:.4f} | "
              f"held-out test AUROC = {curve['test_auc'][-1]:.4f} | {time.time() - t:.1f}s")
    return curve


def test_train():
    """train_step on one batch, then evaluate during 1 epoch on 5,000 training sequences."""
    global roc_auc_score
    torch.manual_seed(SEED)
    m = GlobalCNN().to(DEVICE)
    o = optim.Adam(m.parameters(), lr=1e-3)
    c = nn.BCEWithLogitsLoss()
    Xb, yb = next(iter(train_loader))
    Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
    for q in m.parameters():
        q.grad = torch.full_like(q, 1e6)     # left-over gradients of an earlier step: they must be cleared
    before = [q.detach().clone() for q in m.parameters()]
    l = train_step(m, o, c, Xb, yb)
    assert isinstance(l, float), f"❌ train_step returned a {type(l).__name__}, want the loss as a plain number"
    assert np.isfinite(l), f"❌ train_step returned the loss {l}, want a finite number"
    assert any(not torch.equal(b0, q.detach()) for b0, q in zip(before, m.parameters())), \
        "❌ the weights did not change in train_step: it must compute the gradients and take the optimizer step"
    assert all(q.grad.abs().max().item() < 1e5 for q in m.parameters()), \
        "❌ the gradients of an earlier step were still there: train_step must clear the gradients first"
    idx = torch.from_numpy(np.random.default_rng(0).choice(len(X_train), 5000, replace=False))
    small_loader = DataLoader(TensorDataset(X_train[idx], y_train[idx]), batch_size=256, shuffle=True, drop_last=True)
    seen, real_roc_auc_score = [], roc_auc_score

    def spy(y_true, y_score):        # keeps what evaluate passes to roc_auc_score
        seen.append(np.asarray(y_score))
        return real_roc_auc_score(y_true, y_score)

    roc_auc_score = spy
    try:
        torch.manual_seed(SEED)
        curve = train(GlobalCNN(), small_loader, test_seqs, test_y, epochs=1)
    finally:
        roc_auc_score = real_roc_auc_score
    assert seen, "❌ evaluate never called roc_auc_score"
    for pr in seen:
        assert pr.shape == (len(test_seqs),), \
            f"❌ evaluate gave scores of shape {pr.shape}, want ({len(test_seqs)},): one score per sequence it was given"
        assert np.isfinite(pr).all(), "❌ the probabilities contain NaN or infinite values"
        assert pr.min() >= 0 and pr.max() <= 1, \
            (f"🤔 the evaluated scores range from {pr.min():.2f} to {pr.max():.2f}, so they are logits. "
             "Turn them into probabilities between 0 and 1 with the sigmoid")
    auc = curve["test_auc"][-1]
    assert 0.5 < auc <= 1, f"❌ the held-out AUROC after 1 epoch on 5,000 sequences is {auc:.3f}, want between 0.5 and 1"
    print(f"🚀 Training works! TODO-3 passed: train_step clears the gradients and moves the weights; evaluate turns "
          f"the scores into probabilities (held-out AUROC {auc:.3f} after 1 epoch on 5,000 sequences)")


test_train()

# %%
#@title ▶ Train, then plot the learning curves { display-mode: "form" }
EPOCHS = 10
torch.manual_seed(SEED)
model = GlobalCNN()
curve = train(model, train_loader, test_seqs, test_y, epochs=EPOCHS)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
ax[0].plot(curve["epoch"], curve["train_auc"], marker="o", ms=4, c=MIDGREY, label="Training sequences (1,000)")
ax[0].plot(curve["epoch"], curve["test_auc"], marker="o", ms=4, c=CNN_C, label="Held-out test sequences (1,000)")
ax[0].axhline(0.5, ls="--", c=MIDGREY, lw=1, label="Coin flip")
ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("AUROC"); ax[0].set_ylim(0.4, 1); ax[0].legend(); ax[0].grid(axis="y")
ax[0].set_title("AUROC during training")
ax[1].plot(curve["epoch"][1:], curve["loss"][1:], marker="o", ms=4, c=CNN_C)
ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Mean training loss"); ax[1].grid(axis="y")
ax[1].set_title("Loss on the training batches")
plt.tight_layout(); plt.show()

# %% [markdown]
# **Reading your curves.** Where do the two AUROC curves start, and how fast do they climb? Does the
# curve on training sequences pull away from the curve on held-out sequences, and if so, from which
# epoch? A growing gap means the model starts to memorise its training data. Is your model good, and
# compared with what? The held-out test set has only 1,000 sequences, so any AUROC on it has a sampling
# uncertainty of about ±0.025 (95% range, part 1 talk): smaller differences, between epochs or between
# your network and your neighbour's, are noise.

# %% [markdown]
# ### The ROC curve (TODO-3c)
#
# The model gives every held-out sequence a probability of "bound". To call a sequence bound you pick a
# **threshold**: everything above it counts as bound. Each threshold gives two numbers: the **true
# positive rate** (the fraction of bound sequences above the threshold) and the **false positive rate**
# (the fraction of unbound sequences above it). The **ROC curve** joins these points for every threshold,
# from the strictest (nothing called bound, bottom left) to the most lenient (everything called bound, top
# right). The **AUROC** is the area under this curve: 0.5 for a coin flip (the diagonal), 1.0 for a model
# that puts every bound sequence above every unbound one.
#
# **Tools:** [`roc_curve`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html),
# [`matplotlib.pyplot.hist`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html),
# and `score_test`, the probabilities of your trained model on the held-out test set (computed below).

# %%
model.eval()
with torch.no_grad():
    score_test = torch.sigmoid(model(encode_batch(test_seqs).to(DEVICE))).cpu().numpy()

# TODO-3c: two plots side by side. Left: the distribution of score_test for the bound and for the unbound
#          test sequences (test_y says which is which). Right: the ROC curve of score_test against test_y,
#          with the diagonal of a coin flip. Keep the false and true positive rates as `fpr` and `tpr`.
# === Your code here ===
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(test_y, score_test)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].hist(score_test[test_y == 0], bins=30, alpha=.7, color=UNBOUND, label="Unbound")
ax[0].hist(score_test[test_y == 1], bins=30, alpha=.7, color=BOUND, label="Bound")
ax[0].set_xlabel("Predicted probability of bound"); ax[0].set_ylabel("Sequences"); ax[0].legend()
ax[1].plot(fpr, tpr, c=CNN_C)
ax[1].plot([0, 1], [0, 1], ls="--", c=MIDGREY)
ax[1].set_xlabel("False positive rate"); ax[1].set_ylabel("True positive rate")
ax[1].set_title(f"ROC curve, AUROC {roc_auc_score(test_y, score_test):.3f}")
plt.tight_layout(); plt.show()
# === End of your code ===

# %%
#@title ▶ Test TODO-3c { display-mode: "form" }
def test_roc():
    f, r = np.asarray(fpr), np.asarray(tpr)
    assert f.shape == r.shape and len(f) > 2, "❌ fpr and tpr should be two lists of the same length, one point per threshold"
    assert f[0] == 0 and r[0] == 0 and np.isclose(f[-1], 1) and np.isclose(r[-1], 1),         "❌ the ROC curve should run from (0, 0) to (1, 1)"
    area = float(np.trapz(r, f)) if hasattr(np, "trapz") else float(np.trapezoid(r, f))
    want = roc_auc_score(test_y, score_test)
    assert abs(area - want) < 1e-6,         f"🤔 the area under your curve is {area:.4f}, but the held-out AUROC is {want:.4f}: use score_test against test_y"
    print(f"🎉 Great plots! TODO-3c passed: the area under your ROC curve equals the held-out AUROC, {want:.3f}")


test_roc()

# %% [markdown]
# # Look inside the CNN (sections 5 to 7)
#
# Your CNN scores well, so it has picked up some signal that separates bound from unbound RNA. This
# half asks what that signal is, and then tests the answer. The cells below use `model`, the CNN
# you trained in section 4.
#
# 1. **Read a motif off the model.** Each first-layer filter scores every window as wide as the filter. We
#    collect the windows where its output (its *activation*) is high and summarise them as a
#    *sequence logo*.
# 2. **Test it.** The logo suggests what a filter responds to. We insert that motif into random RNA
#    and check whether the model's score goes up.

# %% [markdown]
# ## 5. The toolkit (given)
#
# Read the names and descriptions. You will combine these functions in the TODOs below.
#
# | function | what it does |
# |---|---|
# | `filter_activations(model, seqs)` | run only the first convolution layer → one activation per (sequence, filter, position) |
# | `cutoff_for(acts_f)` | one filter's activation threshold: `mean + 0.5 × (max − mean)` |
# | `windows_above(acts_f, seqs, cutoff)` | the windows (as wide as the filter) where that filter's activation exceeds the threshold |
# | `pwm_of(windows)` | the frequency of each letter at each position → a table with one row per position |
# | `consensus_of(pwm)` | the most common letter at each position, as a string |
# | `draw_logo(pwm, title)` | draw the table as a sequence logo |
# | `random_rna(n, rng)` | `n` random RNA sequences of 330 letters to insert motifs into |
# | `plant(seqs, motif, copies)` | insert `copies` copies of `motif` into each sequence (they overwrite letters) |
# | `score_sequences(model, seqs)` | the model's probability of "bound" for each sequence |

# %%
#@title ▶ Motif toolkit { display-mode: "form" }
import logomaker
import matplotlib.pyplot as plt
import pandas as pd

CONV1 = next(l for l in model.net if isinstance(l, nn.Conv2d))
KERNEL = CONV1.kernel_size[1]          # read from the model, so a different width still works


@torch.no_grad()
def filter_activations(model, seqs, batch=512):
    """Run only the first convolution layer and its ReLU. Returns one activation per (sequence, filter, position)."""
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
    """The model's binding probability for each sequence of a list of RNA sequences."""
    model.eval()
    out = []
    for i in range(0, len(seqs), batch):
        X = encode_batch(seqs[i:i + batch])
        out.append(torch.sigmoid(model(X.to(DEVICE))).cpu().numpy())
    return np.concatenate(out)


print(f"toolkit ready (filter width = {KERNEL})")

# %% [markdown]
# ## 6. Read a motif off the model (TODO-4)
#
# For each filter: compute its threshold, collect the windows above it, turn them into a letter
# frequency table, and record its consensus (the most frequent letter at each position). You do this
# first for one filter, filter 0, step by step (TODO-4a), then for every filter in a loop (TODO-4b).
# Keep the tables; the cells after the test draw them.

# %%
#@title ▶ The activations of every filter on bound RNA { display-mode: "form" }
pos_seqs = [s for s, y in zip(test_seqs, test_y) if y == 1] + \
           [s for s, y in zip(train_seqs, train_y) if y == 1][:5000]
acts = filter_activations(model, pos_seqs)
n_filters = acts.shape[1]
print("activations:", acts.shape, "-> (sequences, filters, positions) for", len(pos_seqs), "bound sequences")

# %%
# TODO-4a: filter 0, step by step, with the toolkit of section 5. `acts` holds the activations of
#          every filter on the bound sequences `pos_seqs` (sequences, filters, positions). Produce:
#          acts_0       the activations of filter 0 only (one row per sequence, one column per position)
#          cut_0        filter 0's threshold
#          windows_0    the windows of pos_seqs where filter 0 is above its threshold
#          pwm_0        the letter frequency table of these windows
#          consensus_0  the most frequent letter at each position
# === Your code here ===
acts_0 = acts[:, 0, :]
cut_0 = cutoff_for(acts_0)
windows_0 = windows_above(acts_0, pos_seqs, cut_0)
pwm_0 = pwm_of(windows_0)
consensus_0 = consensus_of(pwm_0)
# === End of your code ===
print("activations of filter 0:", acts_0.shape, f"| threshold {cut_0:.3f} (largest activation {acts_0.max():.3f})")
print(f"{len(windows_0):,} windows above the threshold; the first 5:", windows_0[:5])
print(f"\nletter frequency at each of the {KERNEL} positions (each row sums to 1):")
print(pd.DataFrame(pwm_0, columns=list(ALPHABET), index=range(1, KERNEL + 1)).round(2))
print("\nconsensus:", consensus_0)
draw_logo(pwm_0, f"Filter 00, {len(windows_0):,} windows")
plt.show()

# %%
motifs, pwms, n_hits = {}, {}, {}
for f in range(n_filters):
    # TODO-4b: the steps of TODO-4a for filter f, stored as motifs[f] (the consensus),
    #          pwms[f] (the letter frequency table) and n_hits[f] (the number of windows).
    # === Your code here ===
    acts_f = acts[:, f, :]
    windows = windows_above(acts_f, pos_seqs, cutoff_for(acts_f))
    pwms[f] = pwm_of(windows)
    motifs[f] = consensus_of(pwms[f])
    n_hits[f] = len(windows)
    # === End of your code ===
    print(f"filter {f:02d}: {n_hits[f]:>7,} windows | consensus {motifs[f]}")

# %%
#@title ▶ Test TODO-4 { display-mode: "form" }
def test_tables():
    for name, d in (("motifs", motifs), ("pwms", pwms), ("n_hits", n_hits)):
        assert sorted(d) == list(range(n_filters)), \
            f"❌ {name} has entries for the filters {sorted(d)}, want one entry for every filter 0 to {n_filters - 1}"
    for f in range(n_filters):
        a = acts[:, f, :]
        w = windows_above(a, pos_seqs, cutoff_for(a))
        ref = pwm_of(w)
        assert n_hits[f] == len(w), \
            (f"❌ filter {f}: n_hits[{f}] = {n_hits[f]:,}, want {len(w):,}, the number of windows above the "
             f"threshold of filter {f} itself (its activations are acts[:, {f}, :])")
        P = np.asarray(pwms[f])
        assert P.shape == ref.shape, f"❌ filter {f}: pwms[{f}] has shape {P.shape}, want {ref.shape} (positions x letters)"
        assert np.allclose(P, ref, atol=1e-6), \
            f"❌ filter {f}: pwms[{f}] differs from pwm_of(its windows) by up to {np.abs(P - ref).max():.3g}"
        assert motifs[f] == consensus_of(ref), \
            f"❌ filter {f}: motifs[{f}] = {motifs[f]!r}, want {consensus_of(ref)!r}, the consensus of pwms[{f}]"
    print(f"🔬 Well done! TODO-4 passed: window counts, letter tables and consensus match for all {n_filters} filters")


test_tables()

# %%
#@title ▶ Draw the logos { display-mode: "form" }
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
MY_MOTIF = motifs[MY_FILTER]   # its consensus; or type the motif you read off its logo
print(f"filter {MY_FILTER:02d} | testing the motif {MY_MOTIF}")

# %% [markdown]
# **A first control.** Run the same filter on **unbound** RNA. If it gives the same logo there, it picks
# up a property of RNA in general (for example, how U-rich it is) and tells you little about AGO2. In
# that case choose another filter, set `MY_FILTER` above, and rerun that cell and this one.

# %%
#@title ▶ The same filter on bound and on unbound RNA { display-mode: "form" }
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
# ## 7. Now test it (TODO-5)
#
# A logo shows which windows activate one filter. Whether that pattern changes the model's *output* is a
# separate question, and you answer it by intervening: insert your motif into random RNA sequences
# (1, 2, … 7 copies with `plant`) and see whether the model's score changes. First you insert one copy
# into one sequence and look at it (TODO-5a).
#
# **The problem.** Suppose the score goes up. That alone does not show that the model responds to your
# pattern. Every inserted copy also changes which letters the sequence is made of: a U-rich motif adds
# U letters, for example. A model that only prefers U-rich RNA would give the same result.
#
# **Design a control (TODO-5b).** Choose a sequence to insert instead of your motif, so that comparing
# the two answers one question: does the *pattern* matter, or only the letters it adds? Hint: what could
# you build from exactly the letters of your motif? Then you compute the mean score for each number of
# copies, for the motif and for your control (TODO-5c).

# %%
#@title ▶ The random sequences { display-mode: "form" }
rng = np.random.default_rng(2)          # this cell's own random numbers, so a rerun gives the same result
canvas = random_rna(1000, rng)
baseline = score_sequences(model, canvas).mean()
copies = list(range(1, 8))              # 7 copies fit into a 330-letter sequence
print(f"{len(canvas):,} random sequences of {len(canvas[0])} letters | mean score, nothing inserted: {baseline:.3f}")
print(f"your motif: {MY_MOTIF}")

# %%
# TODO-5a: one copy of MY_MOTIF in one sequence. `one` is the first random sequence of `canvas`;
#          `one_planted` is the same sequence with one copy of MY_MOTIF inserted (toolkit, section 5);
#          `score_before` and `score_after` are the model's scores of the two.
# === Your code here ===
one = canvas[0]
one_planted = plant([one], MY_MOTIF, 1)[0]
score_before = score_sequences(model, [one])[0]
score_after = score_sequences(model, [one_planted])[0]
# === End of your code ===
k = len(MY_MOTIF)
print("letters 21 to 80; the inserted letters (copy 1 starts at letter 41) in lower case:")
print("before:", one[20:80])
print("after :", one_planted[20:40] + one_planted[40:40 + k].lower() + one_planted[40 + k:80])
print(f"score before {score_before:.3f} | after {score_after:.3f}")

# %%
# TODO-5b: your control, `control_motif`: the sequence you insert instead of MY_MOTIF (see "Design a
#          control" above), with the same length. In MY_CONTROL_REASON, say in one sentence why
#          comparing the two answers the question.
# === Your code here ===
control_motif = "".join(np.random.default_rng(2).permutation(list(MY_MOTIF)))
# === End of your code ===
MY_CONTROL_REASON = ""
print(f"motif   : {MY_MOTIF}\ncontrol : {control_motif}\nbecause : {MY_CONTROL_REASON or '(not written yet)'}")

# %%
# TODO-5c: for every number of copies in `copies`, the mean score over all sequences of `canvas` with
#          that many copies of MY_MOTIF inserted (`real_scores`), and the same with your control_motif
#          (`control_scores`): two lists, one number per number of copies.
# === Your code here ===
real_scores = [score_sequences(model, plant(canvas, MY_MOTIF, c)).mean() for c in copies]
control_scores = [score_sequences(model, plant(canvas, control_motif, c)).mean() for c in copies]
# === End of your code ===
print("motif  :", [round(float(v), 3) for v in real_scores])
print("control:", [round(float(v), 3) for v in control_scores])

# %%
#@title ▶ Test TODO-5 { display-mode: "form" }
def test_insertion():
    from collections import Counter
    assert isinstance(control_motif, str) and len(control_motif) == len(MY_MOTIF), \
        f"❌ control_motif should be a sequence of {len(MY_MOTIF)} letters, like your motif"
    assert control_motif != MY_MOTIF, "🤔 control_motif is your motif itself: the comparison would show nothing"
    assert Counter(control_motif) == Counter(MY_MOTIF), \
        (f"🤔 your control has other letters than your motif ({dict(Counter(control_motif))} against "
         f"{dict(Counter(MY_MOTIF))}). Then a difference between the two curves could come from the letters alone, "
         "not from the pattern. Which sequence adds exactly the same letters but breaks the pattern?")
    for name, got, motif in (("real_scores", real_scores, MY_MOTIF), ("control_scores", control_scores, control_motif)):
        assert len(got) == len(copies), f"❌ {name} has {len(got)} entries, want {len(copies)}: one per number of copies"
        want = [score_sequences(model, plant(canvas, motif, c)).mean() for c in copies]
        for c, g, w in zip(copies, got, want):
            n_copies = f"{c} copy" if c == 1 else f"{c} copies"
            g = np.asarray(g)
            assert g.size == 1, \
                f"❌ {name}, {n_copies}: {g.size} numbers, want one, the mean score over the {len(canvas):,} sequences"
            g = float(g)
            assert 0 <= g <= 1, f"❌ {name}, {n_copies}: {g:.3f} is not a probability between 0 and 1"
            assert abs(g - w) < 1e-6, \
                (f"❌ {name}, {n_copies}: {g:.4f}, want {w:.4f}, the mean score over canvas with {n_copies} of "
                 f"{motif} inserted")
    print("🌟 Excellent experiment! TODO-5 passed: your control has the same letters as the motif in another "
          f"order, and both lists have {len(copies)} mean scores")


test_insertion()

# %%
#@title ▶ Plot score vs number of copies { display-mode: "form" }
plt.figure(figsize=(5.5, 3.5))
plt.plot(copies, real_scores, "o-", c=INK, label=f"Motif {MY_MOTIF}")
plt.plot(copies, control_scores, "x--", c=BASELINE_C, label=f"Control {control_motif}")
plt.axhline(baseline, ls=":", c=MUTED, lw=1, label="Random RNA, nothing inserted")
plt.xlabel("Copies inserted"); plt.ylabel("Mean predicted probability"); plt.legend(); plt.grid(axis="y")
plt.title("Score vs number of copies"); plt.tight_layout(); plt.show()

# %% [markdown]
# **Reading your result.** Compare the two curves with each other. Your control can move the
# score too, because it contains the same letters. If the curve jumps at one copy number,
# suspect **position**: `plant()` writes copy 1 at letter 40, copy 2 at 85, copy 3 at 130, copy 4 at
# 175, and so on. The next cell checks this.

# %%
#@title ▶ Effect of one copy at each position { display-mode: "form" }
# One copy of the motif at every position, 400 random sequences each; change in mean score vs no motif.
rng = np.random.default_rng(3)
positions = list(range(0, 321, 8))
scan_canvas = random_rna(400, rng)
base_scan = score_sequences(model, scan_canvas).mean()
real_scan = [score_sequences(model, [c[:x] + MY_MOTIF + c[x + len(MY_MOTIF):] for c in scan_canvas]).mean() - base_scan
             for x in positions]
shuf_scan = [score_sequences(model, [c[:x] + control_motif + c[x + len(control_motif):] for c in scan_canvas]).mean() - base_scan
             for x in positions]
plt.figure(figsize=(7, 3.2))
plt.plot(positions, real_scan, "-", lw=2, c=INK, label=f"Motif {MY_MOTIF}")
plt.plot(positions, shuf_scan, "--", c=BASELINE_C, label="Your control")
plt.axhline(0, c=MUTED, lw=1)
for x in (40, 85, 130, 175, 220, 265, 310):
    plt.axvline(x, c=LIGHTGREY, ls=":", lw=1, label="Where plant() inserts copies" if x == 40 else None)
plt.xlabel("Position of the single inserted copy"); plt.ylabel("Change in mean predicted probability")
plt.legend(); plt.title("Effect of one copy by position")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### What to take away
#
# * The logo suggests a pattern. Inserting the motif, next to a control with the same letters in
#   another order, tests whether the model's output responds to the pattern.
# * A strongly activated filter can still lower the score. Its effect on the output passes
#   through every layer above it, so your curve can go down.
# * If the logo looks the same on bound and unbound RNA, that filter picks up a property of RNA in
#   general (for example, how U-rich it is) and tells you little about AGO2.
# * The CNN's last layer has one weight per (filter, **position**). Almost every sequence in this data
#   set, bound or unbound, was cut so that the measured ~40-letter site sits at letters 150–190, with
#   context on both sides (the sequences are 137 to 375 letters long), and the CNN learned to weight
#   that region. A control with the same letters gets much of the same increase there, so part of the effect
#   comes from which letters occupy positions 150–190, whatever their order. A motif's effect on the score
#   depends on where it is placed.
#
# **Discuss with your neighbour:** which control did you choose, and why? Did your motif score higher than it? Did theirs?
