"""Spike step 2: Task 1 CNN - time per epoch, AUROC vs epochs (3 seeds), motifs, planting, controls."""
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from common import OUT, core_of, dinuc_shuffle, encode_batch, load_split, norm, save_json, timer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SEEDS = [0, 1, 2]
KERNEL = 10


class GlobalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 10, (4, KERNEL)), nn.ReLU(), nn.MaxPool2d((1, 3)),
            nn.Conv2d(10, 20, (1, 5)), nn.ReLU(), nn.MaxPool2d((1, 3)),
            nn.Flatten(), nn.Linear(20 * 53, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


T, R = {}, {"device": str(DEVICE), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
with timer(T, "load_and_encode"):
    tr_seqs, tr_y = load_split("train")
    ls_seqs, ls_y = load_split("ls")
    Xtr, Xls = encode_batch(tr_seqs), encode_batch(ls_seqs)
ytr = torch.tensor(tr_y, dtype=torch.float32)


@torch.no_grad()
def score(model, X, bs=1024):
    model.eval()
    return np.concatenate([torch.sigmoid(model(X[i:i + bs].to(DEVICE))).cpu().numpy()
                           for i in range(0, len(X), bs)])


curves, best = {}, None
for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GlobalCNN().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=256, shuffle=True, drop_last=True)
    aucs, ep_times = [], []
    for ep in range(EPOCHS):
        t = time.time()
        model.train()
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            crit(model(X), y).backward()
            opt.step()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        ep_times.append(time.time() - t)
        aucs.append(roc_auc_score(ls_y, score(model, Xls)))
        print(f"seed {seed} ep {ep + 1:02d} auc {aucs[-1]:.4f} ({ep_times[-1]:.1f}s)", flush=True)
    curves[seed] = {"auc": aucs, "sec_per_epoch": float(np.mean(ep_times))}
    if seed == 0:
        best = model
        torch.save(model.state_dict(), OUT / "cnn_seed0.pt")
R["curves"] = curves

# ---- motif readout (fixed helpers) ----
model = best
conv1 = model.net[0]
pos_seqs = [norm(s) for s, l in zip(ls_seqs, ls_y) if l == 1] + \
           [norm(s) for s, l in zip(tr_seqs, tr_y) if l == 1][:5000]
neg_seqs = [norm(s) for s, l in zip(tr_seqs, tr_y) if l == 0][:5500]


@torch.no_grad()
def acts_of(seqs, bs=512):
    out = []
    for i in range(0, len(seqs), bs):
        out.append(torch.relu(conv1(encode_batch(seqs[i:i + bs]).to(DEVICE)))[:, :, 0, :].cpu())
    return torch.cat(out).numpy()


ALPH = "ACGU"


def readout(acts_f, seqs):
    cut = acts_f.mean() + 0.5 * (acts_f.max() - acts_f.mean())
    rows, cols = np.nonzero(acts_f > cut)
    w = [seqs[i][j:j + KERNEL] for i, j in zip(rows, cols)]
    w = [x for x in w if len(x) == KERNEL and set(x) <= set(ALPH)]
    if not w:
        return 0, ""
    arr = np.frombuffer("".join(w).encode(), dtype=np.uint8).reshape(-1, KERNEL)
    pwm = np.stack([(arr == ord(b)).sum(0) for b in ALPH], 1)
    return len(w), "".join(ALPH[i] for i in pwm.argmax(1))


Ap, An = acts_of(pos_seqs), acts_of(neg_seqs)
Wlast = model.net[-1].weight  # not directly interpretable per filter; report mean act diff instead
filters = []
for f in range(Ap.shape[1]):
    n_p, cons_p = readout(Ap[:, f], pos_seqs)
    n_n, cons_n = readout(An[:, f], neg_seqs)
    filters.append({"filter": f, "consensus_pos": cons_p, "n_pos": n_p, "consensus_neg": cons_n, "n_neg": n_n,
                    "mean_max_act_pos": float(Ap[:, f].max(1).mean()), "mean_max_act_neg": float(An[:, f].max(1).mean())})
    print(filters[-1])
R["filters"] = filters

# ---- planting: miRNA sites vs shuffled controls, into random RNA and into real negatives ----
rng = np.random.default_rng(0)
SITES = {"miR-19": "UUUGCAC", "miR-15/16": "UGCUGCU", "miR-30": "UGUUUAC", "let-7": "CUACCUC",
         "polyA-signal": "AAUAAA", "U-run": "UUUUUUU", "GC-run": "GCGCGCG"}


def random_rna(n, L=330):
    return ["".join(ALPH[i] for i in row) for row in rng.integers(0, 4, (n, L))]


def plant(seqs, motif, copies, spacing=45, start=40):
    out = []
    for s in seqs:
        s = list(s)
        for c in range(copies):
            a = start + c * spacing
            s[a:a + len(motif)] = list(motif)
        out.append("".join(s))
    return out


canv = {"random": random_rna(1000), "negatives": [norm(s) for s, l in zip(ls_seqs, ls_y) if l == 0]}
plant_res = {}
for cname, cs in canv.items():
    base = float(score(model, encode_batch(cs)).mean())
    plant_res[cname] = {"baseline": base}
    for name, site in SITES.items():
        shuf = "".join(rng.permutation(list(site)))
        row = {}
        for c in (1, 3, 6):
            row[c] = {"site": float(score(model, encode_batch(plant(cs, site, c))).mean()),
                      "shuffled": float(score(model, encode_batch(plant(cs, shuf, c))).mean())}
        plant_res[cname][name] = row
        print(cname, name, row, flush=True)
R["planting"] = plant_res

# ---- dinucleotide-shuffle control: shuffle positives, keep composition ----
pos_ls = [s for s, l in zip(ls_seqs, ls_y) if l == 1]
neg_ls = [s for s, l in zip(ls_seqs, ls_y) if l == 0]
with timer(T, "dinuc_shuffle"):
    shuf_pos = [dinuc_shuffle(s, rng) for s in pos_ls]
sp = score(model, encode_batch(shuf_pos))
sn = score(model, encode_batch(neg_ls))
R["shuffle_control"] = {
    "auc_real": roc_auc_score(ls_y, score(model, Xls)),
    "auc_shuffled_pos_vs_neg": roc_auc_score([1] * len(sp) + [0] * len(sn), np.concatenate([sp, sn])),
}
print(R["shuffle_control"])
R["timings"] = T
save_json(R, "s2_cnn.json")
