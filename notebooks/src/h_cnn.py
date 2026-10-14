#@title ▶ The CNN from part 1 { display-mode: "form" }
class GlobalCNN(nn.Module):
    """The part 1 network: one-hot (N,1,4,501) -> one binding score (logit) per sequence."""

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


@torch.no_grad()
def score_sequences(model, seqs, batch=1024):
    """The model's binding probability for each sequence (list of RNA strings)."""
    model.eval()
    out = []
    for i in range(0, len(seqs), batch):
        X = encode_batch(seqs[i:i + batch])
        out.append(torch.sigmoid(model(X.to(DEVICE))).cpu().numpy())
    return np.concatenate(out)


def load_cnn(mine="my_cnn.pt", ours="cnn.pt"):
    """Your CNN from part 1 if you uploaded it and it loads; otherwise ours."""
    model = GlobalCNN().to(DEVICE)
    if (DATA_DIR / mine).exists():
        try:
            model.load_state_dict(torch.load(DATA_DIR / mine, map_location=DEVICE))
            print("loaded YOUR CNN from part 1")
            return model.eval()
        except Exception as e:
            print(f"your {mine} doesn't match the part 1 architecture ({type(e).__name__}) - using ours")
    model.load_state_dict(torch.load(fetch(ours), map_location=DEVICE))
    print("loaded the reference CNN")
    return model.eval()
