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


def load_cnn(ours="cnn.pt"):
    """Our trained CNN from part 1."""
    model = GlobalCNN().to(DEVICE)
    model.load_state_dict(torch.load(fetch(ours), map_location=DEVICE))
    return model.eval()
