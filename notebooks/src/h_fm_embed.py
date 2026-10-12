#@title Embedding + probe helpers (written in Slot 5) - run this cell, nothing to change { display-mode: "form" }
@torch.no_grad()
def embed(tok, model, seqs, batch=128):
    """List of DNA strings -> (N, hidden) numpy array: the mean of the last layer over real tokens."""
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


def probe_auc(E_train, y_train, E_test, y_test, C=0.1):
    """Fit a linear probe (logistic regression) on the training vectors; AUROC on the test vectors."""
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=C))
    probe.fit(E_train, y_train)
    return roc_auc_score(y_test, probe.predict_proba(E_test)[:, 1]), probe
