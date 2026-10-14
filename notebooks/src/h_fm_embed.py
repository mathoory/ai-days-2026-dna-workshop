#@title Embedding helper (written in part 2) - run this cell, nothing to change { display-mode: "form" }
@torch.no_grad()
def embed(tok, model, seqs, batch=128):
    """List of DNA strings -> (N, hidden) numpy array: the mean of the last layer over all
    non-padding tokens ([CLS] and [SEP] included)."""
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
