#@title ▶ Foundation-model helpers { display-mode: "form" }
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import transformers
transformers.logging.set_verbosity_error()

FM_REPO = "RaphaelMourad/Mistral-DNA-v1-17M-hg38"            # Mistral-DNA (16M parameters), pretrained on the human genome
NT_REPO = "InstaDeepAI/nucleotide-transformer-500m-human-ref"  # NT-500M (486M parameters)


def to_dna(seq):
    """The DNA language models expect T, not U."""
    return seq.replace("U", "T")


def workshop_data(n_train=12000, seed=0):
    """Train: a fixed random 12k subset of the training file (the same in parts 2 and 3).
    Test: the held-out test set ('ls'), the same 1,000 sequences the part 1 CNN was measured on."""
    seqs, y = read_split("train")
    tr = np.random.default_rng(seed).permutation(len(seqs))[:n_train]
    test_seqs, test_y = read_split("ls")
    return [seqs[i] for i in tr], y[tr], test_seqs, test_y


def load_fm(repo=FM_REPO):
    """Download (first time only) and load a DNA language model + its tokenizer."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    model = AutoModel.from_pretrained(repo, trust_remote_code=True).to(DEVICE).eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    return tok, model
