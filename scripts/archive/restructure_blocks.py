"""Rebuild the three block decks in notebook order: talk, one hands-on period, debrief.

Reuses every existing slide (by title) from the assembled block decks and adds the slides the new
structure needs. Block 2 follows tokenization -> embedding -> transformer -> prediction head.
    python scripts/restructure_blocks.py
"""
import re
from pathlib import Path

S = Path(__file__).resolve().parents[1] / "slides"
NAMES = ["01_block1_build_and_look_inside.qmd", "02_block2_pretrained_models.qmd", "03_block3_shortcuts_and_controls.qmd"]


def units(path):
    text = (S / path).read_text(encoding="utf-8")
    fm, body = text.split("---", 2)[1:]
    slides = {}
    for m in re.finditer(r"^## \d+\.\d+ (.+?)\n(.*?)(?=^## |^::: \{\.center\}|\Z)", body, re.S | re.M):
        title = re.sub(r"\s*\{.*\}\s*$", "", m.group(1)).strip()
        slides[title] = m.group(2).rstrip() + "\n"
    return "---" + fm + "---\n", slides


def deck(front, plan):
    out, sec = [front], 0
    for section, data_name, items in plan:
        sec += 1
        out.append(f'\n::: {{.center}}\n# {sec} {section} {{data-name="{data_name}"}}\n:::\n')
        for k, (title, content) in enumerate(items, 1):
            out.append(f"\n## {sec}.{k} {title}\n{content}")
    return "".join(out)


def main():
    f1, u1 = units(NAMES[0]); f2, u2 = units(NAMES[1]); f3, u3 = units(NAMES[2])
    used = {1: set(), 2: set(), 3: set()}

    def take(u, n, title, new_title=None):
        used[n].add(title)
        return (new_title or title, u[title])

    # ------------------------------------------------------------------ block 1
    insertion_method = '''
- Take the consensus of one filter, insert it 1 to 7 times into 1,000 random RNA sequences, and score them
- **Control**: the same letters in shuffled order, inserted at the same places
- Compare the two curves; the shuffled copy has the same letters, so the difference is what the order adds

::: {.notes}
- Projected: 1 min.
- `plant(seqs, motif, copies)` writes copy n starting at letter 40 + 45(n − 1); 7 copies fit in a 330-letter sequence.
- This turns a logo, which is only a description of what a filter responds to, into an experiment on the whole model's output.
:::
'''
    your_turn_1 = '''
Open **`01_build_and_look_inside.ipynb`** in Colab, select the T4 GPU, run from the top.

| Step | Task | Check | Time |
|:--|:--|:--|:--|
| TODO-1 | one-hot encode a sequence | first 8 columns match the letters | 3 min |
| TODO-2 | build the network from the table | the sanity check prints `torch.Size([2])` | 5 min |
| TODO-3 | held-out AUROC after each epoch | about 0.79 after epoch 1, 0.86 after 10 | 5 min |
| TODO-4 | a logo for every filter (`acts[:, f, :]`, `cutoff_for`, `windows_above`, `pwm_of`) | ten logos | 5 min |
| step | choose `MY_FILTER` and `MY_MOTIF` | | 1 min |
| TODO-5 | insert the motif 1 to 7 times, against a shuffled copy (`plant`) | two curves | 5 min |
| given | position scan, microRNA seed sites | | 2 min |

::: {.notes}
- Projected hands-on total: about 26 min plus Colab start-up. Each TODO has a "Stuck?" cell below it that loads our version.
- Training takes about 4 seconds per epoch on a T4.
- The last cell downloads `my_cnn.pt`, which block 3 can load.
:::
'''
    b1 = [
        ("The task", "Task", [take(u1, 1, "Goals"), take(u1, 1, "What you need"), take(u1, 1, "Today"),
                              take(u1, 1, "The seed rule"), take(u1, 1, "The data"),
                              take(u1, 1, "Seed matches are enriched in bound sites")]),
        ("Learning and testing", "Learning", [take(u1, 1, "What training does"), take(u1, 1, "Train, validate, test"),
                                              take(u1, 1, "AUROC"), take(u1, 1, "Noise in a held-out AUROC"),
                                              take(u1, 1, "Fit vs held-out")]),
        ("The CNN", "CNN", [take(u1, 1, "One-hot encoding", "One-hot encoding (TODO-1)"),
                            take(u1, 1, "A filter computes a dot product"), take(u1, 1, "The filter along one sequence"),
                            take(u1, 1, "The network (TODO-2)"), take(u1, 1, "Training changes the filters"),
                            take(u1, 1, "The training step and the evaluation (TODO-3)")]),
        ("Look inside", "Look inside", [take(u1, 1, "From a filter to a sequence logo", "From a filter to a sequence logo (TODO-4)"),
                                        ("Test a logo: insert it, against a shuffled copy (TODO-5)", insertion_method),
                                        ("Your turn", your_turn_1)]),
        ("What the CNN learned", "Results", [take(u1, 1, "Training curve"), take(u1, 1, "The last layer weighs positions"),
                                             take(u1, 1, "All ten filters"), take(u1, 1, "The same filter on unbound sequences"),
                                             take(u1, 1, "Insert the motif (TODO-2)", "Insert the motif"),
                                             take(u1, 1, "Position matters"), take(u1, 1, "Does the CNN respond to seed sites?")]),
    ]
    used[1] |= {"Your turn"}

    # ------------------------------------------------------------------ block 2
    stages = '''
| Stage | The CNN (block 1) | A DNA language model |
|:--|:--|:--|
| **Tokenization** | single letters | k-mers or learned BPE tokens |
| **Embedding** | a one-hot column of 4 numbers | a learned vector per token (256 numbers in Mistral-DNA) |
| **Body** | two convolution layers | a transformer (8 layers in Mistral-DNA) |
| **Prediction head** | a linear layer → one logit | pretraining: a probability for every token; our task: a linear layer → one logit |

::: {.notes}
- Projected: 2 min.
- The same four stages describe every sequence model today. This block goes through them in order; the transformer stays a black box.
- The CNN's stages were fixed by us (one-hot) or trained on 92,000 labels. A language model learns its embedding and body from unlabelled genome sequence first.
:::
'''
    embedding = '''
- Each token in the vocabulary has a row in a learned table $E$ (4,096 tokens × 256 numbers in Mistral-DNA)
- Token number $i$ becomes the vector $e_i$, row $i$ of $E$. With a one-hot vector $x_i$: $\;e_i = x_i^\\top E$
- One-hot encoding is the special case where the table is fixed and the vectors have one number per letter

::: {.notes}
- Projected: 2 min.
- The table is learned during pretraining, so tokens that occur in similar contexts end up with similar vectors.
- Multiplying a one-hot vector by $E$ just selects a row: an embedding layer is a lookup.
- These are the input vectors. The next slides follow them through the transformer and back out as one vector per sequence.
:::
'''
    transformer = '''
- **In**: one vector per token. **Out**: one vector per token, the same size
- Each output vector depends on all the tokens in the sequence (attention), not only on its own token
- Mistral-DNA: 8 layers, 256 numbers per token, 8 attention heads, up to 512 tokens, 16M parameters
- Today it is a black box: a function from token vectors to context-dependent token vectors

::: {.notes}
- Projected: 2 min.
- Inside each layer, attention mixes information between positions and a small feed-forward network transforms each position. Mistral-DNA's feed-forward part is a mixture of 8 experts.
- For the exercise we need only its last layer's output, shown two slides back as a 26 × 256 matrix.
- The CNN's convolutions see a 10-letter window and then a few pooled neighbours; a transformer layer can relate any two tokens in the input.
:::
'''
    heads = '''
| Use | Head on top of the transformer | Output |
|:--|:--|:--|
| Pretraining, next token | linear layer 256 → 4,096, softmax | a probability for every possible next token |
| Pretraining, masked tokens | linear layer → vocabulary, softmax | a probability for every possible hidden token |
| Our task | linear layer 256 → 1, sigmoid | P(bound), as the CNN's last layer |

- **Linear probe**: the task head trained on the frozen model's averaged embedding, which is logistic regression

::: {.notes}
- Projected: 2 min.
- After pretraining the vocabulary head is discarded and a new task head is attached. The body (and the embedding table) is what transfers.
- Frozen body, new linear head: a linear probe (this block's notebook). Body adapted too: fine-tuning (block 3, LoRA).
:::
'''
    hf_code = u2["From a sequence to an embedding with Hugging Face"]
    b2 = [
        ("From sequence to prediction", "Overview", [("Four stages, in the CNN and in a language model", stages),
                                                      take(u2, 2, "Pretrain once, reuse many times")]),
        ("Tokenization", "Tokens", [take(u2, 2, "Each model cuts DNA into different tokens", "Three ways to cut a sequence into tokens")]),
        ("Embedding", "Embedding", [("Token embeddings", embedding),
                                    take(u2, 2, "From tokens to an embedding", "From token vectors to one sequence embedding")]),
        ("The transformer", "Transformer", [("The transformer, as a black box", transformer),
                                            take(u2, 2, "Two pretraining objectives"), take(u2, 2, "The landscape, 2021 to 2025")]),
        ("Prediction heads", "Heads", [("Prediction heads", heads), take(u2, 2, "Four ways to adapt a pretrained model"),
                                       take(u2, 2, "LoRA")]),
        ("Hands-on", "Hands-on", [take(u2, 2, "From a sequence to an embedding with Hugging Face"), take(u2, 2, "Your turn")]),
        ("What the probe found", "Results", [take(u2, 2, "What the embeddings vary with"),
                                             take(u2, 2, "Frozen pretrained models against the CNN"),
                                             take(u2, 2, "Pretraining from random weights")]),
    ]

    # ------------------------------------------------------------------ block 3
    your_turn_3 = '''
Open **`03_shortcuts_and_controls.ipynb`**. First cell: `USE_MY_FILES = True` to upload your CNN and vectors, or use ours.

| Step | Task | Check | Time |
|:--|:--|:--|:--|
| TODO-1 | AUROC from G+C and U (`gc_u`, `probe_auc`) | about 0.66 | 4 min |
| TODO-2 | AUROC from 5-letter word counts (`kmer_counts(seqs, k=5)`) | about 0.74 | 4 min |
| TODO-3 | shuffle test, CNN and probe (`dinuc_shuffle`; reuse the fitted probe) | about 0.67 and 0.77 | 6 min |
| TODO-4 | `shortcut_scores()` on both test sets | G+C/U 0.64 → 0.51, words 0.76 → 0.73, probe 0.81 → 0.77 | 6 min |
| given | retrain the CNN on both test sets | about 0.84 on both | 2 min |
| TODO-5 | the LoRA training step (`batch_logits`, `crit`, `opt`, `scaler`) | about 0.80 | 2 min + 6–8 min training |
| given | how many labels does each model need? (precomputed) | | 1 min |

::: {.notes}
- Projected hands-on total: about 31 min plus start-up, including LoRA training. Each TODO has a "Stuck?" cell.
- TODO-3: score the shuffled sequences with the probe already fitted on real sequences.
- TODO-5: start the LoRA cell as soon as it is written and read on while it trains.
:::
'''
    b3 = [
        ("Simpler explanations", "Shortcuts", [take(u3, 3, "The two classes differ in composition"),
                                               take(u3, 3, "The dinucleotide shuffle test")]),
        ("A control", "Control", [take(u3, 3, "Composition-matched data")]),
        ("Hands-on", "Hands-on", [("Your turn", your_turn_3)]),
        ("What broke, what survived", "Results", [take(u3, 3, "Scores before and after shuffling"),
                                                   take(u3, 3, "Shuffling lowers the CNN, barely moves the probe"),
                                                   take(u3, 3, "Discussion"), take(u3, 3, "Before and after matching"),
                                                   take(u3, 3, "Fine-tuning against the CNN"),
                                                   take(u3, 3, "How many labels does each model need?")]),
        ("Wrap-up", "Wrap-up", [take(u3, 3, "The day in eight results"), take(u3, 3, "Ten pretrained models on matched data"),
                                take(u3, 3, "Five points to take home"), take(u3, 3, "When to use a pretrained model"),
                                take(u3, 3, "From a question to a model"), take(u3, 3, "Feedback"),
                                take(u3, 3, "Credits and further reading")]),
    ]
    used[3] |= {"Your turn"}

    for n, (u, front, plan, name) in enumerate(((u1, f1, b1, NAMES[0]), (u2, f2, b2, NAMES[1]), (u3, f3, b3, NAMES[2])), 1):
        missing = set(u) - used[n]
        if missing:
            print(name, "NOT PLACED:", missing)
        (S / name).write_text(deck(front, plan), encoding="utf-8")
        print(name, sum(len(p[2]) for p in plan), "slides")


if __name__ == "__main__":
    main()
