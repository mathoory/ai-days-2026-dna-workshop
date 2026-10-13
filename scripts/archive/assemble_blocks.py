"""Assemble the three block decks from the per-slot decks (kept in slides/archive/), keeping every slide.

Sections and slides are renumbered; slot references become block references; presenter markers go
(Mattan presents all three blocks). Block 1 gets a new opening section written below.
    python scripts/assemble_blocks.py
"""
import re
from pathlib import Path

S = Path(__file__).resolve().parents[1] / "slides"
A = S / "archive"

SLOT_TO_BLOCK = {"1": "block 1", "2": "block 1", "3": "block 1", "4": "block 2", "5": "block 2",
                 "6": "block 3", "7": "block 3", "8": "the wrap-up"}


def body(name):
    t = (A / name).read_text(encoding="utf-8")
    return t.split("---", 2)[2].strip("\n")


def sections(text):
    """Split a deck body into sections, each starting with a ::: {.center} divider."""
    parts = re.split(r"(?=^::: \{\.center\}\n# )", text, flags=re.M)
    return [p.strip("\n") for p in parts if p.strip()]


def renumber(secs):
    out = []
    for n, sec in enumerate(secs, 1):
        sec = re.sub(r"^# \d+ ", f"# {n} ", sec, count=1, flags=re.M)
        k = 0

        def slide(m):
            nonlocal k
            k += 1
            return f"## {n}.{k} "
        sec = re.sub(r"^## \d+\.\d+ ", slide, sec, flags=re.M)
        out.append(sec)
    return "\n\n".join(out)


def clean(text):
    text = re.sub(r"Slot (\d)", lambda m: SLOT_TO_BLOCK[m.group(1)], text)
    text = re.sub(r"Slots (\d) to (\d)", lambda m: f"{SLOT_TO_BLOCK[m.group(1)]} to {SLOT_TO_BLOCK[m.group(2)].replace('block ', '')}", text)
    text = re.sub(r"^Presenter: \w+\.( Drafted by Mattan\.?)?\n\n?", "", text, flags=re.M)
    text = re.sub(r"^Hand over to Roee\.\n?", "", text, flags=re.M)
    text = text.replace("block 1, block 1", "block 1")
    return text


def front(title_line):
    return ('---\nsubtitle: "' + title_line + '<br><br>AI Days 2026<br><br> Faculty of Biotechnology and Food '
            'Engineering<br> Technion Israel Institute of Technology <br><br> Mattan Hoory"\n---\n\n')


OPENING = r'''::: {.center}
# 1 The task {data-name="Task"}
:::

## 1.1 Goals

::: columns
::: {.column width="50%"}
**We will**

- see what happens inside a model that reads sequence: encoding, filters, embeddings, fine-tuning
- read a result: what a number means and when a difference is real
- test whether a model uses the biology or a shortcut
:::
::: {.column width="50%"}
**We will not**

- write models from scratch
- cover PyTorch or Hugging Face in depth
- open up transformer internals
:::
:::

::: {.notes}
- Projected: 1 min.
- Every block ends with a result to interpret.
- Most code is given. The TODOs are short and each one says what to write and what number to expect. An AI assistant can help with a TODO; check its answer against that number.
- Sit next to someone with a different background: a programmer next to a biologist.
:::

## 1.2 What you need

::: columns
::: {.column width="58%"}
- A laptop and a Google account
- Colab with a GPU: *Runtime → Change runtime type → T4 GPU*
- The notebooks: **[github.com/mathoory/ai-days-2026-dna-workshop]{.nowrap}**
- Nothing to install; the data download by themselves
- Stuck? Raise a hand, or load our files and continue
:::
::: {.column width="42%"}
![](assets/custom/qr_repo.png){width="80%"}
:::
:::

::: {.notes}
- Projected: 1 min.
- Every notebook runs on its own. At the end of a notebook participants can download their results and upload them in the next; otherwise the notebook loads ours.
:::

## 1.3 Today

| Time | Block | Notebook |
|:--|:--|:--|
| 11:00 | 1. Build a CNN and look inside it | `01_build_and_look_inside` |
| 11:50 | Break | |
| 12:00 | 2. Pretrained DNA models | `02_pretrained_model` |
| 12:50 | Break | |
| 13:00 | 3. Shortcuts and controls | `03_shortcuts_and_controls` |
| 13:50 | Wrap-up and feedback | |

::: {.notes}
- Projected: 1 min.
- Each block: a talk, then the notebook, then we read the results together.
- One data set all day: about 92,000 RNA sequences where AGO2 did or did not bind.
:::

## 1.4 The seed rule

- AGO2 carries a microRNA. Its **seed** (letters 2 to 7) pairs with the target mRNA; this site also pairs letter 8

![](assets/custom/seed_pairing.png){fig-align="center" height="480px"}

::: {.notes}
- Projected: 2 min (a recap if the opening lecture covered it).
- The top row is a real bound sequence from our data. It contains the miR-16 site UGCUGCU, which pairs with microRNA letters 2 to 8.
- Where AGO2 binds, the mRNA is degraded or not translated, so less protein is made.
- Hypothesis for the day: a model that learned AGO2 targeting should respond to seed sites like this one. We test it at the end of this block.
:::

## 1.5 The data

- **Bound**: about 48,000 sites where CLIP-seq found AGO2. **Unbound**: about 44,000 random positions in the same genes
- Each sequence is about 330 letters, with the 40-letter measured site at letters 150 to 190

![](assets/custom/labels_raster.png){fig-align="center" height="500px"}

::: {.notes}
- Projected: 2 min.
- CLIP-seq: UV light crosslinks proteins to the RNA they touch, an antibody pulls down AGO2, and the attached RNA fragments are sequenced.
- The data set is the GraphProt CLIP benchmark (Maticzka et al. 2014), used by Pan and Shen (2018), whose CNN we follow. The unbound sequences are the bound sites moved to random positions in the same genes.
- 99% of sequences have the measured site starting exactly at letter 150. Sequences are 137 to 375 letters long.
- Each row is one sequence, dark for G or C, light for A or U; 150 of each class, sorted by G+C. The white stripes on the right are sequence ends.
- The bound block is slightly richer in A and U: mean G+C 0.41 vs 0.45 over all sequences. AGO2 sites are often in AU-rich 3′ UTRs, so part of this may be biology and part may come from how the unbound positions were drawn. Block 3 tests that.
:::

## 1.6 Seed matches are enriched in bound sites

![](assets/custom/kmer_enrichment.png){fig-align="center" height="620px"}

::: {.notes}
- Projected: 2 min.
- Counted inside the 40-letter measured sites: for every 7-letter word, the fraction of bound sites that contain it divided by the fraction of unbound sites. The 16 most enriched words.
- The top word, UUUGCAC, is the seed match of miR-19 (6 times enriched). miR-30 (UGUUUAC) and miR-15/16 (UGCUGCU, and GCUGCUA, which pairs letters 2 to 7) are also in the list.
- Even the strongest word occurs in only 1 to 2% of bound sites. The signal is real but sparse, which is why a model has to combine many weak cues.
- A model that learned AGO2 targeting should respond to these words. Block 1 ends by testing whether our CNN does.
:::'''


def main():
    # ---- block 1: opening + deck 01 sections 3-4 + deck 02 + deck 03
    d01 = sections(body("01_opening_and_basics.qmd"))
    keep01 = [s for s in d01 if re.search(r"^# \d+ (Learning and testing|Reading a result)", s, re.M)]
    b1 = [s for s in sections(OPENING)] + keep01 + sections(body("02_build_cnn.qmd")) + sections(body("03_look_inside.qmd"))
    t1 = front("Block 1: Build a CNN and look inside it") + clean(renumber(b1)) + "\n"
    t1 = t1.replace("Open **`02_build_cnn.ipynb`** in Colab, select the T4 GPU, run from the top.",
                    "Open **`01_build_and_look_inside.ipynb`** in Colab, select the T4 GPU, run from the top.")
    t1 = t1.replace("Open **`03_look_inside.ipynb`**. First cell: `USE_MY_CNN = True` to upload your `my_cnn.pt`, or `False` to use ours.",
                    "Continue in **`01_build_and_look_inside.ipynb`**: the second half uses the CNN you just trained.")
    t1 = t1.replace("The last cell downloads `my_cnn.pt`, which block 1 can load.",
                    "The last cell of the notebook downloads `my_cnn.pt`, which block 3 can load.")
    (S / "01_block1_build_and_look_inside.qmd").write_text(t1, encoding="utf-8")

    # ---- block 2: deck 04 + deck 05
    b2 = sections(body("04_foundation_models.qmd")) + sections(body("05_pretrained_model.qmd"))
    t2 = front("Block 2: Pretrained DNA models") + clean(renumber(b2)) + "\n"
    t2 = t2.replace("Open **`05_pretrained_model.ipynb`**.", "Open **`02_pretrained_model.ipynb`**.")
    (S / "02_block2_pretrained_models.qmd").write_text(t2, encoding="utf-8")

    # ---- block 3: deck 06 + deck 07 + deck 08 (wrap-up)
    b3 = sections(body("06_break_it.qmd")) + sections(body("07_fix_it.qmd")) + sections(body("08_question_to_model.qmd"))
    t3 = front("Block 3: Shortcuts and controls, and wrap-up") + clean(renumber(b3)) + "\n"
    t3 = t3.replace("Open **`06_break_it.ipynb`**. First cell: `USE_MY_FILES = True` to upload your CNN and vectors, or use ours.",
                    "Open **`03_shortcuts_and_controls.ipynb`**. First cell: `USE_MY_FILES = True` to upload your CNN and vectors, or use ours.")
    t3 = t3.replace("Open **`07_fix_it.ipynb`**. LoRA was introduced in block 2.",
                    "Continue in **`03_shortcuts_and_controls.ipynb`**. LoRA was introduced in block 2.")
    (S / "03_block3_shortcuts_and_controls.qmd").write_text(t3, encoding="utf-8")
    for p in ("01_block1_build_and_look_inside.qmd", "02_block2_pretrained_models.qmd", "03_block3_shortcuts_and_controls.qmd"):
        t = (S / p).read_text(encoding="utf-8")
        print(p, len(re.findall(r"^## ", t, re.M)), "slides,", len(re.findall(r"^# \d", t, re.M)), "sections")


if __name__ == "__main__":
    main()
