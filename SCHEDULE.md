# AI Days 2026: Applications of AI in Biotechnology and Genomics (Thu 15.10.2026)

| Time | Who | Part | Deck | Notebook |
|---|---|---|---|---|
| 10:00–10:50 | Roee | Opening lecture (biology) | Roee's | none |
| 10:50–11:00 | | Break | | |
| 11:00–11:40 | Mattan | 1. Machine learning and CNNs | `01_cnn` | `01_cnn.ipynb` |
| 11:40–11:50 | | Break | | |
| 11:50–12:30 | Mattan | 2. DNA foundation models | `02_foundation_models` | `02_foundation_models.ipynb` |
| 12:30–12:40 | | Break | | |
| 12:40–13:20 | Mattan | 3. Shortcuts, controls and fine-tuning | `03_shortcuts_and_controls` | `03_shortcuts_and_controls.ipynb` |
| 13:20–13:30 | Mattan | Wrap-up and feedback | end of `03_shortcuts_and_controls` | none |
| 13:30–14:00 | | Buffer | | |

## Projected minutes, before any cuts

Talk and debrief: about 1.5 min per content slide, or the slide's own "Projected" note. Notebook: the sum of
the "about N min" section headings (reading, typing and T4 compute). Both are estimates until the Colab test.

| Part | Budget | Talk | Notebook | Debrief | Projected | Over |
|---|---|---|---|---|---|---|
| 1. Machine learning and CNNs | 40 | 32 (task and biology 12, learning and testing 8, CNN 9, look inside 3) | 46 (+5 optional) | 8 | 86 | +46 |
| 2. DNA foundation models | 40 | 20 (overview, tokenization, embedding, transformer, heads) | 32 | 5 | 57 | +17 |
| 3. Shortcuts, controls and fine-tuning | 40 | 10 (shortcuts, control, fine-tuning) | 49 (+2 optional) | 8 | 67 | +27 |
| Wrap-up | 10 | 10 | | | 10 | 0 |
| **Total** | **130** | | | | **220** | **+90 (+60 after the buffer)** |

Measured on a Colab T4 (13.10, no errors): compute is 1.8, 4.1 and 4.7 min for the three notebooks
(excluding the "Stuck?" cells). The longest waits are live pretraining (3.7 min) and LoRA training (3.2 min);
everything else in the minutes above is reading and typing.

Largest items: part 1 notebook sections 7 and 8 (logos 9, insertion 8); part 3 LoRA (7, of which 3 training)
and `shortcut_scores` (8); part 2 TODO-1 embeddings (9) and live pretraining (6, of which 4 computing).

Notebooks for students: https://github.com/mathoory/ai-days-2026-dna-workshop (Open in Colab links in the README).
Data: https://huggingface.co/datasets/yup8/ai-days-2026-ago2
Feedback: https://forms.gle/NfE6ZHj98ri3WYmG6
