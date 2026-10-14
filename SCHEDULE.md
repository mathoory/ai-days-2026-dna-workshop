# AI Days 2026: Applications of AI in Biotechnology and Genomics (Thu 15.10.2026)

| Time | Who | Part | Deck | Notebook |
|---|---|---|---|---|
| 10:00–10:50 | Roee | Opening lecture (biology) | Roee's | none |
| 10:50–11:00 | | Break | | |
| 11:00–11:50 | Mattan | 1. Machine learning and CNNs: talk | `01_cnn` | |
| 11:50–12:00 | | Break | | |
| 12:00–12:25 | Mattan | 1. Notebook | | `01_cnn.ipynb` |
| 12:25–12:55 | Mattan | 2. DNA foundation models: talk | `02_foundation_models` | |
| 12:55–13:15 | Mattan | 2. Notebook | | `02_foundation_models.ipynb` |
| 13:15–13:25 | | Break | | |
| 13:25–13:40 | Mattan | 3. Shortcuts, controls and fine-tuning: talk (wrap-up slides at the end) | `03_shortcuts_and_controls` | |
| 13:40–14:00 | Mattan | 3. Notebook, then feedback | | `03_shortcuts_and_controls.ipynb` |

## Projected minutes, before any cuts

Talk and debrief: about 1.5 min per content slide, or the slide's own "Projected" note. Notebook: the sum of
the "about N min" section headings (reading, typing and T4 compute). Both are estimates until the Colab test.

| Part | Budget | Talk | Notebook | Debrief | Projected | Over |
|---|---|---|---|---|---|---|
| 1. Machine learning and CNNs | 75 (50 talk, 25 notebook) | 32 (task and biology 12, learning and testing 8, CNN 9, look inside 3) | 46 | 8 | 86 | +11 |
| 2. DNA foundation models | 50 (30 talk, 20 notebook) | 20 (overview, tokenization, embedding, transformer, heads) | 32 | 5 | 57 | +7 |
| 3. Shortcuts, controls and fine-tuning | 35 (15 talk, 20 notebook) | 10 (shortcuts, control, fine-tuning) | 39 (+2 optional) | 8 | 57 | +22 |
| Wrap-up | in part 3 | 10 | | | 10 | |
| **Total** | **160 + 30 of breaks (10:50, 11:50, 13:15) = 190, 10:50 to 14:00** | | | | **210** | **+50** |

Measured on a Colab T4 (13.10, no errors): compute is 1.8, 4.1 and 4.7 min for the three notebooks
(excluding the "Stuck?" cells). The longest wait is live pretraining (3.7 min); LoRA training (3.2 min) has since been removed;
everything else in the minutes above is reading and typing.

Largest items: part 1 notebook sections 7 and 8 (logos 9, insertion 8); part 3 LoRA (removed)
and `shortcut_scores` (8); part 2 TODO-1 embeddings (9) and live pretraining (6, of which 4 computing).

Notebooks for students: https://github.com/mathoory/ai-days-2026-dna-workshop (Open in Colab links in the README).
Data: https://huggingface.co/datasets/yup8/ai-days-2026-ago2
Feedback: https://forms.gle/NfE6ZHj98ri3WYmG6
