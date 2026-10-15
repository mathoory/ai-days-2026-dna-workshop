# AI Days 2026: Applications of AI in Biotechnology and Genomics (Thu 15.10.2026)

| Time | Who | Part | Deck | Notebook |
|---|---|---|---|---|
| 10:00–11:30 | Roee | Opening lecture (biology) | Roee's | none |
| 11:30–11:35 | | Break | | |
| 11:35–12:15 | Mattan | 1. Machine learning and CNNs (talk) | `01_cnn` | |
| 12:15–12:20 | | Break | | |
| 12:20–12:40 | Mattan | 1. Machine learning and CNNs, continued (notebook) | | `01_cnn.ipynb` |
| 12:40–13:25 | Mattan | 2. DNA foundation models (25 min talk, 20 min notebook) | `02_foundation_models` | `02_foundation_models.ipynb` |
| 13:25–13:30 | | Break | | |
| 13:30–13:45 | Mattan | 3. Shortcuts, controls and fine-tuning (15 min talk, feedback; notebook at home) | `03_shortcuts_and_controls` | `03_shortcuts_and_controls.ipynb` |
| 13:45–14:00 | | Questions, end | | |

Updated on the day (15.10): the opening lecture ran to 11:30 and part 1's talk to 12:15; part 3's hands-on
is cut (the notebook stays online). 90 + 60 + 45 + 15 + 3 × 5 + 15 of questions = 240 min, 10:00 to 14:00.

## Projected minutes, before any cuts

Talk and debrief: about 1.5 min per content slide, or the slide's own "Projected" note. Notebook: the sum of
the "about N min" section headings (reading, typing and T4 compute). Both are estimates until the Colab test.

| Part | Budget | Talk | Notebook | Debrief | Projected | Over |
|---|---|---|---|---|---|---|
| 1. Machine learning and CNNs | 65 (40 talk, 25 notebook) | 32 (task and biology 12, learning and testing 8, CNN 9, look inside 3) | 46 | 8 | 86 | +21 |
| 2. DNA foundation models | 50 (30 talk, 20 notebook) | 20 (overview, tokenization, embedding, transformer, heads) | 32 | 5 | 57 | +7 |
| 3. Shortcuts, controls and fine-tuning | 35 (15 talk, 20 notebook) | 10 (shortcuts, control, fine-tuning) | 39 (+2 optional) | 8 | 57 | +22 |
| Wrap-up | in part 3 | 10 | | | 10 | |
| **Total** | **150 + 30 of breaks = 180, 11:00 to 14:00** | | | | **210** | **+60** |

Measured on a Colab T4 (13.10, no errors): compute is 1.8, 4.1 and 4.7 min for the three notebooks
(excluding the "Stuck?" cells). The longest wait is live pretraining (3.7 min); LoRA training (3.2 min) has since been removed;
everything else in the minutes above is reading and typing.

Largest items: part 1 notebook sections 7 and 8 (logos 9, insertion 8); part 3 LoRA (removed)
and `shortcut_scores` (8); part 2 TODO-1 embeddings (9) and live pretraining (6, of which 4 computing).

Notebooks for students: https://github.com/mathoory/ai-days-2026-dna-workshop (Open in Colab links in the README).
Data: https://huggingface.co/datasets/yup8/ai-days-2026-ago2
Feedback: https://forms.gle/NfE6ZHj98ri3WYmG6
