# Applications of AI in Biotechnology and Genomics: AI Days 2026 workshop

Technion AI Days, Thu 15.10.2026. Prof. Roee Amit opens at 10:00 until 11:30; the three parts run from 11:35 to
13:45 (parts 1 and 2 a talk, then a notebook; part 3 a talk, its notebook at home), then questions until 14:00. All parts use the AGO2 CLIP-seq data of the GraphProt benchmark (Maticzka et al. 2014).

| part | notebook | deck | you write |
|---|---|---|---|
| 1 Machine learning and CNNs (11:35) | `notebooks/01_cnn.ipynb` | `slides/01_cnn.qmd` | one-hot encoding, the CNN layers, held-out AUROC, a logo for every filter, motif insertion against a shuffled copy |
| 2 DNA foundation models (12:40) | `notebooks/02_foundation_models.ipynb` | `slides/02_foundation_models.qmd` | `embed()`, `probe_auc()` |
| 3 Shortcuts, controls and fine-tuning (13:30) | `notebooks/03_shortcuts_and_controls.ipynb` | `slides/03_shortcuts_and_controls.qmd` | G+C/U baseline, 5-letter word baseline, shuffle test, `shortcut_scores()` on both benchmarks |

Open a notebook in Colab, choose *Runtime → Change runtime type → T4 GPU*, and run from the top. Every
notebook runs on its own: the data and our reference files download from
[Hugging Face](https://huggingface.co/datasets/yup8/ai-days-2026-ago2).

## Layout
```
notebooks/src/NN_*.py      single source of truth (percent format, `# %include h_*.py` for shared helpers)
notebooks/NN_*.ipynb       student versions          (generated)
notebooks/NN_*_solution.ipynb  filled-in versions     (generated, gitignored)
slides/                    reveal.js decks (quarto render NN_*.qmd)
scripts/build.py           src -> solution + student notebooks
scripts/build_colab_test.py  the three solution notebooks in one, timed cell by cell, for a Colab test run
scripts/run_hpc.sh         execute all solution notebooks on gold-srv2 (GPU1)
scripts/spike/             the feasibility spike (REPORT.md) behind the workshop design
data/                      files to host (gitignored): the 4 AGO2 FASTAs, cnn.pt, emb_mistral17m.npz, emb_nt500m.npz
```

## Student work markers
`# TODO-N:` comments, then the student's code between `# === Your code here ===` and
`# === End of your code ===`. `build.py` replaces exactly those regions with `...`; nothing else differs
between the student and solution versions. Given cells have a `#@title` (Colab shows the title and hides
the code). Each TODO states the problem, the rules and the tools (with documentation links), never the answer;
a test cell after it checks the student's code. There are no fallback cells: the solution notebooks go on
the workshop site during the workshop.

## Rebuild
```
python scripts/build.py                  # all notebooks
python scripts/build_colab_test.py       # the Colab timing test
quarto render slides/01_cnn.qmd          # one deck
```

## Measured run times (gold-srv2, NVIDIA A30, 13.10.2026)

| notebook | whole notebook | the slow cells |
|---|---|---|
| `01_cnn` | 32 s | 10-epoch CNN training: 19 s |
| `02_foundation_models` | 5 min (fresh download, NT-500M recomputed) | live pretraining: 3.4 min |
| `03_shortcuts_and_controls` | about 2.8 min (measured with LoRA, since removed) | `shortcut_scores()`: 43 s |

Colab, Tesla T4 (13.10, all three notebooks in one session, no errors): `01_cnn` 1.8 min (training 69 s),
`02_foundation_models` 4.1 min (live pretraining 3.7 min, 1,178 steps), `03_shortcuts_and_controls` 4.7 min
without the "Stuck?" cells (it then included LoRA, 3.2 min, since removed; `shortcut_scores()` 32 s). Peak GPU memory 2.2 GB; the 18 MB data
file downloads in 1.5 s. Torch reports bfloat16 support on the T4, so the float16 + GradScaler branch was
not used there; it runs without errors when forced (tested on the A30).

## Reference results (held-out AUROC)
- **Part 1:** CNN 0.861 (0.789 after epoch 1). Inserted miR-15/16 site 0.53 against 0.30 for shuffled copies.
- **Part 2:** Mistral-DNA probe 0.782; NT-500M probe 0.788; a 105M model 0.554 before and 0.726 after
  3 minutes of pretraining (1,834 steps on the A30).
- **Part 3:** G+C/U 0.660; 5-letter words 0.738. Shuffle test: CNN 0.861 → 0.662, probe 0.782 → 0.765.
  On two 3,000-sequence benchmarks, original → matched:
  - G+C/U: 0.637 → 0.510
  - 5-letter words: 0.762 → 0.725
  - frozen probe: 0.806 → 0.770
  - CNN: 0.844 → 0.838
