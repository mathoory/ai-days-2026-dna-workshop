# Teaching Machines to Read DNA — AI Days 2026 workshop

Technion AI Days, Thu 15.10.2026, 10:00–14:00. Mattan's hands-on slots: 2, 3, 5, 6 and 7.
Roee Amit's theory slots are 1, 4 and 8. All the hands-on slots use the AGO2 CLIP-seq data
from the 00660121 final project (Task 1).

| slot | notebook | deck | students write |
|---|---|---|---|
| 2 Build | `notebooks/02_build_cnn.ipynb` | `slides/02_build_cnn.qmd` | one-hot check, CNN layers, held-out AUROC |
| 3 Look inside | `notebooks/03_look_inside.ipynb` | `slides/03_look_inside.qmd` | motif readout, planting + shuffled control |
| 5 Pretrained model | `notebooks/05_pretrained_model.ipynb` | `slides/05_pretrained_model.qmd` | `embed()`, `probe_auc()` |
| 6 Break it | `notebooks/06_break_it.ipynb` | `slides/06_break_it.qmd` | GC/U baseline, 5-mer baseline, shuffle test |
| 7 Fix it | `notebooks/07_fix_it.ipynb` | `slides/07_fix_it.qmd` | shortcut scores on both benchmarks, LoRA training step |

## Layout
```
notebooks/src/NN_*.py      single source of truth (percent format, `# %include h_*.py` for shared helpers)
notebooks/NN_*.ipynb       student versions          (generated)
notebooks/NN_*_solution.ipynb  filled-in versions     (generated, gitignored)
notebooks/executed/        solution notebooks executed on gold-srv2, with outputs and timings
slides/                    reveal.js decks in the course style (quarto render NN_*.qmd)
scripts/build.py           src -> solution + student notebooks
scripts/run_hpc.sh         execute all solution notebooks on gold-srv2 (GPU1), producing the rejoin files
scripts/spike/             the feasibility spike (REPORT.md) behind the workshop design
data/                      files to host (gitignored): the 4 AGO2 FASTAs, cnn.pt, emb_mistral17m.npz, emb_nt500m.npz
```

## Student work markers
Same convention as the course: `# TODO-N:` comments, then the student's code between
`# === Your code here ===` and `# === End of your code ===`. `build.py` replaces exactly those
regions with `...`; nothing else differs between the student and solution versions.
Given cells start with `#@title … run this cell, nothing to change` (Colab collapses them).
Each TODO names the functions to use, the gotchas, and the number to expect; a "Stuck?" cell after it
loads our version so nobody falls behind.

## Rebuild
```
python scripts/build.py                      # all notebooks
quarto render slides/02_build_cnn.qmd        # one deck
```

## Measured run times (gold-srv2, NVIDIA A30, fresh Hugging Face cache, 12.10.2026)

| slot | whole notebook | the slow cells |
|---|---|---|
| 2 | 23 s | 10-epoch CNN training: 15 s |
| 3 | 10 s | — |
| 5 | 4.2–5 min | NT-500M download + embedding: 40–48 s (students load precomputed by default); live pretraining: 3 min + probes ≈ 3.8 min |
| 6 | 17 s | — |
| 7 | 2.5 min | two CNN trainings: 17 s; LoRA 2 epochs × 20k: 85 s; label-count curve (4 sizes): 21 s |

A T4 is roughly 2–3× slower than an A30. Still to be measured on Colab: slot 5's pretraining cell
and slot 7's LoRA, which take the fp16 + GradScaler branch on a T4 (no bf16).

Reference results (held-out AUROC), reproducible across reruns to three decimals:
- **Slot 2:** CNN 0.861.
- **Slot 5:** probe 0.782; NT-500M 0.788; our model 0.554 before and 0.746 after 3 minutes of
  pretraining.
- **Slot 6:** GC/U 0.660; 5-mers 0.738. Shuffle test: CNN 0.861 → 0.661, probe 0.782 → 0.765.
- **Slot 7:** on two equal-size benchmarks, original → matched.
  - GC/U: 0.637 → 0.510
  - 5-mers: 0.762 → 0.725
  - frozen probe: 0.806 → 0.770
  - CNN: 0.843 → 0.838
  - LoRA, matched only: 0.809
- **Slot 7 label-count curve** (matched data), at 100 / 500 / 2,500 / 12,000 labels:
  - CNN: 0.53 / 0.64 / 0.68 / 0.82
  - pretrained probe: 0.67 / 0.72 / 0.74 / 0.77
  - random-weight probe: 0.50 / 0.50 / 0.51 / 0.54
  - the full 3-draw curve is in `scripts/spike/REPORT.md` §6

## Hosting (open)
The notebooks read their files through `fetch(name)`, which downloads from `DATA_URL` when it is
set and otherwise expects `./data/`. Once the rejoin files in `data/` are hosted, set `DATA_URL`
in `notebooks/src/h_setup.py` and rebuild.
