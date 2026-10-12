# AI Days spike — can AGO2 carry the whole workshop?

Run 12.10.2026 on gold-srv2 (NVIDIA A30, GPU1), torch 2.14 + transformers 5.19 (current
releases, likely close to what Colab ships). Scripts are `s1`–`s6` in this folder; the raw
numbers are in `out/*.json`. All AUROCs are on held-out data. **Short answer: yes, AGO2 works
for every slot**, but the story is a bit different from what we planned.

## 1. Results

| Model (AGO2, held-out) | Original negatives | Composition-matched negatives | Shuffled-sequence control¹ |
|---|---|---|---|
| GC + U content only (2 numbers) | 0.64 | **0.51** (chance) | — |
| 5-letter word counts, whole read | 0.78 | 0.75 | — |
| 5-letter word counts, 40-nt site only | 0.83 | 0.81 | — |
| **CNN** (Task 1 architecture, 10 epochs) | **0.86** | **0.85** | 0.67 |
| Mistral-DNA-17M, frozen + linear probe | 0.80 | 0.75 | **0.77** |
| NT-500M, frozen + linear probe | 0.80 | 0.77 | **0.74** |
| Same two models, random weights (no pretraining) | 0.57 / 0.61 | — | — |
| Mistral-DNA-17M, **LoRA fine-tuned** (3 epochs) | 0.84 | 0.80 | — |
| Our own 5-min pretraining, then probe | 0.72 | — | — |

¹ Real test sequences are shuffled so their letter and letter-pair counts stay the same but
the order is destroyed, then scored against real negatives. A model that still separates them
is reading composition, not sequence.

CNN over 3 seeds, 20 epochs, on the `ls` validation set: 0.85 at epoch 5, 0.867 at epoch 10,
0.87–0.876 at epoch 20. The spread across seeds is ±0.003, so the CNN numbers are stable.

## 2. What it means for the workshop

- **The break is real and clean.** Both foundation models score 0.74–0.77 on
  *shuffled* sequences, about as well as on the real ones (0.78). Their frozen
  embeddings mostly encode composition. The CNN drops to 0.67 on the same test, so it
  reads order. Two numbers from GC/U content alone get 0.64.
- **The big model doesn't win here.** Counting 5-letter words in the 40-nt site (0.83)
  beats both frozen foundation models (0.80), and the CNN (0.86) beats everything.
  NT-500M (486M parameters) ties the 16M Mistral-DNA. This matches KAUST's own
  enhancer result.
- **The fix does its job, but nothing collapses.** On composition-matched negatives,
  GC/U drops to chance (0.51). The CNN loses 0.015, k-mers 0.02, the frozen probes
  0.03–0.045, LoRA 0.035. The foundation models lean on composition the most, but they
  also learned something real. Honest slot 7 message: *the ranking survives the control,
  and the control tells you how much of each score was free.*
- **Fine-tuning helps.** LoRA reaches 0.84 vs the probe's 0.80, closing most of the gap
  to k-mers but not to the CNN. One epoch is already 0.815.
- **Pretraining matters, and you can watch it work.** With random weights the same
  architectures give 0.57–0.61; pretrained, 0.80. Five minutes of our own from-scratch
  pretraining on the Galaxy "verysmall" set already reaches 0.72 (validation loss
  8.37 → 6.09, 2,200 steps, under 1 of 1.3 epochs).
- **The microRNA slide holds, with one change.**
  - Seed sites for miR-19 (`UUUGCAC`, the #1 enriched 7-mer), miR-15/16, miR-30 and
    miR-103/107 are 3–6× enriched in bound sites, but each is in only about 1–2% of them.
  - The CNN's first-layer **logos do not show seeds**; they come out composition-like and
    look the same on negatives.
  - **Planting does show the seed effect:** 6 copies of the miR-15/16 site in real
    negatives push the CNN score from 0.28 to 0.53, vs 0.32 for a shuffled control.
    miR-19 gives 0.40 vs 0.24 and miR-30 0.39 vs 0.18. let-7 and the polyA signal do
    nothing, and GC runs *lower* the score.
  - **Slot 3 therefore becomes:** logos show composition, planting reveals the microRNA
    rule. That's a better lesson: look, then intervene.
  - Also enriched: the polyadenylation signal `AAUAAA`. Bound sites sit near the 3' end
    of genes, which is a location shortcut worth one sentence.

## 3. Timings (A30) and Colab T4 estimates

The T4 estimate is roughly 2–3× the A30 time. Treat it as a guess until measured on Colab.

| Step | A30 | T4 estimate | Notes |
|---|---|---|---|
| Load + one-hot all 93k sequences | 2 s | 5–10 s | |
| CNN, per epoch | 1.4 s | 4–5 s | 10 epochs fit in under 1 min |
| Mistral-DNA-17M: download / load | 10 s / 2 s | ~10 s | small, standard transformers code |
| Mistral-DNA-17M: embed 15k seqs | 4 s | ~10–15 s | 0.3 GB of GPU memory |
| NT-500M: download / load | 55 s / 2 s | 30–60 s | ~2 GB per person, ×100 people |
| NT-500M: embed 15k seqs | 18 s | ~50–60 s | 4.3 GB of GPU memory |
| LoRA Mistral-17M, 20k seqs, per epoch | 0.85 min | ~2–3 min | |
| Pretraining, 5-min budget | 2,200 steps | ~800–1,000 steps | **T4 has no bf16**: needs fp16 + GradScaler, untested |
| Mistral-DNA repo clone (verysmall data) | 48 s | ~20–40 s | 257 MB |
| Composition-matched negatives | 0.4 s | <1 s | |

## 4. Decisions I recommend

| Question | Recommendation |
|---|---|
| AGO2 all day? | **Yes.** No K562 fallback needed. |
| Which foundation model? | **Mistral-DNA-17M** for hands-on: tiny, fast, standard transformers code, and the Galaxy tutorial is CC-BY. NT-500M as one comparison cell ("30× bigger, same score"), precomputed embeddings available as a rejoin file. |
| NT-v2-50M (KAUST's choice) | **Drop.** Its custom code imports a function removed in transformers 5 (`find_pruneable_heads_and_indices`). Patching the config doesn't help. |
| Masked-base fill-in demo | **Drop.** NT-500M's top guess for a masked 6-mer is wrong, at 4% confidence. It's an honest result but takes too long to explain. |
| Pretraining | **Include as a short live cell in slot 5:** start the from-scratch run, watch the loss fall for ~3 min, then probe it (0.72) next to random weights and the real pretrained model. Students load a precomputed checkpoint if their GPU is slow. Needs an fp16 test on Colab. |
| Fine-tuning | **Include LoRA in slot 7, 1 epoch (~2–3 min on T4):** "does fine-tuning get past the shortcut?" Answer: partly (0.80 on matched). |
| Slot 6 "break" test | Lead with the **shuffled-sequence control**: the most striking number in the spike. Then GC/U-only and the k-mer baseline. |

## 5. Not yet verified
- Timings on a real Colab T4, including concurrency and download speed for ~100 users.
- fp16 pretraining on a T4.
- A random-init probe for the exact pretraining configuration: compare against the 0.57
  of the Mistral-17M architecture instead, or measure it in the notebook.
- The planting results are from one CNN seed; slot 3 should show them on the student's own
  model anyway.

## 6. Addendum: when does pretraining pay off? (s7_lowdata.py)

Composition-matched benchmark (Slot 7's), same 3,000-sequence test set. Training sets of n
labelled sequences, 3 random draws each. The CNN is trained from scratch with early stopping on a
10% split of its own n sequences.

| labels n | CNN from scratch | 5-mer counts | Mistral-DNA frozen + probe | random-weight Mistral + probe |
|---:|:--:|:--:|:--:|:--:|
| 100 | 0.54 | 0.63 | **0.68** | 0.50 |
| 250 | 0.55 | 0.65 | **0.71** | 0.50 |
| 500 | 0.58 | 0.65 | **0.73** | 0.50 |
| 1,000 | 0.66 | 0.67 | **0.74** | 0.53 |
| 2,500 | 0.72 | 0.66 | **0.74** | 0.53 |
| 5,000 | **0.77** | 0.70 | 0.75 | 0.54 |
| 12,000 | **0.82** | 0.73 | 0.77 | 0.55 |
| 30,000 | **0.84** | 0.74 | 0.77 | 0.57 |

**The curves cross between 2,500 and 5,000 labels.**
- **Below that, the pretrained model wins clearly:** with 100 labels it reaches 0.68, while the
  CNN is near chance (0.54).
- **Above that, the CNN wins.**
- **Random-weight features are almost useless here** (0.50–0.57), so the low-data advantage comes
  from pretraining, not from the architecture.
