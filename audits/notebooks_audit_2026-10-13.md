# Notebook audit, 13.10.2026 (independent auditor, report only)

Scope: notebooks/src/0*.py, executed solution and student notebooks (A30 run of 13.10), plots, data checks.

## Coverage caveat
The A30 run had `my_cnn.pt` and `my_emb_mistral17m.npz` in DATA_DIR, so every rejoin cell loaded the
student's own files. The default fresh-Colab path (reference `cnn.pt`, reference vectors) and the T4 fp16
code paths were never executed.

## 02_build_cnn
- MAJOR. Title cell: the notebooks never say how the negatives were chosen; Slot 6 Q3 and Slot 7 depend on it. Add one sentence with the procedure from iDeep / iONMF (Stražar et al. 2016).
- MINOR. "AGO2 is a protein that binds RNA": add that it carries a microRNA and binds RNA that pairs with it.
- MINOR. Define "crosslinks" (UV, then pull-down with an antibody).
- MINOR. Padding to 501 while the longest sequence is 375: explain (iDeep input size).
- MINOR. TODO-1: show the indexing `X[0, 0, :, :8]`.
- MINOR. Define ReLU, channel and logit where first used. "each filter acts as a detector for one motif" overstates; say "responds most strongly to one letter pattern (a candidate motif)".
- MINOR. Naming: `val_seqs` / "held-out validation set" / "test set" for the same 1,000 sequences.
- MAJOR. TODO-3 instructions omit `.cpu().numpy()`; `roc_auc_score` on a CUDA tensor fails on a T4. Also show `with torch.no_grad():`.
- MAJOR. No "Stuck?" cells: a failed TODO-2 leaves no training, no AUROC, no plot.
- MINOR. "Two training runs differ by less than 0.01" is unsupported in the notebook. "±0.025" should say "(95% range)".

## 03_look_inside
- MINOR. Pin `logomaker`.
- MAJOR. TODO-1 never says how to get one filter's activations: add `acts_f = acts[:, f, :]`, sequences = `pos_seqs`.
- MINOR. Example "UUUGCACUUA" primes the miR-19 site; use a neutral example.
- MINOR. A mistyped `MY_MOTIF` silently falls back to filter 0; set `MY_FILTER` explicitly or print it.
- MINOR. Filter 00 fails its own bound/unbound control; tell students to pick another filter if the logos match.
- MINOR. "Dose-response" undefined; retitle "Score vs number of copies".
- MINOR (code). `plant()` copy 8 at letter 355 appends past the 330-letter canvas; skip copies that do not fit. Shared `_rng` makes numbers depend on cell order.
- MINOR. Position scan: the shuffled copy gets most of the same boost (composition in 150–190); say so.
- MAJOR. "Every sequence … site at letters 150–190" contradicts lengths 137–375; say "almost every".
- MINOR. Use "(filter, position)" not "(channel, position)". "A logo is a hypothesis" repeated; one colon-reveal sentence.
- MAJOR. Bonus reading text vs output: output 0.527 vs 0.290 ± 0.057; miR-19 +0.016, miR-30 +0.034, let-7 +0.028, all within 1 SD. Rewrite and regenerate on the reference-CNN path. Define "±" as SD over 10 shuffles. Tone down the punchline.

## 05_pretrained_model
- MAJOR (science). Intro describes next-token pretraining for both models; Nucleotide Transformer is masked.
- MINOR. "(from Roee's Slot 4)" → "(Slot 4)". The average includes [CLS]/[SEP]; say so. Name BPE.
- MINOR. Reference embed uses fp16 autocast; mention it is optional.
- MAJOR. The TODO-1 rescue include also defines `probe_auc`, so an empty TODO-2 prints a false pass (0.782).
- MINOR (plot). PCA: blue G+C colour scale clashes with blue "Bound"; use grey→ink. Bound drawn over unbound. "direction 1" undefined. Give the expected answer (G+C, r = +0.52).
- MAJOR. Wrong hint: "above 0.95 → you scored the training set". Training-set AUROC is ~0.82 (NT 0.88).
- MINOR. Explain `C=0.1`. Unequal labels vs CNN (92k vs 12k).
- MINOR. `have_precomputed` condition always True (dead code).
- MINOR. "Mixtral-style" undefined; explain why the 105M model loses to the 16M one.
- BLOCKER until tested on a T4. Pretraining fp16 + GradScaler path never run (MoE, lr 5e-4); NaN embeddings would crash the probe. T4 does ~2.5× fewer steps. Test on Colab; consider 4 minutes and a NaN guard.
- MINOR. Text bars start at 0.5 without a label.
- MINOR. "Before Slot 6" heading vs content; "regularity" sentence; "close to a coin flip" for 0.554.
- MAJOR. "both probes score below the CNN (~0.86)": the CNN had ~8× more labels; say so and point to Slot 7.

## 06_break_it
- MINOR. "unbound AGO2 sites" → "unbound RNA". Define "shortcut".
- MAJOR (interpretation). The shuffle floor is the composition score (~0.66), not 0.5; "longer-range" is wrong (motifs are short; the shuffle also moves the site out of 150–190).
- MINOR. "GC" vs "G+C". Length is also a weak shortcut (AUROC 0.54); mention it.
- MINOR (rendering). Tools/Check lines run together; blank uncommented line inside a markdown cell; "Shortcut #1".
- MINOR. CNN check says 0.66; output 0.666, scoreboard 0.67. Probe fitted on the student's vectors but shuffled vectors from the reference embed.
- MAJOR. No "Stuck?" cells; a failed TODO-3 crashes the scoreboard.
- MINOR (plot). All-caps label; bars from 0.45 exaggerate differences.

## 07_fix_it
- MAJOR (science). "comes from how the negatives were chosen" is asserted; AGO2 sites are AU-rich 3′UTRs, so part may be biology.
- MAJOR (wrong). Matching selects a subset of both classes (drops ~30% of positives); it does not create new negatives. "Composition carries no information" holds for single-letter composition only.
- MAJOR. "differ only in how the negatives were chosen" (positives differ too). Explain why "original" numbers differ from Slot 6 (new 3,000-sequence test sets, 6-epoch CNN on 65k).
- MINOR (plot). Histograms should share x.
- MINOR. TODO-1 "must not use global data" is confusing (it must use `tok`, `fm`).
- MINOR. "Updating all 16M weights is expensive" is not true at 16M; motivate LoRA by large models. State LoRA 20k vs probe 12k labels.
- MINOR. Pin peft and transformers.
- BLOCKER until tested on a T4. TODO-2 fp16 autocast without a GradScaler; only bf16 was run; re-time (expect 5–8 min).
- MAJOR. TODO-2 rescue continues training the same model (the solution run shows 4 epochs); rebuild the model in a function.
- MAJOR. §5 "With about 65,000 labelled sequences … beats the frozen model (0.84 vs 0.77)": the probe saw 12k; compare at 12k (0.82 vs 0.77).
- MINOR (plot). Pretrained and random-weight probes both mid grey.
- MAJOR. Takeaway: "composition" → "short-word composition"; add the CNN surviving the control (0.84 → 0.84); "fine-tuned did better than the CNN at both ends" is unsupported in the notebook (student LoRA 0.81 < CNN 0.84); state the offline NT-500M result explicitly. "CNN close to a coin flip" only at 100 labels.

## Cross-notebook
- Model names: "Mistral-DNA (16M)" everywhere, explain 17M once.
- One term for the 1,000-sequence set; flag the 07 switch to 3,000-sequence test sets.
- "G+C" everywhere; "bound/unbound sequences" everywhere.
- Numbers carried between slots differ (probe 0.78 vs 0.81, CNN 0.87 vs 0.84) without explanation.
- Unequal-label comparisons recur (05, 07 §5, 07 LoRA vs probe).
- Rescue cells inconsistent: none in 02 and 06; 03 only TODO-1; 05 leaks `probe_auc`; 07 does not reset.
- Untested paths: fresh-Colab default and T4 fp16.
- Style mostly clean; remaining tells: colon reveals, repeated aphorism, three question headings in a row, a punchline ending.

## Top 10
1. Run 05 §3 and 07 §3 on a fresh Colab T4; GradScaler for fp16 LoRA; pin transformers/peft/logomaker; re-time.
2. 07 takeaways (unsupported fine-tuning claim; CNN survives the control; short-word composition).
3. 06 shuffle-test logic (floor ≈ 0.66, not 0.5).
4. 07 matching description (subset of both classes) and why "original" numbers differ from Slot 6.
5. 05 TODO-2 false pass via the rescue include.
6. 05 wrong training-set diagnostic (≈0.82, not >0.95).
7. 03 bonus text vs output, regenerated on the reference CNN.
8. Rescue cells for 02 and 06; 07 rescue rebuilds the LoRA model.
9. Where the negatives come from (02); do not assert the composition gap is purely an artefact (07).
10. Unequal-label comparisons; `.cpu().numpy()` and `acts[:, f, :]` hints; "almost every" sequence.
