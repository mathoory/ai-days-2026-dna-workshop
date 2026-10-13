# Merged audit (decks + notebooks as one day), 2026-10-13

Scope: `slides/0*.qmd` (text, notes, rendered screenshots in `shots3/`), `notebooks/src/*.py`, executed solution and student notebooks, `scripts/spike/`, `README.md`, `DESIGN.md`, `AUDIT_FIXES.md`, and the data files in `data/`. Nothing was edited apart from this file. Timings are A30 times from the executed notebooks' cell metadata ×2.5 for a T4.

Severity: **BLOCKER** = the slot fails or the day teaches something false. **MAJOR** = a participant will notice a contradiction, or a conclusion is not supported by the evidence shown. **MINOR** = polish.

---

## Slot 1 (10:00, deck 01, no notebook)

- **MAJOR. 01 §3.2, visible text: "Today the 1,000 held-out sequences (the `ls` file) are the test set in every slot."** This is false for Slot 7, which tests on 3,000 sequences drawn from the *train* file (`07_fix_it.py` l.71–77), and for the Slot 7 offline figures. Fix: "...in Slots 2 to 6. Slot 7 builds its own 3,000-sequence test sets." Also, `h_data.read_split` says `'ls' (the held-out validation set)`, nb02/03 name it `val_seqs/val_y`, and the spike REPORT says "ls validation set". This is the same set the slide calls "test" and contrasts with "validation". Rename `val_*` to `test_*` and fix the docstring (AUDIT_FIXES N6 was closed for the prose but not for the code).
- **MAJOR. Data leakage in the split is never mentioned, although the notes discuss it as a general risk.** The 01 §3.2 notes say "Real projects split by gene or chromosome." The workshop data are not split that way, and the overlap is unequal between the classes. From the FASTA headers: 235/500 held-out **bound** windows have a training window whose centre lies within 150 nt (≥180 shared letters), against 68/500 **unbound**. I fitted slot 6's 5-mer probe (12k training sequences, C=0.1; it reproduces nb06's 0.738). Its AUROC is **0.78** on the bound sequences that have a nearby training window and **0.70** on those that do not, each against all unbound. This could be leakage or gene-level confounding (genes with many sites). Either way the room's coders will ask about it. Fix: one sentence in the 01 §3.2 notes ("our split is random by row; 47% of held-out bound windows overlap a training window"). Then either report the CNN on the non-overlapping half once, or say it is untested.
- **MAJOR. §4.2 (bootstrap figure) and §4.3 use the CNN 0.86 vs the probes 0.78/0.79 as "a real gap".** The CNN was trained on 92k labels and the probes on 12k. The gap survives at equal labels (Slot 7 curve at 12k: CNN 0.82 vs probe 0.77), but nothing in Slot 1 says so. It also names "Mistral-DNA", "probe" and "Nucleotide Transformer" 4 hours before they are introduced. Fix: add "(CNN: 92k labels; probes: 12k; Slot 7 compares at equal labels)" to the notes, or replace the probes with the two-number model (0.66) and a second CNN seed. Those make the same point and need no forward reference.
- **MAJOR (missing). How the negatives were chosen is never stated.** §2.2 only says "chosen by the authors... from places the experiment did not flag". Take-home #1 (deck 08) and the Slot 6 discussion question ("Where do the negatives come from?") depend on it. The headers show positives carry transcript IDs (`NM_…`, `AK…`) while negatives are `neg_N` with coordinates only. Fix: one notes bullet in 01 §2.2 with the actual sampling rule from the source (iDeep reuses the iONMF set, Stražar et al. 2016; check the paper, do not paraphrase from memory). Repeat it in Slot 6 §2.3.
- **MINOR. §1.6 notes:** "about 92,000 RNA stretches" is correct for training (92,346); the 1,000 `ls` sequences are separate. Fine. §4.2 notes give "on 3,000 (Slot 7), below about 0.02", while 07 §2.1 notes say "noise is about ±0.015". Both can be true (one is a CI half-width, the other a difference threshold), but say which is which.
- **MINOR. Colab setup has no slot.** §1.5 asks for "Colab with a GPU", but no time is reserved for 40–100 people to sign in, open the repo and get a T4. Fix: in the last 3 minutes of Slot 1, have everyone open nb02 and run the two setup cells (which also download the 36 MB of data). That moves the slowest step out of Slot 2.

## Slot 2 (10:30, deck 02 + nb02)

Alignment: 2,491 parameters, the shapes table, 492 positions, the epoch-1 value of 0.79 and the four training lines all match the notebook. The table in §1.4 is identical to the nb02 markdown table.

- **MAJOR. No rescue cells in nb02.** If TODO-2 or TODO-3 is wrong, the training cell, the curve and `my_cnn.pt` all fail (student executed notebook: `TypeError ellipsis is not a Module subclass`, then `NameError history`). nb03/05/07 have "Stuck?" cells; the first hands-on slot, where novices are most likely to fail, has none. Fix: add "Stuck?" cells after TODO-2 (defines `GlobalCNN`) and TODO-3 (defines `train`).
- **MAJOR. TODO-3 omits the step novices miss:** the probabilities must come back to the CPU (`.cpu().numpy()`) before `roc_auc_score`. The TODO text says "move X_val to DEVICE" but not back, and §1.6 notes say only "sigmoid of the outputs, then roc_auc_score". On the T4 this raises a CUDA-tensor error. Fix: add step 2b, "...then `.cpu().numpy()`". nb05 lists this as a gotcha; nb02 needs it more.
- **MINOR. Numbers students see vs slides.** nb02 prints a final AUROC of **0.867** (solution run, SEED=42; it shows as 0.87 to two decimals). The slides, nb02 prose and the README ("CNN 0.861, reproducible to three decimals") say 0.86. The reference `cnn.pt` (0.861) is not the model the notebook trains. Within noise, and §1.7 notes say so, but the README claim is wrong. Fix the README; optionally say "0.86 to 0.87".
- **MINOR. §1.6 figure** (`filter_learning`) shows epoch 3 = 0.85. The §1.8 curve and the notebook show 0.84 at epoch 3. They are different runs; drop the per-epoch AUROCs from the §1.6 titles, or use the §1.8 run.
- **MINOR. Unused forward reference.** §1.2 notes: "Its strongest response is only 1.07 times larger on bound than on unbound sequences, which matters in Slot 6." Slot 6 never mentions filters. Cut it, or reference it in 06 §2.1.
- **MINOR. §1.5 (last-layer weights) is used only in Slot 3** (position scan). It costs ~2 min in an over-full slot. Move it to deck 03 just before §2.2, where "the CNN weights positions" is the punchline.
- **MINOR. Why 501?** The maximum length is 375 (nb02 prints "length: 137-375"), so every sequence gets ≥126 padding columns. Someone will ask. Add one notes line: "501 is the architecture from the course; padding columns are uniform." Also note that positives vary more in length (SD 29 vs 15 nt). Since padding sits at the end, the CNN can read length. Slot 7 matches on length, but Slot 6 never names it as a shortcut.

## Slot 3 (11:00, deck 03 + nb03)

- **MAJOR. nb03's closing markdown contradicts the cell directly above it.** It says "in our reference run ... 0.53 against 0.31 ± 0.09", while the executed cell prints `0.527 | shuffled: 0.290 ± 0.057`. For miR-19, the slide (§3.1 figure and notes) gives 0.42 vs 0.34 ± 0.06; the notebook prints 0.395 vs 0.379 ± 0.062. The controls come from `_rng`, whose state depends on how many earlier cells ran (TODO-1, the Stuck cell, the random canvases), so the numbers shift between runs. Fix: give the bonus cell its own `np.random.default_rng(1)`. Regenerate `planting_bars.png` from the notebook's code path, and quote the result as "about 0.53 vs about 0.3".
- **MAJOR. The deck shows the TODO-2 answer before the TODO.** §2.1 (dose response, with the motif UACACUCUAC that nb03 uses by default) and §2.2 (position scan, which explains the jump at copy 4) come before "2.3 Your turn". The notebook's "Reading your result" then asks students to discover the position effect themselves. Fix: put "Your turn" right after §1.3 and show §2.1/§2.2 after the notebook ("show after they have run it", as deck 02 §1.8 does).
- **MINOR. §1.1 notes:** "roughly 25,000 windows ... The notebook's count differs slightly". The figure says 24,667; nb03 prints 35,191 for filter 00. That is 43% more, not "slightly". It happens because nb03 scans 500 `ls` + 5,000 train bound sequences. Say "the notebook scans more sequences, so it collects more windows (about 35,000)".
- **MINOR. Filter numbers only match for the reference CNN.** §1.2 notes ("filters 3 and 9 are G/C-rich") and §1.3 ("filter 0") are only true with `USE_MY_CNN = False`. With their own CNN, students get different filter indices. Say so on §2.3.
- **MINOR. The tools for TODO-1 are almost complete.** The one gap: the toolkit table says `cutoff_for(acts_f)`, but nothing says `acts_f = acts[:, f, :]`. Add it to the TODO comment.
- **MINOR. Edge case in `plant()`:** copies 5–6 are written at 220/265, and the bonus plants 6 copies into real unbound sequences (lengths 240–373). If a slice starts beyond the end, Python appends the motif at the end, which changes length and position silently. That is harmless for the default canvas but affects the bonus. Clip or skip copies that do not fit.
- **MINOR. Single-seed claim.** "The CNN responds to one seed site" (here and in deck 08) rests on one CNN seed (spike REPORT §5 says so). Students with their own CNN may not see it. Add "with our CNN; check yours" to the nb03 bonus text.
- **MINOR. The position confound applies to the bonus too.** Of the 6 copies, only copy 4 (letter 175) lands in the sensitive 140–190 window, which follows directly from §2.2. One notes line would close the loop.
- **MINOR. "First cell: `USE_MY_CNN = True`"** (§2.3). It is the 4th code cell, after setup/data/CNN helpers. Say "the Rejoin cell". Same in deck 06 §1.3.

## Slot 4 (11:30, deck 04, theory)

- **MINOR. Under-filled:** 6 content slides for 30 minutes, while Slot 7 is ~14 min over (see timing). Move the LoRA mechanics slide (07 §1.2) here, right after §1.6, and consider moving the model scan (07 §2.3) here as "what we'll test". The tokenization (§1.2) and landscape (§1.5) slides are not used by any notebook. That is fine for theory, but they are the cheapest to shorten if Roee needs time.
- **MINOR. §1.6 says LoRA trains "1 to 10% of parameters"**; nb07 prints 1.6%. Consistent.
- **MINOR. §1.4 notes** list Mistral-DNA as autoregressive, which matches nb05 (F6 fixed).

## Slot 5 (12:15, deck 05 + nb05)

Alignment: probe 0.782 / NT 0.788 / PCA r=+0.52 / 12,000 + 1,000 sequences / 4,096 tokens / ln 4096 ≈ 8.3 all match the executed notebook.

- **MAJOR. §2.2 "Frozen and probed, the pretrained models lose to a 2,491-parameter model trained on the task"** compares a CNN trained on 92k labels with probes trained on 12k. nb06's intro table states the label counts; this slide does not. Fix: "CNN: 92k labels; probes: 12k. Slot 7 compares at equal labels (the CNN still wins at 12k: 0.82 vs 0.77)."
- **MAJOR. The live pretraining cell is untested on a T4** (AUDIT_FIXES B5 open; README "still to be measured"). The T4 takes the fp16 + GradScaler branch. On the A30 the cell took 212 s (180 s training + ~30 s for the two 13k-sequence embeddings with a 105M model + tokenizing 100k genome pieces). On a T4, expect ~5 min and ~400 steps, so the probe gain will be smaller than the slide's 0.72. Fix: run it once on Colab before Thursday. Have a precomputed "after" checkpoint or number ready to show if a T4 stalls.
- **MINOR. Ambiguous "Random weights" (0.55) in the §2.2 figure.** It is the 105M Mixtral-config model, not Mistral-DNA-16M; Slot 7's "random weights" line is Mistral-DNA-16M (0.50–0.54). Label it "105M model, random weights" and "same model, 3 min pretraining".
- **MINOR. Rounding.** The notebook prints 0.727 after pretraining (shown as 0.73); the slide and figure say 0.72. The README says 0.746, and the §2.3 notes say "another run gave 0.75". Say "0.72 to 0.75 depending on the run" once.
- **MINOR. "Blind guess among 4,096 tokens"** (figure) vs "Uniform guess" (nb05 legend and markdown, deck 04/05 notes). Pick one.
- **MINOR. §1.1 code card** calls `from_pretrained(name)` without `trust_remote_code=True`; `h_fm.load_fm` uses it. Either add it or say "our helper adds one flag".

## Slot 6 (12:45, deck 06 + nb06)

Alignment: 0.66 / 0.74 / 0.666 / 0.765 match the executed notebook (0.660, 0.738, 0.666, 0.765). "Nearly half the distance" (0.16/0.36 = 0.44) is correct.

- **MAJOR. "Composition" changes meaning between Slots 6, 7 and 8, and the conclusions collide.**
  - 06 §2.2 notes: "nearly everything the probe uses is composition at that level, not longer words and not order". Here "that level" means letter-pair counts, which is what the dinucleotide shuffle keeps.
  - 07 §1.1 notes / nb07 intro: after matching G+C, U and length, "composition carries no information about the label, so any score above 0.5 must come from something else."
  - 08 §1.1 notes: "Matched negatives: the control that removes the composition shortcut." 08 take-home #5: "frozen, they mostly encode composition."
  - The data: on matched negatives the frozen probe still scores **0.77** and 5-mer counts **0.725**. Both are composition features in the Slot 6 sense, and matching on G+C/U/length does not remove them.
  - So a careful participant concludes either that the probe is not "mostly composition", or that the matched control did not remove composition. Fix: define two levels everywhere. Single-letter composition (G+C, U) and length are what matching removes. Letter-pair and word composition are what the shuffle keeps, and they remain informative after matching. Rewrite 07 §1.1 notes to "G+C and U carry no information", and 08 #5 to "frozen, they mostly encode letter and letter-pair composition". Best: also run the shuffle test on the matched benchmark (one extra line in nb07), which would show directly what the probe reads after matching.
- **MAJOR. "Shortcut #2: counting words"** (nb06 heading) vs "simpler explanations" (deck 06). Five-letter word counts include seed-site fragments (a 7-mer seed contains three 5-mers), so a word-count model can read real motifs, just not their position. Calling it a shortcut contradicts Slot 3, where a 7-letter site is the biology. Fix: in nb06, call it "Baseline 2: counting words". Add one line: "word counts can capture motifs, but not where they are".
- **MINOR. Unexplained feature in the §2.1 figure.** The Mistral panel has a tall spike of *unbound* sequences at score ≈ 0. Someone will ask what they are, and that cluster contributes to the probe's shuffle AUROC. Either check what those sequences are (e.g. repeats or very high G+C) and add a note, or say "a group of unbound sequences that the probe scores near 0".
- **MINOR. No rescue cells.** In the student notebook, a missing TODO-1/2/3 breaks the scoreboard (`NameError auc_shuf_cnn`). Add a single "Stuck?" cell before the scoreboard.
- **MINOR. Length is never named as a shortcut** in Slot 6, although DESIGN.md lists it and Slot 7 matches on it (see Slot 2).
- **MINOR. §1.3 table has "Expect" instead of "Time"**, unlike every other "Your turn" slide. Keep both columns.

## Slot 7 (13:15, 25 min, deck 07 + nb07)

Alignment: 67,764 sequences, 0.64→0.51, 0.76→0.72/0.73, 0.81→0.77 and 0.84→0.84 match the notebook. 8,192 LoRA weights (2·256·16) and 262k/16.0M (1.6%) are correct. The label curve at 12k (CNN 0.82, Mistral 0.77) matches.

- **BLOCKER. It does not fit in 25 minutes** (see the timing table: ~38 min). LoRA alone is 156 s on the A30 (nb07 metadata; the README's "85 s" is outdated), so 6–8 min on a T4. If TODO-2 fails, the "Stuck?" cell reruns it, and because `clf` is not reset it continues training from where the failed attempt stopped. The deck's "3 to 4 minutes on a T4" (§1.2 notes) and nb07's "~4 minutes" are both too low. Fix (pick two): LoRA 1 epoch on 10k sequences (the spike's own recommendation: "1 epoch, ~2–3 min on T4", already 0.815 after one epoch); make TODO-2 a given cell started at the beginning of the slot; make the label-count cell optional; move 07 §1.2 to deck 04 and 07 §2.3 to deck 08.
- **BLOCKER. fp16 LoRA on the T4 has no GradScaler.** `batch_logits` autocasts to float16 when bf16 is unavailable (T4), and the TODO-2/Stuck loops call `loss.backward(); opt.step()` without `torch.amp.GradScaler`. The README says Slot 7 "take[s] the fp16 + GradScaler branch on a T4". It does not: only nb05 has a scaler. Gradient underflow is plausible, so the "≈ 0.80" check may silently fail on Colab. Fix: add the scaler (as nb05 does) or run LoRA in fp32 on the T4 (16M parameters, still fast). Then do the B5 smoke test.
- **MAJOR. Different test set from Slots 2–6, never explained, so the numbers jump.** The CNN goes from 0.867 (Slot 6) to 0.843 ("original"), the probe from 0.782 to 0.806, and word counts from 0.738 to 0.762. A participant will ask why the probe got better and the CNN worse. Fix: one line on §1.3 and in nb07 §1: "New test set: 3,000 sequences from the training file, so numbers differ from the `ls` results (and the CNN trains 6 epochs here)."
- **MAJOR. Unequal label counts inside the before/after table and the dumbbell.** The CNN trains on 64,764 sequences; word counts and the probe use the first 12,000 (`shortcut_scores(..., n=12000)`). The figure puts them on one axis. The slide's point (the drop under matching) survives, but the vertical ranking invites a false comparison. Fix: put "CNN: 65k labels; others: 12k" on the figure or in the notes. The model-scan notes already handle this well for §2.3.
- **MAJOR. "In each bin keep as many unbound as bound" hides that bound sequences are dropped too.** Matching keeps min(bound, unbound) per bin: 33,882 of 48,095 bound survive, and the most A/U-rich bound sites go. So "only the negatives differ" (§1.1 notes, nb07 markdown "differ only in how the negatives were chosen") is not exact: the positives in the matched benchmark are a composition-typical subset. Fix: "keep the same number of each class in every bin (14,000 bound sequences with no unbound match are dropped)".
- **MINOR. The §2.4 label-curve figure (offline, 3 draws, 8 sizes, word counts, NT diamonds) differs from the notebook's curve** (one draw, 4 sizes, no word counts). At 500 labels the CNN is 0.58 on the slide, 0.637 in the notebook and 0.66 on §2.2. The notes explain the 0.58 vs 0.66 gap but not the notebook. Fix: title the slide "offline, 3 draws per size"; one notes line: "your notebook runs one draw at 4 sizes".
- **MINOR. 0.725 is shown as 0.72** (dumbbell figure, deck) **and 0.73** (nb07 TODO-1 check text). Use one.
- **MINOR. TODO-1 requires typing three long exact dict keys** (`"5-letter word counts (1,024 numbers)"`). Pre-fill the dict with the keys and `...` values. This takes ~2 min off an over-full slot.
- **MINOR. §2.5 "No labels yet: rank variants by how unexpected the model finds them".** Nothing in the day shows this. Either cut it or add "(not shown today)".
- **MINOR (missing baseline).** The spike's strongest simple model, 5-mer counts on the 40-nt site only (0.83 original / 0.81 matched), is above every frozen probe and close to the CNN. It is never shown. If "fine-tuned beats a task model" is a take-home, this is the baseline a skeptic will ask about. Add it to the model-scan notes or to §2.1.
- **MINOR. Model naming.** "Mistral-DNA-17M" appears in `model_scan.png`, while the deck and notebooks say "Mistral-DNA (16M)". Use one (the AUDIT F9 fix did not reach this figure).

## Slot 8 (13:40, 20 min, deck 08)

- **MAJOR.** Take-home #5 and the recap note "Matched negatives: the control that removes the composition shortcut" carry the composition contradiction above. Rewrite once Slot 6/7 wording is fixed.
- **MINOR.** "The CNN responds to the miR-16 site" (recap) vs "miR-15/16" (deck 03, nb03). Pick one. "Test a model like a sample" (#4) is unclear; "Test a model the way you test a sample: insert, shuffle, match" reads better.
- **MINOR.** The recap thumbnails are unreadable at this size (the screenshot shows 8 panels at ~300 px). Accept them as visual reminders and have the presenter say the numbers, or cut to 4 panels.
- **MINOR.** The slot has ~8 min of slack (4 slides + 2 min feedback). Use it to absorb 07 §2.3/§2.4.

---

## Timing (T4 ≈ 2.5 × A30 cell time)

| Slot | Budget | Deck talk | Notebook compute (T4) | Student TODO/think time | Total | Verdict |
|---|---|---|---|---|---|---|
| 1 | 30 | ~26 (13 content slides) | 0 | 3 (Colab setup, if added) | ~29 | fits |
| 2 | 30 | ~13 (§1.1–1.7) + 1 (§1.8) | ~1.5 (connect, data 36 MB, encode, 10 epochs × ~5 s) | 15 (slide's own budget) + 1 download | ~31–32 | tight; move §1.5 to deck 03, pre-run setup in Slot 1 |
| 3 | 30 | ~10 (7 slides incl. bonus) | ~1 (pip logomaker, load, scans) + 1 upload | 20 (8+2+7+3) | ~32 | over; show §2.1/2.2 after the notebook (they overlap with it anyway); make bonus presenter-run |
| 4 | 30 | ~15–18 (6 slides) | 0 | 0 | ~18 | under; absorb 07 §1.2 (LoRA) |
| 5 | 30 | ~4 + ~5 (results, during pretraining) | ~1.5 (model, embed 13k, PCA, NT npz) + ~5 (pretrain cell) | 12 (7+5) + 2 | ~28–30 | fits only if results talk overlaps the pretraining cell; fp16 untested |
| 6 | 30 | ~6 + ~6 (results + discussion) | ~1 | 11 (3+3+5) | ~25 | fits (4 min slack) |
| 7 | **25** | ~6 + ~9 (5 results slides) | ~1.5 (TODO-1 embeds 2×15k + k-mers) + ~1 (CNN ×2) + **6–8 (LoRA)** + ~2 (label curve) | 8 + 3 | **~37–40** | **over by 12–15**; see Slot 7 BLOCKER |
| 8 | 20 | ~10 + 2 feedback | 0 | 0 | ~12 | slack 8 |

The day as a whole can absorb the overrun only by moving material: 07 §1.2 → deck 04, 07 §2.3/§2.4 → deck 08 (Roee), LoRA → 1 epoch / given cell, label curve → optional.

## Story of the day

The arc holds: build a CNN (2), interrogate it (3), borrow a pretrained model (4–5), attack both with baselines and a shuffle (6), change the data and compare at equal labels (7), summarise (8). Each slot hands a concrete object to the next (`my_cnn.pt`, embeddings, `probe`), and the rejoin cells make every slot independent. The Slot 1 logistic-regression slide on (G+C, U) and the Slot 6 two-number model are the same model, which works well. The Slot 1 "fit vs held-out" slide pays off in nb05's ">0.95 means you scored the training set" check.

Where it breaks:

1. **"Composition" is undefined, and the three statements about it conflict** (06 §2.2, 07 §1.1, 08 #5). This is the scientific core of the afternoon. See Slot 6.
2. **Test sets change silently** (`ls` 1,000 in Slots 2–6; 3,000 from the train file in Slot 7; offline sets in Slot 7 figures), against an explicit Slot 1 promise.
3. **Unequal label counts drive the first verdict** (Slot 1 §4.2, Slot 5 §2.2: CNN 92k vs probes 12k). The fair comparison arrives only in Slot 7. The verdict survives, but it should be flagged when first stated.
4. **Offline vs in-notebook results.** The headline claims of Slot 7/8 ("fine-tuning wins", 0.87; the ten-model scan; the 3-draw label curve) are offline A30 runs with NT-500M/DNABERT-2. What students run gives Mistral-DNA LoRA 0.81 vs CNN 0.84 on matched data, i.e. **fine-tuning loses to the CNN in their own notebook**. The notes say "Students fine-tune the smaller Mistral-DNA (about 0.80)", but nobody tells students that their result points the other way from the slide title "Fine-tuning against the CNN". Say it on the slide: "In your notebook, Mistral-DNA LoRA 0.81 < CNN 0.84; the larger NT-500M is what beats it."
5. **Single runs.** Planting (one CNN seed, controls that depend on RNG state), the dumbbell (one split; drops of 0.036 on two different 3,000-sequence test sets, close to the ±0.02 rule), and the model scan (LoRA "one run each", HyenaDNA's drop) are single runs. The decks label some of these. The dumbbell and the planting are not labelled.
6. **Missing:** how the negatives were sampled (take-home #1 depends on it); the row-level split and the asymmetric overlap with training windows (Slot 1); length as a shortcut (Slot 6). The fine-tuned 0.87 models were never shuffle-tested; 07 §2.5 notes say so, and that should go on a slide, since take-home #5 rests on them.
7. **Redundant:** 06 §2.1 and §2.2 both report the shuffle AUROCs, and 05 §2.2 repeats 01 §4.2. Acceptable, but these are the first cuts if Slot 7 content moves.

## Language and terminology (across decks + notebooks)

| Concept | Variants found | Use |
|---|---|---|
| the 1,000 `ls` sequences | test set (decks, nb prose), held-out (decks), validation (`h_data` docstring, spike REPORT), `val_seqs`/`val_y` (nb02/03 code) | "held-out test set"; rename variables |
| intervention | insert/inserted (deck 03 titles, figures, deck 08 #4), plant/planted (nb03 code, prose, axis labels) | keep `plant()` and say "plant (insert)" once, then one word |
| models | Mistral-DNA (16M), Mistral-DNA-17M (model_scan, file `my_emb_mistral17m.npz`), Nucleotide Transformer (486M), NT-500M | first use full name + size, then Mistral-DNA / NT-500M |
| microRNA | miR-15/16, miR-16 (deck 01 notes, deck 08, deck 06 notes) | miR-15/16 |
| G+C | G+C (decks), GC (nb06 cell title "Plot GC and U content", `dinuc_shuffle` docstring "same GC") | G+C |
| reference loss | "blind guess" (fig), "uniform guess" (nb, notes) | uniform guess |
| composition | single-letter, letter-pair, word-level all called "composition" | define two levels (see Slot 6) |

AI-writing tells: visible slide text is clean (no em dashes or middle dots in `.qmd`). One en dash in nb03 l.285 ("150–190") is fine. Contrast constructions in notes: 01 l.142 "not from AGO2", 02 l.72 "not only which motifs", 06 l.72 "not longer words and not order" (which is also factually contradicted by Slot 7, see above), and 06 l.89 "Slot 7 changes the data, not the model". These are acceptable in notes, but the 06 l.72 one should be rewritten for content anyway. No hype found.

---

## The 15 most important fixes, ranked

1. **Slot 7 time (BLOCKER):** LoRA → 1 epoch × 10k, or a given cell started at the beginning of the slot; label curve optional; move 07 §1.2 to deck 04 and 07 §2.3 to deck 08. Correct "3–4 min on a T4" to the measured time.
2. **Slot 7 fp16 without GradScaler on the T4 (BLOCKER):** add `torch.amp.GradScaler` to the TODO-2 and Stuck loops (as in nb05), or run LoRA in fp32. Fix the README claim. Reset `clf` in the Stuck cell.
3. **Smoke-test nb05 and nb07 on a real Colab T4 before Thursday** (AUDIT_FIXES B5 still open). Record the real times and the pretraining/LoRA AUROCs, and quote those on 05 §2.3 and 07 §1.2.
4. **Define "composition" in two levels and reconcile 06 §2.2, 07 §1.1 (notes + nb07 intro), 08 recap and take-home #5.** Ideally add the shuffle test on the matched benchmark to nb07.
5. **State the test set per slot:** fix 01 §3.2 ("Slots 2–6"); add "new 3,000-sequence test set" to 07 §1.3 and nb07 §1; rename `val_*` → `test_*` and fix the `read_split` docstring.
6. **Flag label counts where the CNN is first compared with probes** (01 §4.2, 05 §2.2), and on the 07 dumbbell / before-after table.
7. **Tell students that their own LoRA result (Mistral-DNA 0.81) is below the CNN (0.84)** and that the 0.87 win is NT-500M offline (07 §2.2).
8. **Add how the negatives were sampled** (from the source paper) to 01 §2.2 notes and Slot 6 §2.3.
9. **Mention the row-level split and the asymmetric train/test window overlap** (47% of held-out bound vs 14% unbound within 150 nt; 5-mer probe 0.78 vs 0.70 on the two halves). Ideally report the CNN on the non-overlapping half.
10. **Add "Stuck?" cells to nb02 (TODO-2, TODO-3) and nb06 (before the scoreboard).** Add `.cpu().numpy()` to nb02 TODO-3.
11. **Deck 03: move "Your turn" before §2.1/§2.2** so the slides don't give away TODO-2's result.
12. **nb03 bonus: give it its own RNG seed, regenerate `planting_bars.png` from the same code path, and make the markdown quote what the cell prints.** Label it single-seed.
13. **nb06: rename "Shortcut #2: counting words" to a baseline**, and say that word counts can capture motifs but not position.
14. **Fix the matching description** ("same number of each class per bin; ~14,000 bound dropped") in 07 §1.1 and nb07, and drop "differ only in the negatives".
15. **Terminology pass:** held-out test set, plant/insert, Mistral-DNA (16M) everywhere including `model_scan.png`, miR-15/16, G+C, "uniform guess", one rounding for 0.725. Also correct the README's "CNN 0.861 reproducible" (the notebook trains to 0.867) and its LoRA time (85 s → 156 s).
