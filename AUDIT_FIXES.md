# Audit fixes: one by one

Two independent audits, run 2026-10-12:
- **Content audit (C):** cringe, slogans, missing intuition, factual errors. Checked against Mattan's tut decks.
- **Alignment audit (A):** do the decks and notebooks agree, and is every tool introduced before it's needed?

**Mattan's decisions (2026-10-12):**
- **Hosting:** lab SharePoint per-file links if anonymous download works, else a Hugging Face dataset repo.
- **Rejoin:** each notebook ends with "download your outputs", and the next one starts with "upload yours, or use ours".
- **Slot 7:** keep everything for now; Mattan trims later.
- **AI cells:** drop the "AI-assistant cell" approach. Write clear TODO descriptions plus hints on gotchas worth telling an AI. Run another audit after the fixes.
- **New topic 17:** how to interpret results.
- **New:** a goals slide in the spirit of tut0's "Goals: YES / NO".
- **Full-data fine-tuning:** wait. Finish what's queued, then run the KAUST notebooks.

Status: ☐ open · ◐ partly done · ☑ done · ✗ won't do (reason given)

## Factual errors (fix first)
- ☑ F1 `seed_pairing` was off by one letter → redrawn from a real bound sequence (miR-16 / UGCUGCU), checked (C)
- ☑ F2 `shuffle_demo` claimed to keep letter pairs but didn't → redrawn with the real `dinuc_shuffle` + all 16 pair counts (C)
- ☑ F3 `auroc` used synthetic scores outside [0,1] → real CNN scores on the 1,000 held-out sequences (C)
- ☑ F4 `embedding` used random vectors and averaged lookup rows → replaced by `pipeline_real` (real tokens, ids, last layer, mean) (C)
- ☑ F5 `tokens` were typed by hand → real tokenizers. Mistral-DNA and DNABERT-2 really do share one (verified) (C)
- ☑ F6 Notebook 05 said Mistral-DNA learns by "hiding letters" → it predicts the next chunk (A)
- ☑ F7 Deck 05 quoted a loss "from ~8.4" that's never visible (the curve starts at ~7) → quote what's shown, and give ln 4096 = 8.3 as the "guessing" reference (A, C)
- ☑ F8 Slot 3's dose-response is confounded with position (`plant()` writes at 40/85/130/175…) → new `position_scan` slide explains it (C, A)
- ☑ F9 `dumbbell` clipped labels; Mistral-DNA called 16M vs 17M; one palette with clashing meanings (C)
- ☑ F10 Claims contradicted by the fine-tuning results ("fine-tuned genome models don't beat the CNN", idea #5, "it isn't the model") → rewrite once the 3-seed head-to-head is in (C)
- ☑ F11 Deck 06 claimed "the bigger model… much of that was composition", which notebook 06 never tests → cut or support (A)

## Blockers for Colab
- ◐ B1 Hosting: per-file URLs (SharePoint) or one prefix (Hugging Face), via `fetch()` (A)
- ☑ B2 Rejoin: download-your-outputs cell at the end; upload-or-use-ours cell at the start (A)
- ☑ B3 `load_cnn` must fall back to our CNN on any load error. Drop the "change the number of filters" knob (A)
- ☑ B4 Notebook 05 forces the 2 GB NT download when no URL is set → load the precomputed file first (A)
- ☐ B5 Smoke-test all five notebooks on a real Colab T4, including the fp16 pretraining and LoRA paths (A)

## Notebooks
- ☑ N1 Drop the 🤖/AI markers. Every TODO gets a description, the exact names/signatures, an expected value, and "gotchas to tell your AI" (Mattan, A)
- ☑ N2 05 TODO-1 check: add padding invariance (`embed([s])` ≈ `embed([s, longer])[0]`) (A)
- ☑ N3 05/07: state that `probe_auc` returns `(auc, probe)`; give the `embed` and `kmer_counts` signatures (A)
- ☑ N4 Rescue cells where later cells depend on a TODO (05 `embed`/`probe_auc`, 07 `shortcut_scores`) (A)
- ☑ N5 03: make the `MY_MOTIF` step explicit; say the two shuffles differ (03 shuffles the motif's letters, 06 does a letter-pair shuffle) (A)
- ☑ N6 One name for the 1,000 `ls` sequences: "the held-out test set" (A)
- ☑ N7 06: expected values in the checks (A)

## Decks: cut (slogans / forced / no added value)
- ☑ D1 01: Venn "room" slide (say it), thesis text slide, why_ago2 / why_predict cards, timeline, dictionary (C, Mattan)
- ☑ D2 01: "most workshops stop at Evaluate · we don't", the hero tagline (C)
- ☑ D3 03: look_vs_test cartoon (C)
- ☑ D4 04: "DNA is not English", heads, anatomy cartoon, footer slogans (C, Mattan)
- ☑ D5 05: agent-loop cycle (C, Mattan)
- ☑ D6 06: three-attacks cards (C)
- ☑ D7 07: when_fm cards, "yours" (C)
- ☑ D8 Remove duplicate figures (network ×2, anatomy ×2) and thin out the "you are here" strips (C)

## Decks: replace with real mechanism (intuition)
- ☑ R1 01: hero = a real bound sequence + its real CNN score (C)
- ☑ R2 01: seed pairing moves here and states the rule *before* any model (C)
- ☑ R3 01: data = real sequences as pixels (`labels_raster`) (C)
- ☑ R4 01: what training does = logistic regression on (G+C, U), line moving + loss (`lr_training`) (C)
- ☑ R5 02: the real filter × window dot product (`filter_dot`), activation along a sequence (`activation_track`), the network with real shapes + last-layer weights (`network`), a filter learning (`filter_learning`) (C)
- ☑ R6 03: filter weights vs its logo (`filter_logo`), position scan (C)
- ☑ R7 04: real tokenizers, real sequence → vector pipeline, real next-token prediction (C)
- ☑ R8 05: what the vectors encode (`emb_pca`, coloured by G+C) (C)
- ☑ R9 06: score histograms under the shuffle (`shuffle_scores`) (C)
- ☑ R10 07: before/after and model comparison with frozen → fine-tuned arrows, seed spread, labelled "run offline" (C)
- ☑ R11 08: recap built from the day's real figures (C)

## Decks: missing before use (alignment)
- ☑ M1 02: ReLU in the layer list; sigmoid (score → probability); `model.eval()`/`no_grad` = measuring mode; the 4 training lines (`zero_grad`, `backward`, `step`) (A)
- ☑ M2 03: a "Your turn" slide (notebook, steps, `MY_MOTIF`, rejoin); logo y-axis in bits (A)
- ☑ M3 04: padding + attention mask, mean over real tokens, special tokens ([CLS]/[SEP]), the linear probe = logistic regression, standardising (A)
- ☑ M4 05: PCA = "a 2-D map of the vectors" (A)
- ☑ M5 07: name `batch_logits`, `crit`, `opt` on the slide; the CNN retrain on the TODO card (A)

## New content
- ☑ G1 01: Goals slide, YES / NO, tut0-style (Mattan)
- ☑ T17 Interpreting results:
  - 01 section "Reading a result": AUROC from real distributions; how big a difference must be (bootstrap ±0.025 on 1,000 sequences); fit vs held-out (`probe_trap`)
  - a "how to read your result" line on every hands-on results slide (Mattan)

## Later
- ☑ L1 Third audit after the fixes (Mattan): done; its items are listed below as A3-*
- ☐ L2 Full-data fine-tuning grid (after the KAUST notebooks)
- ☐ L3 Run the shuffle/planting tests on the fine-tuned 0.87 models before claiming they "read AGO2" (C)

## Notes on how items were closed (2026-10-12)
- **B1:** `fetch()` takes per-file links (`FILE_URLS`, for SharePoint) or one folder URL (`DATA_URL`, for Hugging Face). Still needs the actual upload and links.
- **F10:** fair 3-seed head-to-head (`scripts/spike/s10_headtohead.py`): NT-500M LoRA 0.869 vs CNN 0.835 at 20k labels; 0.750 vs 0.655 at 500. Decks 05/07/08 and idea #5 rewritten to match.
- **F8:** the position scan was added to notebook 03 as a given cell, and as deck 03 slide 2.2. Both bound and unbound sequences have the measured site at letters 150–190 (verified), so the slides now say "learned where to look", not "shortcut".
- **N1:** all 🤖 markers are gone. TODOs now say what to build, with exact names, the return type, "gotchas to tell your assistant", and an expected number.
- **N4:** "Stuck?" cells after 05 TODO-1/TODO-2 and 07 TODO-1.
- **T17:** deck 01 section 4 (AUROC from real scores · bootstrap ±0.025 · fit vs held-out), plus a "reading the result" line in the notes of every results slide and in notebooks 02, 03, 05.
- **The CNN's parameter count** is 2,491 (stated in deck 01).
- **Removed figure files** (no longer referenced): room, why_ago2, why_predict, labels (synthetic), dictionary, dogma, ml_loop, look_vs_test, dna_not_english, heads, anatomy, embedding, agent_loop, three_attacks, when_fm, yours, traps, conv_slide.

## Third audit (2026-10-12)
- ☑ A3-1 One canonical CNN: local `data/cnn.pt` differed from the HPC one (the source of the figure/notebook mismatches: window count, epoch numbers, 0.838). Replaced by the HPC model; every figure re-rendered from it (CNN 0.861 in both).
- ☑ A3-2 Notebook 05 renamed `05_pretrained_model` everywhere (src, run scripts, deck file, TODO card, README, DESIGN, agenda box).
- ☑ A3-3 Upload of `my_cnn (1).pt`-style browser renames normalised in `upload_yours()`.
- ☑ A3-4 01: "rejoin at the next cell" → "load our files"; seed footer no longer says "7 letters decide"; lr_training y-label unclipped; splits figure + notes admit the held-out set is watched every epoch; paired-comparison note on 4.2; G+C 0.42/0.46 (sample) vs 0.41/0.45 (all) both stated.
- ☑ A3-5 02: one noise figure everywhere (±0.025); learning curve = executed run; filter_dot title overlap; activation_track label clipped.
- ☑ A3-6 03: window count; filter 00 described as composition; planting claims only miR-15/16 (miR-19 within the shuffle spread), SD bars, label moved into the legend.
- ☑ A3-7 04: trad_vs_fm cartoon replaced by a two-line text slide; pipeline_real redrawn (token labels unclipped; the average is a row under the matrix); "9 models" → 10; "no labels" scoped to pretraining.
- ☑ A3-8 05: PCA notes no longer say "same cloud" / "biggest thing"; colourbar/label overlap fixed.
- ☑ A3-9 06: "a third of the way" → nearly half (0.44); the shuffle keeps letter-pair counts, so the probe reads composition, not 5-mers.
- ☑ A3-10 07: CNN line in model_scan = all ~64k matched sequences (noted); CNN-at-500 0.58 vs 0.66 explained (draw-to-draw spread 0.51–0.68); "no labels" bullet toned down; dumbbell 0.51 label off the line; notebook takeaway rewritten (fine-tuned NT-500M 0.87 vs 0.84 / 0.75 vs 0.66, offline).
- ☑ A3-11 08: recap adds planting and head-to-head panels; idea 5 shortened (no longer clipped).
- ☐ A3-12 Survey QR + form link (needs the form; Mattan)
