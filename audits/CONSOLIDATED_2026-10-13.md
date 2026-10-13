# Consolidated audit, 13.10.2026

Three independent audits: [notebooks](notebooks_audit_2026-10-13.md), [decks](decks_audit_2026-10-13.md),
[merged, decks + notebooks](merged_audit_2026-10-13.md). Below, the findings that more than one auditor raised or
that change the story, ranked. Nothing here has been applied yet.

## A. Can break on the day
1. **Slot 7 does not fit 25 minutes** (estimate ~38). LoRA took 156 s on the A30, so 6–8 min on a T4 (decks say 3–4).
   Options: 1 epoch on 10k sequences, or make LoRA a given cell started at the top of the slot; make the label-count
   cell optional; move the LoRA slide to deck 04 and the model scan to deck 08 (slot 4 has ~12 min spare, slot 8 ~8).
2. **T4 fp16 paths never run.** LoRA (nb07) uses fp16 autocast without a GradScaler; pretraining (nb05) has one but is
   untested on fp16. Add the scaler (or run LoRA in fp32), then smoke-test nb05 and nb07 on a fresh Colab T4 with the
   default reference files (also never executed: the A30 run used the student's own files).
3. **Rescue cells.** nb02 and nb06 have none; nb05's TODO-1 rescue also defines `probe_auc` (an empty TODO-2 prints a
   pass); nb07's TODO-2 rescue keeps training the half-trained model. nb02 TODO-3 must mention `.cpu().numpy()`;
   nb03 TODO-1 must show `acts[:, f, :]`.

## B. Claims and numbers that are wrong or unsupported
4. **"Composition" means different things in slots 6, 7, 8.** Matching removes single-letter composition only; the
   frozen probe still scores 0.77 and 5-letter word counts 0.725 on matched data. Fix the wording in deck 06/07/08
   and nb06/07 ("single-letter" vs "short-word" composition), drop "composition carries no information" and
   "frozen models mostly encode composition".
5. **Shuffle-test floor is ~0.66, not 0.5** (shuffled bound sequences keep their composition). The shuffle also moves
   sites out of letters 150–190, so the CNN's drop mixes order and position. Deck 06 notes, nb06 intro and §3.
6. **Fine-tuning claim.** The students' own LoRA (Mistral-DNA 0.81) is below their CNN (0.84). "Fine-tuning wins" is an
   offline NT-500M result on one test set without a shuffle test. Say so on deck 07 and 08, and in nb07's takeaway.
7. **Unequal label counts.** CNN (92k) vs probes (12k) on decks 01 §4.2, 05 §2.2, nb05; 65k vs 12k in deck 07 / nb07 §5.
   At equal labels (12k) it is 0.82 vs 0.77, so the verdict holds; state the counts.
8. **Unnamed second test set.** Slot 7 uses 3,000 sequences from the training file, not the 1,000 `ls` sequences, which
   is why its "original" numbers differ (CNN 0.84 vs 0.86, probe 0.81 vs 0.78). Name it on deck 07 §2.1 and in nb07;
   fix deck 01 §3.2 ("test set in every slot"). Rename `val_*` to `test_*`.
9. **Specific numbers:** frozen probe in the head-to-head is one run, not three seeds (deck 07 §2.2); CNN at 500 labels
   0.66 vs 0.58 on consecutive slides; planting control 0.29 ± 0.06 in the notebook vs 0.31 ± 0.09 on deck 03 (random
   state), and the miR-19/30/let-7 wording; README 0.861 vs notebook 0.867; data citation is Stražar et al. 2016
   (iDeep is Pan and Shen 2017, not 2018).
10. **Matching description.** It selects a subset of both classes (drops ~14,000 bound sequences), it does not create
    new negatives; "only the negatives differ" is not exact.
11. **Cause of the G+C gap is asserted, not shown** (decks 01, 06, 08; nb07). AGO2 sites are AU-rich 3′UTRs, so part
    may be biology. Also: **how the negatives were chosen is never stated**, though take-home 1 and the slot 6
    discussion depend on it.
12. **Science wording:** deck 04 §1.4 shows only next-token pretraining (Nucleotide Transformer is masked); nb05 intro
    the same; embeddings average includes [CLS]/[SEP]; probes used on deck 01 before slot 4 defines them; nb05 hint
    "above 0.95 means you scored the training set" is wrong (training AUROC ≈ 0.82).

## C. New finding to decide on
13. **Possible leakage in the random split** (merged auditor): 47% of held-out bound windows have a training window
    centred within 150 nt, against 14% of unbound. The 5-letter word-count probe scores 0.78 on the overlapping half
    and 0.70 on the rest. Could be leakage or gene-level confounding; untested on the CNN. Related (deck auditor):
    word counts inside the 40-letter site alone reach 0.83, close to the CNN's 0.86, so what the CNN adds may be mostly
    where it looks. Both matter for what the day claims.

## D. Decks: content and layout
14. Slot 8: two content slides for 20 minutes; recap thumbnails unreadable (replace with a slot / finding / number
    table); take-home list reads as slogans; `notes_for_roee_slot8.md` is stale; consider moving deck 07 §2.5 there.
15. "Your turn" slides: put function names, expected numbers and the main pitfall on the slide, not only in notes;
    evaluation code on deck 02 §1.8; masked mean and U→T on deck 05 §1.1. Deck 03 shows the TODO-2 result before
    "Your turn". Slot 6 discussion questions only in notes. The "experiment with an AI assistant" goal is never
    delivered.
16. Colour rules: blue still used for non-bound/non-fine-tuned things on ~7 slides; PCA G+C scale clashes with
    "Bound"; off-palette green/red on the Goals slides; small tick labels (13–15 px); legend overlap on deck 06 §2.1.
17. Chrome: one-item tables of contents (decks 02, 04, 08), menu offset, pale slide numbers, hamburger icon under the
    logo.

## Timing estimate (T4 ≈ 2.5 × A30; minutes, budget / estimate)
slot 1 30/~29 · slot 2 30/~31–32 · slot 3 30/~32 · slot 4 30/~18 · slot 5 30/~28–30 · slot 6 30/~25 ·
slot 7 25/~37–40 · slot 8 20/~12
