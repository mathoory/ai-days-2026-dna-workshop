# Deck audit, 13.10.2026 (independent auditor, report only)

Scope: `slides/0*.qmd` (visible text and notes), all 89 rendered slides in `shots3/`, figures in `slides/assets/custom/`, numbers cross-checked against the brief and `scripts/spike/out/*.json`. Slides that show only on a fragment (01/05, 07/11) were judged from source.

---

## 01 Opening and basics (Mattan, Roee)

**01 Table of contents**
- MINOR. Four links in the top 40% of the slide, the rest empty. Menu bar names ("ML", "Results") do not match the section titles ("Learning and testing", "Reading a result"). Fix: drop the TOC slide (the agenda at 1.6 does that job) or rename `data-name` to match the titles ("Learning", "Reading results").

**03 1.1 Teaching machines to read DNA**
- MAJOR (accuracy). "A CNN trained on these sequences gives this one a score of 0.82" sits above 28 letters. The note says the CNN reads 330 letters, so the slide implies that the 28 letters were scored. Fix: "gives the 330-letter sequence this stretch comes from a score of 0.82".
- MAJOR (figure). The blue UGCUGCU in the hero answers the question the slide asks, and nothing in the workshop shows that this one site drives the 0.82. Slot 3 shows an effect only for 6 inserted copies on unbound backgrounds, and the CNN responds mainly to letters 150–190. Fix: show all letters in ink and reveal the site in 2.1, or keep the blue and call it "the hypothesis" in the notes. Optional: delete the site in silico and report the score change.
- MINOR. The title says DNA and the letters are RNA (U). Fix: one line in the notes ("the data are RNA; the models read DNA letters, U becomes T").
- MINOR (layout). The figure is 400 px tall and leaves about 180 px of white space below it. Fix: height 520 px.

**04 1.2 About me**
- MINOR (layout). The photo runs to y≈886 and its left edge touches the logo at x≈107. Fix: set the photo height to about 620 px, or move the column right.
- MINOR. Roee has no intro slide, although he presents three slots. Fix: "About us" with two lines.
- MINOR (notes). "QA, then developer, then a B.Sc." is a triplet for rhythm. Cut it to "B.Sc. in CS (bioinformatics), now M.Sc. in Roee's lab".

**05 1.3 Goals: YES / 06 1.4 Goals: NO**
- MAJOR. The goal "**Experiment** with an AI assistant as a coding partner" is not delivered: no deck has an AI-assistant step. Only the 1.4 note says "Use an AI assistant if you get stuck". Fix: cut the goal, or add an explicit assistant task to one notebook and its "Your turn" slide.
- MINOR (style). The green #188038 and red #D93025 fall outside the ink/grey/accent palette. Fix: plain ink "Goals" with two columns "We will" / "We will not" on one slide. That also removes slide 06, which is 70% empty.
- MINOR. Only the first NO bullet has a bold lead ("**Code** from scratch"). Fix: bold all three leads or none.
- MINOR (notes). "The aim is intuition. The code is how we get there." is a slogan. Cut it.

**07 1.5 What you need**
- MINOR. The URL breaks across lines ("ai-days-\n2026-dna-workshop"). Fix: shorten the text column or set the URL in its own `<span style="white-space:nowrap">`.

**08 1.6 Today**
- MINOR. The "Who" column mixes names and "hands-on". Fix: put "Mattan" in each hands-on row and use a separate "Type" column, or drop the column.
- MINOR (notes). "The arc: build…, look…, borrow…, try…, then fix" is a colon-reveal and a list for rhythm. Fix: "We build a CNN, inspect it, compare it with pretrained models, test both for shortcuts and then change the data."

**10 2.1 A seven-letter address**
- MINOR (science). "Its **seed** (letters 2 to 8)": the canonical seed is nucleotides 2–7, and position 8 extends it to a 7mer-m8 site. Fix: "Its seed (letters 2 to 7, often with 8) pairs…".
- MINOR (figure). The miRNA strand is labelled 5′ on the right but has no 3′ on the left. Fix: add 3′.
- MINOR. The title is a forced analogy ("address"). Fix: "2.1 The seed rule", which matches the name 08/1.1 uses.
- MINOR (notes). The "Why it matters:" and "Why predict what we can measure:" leads are colon-reveals. Fix: turn them into plain sentences.

**11 2.2 The data**
- MAJOR (science, notes). "That difference comes from how the negatives were picked, not from AGO2." This is stated as fact without support, and it is also a "not X but Y" construction. AGO2 sites sit in AU-rich 3′UTRs, so part of the difference may be biology. Fix: "Part of this difference may come from how the negatives were picked; Slot 7 tests that."
- MINOR (notes). "each pixel one letter, coloured by base" is wrong: the figure is binary ("G or C" / "A or U"). Fix the note.
- MINOR (notes). "the measured 40-letter site in the middle (letters 150 to 190)" cannot hold for every sequence, because lengths run from 137 to 375 (s1_shortcuts). Fix: "in almost every sequence".
- MINOR. "Unbound: about 44,000 chosen by the authors": say which authors (the data set authors, Stražar et al. 2016) and say how in one clause.
- MINOR (figure). The class difference, which is the point of the slide, cannot be seen in the raster. The white stripes on the right of the bound panel are sequence ends (length), which nobody explains. Fix: say in the notes that the stripes are sequence ends, or replace the raster with the G+C histogram from 06/03.

**13 3.1 What training does**
- MINOR (readability). Tick labels are about 14 px and the points are 2 to 3 px, so they cannot be read from the back of the room. The figure uses only about 60% of the width. Fix: drop the step-10 panel and enlarge the other three (ticks ≥ 20 px).
- MINOR (notes). "two numbers carry a little information, not much" is a colon-reveal plus a contrast. Fix: "The two numbers carry some information: the loss falls from 0.69 to 0.67."

**14 3.2 Train, validate, test**
- MAJOR (accuracy). "Today the 1,000 held-out sequences (the `ls` file) are the test set in every slot" is false. Slot 7 uses 3,000-sequence test sets on two new benchmarks, and the head-to-head and the label curve use other splits. Fix: "Slots 2 to 6 use the same 1,000 held-out sequences; Slot 7 builds new test sets."
- MINOR. "`ls` file" is internal jargon. Fix: drop it from the slide.
- MINOR (layout). Text only, with the bottom 55% empty, while `splits.png` exists in assets. Fix: add it.
- MINOR (notes). "An honest caveat:" is filler. Fix: start at "Slot 2 prints…".

**16 4.1 AUROC**
- MINOR (notes). "bound in green" is wrong: bound is blue. Fix the note.
- MINOR (figure). The "Bound" label collides with the tallest bar, and "Unbound" sits on the grey bars. Fix: move both labels outside the bars, or into a legend above the plot.

**17 4.2 How big is a difference?**
- MAJOR (order). The slide compares "Mistral-DNA + probe" and "Nucleotide Transformer + probe" 3.5 hours before probes and embeddings are defined (Slot 4). Fix: compare the CNN at 10 epochs, the CNN at 1 epoch and the G+C/U model, or add a one-line definition. Same problem at 18.
- MINOR. "moves by about ±0.025 by chance" should say what ±0.025 is. Fix: "95% of resampled AUROCs fall within ±0.025".
- MINOR (title). A rhetorical question. Fix: "4.2 Noise in a test-set AUROC".

**18 4.3 Fit vs held-out**
- MAJOR (wording). "Only the held-out number means anything" is a slogan and overstated: the training score diagnoses overfitting, as this slide shows. Fix: "Report the held-out score; a large gap to the training score means memorisation."
- MINOR (style). Blue (the accent) marks "held-out", which breaks the rule that blue means bound or fine-tuned. Fix: training = light grey, held-out = ink.
- MINOR. The figure uses a probe (undefined at this point). Fix: two CNNs, or logistic regression on 5-letter counts with 600 vs 12,000 sequences.

---

## 02 Build a CNN (Mattan)

**01 Table of contents / 02 "1 Build"**
- MAJOR (layout, all one-section decks: 02, 04, 08). The TOC has one link and the divider has one section. That is two empty slides out of eleven, and the menu bar holds a single item. Fix: set `toc: false` for these decks and drop the divider, or split the deck into "Encode / Build / Train" sections that carry content.

**03 1.1 One-hot encoding**
- MINOR. "A sequence of length L becomes a 4 × L matrix; we pad to L = 501" uses L for two different lengths, and 501 is unexplained. Fix: "…a 4 × L matrix. All sequences are padded to 501 columns (the iDeep input size; the longest is 375) with columns of 0.25."

**04 1.2 A filter computes a dot product**
- MAJOR (readability). The "Weight × input" values (+0.15+0.08+0.13…) are about 12 px and run into each other, and the weight heatmap has no colour scale, so the sign cannot be read. Fix: print the 10 contributing weights as one row of numbers under the window at ≥ 22 px, and add a small colour bar (grey −, blue +).
- MINOR. The input-window cells are dark grey instead of ink. Use ink.
- MINOR (notes). "only 1.07 times larger … which matters in Slot 6" is a vague promise. Fix: say what it implies, or cut it.

**05 1.3 The filter along one sequence**
- MINOR (figure). Three bars are blue and the rest grey, but several grey bars (0.81 at ~55, 0.80 at ~265) are as tall as a blue one (0.86), so the reason for the blue is unclear. Fix: colour only the maximum, or colour all bars above a stated cutoff and draw the cutoff line.

**06 1.4 The network (TODO-2)**
- MINOR (layout). The table takes 45% of the width and the lower 40% is empty. `network.png` exists. Fix: put the diagram on the right.
- MINOR (style). Inline code uses the Quarto default purple-blue (#4758AB), a second blue next to the accent. Fix: set `.reveal code { color: var(--ink) }`. This affects all decks.

**07 1.5 The last layer weighs positions**
- MINOR (notes). "columns 15 to 20, letters roughly 135 to 180" does not match the mapping. Pooled column j covers letters ≈ 9j to 9j+29, and the strongest columns (17 to 20) map to ≈ 150–210. Fix: "letters roughly 150 to 210".
- MINOR (figure). Columns ≥ 41 (letters > ~370) show flat bands: these are padding columns. Nobody explains them, and a student will ask. Fix: shade them as padding in the figure, or cover them in one note line.
- MINOR. Blue for positive weights clashes with blue = bound. Accepted for diverging maps elsewhere, but say "blue = positive" with a labelled colour bar (there is one here; the other weight maps lack it).

**08 1.6 Training changes the filters**
- MINOR. Which filter is shown is not said, and there is no colour scale. Fix: panel title "Filter 7" (or whichever it is) and a shared colour bar.
- MAJOR (Req. 5). TODO-3 asks for an evaluation loop, but the needed calls appear only in the notes: `model.eval()`, `torch.no_grad()`, `torch.sigmoid`, `.cpu().numpy()`, `roc_auc_score`. The notebook audit found that omitting `.cpu()` crashes on a T4. Fix: put the four training lines and the three evaluation lines on this slide as a code block.

**09 1.7 Your turn**
- MAJOR (Req. 5). It has no function or variable names, nothing on rejoining if TODO-2 fails, and no mention of the `my_cnn.pt` download that Slot 3 needs. Fix: add an "Expect / check" column (TODO-1: first columns match letters; TODO-2: 2,491 parameters; TODO-3: about 0.79 after epoch 1) and one line: "Last cell saves `my_cnn.pt` for Slot 3."
- MINOR (layout). The table fills 50% of the width and 45% of the height. Fix: enlarge it or add the expect column.

**10 1.8 Training curve**
- MINOR. Only one seed is shown, while the notes cite three seeds within 0.01. Fix: draw all three runs as thin lines.

---

## 03 Look inside (Mattan)

**01 Table of contents**
- MINOR. The menu reads "Look / Test / microRNA" and the sections read "Look / Test / Back to the seed". Fix: `data-name="Seed sites"`.

**03 1.1 From a filter to a sequence logo**
- MAJOR (Req. 5). The helper names `cutoff_for`, `windows_above`, `pwm_of`, `consensus_of` and the cutoff rule ("high" = ?) are only in the notes. Fix: one line on the slide: "`cutoff_for(f)` → `windows_above(f, cut)` → `pwm_of(windows)` → logo; cutoff = …".
- MINOR. The panel title "…windows where it fires" uses jargon. Fix: "…windows with the highest output".
- MINOR. The right panel duplicates the left panel of 1.3. Fix: drop the logo here and show the weights plus the window-collection step, or merge 1.1 and 1.3.

**04 1.2 All ten filters**
- MINOR (notes). "two G/C-rich ones (filters 3 and 9)" undercounts: filters 4, 5 and 6 have tall G and filter 8 a tall C. Fix: "several have one dominant G or C (4, 5, 6, 8, 9)".
- MINOR. The axis ticks are about 13 px. Fix: drop the y ticks and keep one "2 bits" reference.

**05 1.3 The same filter on unbound sequences**
- MINOR (science, notes). "not specific to AGO2-bound sequences" does not follow. Identical logos show what the top windows look like, not how often they occur in each class. Fix: "The windows it responds to look the same in both classes; whether they are more frequent in bound sequences needs a count."

**07 2.1 Insert the motif, against a shuffled control (TODO-2)**
- MINOR (figure). The legend is in Roboto Mono, while every other figure legend uses Roboto. The y axis is truncated at 0.40. The fall from 0.72 to 0.60 at 5 to 8 copies is not explained anywhere. Fix: Roboto legend, y from 0.3, one note line on the decline.
- MINOR. The x axis "Copies inserted" hides the confound the notes explain (copy 4 lands at letter 175). Fix: label the x ticks with the insertion letter ("4 @175").
- MINOR. The title is long. Fix: "2.1 Insert the motif (TODO-2)".

**08 2.2 Position matters**
- MINOR (figure). Seven unlabelled black ticks on the x axis mark where `plant()` puts copies 1 to 7; nobody can read that from the slide. Fix: label them 1–7. The shuffled sequence here (CUAACCUCUA) differs from the one in 2.1 (AUCCACAUUC). Fix: use the same control in both.
- MINOR (notes). "the region where every sequence in the data has its measured site": change "every" to "almost every" (see 01/11).
- MINOR (notes). "It is not an error in the model. It does mean that…" reads like the writer arguing with itself. Fix: "This is how the data were built, and it changes the answer to 'does this motif matter?'."

**09 2.3 Your turn**
- MAJOR (Req. 5). The slide does not name `plant()`, the shuffle helper, `MY_FILTER`, or what to do when a filter's bound and unbound logos match (the notebook audit says filter 00 fails its own control). Fix: add "Expect" and "Names" columns (`plant(seq, motif, n)`; expected jump at 4 copies).

**11 3.1 Does the CNN respond to seed sites?**
- MAJOR (numbers). The slide and notes give a shuffled control of "0.31 ± 0.09". The notebook output in the notebook audit gives 0.290 ± 0.057, and the brief gives 0.29–0.31. The spread differs by 60%, and "a clear effect" depends on it. Fix: regenerate the figure from the reference-CNN notebook path and quote that number.
- MINOR (style). miR-15/16 is blue and the other site bars are ink: one series, two colours, and ink means CNN elsewhere. Fix: all site bars ink, with an annotation or bold label for miR-15/16. Bars on a y axis truncated at 0.20 exaggerate the effect; start at 0.
- MINOR (notes). "3 to 6 times enriched … occur in only 1 to 2%" holds only inside the 40-letter site (s1 `ratio_core`, `pos_core`; let-7 is 0.8%). Over the full sequence it is 1.2 to 2.3 times. Fix: "in the 40-letter site, 3 to 6 times enriched, in about 1 to 2% of bound sequences".
- MINOR (notes). "The logos did not show it; the experiment did." is a contrast slogan. "A clear effect." is a fragment. Fix: "The logos did not reveal this site; the insertion test did."

---

## 04 Foundation models (Roee)

**01 Table of contents / 02 divider**
- MAJOR (layout). One link and one section ("Pretrained"). Same fix as 02/01.

**03 1.1 Pretrain once, reuse many times**
- MAJOR (layout). This text-only slide opens a theory slot with the lower 62% empty, while `pretraining.png`, `adapt.png` and `trad_vs_fm.png` exist. Fix: add one two-stage diagram (genome → pretrained model → small labelled set → head).
- MINOR. "The claim to test:" is a colon-reveal bold lead. Fix: "**Claim**: what it learns transfers…".
- MINOR (notes). "Until now every task … trained a model from random weights" is overgeneral. Fix: "Most tasks…". "Drafted by Mattan, adapt freely." is meta-talk in the notes. Move it out of the deck.

**04 1.2 Tokenization: where does a word end?**
- MINOR. The "Same, shifted by one" row does not name the tokenizer. Fix: "Nucleotide Transformer, shifted by one".
- MINOR. The Mistral-DNA and DNABERT-2 rows are identical. Fix: merge them into one row, "Mistral-DNA, DNABERT-2 (BPE)", and use the space to enlarge the letters.
- MINOR. The blue letters (miR-19 site, in DNA letters TTTGCAC) are not explained on the slide, and the title is a rhetorical question. Fix: "1.2 Each model cuts DNA into different tokens".

**05 1.3 From tokens to an embedding**
- MAJOR (accuracy). "averaged over the real (non-padding) tokens" is not what the code does: the average includes [CLS] and [SEP] (notebook audit, `figs.py` mask). Fix: "averaged over all non-padding tokens, including [CLS] and [SEP]".
- MAJOR (readability). The 26 token labels on the y axis are about 9 px. Fix: label 5 tokens at ≥ 18 px, or drop the token labels and keep "26 tokens".
- MINOR. A few always-high dimensions (vertical blue stripes) are not explained. Fix: add one note line ("a few dimensions are large for every token, common in transformers") or clip the colour scale.

**06 1.4 The pretraining task**
- MAJOR (science). Only the next-token loss is on the slide. Nucleotide Transformer, the 0.79 model in Slot 5 and the 0.87 fine-tuned model in Slot 7, is masked. Masked prediction appears only in the notes. Fix: two equations, autoregressive $-\log p(x_t \mid x_{<t})$ and masked $-\log p(x_t \mid x_{\setminus t})$, each naming its models. `s6_mlm.json` has a real NT masked-token example to use.
- MINOR (figure). "(not in the top 8)" is a caption phrase inside the figure, in monospaced blue. Fix: drop it and let the speaker say it, or plot the true token as the ninth bar.

**07 1.5 The landscape, 2021 to 2025**
- MINOR (notes). "Slot 7 shows ten of these models": the ten are mostly NT and HyenaDNA variants, not ten of the points shown. Fix: "ten models from these families".

**08 1.6 Four ways to adapt a pretrained model**
- MINOR (accuracy). The probe's cost of "seconds" ignores the forward pass needed to embed (about 20 s to 2 min in s8). LoRA at "1 to 10% of parameters" is high for typical setups (0.1–1%); this workshop measured 0.9–8.5% (s9_rerun). Fix: "a new output layer only (after one embedding pass)"; "about 1% of parameters here".
- MINOR (layout). The table fills 55% of the height. "Partial fine-tuning" is never used and could be cut.
- MINOR (deck-level). This 30-minute theory slot has six content slides and nothing on why transfer should help AGO2 (what the genome teaches a model about UTRs). Fix: add one slide linking pretraining to this task, or move 07/2.5 "When to use a pretrained model" here as a preview.

---

## 05 A pretrained DNA model (Mattan)

**03 1.1 Loading a model from Hugging Face**
- MAJOR (Req. 5). The code stops at `hidden_states[-1]` (one vector per token). The step TODO-1 needs, the masked mean, is not shown, and neither is the U→T conversion. Fix: add two lines:
  `m = ids["attention_mask"].unsqueeze(-1)` and `vec = (vecs * m).sum(1) / m.sum(1)`, plus `seq.replace("U", "T")`.
- MINOR (layout). The lower 45% is empty. The code could be 28 px.

**04 1.2 Your turn**
- MAJOR (Req. 5). It has no expected numbers (Slot 6 has an "Expect" column, this deck does not). `probe_auc()` returning `(auc, probe)`, standardisation, `C=0.1` and the 12,000-sequence training subset are only in the notes. Fix: an "Expect" column (TODO-1: shape (N, 256) and the batch-invariance check passes; TODO-2: about 0.78) and one line of signatures.
- MINOR (layout). The Step column takes 30% of the width while the Task text wraps. Fix: `tbl-colwidths="[15,65,20]"`.

**06 2.1 What the embeddings vary with**
- MAJOR (style). The G+C colour scale is blue, the same hue as Bound in the right panel, so readers will map dark blue to bound. Fix: a grey-to-ink sequential map for G+C.
- MINOR (figure). Bound points are drawn over unbound. Fix: plot in random order, or smaller and semi-transparent.
- MINOR (notes). "The first principal component correlates with G+C at r = 0.52" is the weaker part of the story: in the left panel the G+C gradient runs along the upper arm (mostly PC2). Fix: give r for both PCs, or say "G+C changes smoothly across the map".

**07 2.2 Frozen pretrained models against the CNN**
- MAJOR (fairness). The CNN (0.86) was trained on about 92,000 labels and the probes on 12,000, and the slide does not say so. The notes go further: "the pretrained models lose to a 2,491-parameter model trained on the task". At 12,000 labels the gap is smaller (Slot 7 label curve: CNN 0.82 vs probe 0.77). Fix: add label counts to the bar labels ("CNN, 92k labels"; "probe, 12k labels") and point to 07/2.4.
- MINOR (style). "Pretrained 3 min (live cell)" is in accent blue, which means fine-tuned elsewhere. It is a frozen probe. Fix: mid grey with an ink outline, or order the bars by AUROC and keep blue out of this figure.
- MINOR. "Random weights" does not say which model. Fix: "105M model, random weights".

**08 2.3 Pretraining from random weights**
- MINOR (science, notes). The 8.3 → 6.4 drop mostly reflects learning token frequencies; a unigram model would already sit far below 8.3. Fix: draw the unigram-entropy line on the figure, or say it. Also, "The released model trained far longer and reaches 0.78" compares two different architectures (105M MoE vs 16M). Fix: "a different, longer-trained model reaches 0.78".
- MINOR (figure). "Blind guess among 4,096 tokens" → "uniform guess (ln 4096)". The loss line in accent blue has no class meaning here. Use ink.
- MINOR (notes). The 0.72 after 3 minutes is not far above G+C/U alone (0.66). Say so before Slot 6, or it reads as learned biology.

---

## 06 Break it (Mattan)

**03 1.1 The two classes differ in composition**
- MINOR (figure). The "Bound mean 0.41" label is drawn over the blue bars. The overlap of the two alpha histograms creates a third colour. Fix: step outlines for one class, label outside the bars.
- MINOR (notes). "Three tests, each one a model that should do poorly…": the shuffle test is not a model. Fix: "two simple models and one stress test".

**04 1.2 The dinucleotide shuffle test**
- MAJOR (science, notes). "If it falls towards 0.5, it relies on the order of letters" is wrong. Shuffled bound sequences keep their composition, so the floor is the composition score (about 0.66), not 0.5. Fix: "If it falls to the level of the composition-only model (0.66), the model uses nothing beyond letter and letter-pair counts."
- MINOR (accuracy). "destroys longer patterns" overstates it: the shuffle randomises longer patterns subject to the pair counts. It also moves sites out of letters 150–190, which matters for a position-weighting CNN. Fix: "keeps every letter and letter-pair count, scrambles everything longer, including where sites sit".
- MINOR (figure). Sixteen pairs of identical bars spend half the slide showing "equal". Fix: keep the two sequences, shrink the bar chart to half height, or replace it with "16 of 16 pair counts identical".

**05 1.3 Your turn**
- MAJOR (Req. 5). No helper names (k-mer counter, `dinuc_shuffle`, the Slot 5 probe), and the instruction to reuse the fitted probe, the most likely mistake, is only in the notes. Fix: add "Use the probe fitted on real sequences; do not refit" and the function names to the slide. Add a Time column, as in the other decks.

**07 2.1 Scores before and after shuffling**
- MAJOR (figure). The legend box overlaps the tall unbound spike at score 0 in the right panel. The spike itself, a block of unbound sequences scored about 0 by the probe, is not explained and will draw questions. Fix: move the legend to the upper right of the left panel, and find out what those sequences are (likely the high-G+C arm in 05/2.1) and say it in the notes.
- MINOR. The two-line panel titles carry the result ("shuffled vs unbound: AUROC 0.66"), which is fine. The y axes are hidden, and the panels may differ in scale. State "density, per panel".

**08 2.2 Summary**
- MINOR (title). "Summary" carries no information. Fix: "2.2 Shuffling lowers the CNN, barely moves the probe".
- MINOR (style). The G+C/U line is dashed accent blue, but `deckstyle.SHORTCUT_C = MID`, so the figure breaks its own style constant. Fix: mid grey.
- MINOR. The "Shuffled" bars mean "shuffled bound vs real unbound", which can be misread as a model trained on shuffled data. Fix: bar label "bound shuffled".
- MAJOR (science, notes). "Without letter order the CNN retains only composition information" overstates it. The shuffle also moves the site out of 150–190, and the CNN weights positions (02/1.5, 03/2.2), so the drop mixes order and position. Fix: "After the shuffle the CNN scores like the composition-only model; the shuffle removes both letter order and site position." Also, "nearly everything the probe uses is composition at that level, not longer words and not order" is a "not X and not Y" construction. Fix: "The probe's score survives the shuffle, so most of what it uses is letter and letter-pair composition."

**09 2.3 Discussion**
- MAJOR (Req. 5). The three discussion questions students are meant to discuss in pairs exist only in the notes. The slide shows one rhetorical question on an empty page. Fix: put the three questions on the slide.
- MAJOR (science, notes). Q3 asserts "Bound and unbound sequences differ in G+C because of how the negatives were chosen", which is the unproven claim from 01/11, presented as a premise. Fix: "Could the G+C difference come from how the negatives were chosen? How would you test that?" That question sets up Slot 7.
- MINOR (notes). "Slot 7 changes the data, not the model." is a "not X but Y" slogan. Fix: "Slot 7 keeps the models and changes the negatives."

---

## 07 Fix it (Mattan)

**03 1.1 Composition-matched negatives**
- MAJOR (science, notes). "In the matched set, composition carries no information about the label, so any score above 0.5 must come from something else" is false. Matching uses G+C, U and length only. The G vs C split and all letter-pair counts are unmatched, and word counts still score 0.73 on the matched set. Fix: "In the matched set, G+C, U and length carry no information; letter-pair and longer composition still can."
- MINOR. Matching discards 30% of the bound sequences (pos_kept_frac 0.70), and the ones kept are those that look like the negatives. Fix: add one note line on what this does to the positives.

**04 1.2 LoRA: fine-tuning with few trainable weights**
- MINOR. The rank is never named. Fix: "rank r = 16" next to the equation.
- MINOR (figure). The 256 × 256 W texture carries only "large". Fix: shrink it to 60% and enlarge B and A, which carry the message (B is all zeros at the start).

**05 1.3 Your turn**
- MAJOR (Req. 5). No expected numbers (the notes have "about 0.51 on matched data"), no benchmark variable names, and `batch_logits`, `crit`, `opt` appear only in the notes. Fix: an "Expect" column (TODO-1: 0.64/0.51, 0.76/0.73, 0.81/0.77; TODO-2: about 0.80 after 2 passes) and one line: "TODO-2: the four lines from Slot 2 with `batch_logits`, `crit`, `opt`."
- MINOR (timing). The tasks total 17.5 minutes plus 3 to 4 minutes of LoRA training inside a 25-minute slot that also holds 6 slides. Cut either 2.3 or 2.4 from the live talk, or make one of them a "look at home" slide.

**07 2.1 Before and after matching**
- MAJOR (numbers). The original-negative values here (probe 0.81, word counts 0.76, G+C/U 0.64, CNN 0.84) differ from the numbers students saw an hour earlier (0.78, 0.74, 0.66, 0.86), and the slide does not say why (a different 3,000-sequence test set). Fix: x label "Held-out AUROC, new 3,000-sequence test set (not the Slot 2–6 set)".
- MINOR (cross-check). The CNN is shown as 0.84 → 0.84, while `s5_fix_build…json` has 0.863 → 0.849. Confirm which run the figure uses and say so in the notes.
- MINOR (style). The G+C/U row is accent blue (shortcut colour should be mid grey). Fix: mid grey.
- MINOR (notes). "noise is about ±0.015" vs 01/4.2 "below about 0.02": pick one (±0.015, 95% interval).

**08 2.2 Fine-tuning against the CNN**
- MAJOR (accuracy, notes). "three seeds each" and "The seeds (dots) do not overlap" are wrong for the frozen probe, which has one run (s10: `frozen_probe.runs` has length 1, sd 0). At 500 labels the frozen point (0.733) lies 0.007 below the lowest LoRA seed (0.740). Fix: "CNN and LoRA: three seeds; frozen probe: one run. LoRA and CNN seeds do not overlap."
- MINOR (style). Frozen is light grey here and mid grey in 2.3 and 05/2.2. Fix: mid grey.
- MINOR (notes). "Run offline on our A30 server with Nucleotide Transformer 500M": name it as in 2.3 ("NT-500M, human reference").

**09 2.3 Does model choice matter?**
- MINOR (notes). "trained on all of roughly 64,000 matched training sequences": s5 gives a matched train size of 60,988. Fix: "about 61,000".
- MINOR. The labels "NT-500M" and "NT-v2-500M" sit side by side with no explanation. Fix: "NT-500M (human ref.)" vs "NT-v2-500M (multi-species)".
- MINOR (title). A rhetorical question. Fix: "2.3 Ten models, frozen and fine-tuned". The notes line "One recipe does not suit every architecture; fine-tuning needs its own tuning." is a slogan. Keep only the facts (HyenaDNA 0.78 → 0.72).

**10 2.4 How many labels does each model need?**
- MAJOR (consistency). On consecutive slides the CNN at 500 labels is 0.66 (2.2) and 0.58 (2.4), and the audience sees both. Fix: show the same draws on both slides, or put the spread on the slide ("3 draws: 0.51–0.68").
- MINOR (figure). The curves are Mistral-DNA (frozen) but the diamonds are NT-500M (fine-tuned), so model and adaptation change at once. The "NT-500M fine-tuned" label floats at the top centre, far from both diamonds. "Random weights" does not say which model. Fix: label each diamond directly, and "Mistral-DNA, random weights".
- MINOR (notes). "Which model is best depends on how many labels you have." and "With little data, simple features do well." are slogans. Cut them.

**11 2.5 When to use a pretrained model**
- MAJOR (evidence). "**No labels yet**: rank variants by how unexpected the model finds them" was never tested here. Zero-shot likelihood scoring is weak for regulatory variants in the very paper the deck cites (Tang and Koo 2024). Fix: cut the bullet, or add "(not tested today; mixed evidence)".
- MINOR. The three bullets are a triplet with colon leads, and the first bullet is itself a triplet ("a new cell type, a rare condition, a pilot experiment"). Fix: two bullets with numbers from 2.4 ("below about 2,500 labels: frozen probe; with 20,000: fine-tune, keep the CNN as baseline").
- MINOR (structure). This is Roee's wrap-up material placed in Mattan's slot. Fix: move it to Slot 8.
- MINOR (notes). "'Labels are cheap now' holds only…" argues with a quote nobody made. Cut it.

---

## 08 What we saw (Roee, Mattan)

**01 / 02** – MAJOR (layout). A one-link TOC and a one-section divider, as in 02/01.

**03 1.1 What we saw**
- MAJOR (readability). Eight thumbnails of earlier figures with tick text of about 5 px: nothing on the slide can be read, and the panel titles sit at different heights (y = 171–204, 524–543). Fix: replace the grid with an eight-row table: slot | finding | number (seed rule; filter = dot product; miR-15/16 0.53 vs 0.29–0.31; position 150–190; PC1 vs G+C r = 0.52; shuffle CNN 0.86 → 0.66; matched G+C/U 0.64 → 0.51; LoRA 0.87 vs CNN 0.84).
- MAJOR (overclaim). The panel title "Fine-tuning wins" is a slogan. The evidence is one benchmark, one test set, NT-500M and DNABERT-2 only (HyenaDNA lost), with no shuffle test (07/2.5 notes say so). Fix: "Fine-tuned NT-500M: 0.87 vs CNN 0.84".
- MINOR. "The CNN responds to miR-16" vs "miR-15/16" in Slot 3. Use one name.

**04 1.2 Five points to take home**
- MAJOR (Req. 1). All five bullets use the same "statement; **bold reveal**" pattern. "Test a model like a sample: insert, shuffle, match" is a forced analogy, a colon-reveal and a triplet. "DNA has no words" is a slogan, and the data are RNA. "the negatives were chosen by a person" is rhetorical (they were chosen by a sampling procedure). "Models use the easiest signal available" anthropomorphises. Fix: plain sentences with numbers, e.g. "The negatives came from a sampling procedure; G+C and U alone gave 0.66 on them."
- MAJOR (overclaim). "frozen, they mostly encode composition" is contradicted by the matched benchmark: the frozen probe still scores 0.77 when G+C/U scores 0.51. What the shuffle shows is that the probe's discriminative signal survives letter-pair-preserving shuffles. Fix: "Frozen, their AUROC barely changed when letter order was destroyed; fine-tuned NT-500M reached 0.87 vs the CNN's 0.84."
- MINOR (notes). "Training takes four lines of code. Whether the model learned anything real depends on how you test it." is a slogan. Fine as a spoken close, but cut it from the notes, or keep it and drop the five bold reveals.

**05 1.3 Feedback**
- MINOR. "what worked, what did not, what was missing" is a triplet. The repo URL breaks mid-word. Fix: "What should change for your background?" and a nowrap URL.

**06 1.4 Credits and further reading**
- MAJOR (attribution). "Pan and Shen (2018) iDeep: the AGO2 CLIP-seq data": iDeep is Pan and Shen 2017 (BMC Bioinformatics), and the benchmark data come from Stražar et al. 2016 (iONMF, Bioinformatics), built on published AGO2 CLIP experiments. Fix: cite Stražar et al. 2016 for the data, and Pan and Shen 2017 for iDeep.
- MINOR. Missing: Altschul and Erickson 1985 (dinucleotide shuffle, named in 06 notes), the Mistral-DNA citation, Evo/Evo 2 (shown in 04/1.5) and logomaker. Add them.

**Deck-level**
- MAJOR. This is a 20-minute slot for the senior presenter with two content slides, one of them unreadable. `notes_for_roee_slot8.md`, the brief Roee may read, is stale and contradicts the decks. It says that with plenty of labels the CNN still wins (0.84 vs 0.81), that with 100 labels the pretrained model wins (0.68 vs 0.54), and that an AI assistant built the pretrained model in Slot 5. Fix: rewrite the brief from the current numbers (LoRA 0.87 vs CNN 0.84 at 20k; 0.75 vs 0.66 at 500), move 07/2.5 here, and add one slide on what to do with a new question (choose negatives, baseline, shuffle, match), which would justify the file name `question_to_model`.

---

## Cross-deck

**Numbers**
- 0.86 (CNN, 1,000 ls) ✓ 01/17, 02/10, 05/07, 06/08. 0.78 / 0.79 probes ✓ (s8: 0.782, 0.788 NT-500M human ref). 0.66 / 0.74 ✓. Shuffle 0.66 / 0.77 ✓ (s8 Mistral shuffle 0.765). Matched 0.64 → 0.51, 0.76 → 0.73, 0.81 → 0.77, 0.84 → 0.84 ✓ against the brief. The dumbbell rounds 0.72 for word counts while the brief has 0.73. Check the source. Head-to-head ✓ against s10 (0.835, 0.783, 0.869; 0.655, 0.733, 0.750). Label curve ✓ against s7. Model scan ✓ against s8/s9_rerun.
- Two test sets with different numbers for the same model (Slots 2–6: 1,000 sequences; Slot 7: 3,000) are never named on a slide. Fix 01/3.2 and 07/2.1 as above.
- CNN at 500 labels is 0.66 vs 0.58 (07/2.2 vs 07/2.4).
- Planting control spread: slide 0.31 ± 0.09 vs notebook 0.290 ± 0.057.
- Unused but notable: s1 `core:k5` gives 0.83 on the ls set (counts of 5-letter words in the 40-letter site alone). Order-free counts restricted to the site nearly match the CNN, which supports the "position" story and weakens "the CNN uses letter order". Consider one note line in 06/2.2.

**Terms**
- Model names: "Nucleotide Transformer", "NT-500M", "486M-parameter model", "Nucleotide Transformer 500M", "NT-500M (human ref)" next to "NT-v2-500M". Pick "NT-500M (human reference)" and use it everywhere. "Mistral-DNA", "Mistral-DNA, 16M", "Mistral-DNA-17M": define once (16M parameters, repo name 17M).
- "held-out" / "test" / "ls file" for the same set. "five-letter word counts" / "5-letter word counts". "G+C and U only" / "G+C/U" / "two numbers". "miR-16" / "miR-15/16". "probe" / "linear probe" / "frozen + probe". Pick one form of each.
- DNA vs RNA: the title says DNA, the data are RNA, and U→T conversion appears only in a Slot 4 note.

**Colour semantics** (`deckstyle`: bound = accent, CNN = ink, frozen = mid grey, fine-tuned = accent, shortcut = mid grey). Violations:
- Accent used for non-bound, non-fine-tuned items: 01/18 (held-out), 03/11 (miR-15/16), 05/06 (G+C scale), 05/07 (3-min pretrain), 05/08 (loss line), 06/08 and 07/07 (G+C/U shortcut).
- Frozen probe in light grey: 07/08.
- Off-palette green and red: 01/05–06.
- The second blue of inline code (#4758AB) appears in every deck.

**Chrome and layout**
- One-section decks (02, 04, 08) carry a one-item TOC, a divider and a one-item menu bar. The TOC adds little in the others too.
- The first menu item renders about 3 px higher than the rest in every deck (Welcome, Look, Load, Baselines, Control, Wrap-up).
- Slide numbers render pale (about #BDC1C6) despite `--muted`. Check that `.slide-number` is not inheriting opacity.
- A blue hamburger (reveal menu) icon sits under the logo on every slide. The reference seminar has none. Fix: `menu: false`.
- Title sizes are consistent (h2 1.5em). Most matplotlib tick labels are 13–15 px on a 1600-wide slide, which is too small for the back of a room. Raise the base font in `deckstyle.py` by about 30%.

**"Your turn" slides**: column sets vary (Time / Expect / Time). Every one should have Step | Task | Names | Expect | Time, because the notes hold exactly the information students need before opening the notebook.

**Writing tells (Req. 1)**: none of the requested patterns remain in the visible slide text except 08/04 and 07/11. In the notes:
- "not X but Y" constructions: 01/11 ("not from AGO2"), 02/07 ("not only which motifs"), 05/03 ("not the laptop"), 06/08 ("not longer words and not order"), 06/09 ("changes the data, not the model").
- Colon-reveals: "An honest caveat:", "Why it matters:", "The arc:", "Hypothesis for the day:", "Rule of thumb for today:".
- Slogans: "Only the held-out number means anything", "Fine-tuning wins", "A clear effect.", "The logos did not show it; the experiment did.", "Which model is best depends on how many labels you have."
- A rhetorical question with its answer: 02/10 notes, "Next: the model scores 0.86. What did it learn?"
- Forced analogies: "address", "like a sample", "DNA has no words".

No em dashes or middle dots in the sources.

---

## Top 15 fixes, ranked

1. **08/04 point 5 and 08/03 "Fine-tuning wins"**: remove the overclaims. Frozen probes still score 0.77 on matched negatives, and fine-tuning beat the CNN by 0.03 on one test set with no shuffle test. Restate both with numbers.
2. **`notes_for_roee_slot8.md` and the Slot 8 deck**: rewrite the brief from the current numbers (it says the CNN wins with many labels). Replace the unreadable thumbnail grid with a slot | finding | number table. Move 07/2.5 into Slot 8 so the 20-minute slot has content.
3. **04/1.4**: put the masked-LM objective on the slide next to the next-token loss. NT, the model behind the 0.79 and the 0.87, is masked.
4. **06/1.2 notes and 06/2.2 notes**: the shuffle floor is the composition score (0.66), not 0.5, and the shuffle also moves sites out of 150–190. Rewrite "relies on order" and "retains only composition".
5. **07/1.1 notes**: "composition carries no information … any score above 0.5 must come from something else" is false (only G+C, U and length are matched; word counts still give 0.73).
6. **01/11, 06/09 Q3, 08/04 point 1**: stop asserting that the G+C gap comes from negative selection. Present it as the hypothesis Slot 7 tests.
7. **07/08 notes**: the frozen probe has one run, not three seeds. Correct the claim that the dots do not overlap.
8. **07/2.1 and 01/3.2**: name the second test set on the slides. 0.81 / 0.76 / 0.84 vs 0.78 / 0.74 / 0.86 will otherwise look like errors.
9. **07/2.2 vs 07/2.4**: the CNN at 500 labels is 0.66 on one slide and 0.58 on the next. Use the same draws or show the range on the slide.
10. **All "Your turn" slides (02/09, 03/09, 05/04, 06/05, 07/05)**: add Names and Expect columns. Move the code TODO-3 needs (eval, `.cpu()`) to 02/08 and the masked mean plus U→T to 05/03.
11. **05/2.2**: show that the CNN had about 92k labels and the probes 12k. Point to 07/2.4.
12. **06/2.1**: move the legend off the probe's unbound spike at 0, and explain the spike.
13. **01/4.2–4.3**: do not use probes before Slot 4 defines them. Use CNN and G+C models for the noise and overfitting examples. Replace "Only the held-out number means anything".
14. **Colour semantics across decks**: accent only for bound and fine-tuned, mid grey for shortcut and frozen. Fix 01/18, 03/11, 05/06 (G+C scale), 05/07, 06/08, 07/07, 07/08, the 01/05–06 green and red, and the inline-code colour.
15. **Chrome**: drop the one-item TOC and divider slides in decks 02, 04 and 08. Fix the misaligned first menu item, the pale slide numbers and the hamburger icon. Remove the "AI assistant" goal (01/05) or deliver it. Correct the data citation (Stražar et al. 2016; iDeep 2017).
