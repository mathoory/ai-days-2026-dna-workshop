# Deck review, 13.10.2026

Scope: `slides/01_cnn.qmd`, `slides/02_foundation_models.qmd`, `slides/03_shortcuts_and_controls.qmd`, slide by slide
(slide text and speaker notes), against `notebooks/src/01_cnn.py`, `02_foundation_models.py`,
`03_shortcuts_and_controls.py` (+ `h_*.py`), `SCHEDULE.md`, the data files, and the rendered figures in
`slides/assets/custom/` that the decks use. Slide references are "deck N.M", with deck 1 = `01_cnn`, deck 2 =
`02_foundation_models`, deck 3 = `03_shortcuts_and_controls`. No file was edited.

Checks that came back clean: no em dashes or middle dots in any deck; all `## N.M` titles are sentence case;
no "block" used for the units; the "Today" table (deck 1 1.4) matches `SCHEDULE.md`; every "Your turn" time
matches the notebook section headings (46/51, 33, 53/55 min); every function and variable name quoted in a
"Your turn" table exists in the notebook; every figure the decks reference exists; the cross-references
"slide 5.2", "slide 5.6", "slide 1.5" (deck 1), "slide 3.2" (deck 2) and "slide 4.6", "Section 2" (deck 3)
point to the intended slides in the current order.

---

## Must fix

**M1. Two figures still print old slot names on screen.**
- deck 2 7.2, `results_05.png`: the CNN bar is labelled **"CNN, Slot 2"**.
- deck 3 4.4, `dumbbell.png`: x-axis reads **"Held-out AUROC on Slot 7's new 3,000-sequence test sets"**.
- Fix in `slides/figures/figs.py`: line 659 `("CNN, Slot 2", 0.861, INK)` → `("CNN (part 1)", 0.861, INK)`;
  line 886 → `ax.set_xlabel("Held-out AUROC on part 3's two 3,000-sequence test sets")`. Regenerate both.
- Also rename the files that carry old slot numbers: `results_05.png` → `results_part2.png` (deck 2 7.2),
  `scoreboard_06.png` → `shuffle_scoreboard.png` (deck 3 4.2), and update the two `![](...)` paths.
- The older generator scripts still contain slot strings and write to the same filenames
  (`deck05_08.py` lines 91, 161: "your CNN from Slot 2", "CNN (Slot 2)", "Mistral-DNA + probe (Slot 5)";
  `intuition.py` 283; `intuition2.py` 265 "where Slot 3's plant() writes copies"; `intuition3.py` 60, 88-89;
  `deck01.py` 69-83 "theory · Prof. Roee Amit"). Rerunning any of them overwrites current figures with slot
  labels. Move them to `slides/figures/archive/` or strip the strings.

**M2. Part 1 is not self-contained for someone who missed the biology lecture (deck 1 1.5, 1.6).**
The only biology is one bullet on 1.5. Never stated anywhere in deck 1: what RNA and mRNA are, that the
letters are A, C, G, U (and why a "DNA" workshop uses U), what AGO2 is (a protein), what a microRNA is, the
pairing rule (A with U, G with C) needed to read `seed_pairing.png`, what "binds" means, and what a 3′ UTR is
(used in 1.6 notes). CLIP-seq is on the slide but defined only in the notes. The archived deck had the pairing
rule and "microRNA of about 22 letters" in its notes; both were lost.
Fix: insert a slide before the seed rule (renumber 1.5 to 1.7 → 1.6 to 1.8):

> ## 1.5 RNA, microRNA and AGO2
> - A gene's DNA is copied into a messenger RNA (mRNA), which the cell uses to make a protein. RNA is written in A, C, G and U; U takes the place of DNA's T
> - AGO2 is a protein that holds a microRNA, an RNA of about 22 letters. It binds mRNAs that pair with that microRNA (A with U, G with C), and the bound mRNA yields less protein
> - Our sequences are RNA, so the notebooks use U; the DNA models of part 2 read T

and change deck 1 1.6 bullet 1 to:
> - **Bound**: about 48,000 sites where CLIP-seq found AGO2 (AGO2 is crosslinked to the RNA it touches, pulled down, and that RNA is sequenced). **Unbound**: about 44,000 random positions in the same genes

Add to 1.6 notes: "3′ UTR: the part of an mRNA after its protein-coding letters."

**M3. Deck 2 uses concepts before introducing them and does not follow the spine order
(tokenization → embedding → transformer → prediction head).**
- 1.1 uses "language model", "pretraining" (prediction-head row) and "Mistral-DNA" before 1.2 introduces
  pretraining; "foundation model", the title of the part, is defined only in the 1.2 notes.
- 3.2 defines the sequence embedding as "the last layer's output" before the transformer appears (4.1).
  It also reuses the word "embedding" one slide after 3.1 used it for the token table, so the same word
  means the input lookup on 3.1 and the pooled transformer output on 3.2, 6.1, 6.2 and 7.1.
- "Linear probe" is defined twice (3.2 bullet 2 and 5.1 bullet).
- 4.2 (pretraining objectives) and 4.3 (landscape) are about pretraining and models, not the transformer,
  yet sit in section 4 "The transformer".
- 3.1 notes: "The next slides follow them through the transformer and back out as one vector per sequence",
  but the next slide (3.2) jumps to the pooled output first.

Proposed order (numbers after the move):
1. 1.1 Pretrain once, reuse many times (old 1.2), with an added first bullet:
   "A **foundation model** is pretrained on unlabelled sequence, then adapted to many tasks. A DNA
   **language model** is one pretrained to predict tokens of genome sequence from the other tokens."
   Name the running example: "Today's example: **Mistral-DNA**, 16M parameters, pretrained on the human genome."
2. 1.2 Four stages (old 1.1).
3. Section 2 Tokenization (see S1).
4. 3.1 Token embeddings (see S2).
5. 4.1 The transformer, as a black box.
6. 4.2 From token vectors to one sequence embedding (old 3.2). Title it so, change bullet 1 to
   "**Sequence embedding**: the transformer's last-layer output, averaged over all non-padding tokens
   (including the special start and end tokens `[CLS]` and `[SEP]`)", and delete the "Linear probe" bullet.
   Update old 4.1 notes "the token-by-256 matrix on slide 3.2" → "on the next slide".
7. Section 5 Prediction heads: 5.1 heads table (see S3), 5.2 two pretraining objectives (old 4.2: the heads
   pretraining trains), 5.3 landscape (old 4.3).
8. Sections 6 and 7 unchanged. Old 5.2 and 5.3 (adaptation, LoRA) move to deck 3 (M4).

**M4. LoRA is taught in part 2 but used only in part 3, and deck 3 has no fine-tuning slide.**
Deck 3's title is "Shortcuts, controls and fine-tuning", its spine includes LoRA, and TODO-5 (11 min) is the LoRA
training step, but deck 3's talk is three slides (1.1, 1.2, 2.1) and none mentions fine-tuning. The LoRA slide
(deck 2 5.3) is presented about an hour earlier, before a notebook that does not use it.
Fix: move deck 2 5.2 "Four ways to adapt a pretrained model" and 5.3 "LoRA" to deck 3 as a new section
"# 3 Fine-tuning" (3.1, 3.2) before Hands-on; renumber Hands-on → 4, Results → 5, Wrap-up → 6. Then update:
deck 3 old 5.4 notes "(slide 4.6)" → "(slide 5.6)"; deck 2 5.1 notes keep "Body adapted too: fine-tuning
(part 3, LoRA)"; move "Next: load a pretrained model and probe it." from deck 2 5.3 notes to the end of the
last section-5 slide. In the moved LoRA notes change "so the students practise the method" →
"so you practise the method".

**M5. Deck 1 2.2 uses part 3 concepts and an undefined term on the slide.**
Bullet 4 mentions "the shuffle test" and "the matched data" (introduced in part 3); "epochs" is never
defined in deck 1. Replace the bullets with:
> - **Train**: fit the parameters
> - **Validation**: choose settings, such as the number of epochs (passes over the training data)
> - **Test**: report once; never used for choices
> - Parts 1 and 2 use the same **1,000 held-out sequences** (500 bound, 500 unbound) as the test set

Part 3's extra test sets are already explained in deck 3 2.1 notes and 4.4 notes.

**M6. Wrong cross-references left over from the old decks (speaker notes, spoken aloud).**
- deck 1 2.4 notes: "the two composition numbers of 3.1 (0.66)". 3.1 is now One-hot encoding (old deck 01
  numbering, where 3.1 was "What training does"). → "the two composition numbers of slide 2.1 (0.66)".
- deck 1 3.4 notes: "20 × 53 = 1,060 weights. The next slide shows them." The next slide is 3.5 (filters);
  in old `02_build_cnn` the next slide was "The last layer weighs positions". → "Slide 5.2 shows them after training."
- deck 1 5.4 notes: "Whether they occur more often in bound sequences needs a count or a test, which is what the
  next section does." The next item is slide 5.5 in the same section, and it is an insertion test, not a count.
  → "Whether the filter's pattern changes the model's output needs a test; the next two slides insert it."
- deck 2 7.3 notes: "the two composition numbers of part 3 (0.66)". They were first shown in part 1.
  → "the two composition numbers of part 1, slide 2.4 (0.66)".

**M7. deck 1 2.3 notes address the wrong audience: "For the Medical Diagnostics students: ...".**
Course leftover. → "For anyone who knows diagnostic tests: this is the ROC curve, sensitivity against 1 minus
specificity."

**M8. deck 2 6.1: the code on the slide fails as written.**
`model(**ids, output_hidden_states=True)` passes every key the tokenizer returns; the notebook's TODO-1 gotcha
says "the tokenizer returns more keys than the model accepts; pass only `input_ids` and `attention_mask`".
`AutoModel.from_pretrained(name)` also lacks `trust_remote_code=True`, which `h_fm.py` uses. Replace:
```python
tok   = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
model = AutoModel.from_pretrained(name, trust_remote_code=True)
...
h   = model(input_ids=ids["input_ids"], attention_mask=ids["attention_mask"],
            output_hidden_states=True).hidden_states[-1]           # (1, tokens, 256)
```

**M9. Expected values on "Your turn" slides disagree with the debrief and wrap-up (see the table at the end).**
- deck 3 3.1 TODO-3 "about 0.67 and 0.77" vs deck 3 4.1, 4.2 notes and 5.1 table "0.66" (README: 0.661).
  → "about 0.66 and 0.77". The notebook check (`03_shortcuts_and_controls.py` line 217) also says 0.67.
- deck 3 3.1 TODO-5 "about 0.80" vs deck 3 4.5 notes "about 0.81" (notebook takeaway line 683 "≈ 0.81",
  check line 527 "≈ 0.80"; README 0.809). Pick 0.81 everywhere.

**M10. Projected times far exceed the 40-minute parts that deck 1 1.4 announces.**
`SCHEDULE.md` projects 84, 59 and 67 min for parts 1-3 (+90 in total, +60 after the buffer). Not a slide
error, but every "Your turn" table above promises times the day cannot hold. Duplicates this review found,
which are the cheapest cuts: deck 1 2.1 and 3.5 both explain gradient-descent training; deck 1 5.3 and 5.4
repeat what participants just plotted; deck 2 3.2 and 5.1 both define the probe; deck 2 4.3 (landscape)
overlaps deck 3 5.2 (ten models); deck 3 4.1 and 4.2 show the same two numbers.

---

## Should fix

**S1. Tokenization (deck 2 2.1) is one bullet and one figure; the spine asks for examples.**
k-mers are given only a definition; BPE is "frequent substrings, a learned vocabulary of 4,096", with no
statement of how the vocabulary is built (the notebook, section 1, explains the merges; the deck does not).
Split into three slides, keeping `tokens.png` on the last:
> ## 2.1 Single letters
> - The CNN read one token per letter, each a one-hot column (part 1): a vocabulary of 4, and a 330-letter sequence is 330 tokens
>
> ## 2.2 k-mers
> - A k-mer is a fixed word of k letters. Nucleotide Transformer cuts DNA into consecutive 6-mers: `AAACTG CAGAAT TTTGCA ...`, a vocabulary of 4^6 = 4,096
> - Shift the sequence by one letter and every token changes: `AACTGC AGAATT TTGCAC ...`
>
> ## 2.3 Byte-pair encoding (BPE)
> - Start from the 4 letters. Count every pair of neighbouring tokens in the genome, merge the most frequent pair into one new token, and repeat until the vocabulary has 4,096 tokens
> - Frequent stretches such as `TTTT` become one token, rare ones stay short, so token lengths vary
> - Mistral-DNA and DNABERT-2 share one BPE vocabulary

(The 6-mer strings above are read off `tokens.png`, rows 4 and 5.)

**S2. The embedding slide (deck 2 3.1) is imprecise and puts the explanation in the notes.**
Bullet 3, "One-hot encoding is the special case where the table is fixed and the vectors have one number per
letter", is wrong as worded (each one-hot vector has 4 numbers, one per possible letter). The only statement
of what the learned vectors capture ("tokens that occur in similar contexts end up with similar vectors") is
in the notes. Replace bullet 3 and add a bullet:
> - The CNN's one-hot encoding is the same lookup with a fixed 4 × 4 identity table: 4 tokens, 4 numbers each, nothing learned
> - The table is learned during pretraining, so tokens that occur in similar contexts end up with similar vectors

Also use "sequence embedding" for the pooled vector throughout: deck 2 6.1 code comment `# (1, 256): the
embedding` → `# (1, 256): the sequence embedding`; 6.2 TODO-1 row "one 256-number vector per sequence" is fine;
7.1 title "What the embeddings vary with" → "What the sequence embeddings vary with".

**S3. The prediction-head slide (deck 2 5.1) does not say what the task head reads, and uses "softmax" undefined.**
Change the "Our task" row to `| Our task | average the last layer over tokens, then linear layer 256 → 1, sigmoid | P(bound), as the CNN's last layer |`
and add a bullet: "**Softmax**: turns one score per token into probabilities that sum to 1". Delete the probe
bullet from old 3.2 (M3) so 5.1 is the only definition.

**S4. "Overfitting" never appears in part 1 (spine item).**
deck 1 2.5 describes it without naming it; "regularisation" is used in its notes undefined.
Title → "## 2.5 Overfitting". Bullet → "A model that scores much higher on its training data than on held-out
data has **overfit**: it memorised its training examples. Report the held-out score." Add to the notes:
"Regularisation penalises large weights, which limits how closely a model can fit its training data."

**S5. Terms used on slides but defined only in speaker notes, or nowhere.**

| deck, slide | term | where defined now | proposed slide text |
|---|---|---|---|
| 1, 2.1 | σ (sigmoid), loss | notes only | add bullet: "$\sigma(z) = 1/(1+e^{-z})$ turns any score into a probability between 0 and 1. Loss: binary cross-entropy, minus the log of the probability given to the true label" |
| 1, 3.4 | logit | notes only | last table row "one score, a **logit** (any real number; the sigmoid turns it into a probability)" |
| 1, 4.2 | motif, consensus | 4.1 notes, notebook | bullet 1 → "Take the consensus of one filter (its most frequent letter at each position), a candidate **motif**, and insert it 1 to 7 times ..." |
| 2, 3.2 / new 4.2 | `[CLS]`, `[SEP]` | notebook only | see M3 item 6 |
| 2, 4.1 | "8 attention heads" | nowhere | delete "8 attention heads," (the transformer is a black box) |
| 3, 1.1 | shortcut | notes only | add bullet: "A **shortcut** is a simple property that differs between the classes in this data set, such as G+C" |
| 3, 1.2 | dinucleotide | implicit | bullet → "A dinucleotide is a pair of neighbouring letters. The shuffle keeps every letter and dinucleotide count and scrambles everything longer, including where sites sit" |
| 1, 1.6 notes; 3, 1.1 notes | 3′ UTR | nowhere in decks (notebook 3) | see M2 |
| 1, 5.1 notes; 3, 4.5 notes | (random) seed | nowhere | see S6 |

**S6. "Seed" has two meanings and the second is never defined.**
"Seed" is the microRNA seed (deck 1 1.5, a key term) and also a random seed: deck 1 5.1 notes "Three other seeds",
deck 3 4.5 notes "three seeds each ... The CNN and LoRA seeds do not overlap", and `headtohead.png` legend
"(3 seeds)". Replace in notes:
- deck 1 5.1: "Three other training runs, each from different random starting weights, follow the same path and end within 0.01 of each other."
- deck 3 4.5: "CNN and LoRA have three runs each, from different random starting weights (dots); the frozen probe has one run. The CNN and LoRA runs do not overlap at either label count."
- `headtohead.png` (figs.py `headtohead()`): "(3 seeds)" → "(3 runs)".

**S7. Deck 1 section 2 shows CNN results before the CNN exists.**
deck 1 2.3 (score histograms "the CNN's scores") and 2.4 (CNN 0.86 in `auroc_ci.png`) come before section 3
builds the CNN; 2.1 notes give "Our CNN has 2,491 parameters". Add to 2.3 notes, first line: "These are the
scores of the CNN we build in the next section; for now it is any model that outputs a probability."

**S8. Debrief gaps: results the notebook produced but no slide discusses, and slides on results it did not produce.**
- deck 3: the participants' own LoRA result (TODO-5, Mistral-DNA ≈ 0.81 on matched data) appears only in 4.5
  notes. Add it to the slide 4.5 title area or as a row of the wrap-up table, e.g. deck 3 5.1 new row
  `| 3 | Mistral-DNA fine-tuned with LoRA (20,000 labels) | 0.81 on matched data, frozen 0.77, CNN 0.84 |`.
- deck 1 5.2 (last-layer weights) and deck 3 4.1 (score histograms) are not produced by the notebooks; deck 3 4.6
  shows 5-letter word counts and sizes up to 30,000, while the notebook plots three lines up to 12,000. Say so
  in the notes: deck 1 5.2 "Our run; the notebook does not draw this."; deck 3 4.1 "Our run; the notebook prints
  only the two AUROCs."; deck 3 4.6 "The notebook plots three of these lines, up to 12,000 labels."
- deck 1 5.7 debriefs the optional bonus cell; deck 1 1.5 notes promise "We test it at the end of this part".
  → "The last slide of this part tests it (an optional cell in the notebook)."

**S9. deck 3 4.3, question 3 asks for a test the participants have already run.**
In the notebook, the matched control (section 4) follows the discussion; in the debrief the participants have
already done it. Align with the notebook's question 3:
> 3. Which part of the G+C difference could be AGO2 biology (3′ UTRs are A/U-rich), and which part could come from how the unbound positions were drawn?

Notes → "Question 3 connects the shuffle results with the matched data on the next slide."

**S10. Comparisons without label counts or test-set sizes (style rule: claims proportional to evidence).**
- deck 3 5.3 item 3 → "Check simple baselines first: on the 1,000 held-out sequences, two composition numbers reached 0.66 and 5-letter word counts 0.74 (12,000 labels each), against 0.86 for the CNN (92,346 labels)"
- deck 3 5.3 item 5 → "Frozen Mistral-DNA with a probe scored 0.77 on the 3,000-sequence matched test set, and 0.77 against 0.78 under the shuffle. In one offline benchmark on the same matched test set, NT-500M fine-tuned with LoRA reached 0.87 against the CNN's 0.84 (20,000 labels, 3 runs each); it has not been shuffle-tested"
- deck 3 5.4 bullet 2 → "**With 20,000 labels**: fine-tuned NT-500M (0.87) beat the CNN (0.84) in one offline benchmark (3,000 matched test sequences, 3 runs each); keep a small task model as the baseline"
- deck 3 4.5 notes bullet 1: add "on the same 3,000-sequence matched test set".
- deck 3 5.2 notes: add "All values on the 3,000-sequence matched test set."

**S11. deck 3 3.1: "First cell: `USE_MY_FILES = True`".** The flag is in the rejoin cell after seven setup cells.
→ "In the rejoin cell at the end of section 0, set `USE_MY_FILES = True` to upload your CNN and vectors, or keep `False` to use ours."

**S12. deck 3 1.2 notes misstate what a composition-only model does under the shuffle.**
"a model that uses only composition still scores about as well as the composition model (0.66)" conflates two
things: a composition-only model keeps its own score; 0.66 is the floor for a model that loses everything beyond
composition. → "A model that uses only letter and letter-pair composition keeps its score. A model that also uses
longer patterns loses that part and drops towards the composition model's 0.66, not towards 0.5."

**S13. The new notebooks have never been executed.** `notebooks/executed/` holds only the old slot notebooks
(`02_build_cnn` ... `07_fix_it`), and the `*_solution.ipynb` files have no outputs. Every debrief number was
checked here against the old runs, the README and the figures, not against a run of `01_cnn`,
`02_foundation_models` and `03_shortcuts_and_controls`. Run all three once (ideally on a Colab T4) and reconcile
the values in the table below.

**S14. Slogans and colon-reveal sentences (Mattan's style rules).**
- deck 1 1.7 notes: "The signal is real but sparse, which is why a model has to combine many weak cues." (slogan,
  and the "has to" is not shown) → "Even the most enriched word occurs in only 1 to 2% of bound sites, so no single word separates the classes."
- deck 1 2.4 notes: "The bars do not overlap: these are real differences." → "The bars do not overlap, so these differences are larger than the sampling noise."
- deck 1 2.5 notes: "This is the first thing to check ...: which data was the score computed on?" → "In any evaluation code, including code an AI assistant wrote, check first which data the score was computed on."
- deck 1 5.6 notes: "a good part of the effect is the letters' composition in that region rather than their order" → "so a good part of the effect comes from which letters sit in that region, whatever their order" (the notebook's wording).
- deck 1 5.7 notes: "The logos did not reveal this site; the insertion test did." → "None of the ten logos shows this site; the insertion test detected the response."
- deck 2 3.1 notes: "Multiplying a one-hot vector by $E$ just selects a row: an embedding layer is a lookup." → "Multiplying a one-hot vector by $E$ selects one row of $E$, so an embedding layer is implemented as a table lookup."
- deck 2 4.1 bullet 4 repeats bullet 1 behind a colon ("Today it is a black box: a function from token vectors to context-dependent token vectors"). Delete it; the title already says black box.
- deck 3 4.2 notes: "close to the CNN: much of what the CNN adds may be where it looks." → "close to the CNN, which suggests that much of what the CNN adds is where it looks."
- deck 3 5.3 item 1: "The negatives are part of the experiment: here they were ..." → "Here the negatives were random positions in the same genes, and G+C and U alone separated them from bound sites with AUROC 0.66 (1,000 held-out sequences)"
- deck 3 5.3 item 2: "How a sequence is encoded ... decides what a model can see" (slogan; not tested today as stated) → "The encoding limits what a model can use: one-hot letters keep order and position, 5-letter word counts drop both, and each tokenizer splits the same seed site differently"
- deck 3 5.3 notes: "Training a model takes a few lines of code. What it learned is found out by testing it." Punchline; delete.

---

## Minor

- deck 1 1.3 bullet "Stuck? Raise a hand, or load our files and continue" (question-then-answer) → "If a cell fails, raise a hand, or run its "Stuck?" cell and continue".
- deck 1 1.4 notes: "about 92,000 RNA sequences" → "92,346 training sequences plus 1,000 held out" (the data set is 93,346 in total).
- deck 1 2.1 notes: "The two numbers carry some information; part 3 measures how much." Slide 2.4 already gives 0.66. → "Slide 2.4 measures how much (0.66)."
- deck 1 2.4 notes, rule of thumb: ±0.025 → "ignore below 0.03" but ±0.015 → "ignore below 0.015"; the two thresholds are not on the same footing. → "on 3,000 (part 3), below about 0.02".
- deck 1 3.2 notes: "It responds to something common in both classes, which we test properly in part 3." Part 3 does not test this filter. → "Part 3 tests composition effects of this kind with a shuffle and with matched data."
- deck 1 3.2/4.1: 3.2 shows filter 7 (`filter_dot.png`), 4.1-5.6 use filter 0. Say "filter 7" in the 3.2 notes so the switch is visible.
- deck 1 3.6: the slide code names the probabilities `p`; the notebook TODO-3 and the given cell use `probs`. → `probs = torch.sigmoid(...)`, `auc = roc_auc_score(test_y, probs)`.
- deck 1 3.6 notes: "on Colab that is the most common error in this TODO" is a prediction, not an observation. → "a common error in this TODO".
- deck 1 4.3 TODO-5 row lists "position scan" as part of the TODO; it is a given cell. → "...(`plant`); then run the given position scan".
- deck 1 5.1 notes: "Next: what did the CNN learn?" (rhetorical) → "Next: the trained last-layer weights."
- deck 1 5.5 notes: "falls again with more copies" overstates `dose_response.png` (0.72 to 0.68). → "decreases slightly with more copies (0.72 to 0.68)".
- deck 1 5.7 and deck 3 4.6: titles phrased as questions. → "5.7 The CNN responds to one of four seed sites"; "4.6 Test AUROC against the number of labels".
- deck 2 5.2: "Full fine-tuning | all weights | hours" contradicts 5.3 notes ("A 16M-parameter model could be fully fine-tuned on a laptop GPU"). → "minutes to days, with model size".
- deck 2 6.1 notes: "Mistral-DNA (16M) has 16M parameters" → "Mistral-DNA has 16M parameters".
- deck 2 7.3 / deck 2 6.2: name the pretraining reference GPU once ("our A30 test GPU"); 7.3 says "our A30", the notebook "our test GPU".
- deck 3 4.1 notes: "the tall grey bar on the right" is at score 0 (left edge of the right panel). → "the tall grey bar at score 0 in the right panel".
- deck 3 4.4 notes: "(CNN 0.84 here, 0.86 before)"; notebook section 4 says "0.86 to 0.87 there". Use one.
- deck 3 4.5 notes: "The students fine-tuned the smaller Mistral-DNA" → "In the notebook you fine-tuned the smaller Mistral-DNA".
- deck 3 5.2 (`model_scan.png`) labels the model "Mistral-DNA-17M"; everywhere else it is "Mistral-DNA (16M)".
- deck 3 5.4 bullet 3: "scoring variants" → "scoring mutations (variants)".
- deck 3 5.7: "Friedman et al. (2009) Genome Research. Chin et al. (2008) Cancer Research." have no titles and are not cited on any slide; add titles or drop. Add a year for "R. Mourad, *Mistral-DNA* models".
- Leftover build folders in `slides/` named after old decks and "blocks": `01_block1_build_and_look_inside_files`, `02_block2_pretrained_models_files`, `03_block3_shortcuts_and_controls_files`, `01_opening_and_basics_files`, `02_build_cnn_files`, `03_look_inside_files`, `04_foundation_models_files`, `05_pretrained_model_files`, `06_break_it_files`, `07_fix_it_files`, `08_question_to_model_files`. Delete before the repo is shown to participants (slides 1.3 and 5.6 link it).
- Unused figures named after old slots in `slides/assets/custom/`: `here_02.png` ... `here_08.png`, `todo_02.png` ... `todo_07.png`, `results_05.png`/`scoreboard_06.png` (until renamed, M1), `agenda.png`, `logistics.png`. Move to an archive folder.

---

## Factual inconsistencies (same quantity, different values)

| Quantity | Values found | Where | Proposed value |
|---|---|---|---|
| CNN AUROC on shuffled bound vs unbound | 0.67 / 0.66 / 0.661 | deck 3 3.1; deck 3 4.1, 4.2, 5.1, `shuffle_scores.png`, `scoreboard_06.png`; README | 0.66 (fix deck 3 3.1 and notebook 3 line 217) |
| Mistral-DNA LoRA, matched test set | 0.80 / 0.81 / 0.79 / 0.809 | deck 3 3.1 and nb 3 check; deck 3 4.5 notes and nb 3 takeaway; deck 3 5.2 notes and `model_scan.png` (offline shared recipe); README | 0.81 for the notebook run; say in 5.2 notes that 0.79 is the offline shared recipe |
| 105M model probe after 3 min pretraining | 0.72 / 0.73 / 0.75 / 0.746 | deck 2 7.2, 7.3, `results_05.png`; nb 2 section 3 ("about 1,100 steps, probe 0.73"); deck 2 7.3 ("another run"); README ("reproducible to three decimals") | one value from the next run, with "(another run: x)" if they differ |
| Pretraining loss: meaning of falling below the dashed line | deck: below "token frequencies only" needs context; dropping below uniform can be frequencies alone | deck 2 7.3 notes, `pretrain_loss.png` vs nb 2 "What to take away" ("below that line ... learned which tokens tend to follow which") | fix the notebook text to match the deck |
| Matched training sequences for the CNN | ~61,000 / about 65,000 (64,764 = 67,764 − 3,000) | deck 3 5.2 notes, `model_scan.png`; nb 3 sections 4, 5 and table footnote | check the offline split; if it is the same file, 65,000 |
| microRNA seed | letters 2 to 7 (site also pairs 8) / "seed (nucleotides 2-8)" | deck 1 1.5; nb 1 bonus cell comment line 611 | nb 1: "seed (letters 2-7); these sites also pair letter 8" |
| Random-weight probe, highest value | 0.57 / about 0.55 | deck 3 4.6 notes (to 30,000 labels); nb 3 takeaway (to 12,000) | state the range in both: "0.50 to 0.55 up to 12,000 labels, 0.57 at 30,000" |
| Label-curve range | 100 to 30,000 / 100 to 12,000 | deck 3 4.6 notes, `label_curve2.png`; nb 3 section 8 | see S8 |
| Size of the data set | "about 92,000" (training file, 92,346) / 93,346 with the held-out set | deck 1 1.4 notes, deck 3 1.1 notes | see Minor |
| CNN on the part 3 original test set vs held-out | "0.84 here, 0.86 before" / "0.86 to 0.87 there" | deck 3 4.4 notes; nb 3 section 4 | 0.86 |

Consistent across decks and notebooks (checked): 2,491 parameters; 0.79 → 0.86 by epoch; 698 sequences → 0.85;
±0.025 / ±0.015; 0.66, 0.74, 0.78, 0.79 on the held-out set; 0.64 → 0.51, 0.76 → 0.73, 0.81 → 0.77,
0.84 → 0.84; 67,764 of 92,346 and about 30% of bound dropped; head-to-head 0.84/0.78/0.87 and 0.66/0.73/0.75;
label curve at 500 (0.58) and 12,000 (0.82 vs 0.77); miR-15/16 0.53 vs 0.31; 24,667 windows; PCA r = 0.81/0.52.

---

## Spine coverage

| Spine item | Covered at | Gap |
|---|---|---|
| Part 1: training | deck 1 2.1, 3.5, 3.6 | sigmoid, loss on slide (S5) |
| train/validation/test | deck 1 2.2 | epoch undefined, part 3 forward reference (M5) |
| AUROC, noise | deck 1 2.3, 2.4 | none |
| overfitting | deck 1 2.5 | not named (S4) |
| one-hot, filters, network, training | deck 1 3.1-3.6 | logit on slide (S5) |
| logos, motif insertion | deck 1 4.1-4.2, 5.3-5.7 | none |
| Part 2: tokenization (one-hot, k-mers, BPE) | deck 2 2.1 | one slide, no k-mer or BPE example (S1) |
| embedding, with the exercise | deck 2 3.1, 3.2, 6.1-6.2 | wording (S2); token vs sequence embedding and order (M3) |
| transformer as black box | deck 2 4.1 | none |
| prediction head | deck 2 5.1 | pooling and softmax missing (S3) |
| probe exercise and results | deck 2 6.2, 7.1-7.3 | none |
| Part 3: composition shortcuts, shuffle test, matched control | deck 3 1.1, 1.2, 2.1, 4.1-4.4 | none |
| LoRA | deck 2 5.3 (wrong part) | M4; own result not on a slide (S8) |
| label curve | deck 3 4.6 | none |
| wrap-up | deck 3 5.1-5.7 | 5.1 row 1 is background, not a result of the day, though the notes say "Each row is a measured number from today" (see S15) |

**S15. deck 3 5.1, row 1** ("AGO2 is guided by a microRNA seed | site UGCUGCU pairs miR-16 letters 2–8") is
background, while the notes say each row is a measured number from today. Replace the row with:
`| 1 | seed-site words are enriched in bound sites | miR-19 site UUUGCAC 6 times enriched, in 1-2% of bound sites |`

---

## Outside the decks, but shown to participants

- `README.md` is the page participants land on from deck 1 1.3 and deck 3 5.6. It still says "Mattan's hands-on
  slots: 2, 3, 5, 6 and 7. Roee Amit's theory slots are 1, 4 and 8", "10:00–14:00", gives part 2 at "(12:00)" and
  part 3 at "(13:00)" (the schedule says 11:50 and 12:40), tells readers to run
  `quarto render slides/02_build_cnn.qmd`, and lists run times and reference results by slot ("Slot 2", "Slot 5",
  "Slot 7 label-count curve", with CNN 0.64 at 500 labels where deck 3 4.6 and the notebook give 0.58). Rewrite
  by part with the current numbers.
- Notebook text found inconsistent with the decks while checking alignment: nb 3 TODO-3 check 0.67 (line 217);
  nb 3 LoRA 0.80 (line 527) vs 0.81 (line 683); nb 2 takeaway on the loss line; nb 1 bonus comment "seed
  (nucleotides 2-8)" (all in the table above).
