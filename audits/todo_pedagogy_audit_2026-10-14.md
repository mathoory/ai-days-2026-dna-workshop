# TODO pedagogy audit, 2026-10-14 (external review)

Scope: the student TODOs in `01_cnn.ipynb`, `02_foundation_models.ipynb` and `03_shortcuts_and_controls.ipynb`, checked against the "Your turn" slides, the speaker notes, the solution notebooks and the executed runs, and against SCHEDULE.md. Audience assumed: graduate students with basic Python and basic linear algebra, no PyTorch, scikit-learn, Hugging Face, broadcasting or statistics beyond the mean. Most have an AI assistant in Colab.

Bottom line: almost every TODO comment already contains the answer, one line per step. A student, or Gemini in Colab, can turn each comment into code without understanding it. As a result, the TODOs mostly test transcription and API calls rather than the ideas. The ideas (padding mask, train/test split, intervention with a shuffled control, shortcut baselines, shuffle test) live in the printed outputs and the markdown, and nothing makes students look at them. The time budget is also not feasible: Part 1 schedules 32 min of talk plus 46 min of notebook in a 40 min slot.

---

## 1. Summary table

Level: E = easy, M = medium, H = hard for this audience. Abstraction: Idea = the student works with the concept; Mixed = concept plus some plumbing; API = mostly library calls, shapes and dtypes. Agent-proof: does anything make the student understand once an assistant has written the code? Value: what the TODO teaches that serves the workshop goals. Time OK: is it feasible in the "Your turn" minutes, without / with an assistant.

| Notebook | TODO | Task | Level | Abstraction | Agent-proof? | Value | Time OK? (no AI / AI) |
|---|---|---|---|---|---|---|---|
| 01 | 1a | `one_hot(seq)` with a loop, 0.25 padding | M | Mixed | No | Medium | Tight / Yes |
| 01 | 1b | `encode_batch`: `np.stack`, `X[:, None]`, `torch.from_numpy` | M | API | No | Low | No (1a+1b in 3 min) / Yes |
| 01 | 2 | Type the 8 layers of the table into `nn.Sequential` | E-M | API (transcription) | No | Low | Yes / Yes |
| 01 | 3a | One step by hand: logits, loss, zero_grad/backward/step, loss again | E | Mixed | Partly (before/after loss printed) | Medium-High | Yes / Yes |
| 01 | 3b | Same four lines inside the loop | E | Transcription | No | Low (repeats 3a) | Yes / Yes |
| 01 | 3c | Eval block: `eval`, `no_grad`, sigmoid, `.cpu().numpy()`, AUROC | M | API | No | Medium | Tight / Yes |
| 01 | 4a | Five toolkit calls on filter 0, each written out in the comment | E | Transcription | No | Low as code (output High) | Yes / Yes |
| 01 | 4b | The same five calls in a loop over filters, fill 3 dicts | E-M | Mixed | No | Low | Tight (4a+4b in 9 min) / Yes |
| 01 | 5a | `plant` one copy into one sequence, score before/after | E | Transcription | No | Low-Medium | Yes / Yes |
| 01 | 5b | Mean score for 1..7 copies, motif and shuffled | M | Idea | No | High | Tight / Yes |
| 02 | 1a | `enc = tok([s], return_tensors="pt")` (answer given) | E | API | No | Low | Yes / Yes |
| 02 | 1b | Run model, `hidden_states[-1].float()` (answer given) | M | API | No | Low-Medium | Yes / Yes |
| 02 | 1c | `masked_mean` with `unsqueeze`, sum, divide | M-H | Idea, but broadcasting | Partly (masked vs `.mean(1)` printed) | High | Tight / Yes |
| 02 | 1d | `embed`: batching loop, tokenizer args, autocast, concat | H | API / plumbing | No | Low | No (1a-1d in 9 min) / Yes |
| 02 | 2 | `probe_auc`: pipeline, fit on train, AUROC on test | M | Mixed | Partly (test catches train scoring) | Medium-High | Yes / Yes |
| 03 | 1a | `gc_u_one`: counts divided by length | E | Idea | No | Medium | Yes / Yes |
| 03 | 1b | `gc_u`: list of pairs to `np.array` | E | Plumbing | No | Low | Yes / Yes |
| 03 | 1c | `probe_auc` on the two features | E | Idea (one line) | No | High (result), Low (code) | Yes / Yes |
| 03 | 2a | `count_words` with a `Counter` of overlapping windows | E-M | Idea | Partly (compare with `kmer_counts`) | Medium | Yes / Yes |
| 03 | 2b | Same as 1c with `kmer_counts` | E | Transcription | No | Low (code) | Yes / Yes |
| 03 | 3a | `pair_counts` (= `count_words(seq, 2)`) and two booleans | E | Transcription | No | Low-Medium | Yes / Yes |
| 03 | 3b | Shuffle-test AUROC for CNN and probe | M-H | Mixed | No (answers 0.66 / 0.77 printed in advance) | High | Tight / Yes |
| 03 | 4 | Five LoRA/GradScaler lines, given verbatim | E | API (transcription) | No | Low | Yes / Yes |

Count: 23 sub-TODOs. Four have high value as exercises: 01-5b, 02-1c, 02-2 and 03-3b. 03-1c's result is also high value, though its code is trivial. Eleven are transcription of code that appears in the comment, the markdown or the slide.

---

## 2. Part 1, `01_cnn.ipynb`

Talk before the notebook: slides 2.1 to 5.2 (loss, gradient descent, batches, AUROC, overfitting, one-hot, convolution, pooling, network table, training code, logo, insertion). "Your turn" (slide 5.3): TODO-1 3 min, TODO-2 7, TODO-3 7, TODO-4 9, TODO-5 8 = 34 min of TODOs, 46 min for the notebook according to the slide notes.

### TODO-1a `one_hot(seq)`
1. Task: build a 4 x 501 array of 0.25 and set the column of each letter to a one-hot column.
2. Level: Medium. Needs `np.full(..., dtype=np.float32)`, 2-D NumPy indexing (`M[:, i] = 0`, `M[r, i] = 1`), `enumerate`, `str.index`. The 2-D column assignment is the only step past basic Python, and it is not shown anywhere in the notebook or the slides.
3. Abstraction: close to the idea (letters become columns), with a little NumPy. Right level.
4. Agent: Gemini solves it from the comment in one go. The print of `M[:, :8]` and of the last-letter/first-padding columns is useful, but it is printed, not asked about.
5. Value: Medium. One-hot encoding is the bridge to "a filter is a dot product". It is worth keeping because the CNN talk relies on it.
6. Wording: the steps say "for each position i and letter b of seq: set column i to 0, then put a 1 in the row", and only then "A letter that is not in ALPHABET keeps its column of 0.25". Read in order, step 2 tells the student to zero every column, and the exception reads as an afterthought. Rewrite: "2. for each position i with letter b: if b is A, C, G or U, set column i to 0 and put a 1 in row ALPHABET.index(b); any other letter (for example N) leaves the column at 0.25."
7. Time: about 3 min with an assistant. Without one, 1a alone takes this audience 5 to 8 min (NumPy column indexing), so 3 min for 1a+1b is not realistic.

### TODO-1b `encode_batch`
1. Task: stack the one-hot arrays, add an axis, convert to a tensor.
2. Level: Medium for this audience. Needs `np.stack`, `X[:, None]` (None-indexing), `torch.from_numpy`, and the idea of a "channel axis", which slide 4.4 introduces after the TODO-1 slide.
3. Abstraction: pure API and shape plumbing. The comment already gives all three calls.
4. Agent: solved instantly. The student learns nothing the shape print does not already say.
5. Value: Low. The cell right after it replaces the student's version with a faster one, so even the code is thrown away.
6. Wording: "add the channel axis: shape (N, 1, 4, MAX_LEN), for example with X[:, None]". For a student without NumPy, `None` as an index looks like a typo. If it stays, add "(None inserts a new axis of size 1)".
7. Time: with 1a it does not fit in 3 min without an assistant.
Recommendation: make it given (merge into the fast `encode_batch` cell) and keep only 1a.

### TODO-2 the network
1. Task: copy the table into `nn.Sequential`, adding `nn.ReLU()` after each Conv2d and `nn.Flatten()`.
2. Level: Easy to Medium. The table gives the exact constructor calls. The student only needs to know that ReLU and Flatten are layers too (`nn.ReLU()`, `nn.Flatten()`), which neither the table nor the comment names. That is the one place a non-PyTorch student gets stuck.
3. Abstraction: transcription. The concepts (why 492, why 1060, why 2,491 parameters) are in the speaker notes. The student never computes them.
4. Agent: solved in one go. The test cell's per-layer shape check is well built and its messages are clear, but it only runs on a correct answer or flags an error the assistant would not make.
5. Value: Low as written. The part worth learning is the shape arithmetic and the parameter count, and both are given.
6. Wording: "build the network from the table above, in order, separated by commas". Add "ReLU is its own layer, nn.ReLU(); Flatten is nn.Flatten()". The table row "`Conv2d(1, 10, (4, 10))` + ReLU" does not tell the student that this is two layers.
7. Time: 7 min is generous with an assistant. Without one it is enough.
Recommendation: make the network given and turn the TODO into "predict, then run". Before running, the student writes down: the width after the first conv (501 - 10 + 1), the number of weights in one first-layer filter (4 x 10 + 1 bias), and the total for 10 filters (410). Then a cell prints the shapes and the parameter count. This uses the dot-product slide (4.2) directly.

### TODO-3a one step by hand
1. Task: `logits = model(X)`, `loss = criterion(logits, y)`, then `zero_grad/backward/step`, then `loss_after`.
2. Level: Easy, given the table in section 4. The concepts (loss, gradient, step) were covered on slides 2.2 to 2.6.
3. Abstraction: good. Each line maps to one idea. This is the best-designed TODO in Part 1.
4. Agent: solved in one go. The printout (logits, labels, loss before and after, `w.grad` shape) is the learning, but nothing asks the student to read it.
5. Value: Medium-High. The before/after loss makes gradient descent concrete.
6. Wording: clear. The TODO is split into two code regions with print statements between them. That is fine, but a student who leaves the first `...` empty gets a `NameError: logits` and no Stuck cell. Add one line: "If this cell fails, skip it: train() below does not need it."
7. Time: 2 to 3 min.
Suggested change: before running, ask "A model that knows nothing predicts 0.5 for everything. What loss do you expect before the step? (slide 2.2: ln 2 = 0.69). Will the loss after one step be higher or lower?" Print the comparison with 0.69.

### TODO-3b the loop body
1. Task: paste the four lines of 3a into the loop.
2. Level: Easy.
3. Abstraction: transcription of 3a.
4. Agent: trivial.
5. Value: Low. It repeats 3a. Its only extra lesson is "the same step, many times", and the given loop already shows that.
6. Wording: fine.
7. Time: 1 min.
Recommendation: merge into given code. Keep 3a as the hands-on step.

### TODO-3c evaluation
1. Task: `model.eval()`, `with torch.no_grad()`, sigmoid, `.cpu().numpy()`, `roc_auc_score`.
2. Level: Medium. Needs the GPU/CPU distinction, tensor-to-NumPy conversion and context managers. The speaker notes concede that leaving out `.cpu().numpy()` is "a common error in this TODO".
3. Abstraction: mostly API. The one idea (evaluate on held-out data, use probabilities) is good, but the work is device plumbing.
4. Agent: solved in one go.
5. Value: Medium. The held-out evaluation idea matters for the whole day. The test cell's spy on `roc_auc_score` (did you pass probabilities, a NumPy array, the right shape?) is a good diagnostic.
6. Wording: "Expected (the training cell below prints it): about 0.79 after epoch 1, about 0.86 after epoch 10." The test cell runs first and prints a different number (1 epoch on 5,000 sequences). Say: "The test cell trains on only 5,000 sequences, so its AUROC is lower; the full training cell gives about 0.79 then 0.86."
7. Time: 3 to 4 min with an assistant. Without one, longer than the 7 min left for all of TODO-3 once 3a is done.
Recommendation: give the line `probs = torch.sigmoid(model(X_test.to(DEVICE))).cpu().numpy()` and leave only `auc = roc_auc_score(test_y, probs)` as the student's line, with a question: "Why test_y and not y_train?" Or make 3c fully given.

### TODO-4a filter 0 step by step
1. Task: five assignments, each spelled out in full in the comment (`acts_0 = acts[:, 0, :]`, `cut_0 = cutoff_for(acts_0)`, ...).
2. Level: Easy. Pure copying.
3. Abstraction: the steps are at the idea level, but the student does not choose any of them.
4. Agent: trivial. Even without an assistant it is copy-paste.
5. Value: Low as code. The output (window count, frequency table, logo) is valuable, and it would be just as valuable from a given cell.
6. Wording: clear.
7. Time: 2 min.

### TODO-4b loop over filters
1. Task: repeat 4a for every f and fill three dicts.
2. Level: Easy to Medium (loop, dicts).
3. Abstraction: plumbing (the three dicts are bookkeeping for the plotting cell).
4. Agent: trivial.
5. Value: Low. What students should do here is look at 10 logos, compare a filter on bound and unbound RNA, and choose `MY_FILTER`. That step is not a TODO; the markdown says "This is not a TODO, but the rest of the notebook uses it".
6. Wording: "fill motifs[f] (consensus string), pwms[f] (the table) and n_hits[f] (number of windows)" is clear.
7. Time: 9 min for 4a+4b is far more than the coding needs. The time goes into reading the logos, which is the right use of it, but the slide schedules it as coding.
Recommendation: make 4a and 4b given (one cell that builds and draws the logos). Turn the filter choice into the exercise: "Pick a filter whose logo has tall columns. Look at the same filter on unbound RNA. Write one sentence: does it look AGO2-specific? Write your MY_MOTIF." This is the interpretation skill the workshop is about.

### TODO-5a one copy in one sequence
1. Task: `plant([one], MY_MOTIF, 1)[0]` and two `score_sequences(...)[0]` calls, all spelled out.
2. Level: Easy.
3. Abstraction: transcription. It does show the list-in/list-out convention, which 5b needs.
4. Agent: trivial.
5. Value: Low to Medium. The printout (before/after letters, before/after score) is a good concrete picture of what "inserting a motif" means. In the executed run with the default filter 0 the score goes down (0.603 to 0.567), which is worth discussing, but nothing prompts the student to notice it.
6. Wording: clear.
7. Time: 2 min.
Recommendation: make it given and add a question: "Did one copy raise the score? Is one sequence enough to tell?"

### TODO-5b mean scores for 1..7 copies
1. Task: two lists of seven means, one for `MY_MOTIF` and one for `shuffled`.
2. Level: Medium. A loop or list comprehension, `.mean()`, and the right variables (`canvas`, `copies`, `shuffled`).
3. Abstraction: at the idea level: an intervention plus a control. This is the strongest coding TODO in Part 1.
4. Agent: one line per list, solved in one go. The plot is where the learning is, and the take-aways are long and subtle (the position at letters 150 to 190, the shuffled control rising too).
5. Value: High. It is the first experiment of the day, and it sets up Part 3.
6. Wording: "The motif's score usually jumps at copy 4 (letter 175)" gives the finding away before the student has seen the plot, and "usually" does not hold for every filter. Remove it. Replace with a prediction prompt (below).
7. Time: 4 to 5 min with an assistant, plus reading. 8 min for 5a+5b plus the position scan is tight.
Suggested change: before running, "Predict: will the shuffled curve stay flat or rise? Why might either curve jump at one particular number of copies?" After the plot: "At which copy number does the curve change most? Which letters of the canvas does that copy occupy?" Then the position-scan cell answers it.

### Given cells and fallbacks in Part 1
- The setup, data and toolkit cells are hidden behind `#@title` forms, and the toolkit table in section 6 describes them. Good.
- Visible code with TODOs (`GlobalCNN`, `train`) is short and readable.
- The "Stuck?" cells redefine the function and rerun the test. Good pattern. Gaps: 1a, 3a, 4a and 5a have no fallback, and the notebook does not say which later cells depend on them. 3a, 4a and 5a define names (`loss_after`, `acts_0`, `one_planted` and so on) that no later cell uses, so they can be skipped, but students will not know that.
- The test cells are thorough, and their messages name the variable and the expected value. Example: "the evaluated scores range from ... so they are logits. Apply torch.sigmoid". These are the best part of the notebooks.
- The section 8 "Reading your result" and "What to take away" texts are dense, three ideas per bullet. A student who reaches them at minute 38 will not read them.

---

## 3. Part 2, `02_foundation_models.ipynb`

Talk: tokenization, embedding table, transformer as a black box, mean over tokens, padding and mask, heads, the linear probe, loading from Hugging Face (slides 1.1 to 6.1). "Your turn" (slide 6.2): TODO-1 9 min, TODO-2 5 min, live pretraining 6 min. The notes project 32 min of notebook.

The notebook intro says "Use an AI assistant ... as much as you like. Each TODO states ... the gotchas to tell your assistant". That is honest, but it makes the AI-proofing problem explicit: the TODOs are written as prompts for an assistant.

### TODO-1a tokenize
1. Task: `enc = tok([s], return_tensors="pt")`, written out in the comment.
2. Level: Easy (copy).
3. Abstraction: API.
4. Agent: trivial; the student can paste the comment.
5. Value: Low as code. The printout (letters, BPE tokens of varying length, ids, mask of all 1s, extra keys) is valuable and ties to slide 2.3.
6. Wording: clear.
7. Time: 1 min.
Recommendation: make it given. Ask: "How many letters does the longest token have? Count the tokens: is it fewer than 60?"

### TODO-1b last layer
1. Task: move ids/mask to DEVICE, call the model with `output_hidden_states=True` inside `no_grad`, take `hidden_states[-1].float()`. All written out.
2. Level: Medium if typed by hand (device, context manager, keyword args); Easy as copying.
3. Abstraction: API. The idea "one 256-number vector per token" comes across through the printout, not through the code.
4. Agent: trivial.
5. Value: Low to Medium.
6. Wording: "(.float(): the model may compute in 16-bit numbers)". For this audience, "16-bit numbers" needs a word: "in a lower-precision number format".
7. Time: 2 min.
Recommendation: make it given. Ask: "How many vectors are there, and why does the count match the token count, not the letter count?"

### TODO-1c `masked_mean`
1. Task: `m = mask.unsqueeze(-1).float(); return (h * m).sum(1) / m.sum(1)`.
2. Level: Medium to Hard. Needs `unsqueeze`, element-wise multiplication of a (B, T, 256) array by a (B, T, 1) array (broadcasting, never taught), `sum(dim)` and division by a (B, 1) array. Slide 6.1's speaker note on `unsqueeze` is for the speaker. Without NumPy fluency, the shape reasoning is the hard part, and the comment hands it over line by line.
3. Abstraction: the idea ("average the vectors of the real tokens") is right, and it is the core concept of the part. The code forces the student through broadcasting to express it. An alternative that stays at the idea level with basic Python: write it for one sequence first, `h[0, mask[0] == 1].mean(0)`, then give the batched version.
4. Agent: solved in one go. The given comparison print ("masked_mean: 0.0000 (padding left out)" against ".mean(1): ... (padding averaged in)") is the best agent-proof element in Part 2, because it shows what goes wrong without the mask. Nothing asks the student to explain it.
5. Value: High. This is the one TODO that teaches something about how sequence models are used, not about the library.
6. Wording: "the sum over the tokens (dimension 1) of h * m: shape (batch, 256); padding adds 0". Add a two-token example in the comment: "e.g. tokens [v1, v2, pad] with mask [1, 1, 0] -> (v1 + v2) / 2".
7. Time: 3 to 5 min with an assistant, 10+ without.
Suggested change: before running the comparison, ask "Which of the two numbers will be 0, and why?"

### TODO-1d `embed`
1. Task: a batching loop with tokenizer arguments (`padding=True, truncation=True, max_length=512`), device moves, the model call, `masked_mean`, `.cpu().numpy()`, appending to a list, `np.concatenate`, optional autocast.
2. Level: Hard for this audience. Seven API concepts, none of them about sequence models.
3. Abstraction: plumbing. The test checks batch-size invariance, which is good engineering but not a workshop goal.
4. Agent: solved in one go, by design (the "gotchas to tell your assistant" are prompt material).
5. Value: Low. It teaches batching and the Hugging Face tokenizer API.
6. Wording: "Optional: run the model inside with torch.autocast(DEVICE.type, dtype=torch.float16): it only makes it faster". The student cannot judge that. Drop it from a student TODO.
7. Time: 9 min for 1a-1d is not feasible without an assistant. SCHEDULE.md itself lists "part 2 TODO-1 embeddings (9)" among the largest items.
Recommendation: make it given (it already exists word for word in the Stuck cell and in Part 3's helper).

### TODO-2 `probe_auc`
1. Task: `make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))`, fit on train, AUROC on test, return `(auc, probe)`.
2. Level: Medium. Needs scikit-learn's fit/predict_proba, `[:, 1]` column indexing, and pipelines. The markdown gives each piece.
3. Abstraction: mixed. The idea (fit on train, score on test; reuse the training-set scaling) is right. The `predict_proba(...)[:, 1]` part is API trivia.
4. Agent: solved in one go. The test cell is well designed: it detects training-set scoring and says so ("auc = ... is the AUROC on the training vectors"). The markdown also gives away that 0.82 means the student scored the training set, so the student never has to find the mistake.
5. Value: Medium to High. It ties to slide 3.4 (overfitting) and to the slide note "check first which data the score was computed on", which is the most important habit for assistant users.
6. Wording: the code cell has no stub: `# TODO-2: the linear probe` and `...`, with the signature only in the markdown. Every other TODO in the workshop gives the `def` line. Add `def probe_auc(E_train, y_train, E_test, y_test):`. The Stuck cell has two `#@title` lines ("Stuck? ..." then "Probe helper (written in part 2)"), and the same happens in the TODO-1 Stuck cell. Remove the second line.
7. Time: 5 min with an assistant, 8 to 10 without.
Suggested change: make it a "predict, then run". The probe is given with an `ON_TRAIN = True` flag. Before running, ask "Will the AUROC on the training vectors be higher or lower than on the test vectors?" Then run both and compare with slide 3.4.

### Live pretraining (not a TODO)
6 min, of which about 4 compute. Good as a "read while it runs" step. The take-away on the loss ("a loss below about 7.1 means the model uses the tokens before it") is the right thing to read during the wait. The cell code is long and visible (about 70 lines including GradScaler and padding sides), but it is in form mode. Fine.

### Given cells in Part 2
The PCA map cell is good and the "Reading the map" text is clear. The phrase "the signs can flip between runs" will confuse readers who do not know PCA; drop it or say "the direction of the axis is arbitrary". The NT-500M cell is fine.

---

## 4. Part 3, `03_shortcuts_and_controls.ipynb`

Talk: composition histogram, word counts, dinucleotide shuffle, matched data, four ways to adapt, LoRA, the LoRA training step (slides 1.1 to 3.3). "Your turn" (slide 4.1): TODO-1 6 min, TODO-2 4, TODO-3 10, TODO-4 7 = 27 min, 39 projected.

This notebook has the strongest pedagogical arc of the three (baselines, a stress test, a control, then fine-tuning). It also has the clearest "Discuss" prompts. The TODOs are mostly easy pure Python, which suits the audience.

### TODO-1a `gc_u_one`
1. Task: `(s.count("G") + s.count("C")) / len(s), s.count("U") / len(s)`.
2. Level: Easy. Pure Python.
3. Abstraction: idea level. Good.
4. Agent: trivial, but so short that a student types it faster than they prompt for it.
5. Value: Medium. It makes "composition" concrete.
6. Wording: clear.
7. Time: 2 min.

### TODO-1b `gc_u`
1. Task: `np.array([gc_u_one(s) for s in seqs])`.
2. Level: Easy (the comment gives the hint).
3. Abstraction: plumbing.
4. Agent: trivial.
5. Value: Low.
6. Wording: clear.
7. Time: 1 min.
Recommendation: merge with 1a (one function that returns the (N, 2) array) or make it given.

### TODO-1c AUROC of two numbers
1. Task: `auc_gc, _ = probe_auc(gc_u(S_train), y_train, gc_u(S_test), y_test)`.
2. Level: Easy.
3. Abstraction: idea level (any feature matrix into the same probe).
4. Agent: trivial. The result (0.66) is printed in the markdown ("Check: auc_gc ≈ 0.66") before the student runs anything, and also on part 1 slide 3.3 and part 3 slide 1.1. There is nothing left to discover.
5. Value: High result, Low code. "Two numbers reach 0.66" is the central surprise of Part 3, and it is spoiled three times.
6. Wording: fine.
7. Time: 1 min.
Suggested change: "predict, then run". Before running: "The CNN scores 0.86 and the probe 0.78. Write down what you expect from two numbers per sequence: 0.5, 0.6, 0.7 or 0.8?" Move the expected value into the test cell message and out of the markdown.

### TODO-2a `count_words`
1. Task: `Counter(seq[i:i + k] for i in range(len(seq) - k + 1))`.
2. Level: Easy to Medium (slicing, range bounds, `Counter`). The off-by-one is the real content, and the comment gives it ("L - k + 1 of them").
3. Abstraction: idea level. Good.
4. Agent: trivial. The comparison table against `kmer_counts` (count vs fraction) is a good check, but it is read-only.
5. Value: Medium. It makes "a model without order" concrete.
6. Wording: clear.
7. Time: 2 to 3 min.

### TODO-2b AUROC on word counts
1. Task: TODO-1c with `kmer_counts`.
2. Level: Easy.
3. Abstraction: transcription of 1c.
4. Agent: trivial.
5. Value: Low as code (the result, 0.74, is printed in the markdown and on slide 3.3 of part 1).
6. Wording: fine.
7. Time: 1 min.
Recommendation: make it given, with a prediction prompt: "More than 0.66 or less? Why?"

### TODO-3a `pair_counts` and two booleans
1. Task: `count_words(seq, 2)` (the comment says so), then `pair_counts(s) == pair_counts(s_shuf)` and `s == s_shuf`.
2. Level: Easy.
3. Abstraction: transcription.
4. Agent: trivial.
5. Value: Low to Medium. Checking what the shuffle keeps is a good habit, but the printed 4 x 4 pair table already shows it, and the next given cell checks G+C too.
6. Wording: "1. write pair_counts(seq): a Counter of the len(seq) - 1 neighbouring letter pairs seq[i:i + 2] (the same as count_words(seq, 2) from TODO-2a)". If it is the same, just call it.
7. Time: 2 min.
Recommendation: make it given; keep the printed pair table and ask "Which counts differ?" (none).

### TODO-3b the shuffle test
1. Task: score `shuffled + unbound` with the CNN (`score_sequences`) and with the probe (`embed` on `to_dna` sequences, `predict_proba[:, 1]`), both against `y_shuf`.
2. Level: Medium to Hard. The student must combine four helpers, keep the list order matching `y_shuf`, remember `to_dna`, and use the already-fitted probe. The markdown names every helper.
3. Abstraction: at the idea level. This is the conceptual centre of the day: same model, input changed in a controlled way.
4. Agent: solved in one go. The markdown "Check: CNN ≈ 0.66, language model ≈ 0.77" gives the result before the experiment, so the Discuss question "Which model was least affected by the shuffle?" is answered in advance.
5. Value: High.
6. Wording: "Gotchas to tell your assistant: use the probe that was already fitted ... do not fit a new one on shuffled data (that would answer a different question)". Good, but say which question: "a new probe would learn what separates shuffled from unbound, not whether the original probe relies on order".
7. Time: 5 to 7 min with an assistant, including the embedding of 1,000 sequences. 10 min for 3a+3b is tight without one.
Bug risk: the test asserts `abs(auc_shuf_cnn - 0.66) <= 0.03`. With `USE_MY_FILES = True` and a student's own CNN (different seed, or a different `EPOCHS` from section 10 of part 1), a correct implementation can fall outside 0.63 to 0.69 and the test will report it as wrong. Check against the recomputed reference only (the test already computes `ref`), and print the 0.66 for comparison instead of asserting it.
Suggested change: remove "≈ 0.66 / ≈ 0.77" from the markdown. Before running: "Predict for each model: will its AUROC stay near its real score, or drop towards 0.66?" Write the prediction into a variable that the next cell prints next to the result.

### TODO-4 LoRA training step
1. Task: type five lines that appear verbatim in the comment and on slide 3.3.
2. Level: Easy (copying). Understanding `GradScaler` is not required, and it is not expected either.
3. Abstraction: API transcription. The LoRA concept (W + BA, about 1.6% trained) is in the talk and in the setup printout; the TODO does not touch it.
4. Agent: trivial.
5. Value: Low. It repeats part 1's TODO-3, with the scaler added as API trivia.
6. Wording: the setup cell says "Half precision ... bfloat16 where the GPU supports it, float16 otherwise (T4)", while section 6 says "With bfloat16 (Colab's T4 included) the scaler does nothing" (slide 3.3 notes say the same). One of the two is wrong, or depends on the PyTorch version (recent PyTorch reports bfloat16 as supported on a T4 through emulation). Students who read both will be confused. Pick one statement and print which path the run took (the setup cell already prints "half precision: ...").
7. Time: 2 min of typing plus about 3 min of training. The 7 min is mostly waiting.
Recommendation: make the step given. Replace the TODO with a prediction: "The frozen probe scores 0.77 and the CNN 0.84 on matched data. Where will the fine-tuned model land?" Students can also count the trainable parameters from the slide 3.2 formula (2 x 256 x 16 per matrix) and compare with the printed number.

### Given cells in Part 3
- The rejoin cell is long and handles three cases (yours, ours, refit on reference vectors). Its comment block is correct but hard to read. Students only need "USE_MY_FILES = True uploads yours". Fine as hidden.
- Sections 4 to 5 (`matched_indices`, `shortcut_scores`, CNN retraining) are given. That is the right call; SCHEDULE.md says `shortcut_scores` cost 8 min when it was a TODO. But the markdown before section 4 is three long paragraphs (two benchmarks, 3,000 test sequences, 65,000 training sequences, why the 0.84 differs from 0.86). Most students will skip it. Cut it to three bullets.
- "**Expected:** on matched, the 2-number model is close to 0.5 (≈ 0.51). In table order: 0.64, 0.76, 0.81 on original and 0.51, 0.73, 0.77 on matched." This hands over the main result of the control. Turn it into "predict before you run".
- The "What to take away" text is good, but it states every conclusion. The Discuss prompts after section 3 are the right format; one such prompt after section 7 would help.

---

## 5. Overall assessment

### Progression within and across parts

Within each notebook the order is sensible: one example, then a batch, then a test. The "first one, then many" pattern (1a/1b, 4a/4b, 5a/5b, 1a-1d) is good in principle. In practice, the "many" step repeats the "one" step and adds only plumbing, so each pair costs twice the time for one idea.

Across the parts the arc is strong: build a model, look inside it with an intervention, use a pretrained model, then test both against shortcuts and a control. Part 3's design (baselines, shuffle, matched data) is the most valuable thing in the workshop.

Concepts used before they are taught:
- 01 TODO-1b uses the "channel axis" and None-indexing; channels are introduced on slide 4.4, after the TODO-1 slide, and None-indexing never.
- 01 TODO-2 needs `nn.ReLU()` and `nn.Flatten()` as separate layers; the table writes "+ ReLU" and "Flatten →".
- 01 TODO-3c needs GPU/CPU device moves; this is explained only in a speaker note ("`.cpu().numpy()` is needed because scikit-learn cannot read a tensor that is still on the GPU").
- 02 TODO-1c needs broadcasting and `unsqueeze`; only a speaker note mentions `unsqueeze`.
- 02 TODO-2 needs `predict_proba(...)[:, 1]` and pipelines; neither is taught, only given.
- 03 TODO-4 needs GradScaler; slide 3.3 covers it, but the setup and slide disagree about whether it is active on a T4.
- The position confound (letters 150 to 190) is the key to reading 01 TODO-5b and 03 TODO-3b. It is introduced only in part 1's take-aways and debrief slides, after the student has read the curve.

### Is the load realistic?

No, and SCHEDULE.md says so itself (210 projected minutes against 130).
- Part 1: 32 min of talk plus 46 min of notebook plus 8 min of debrief in 40 min (+46). With the talk at 32 min, only 8 min would be left for a notebook that needs 1.8 min of compute alone, plus downloading data, setting the runtime and installing logomaker. Even with every TODO made given, the notebook is about 20 min of reading and running.
- Part 2: 20 + 32 + 5 = 57 in 40 (+17).
- Part 3: 10 + 39 + 8 = 57 in 40 (+17), and LoRA training takes 3 min of it.
- The "Your turn" times sum to 34, 14 (+6 live) and 27 min of TODOs. They leave out setup, model downloads and reading, which the slide notes count separately as "about 12 min" per part.

To fit 40 minutes, a part needs about 10 to 12 min of talk, 20 to 22 min of notebook and 5 min of debrief. That means at most 3 to 4 short TODOs per part.

### Given cells, Stuck cells and tests

- Given cells are mostly hidden behind forms and described in tables or markdown. Students do not have to read long code. Good. The exceptions are the visible pretraining cell (part 2) and the long markdown blocks before part 3 section 4 and after part 1 section 8.
- The test cells are the best-built part of the workshop. They check behaviour, not exact code, and their messages name the mistake (logits instead of probabilities, training-set scoring, swapped columns, stale gradients). Two problems: 03 TODO-3 asserts a fixed range that a student's own CNN may miss, and 01 TODO-3's test prints an AUROC that differs from the "Expected 0.79 / 0.86" in the comment.
- The Stuck cells are well designed: they redefine and rerun the test. Gaps: sub-steps without a test (01 1a, 3a, 4a, 5a; 02 1a-1c) have no fallback, and nothing tells the student they can skip them. Two Stuck cells in part 2 carry a second, stray `#@title` line.
- Expected values: almost every TODO prints its target number in advance (0.86, 0.78, 0.66, 0.74, 0.66/0.77, 0.81, and "jumps at copy 4"). That helps debugging, but it removes every chance to predict. Move the numbers into the test-cell messages, where they appear only after the student has a result.

### The AI-assistant problem

Gemini in Colab reads the cell and the comment above it. For 21 of the 23 sub-TODOs, the comment contains a line-by-line recipe, so the assistant fills it in without a prompt. Writing the code is therefore not where learning can happen. It can happen when the student must:
1. commit to a prediction before running (a variable `MY_PREDICTION = ...` that the next cell prints next to the result);
2. answer a one-line question about a printed output (logits vs labels; masked vs unmasked difference; which pair counts differ);
3. make a choice that changes the result (`MY_FILTER`, and that already exists);
4. debug a planted mistake (for example, a given probe that scores the training set, where the student must find why it says 0.82).

Typed answers need no grading. The value is in writing them down, and the debrief can ask two or three tables to read theirs aloud.

---

## 6. The five most important changes, ranked

1. **Cut Part 1 to fit 40 minutes.** Make TODO-1b, TODO-2, TODO-3b, TODO-3c, TODO-4a, TODO-4b and TODO-5a given. Keep TODO-1a (one-hot, about 4 min), TODO-3a (one step by hand, with a "predict the loss: ln 2?" question) and TODO-5b (motif vs shuffled). Turn TODO-2 into "predict the shapes and the parameter count, then run" (410 = 10 x (4 x 10 + 1)). Cut the talk to 12 min (slides 2.1 to 3.4 condensed to four slides; 4.2 to 4.7 to three). Reason: the part is +46 min over, and those TODOs are transcription.

2. **Remove the expected results from the markdown for the four key experiments and replace them with written predictions.** 01 TODO-5b ("usually jumps at copy 4"), 03 TODO-1c ("≈ 0.66"), 03 TODO-3b ("CNN ≈ 0.66, language model ≈ 0.77") and the part 3 section 4 "Expected" line. Add a `MY_PREDICTION` cell before each, printed next to the result. Keep the numbers in the test messages. Reason: these are the workshop's findings. An assistant writes the code anyway, so a prediction is the only way to make the student think about the result before seeing it.

3. **Make 02 TODO-1a, 1b and 1d given; keep TODO-1c (masked mean) as the only coding step of TODO-1, and present it at the idea level first.** Step 1: one sequence, `h[0][mask[0] == 1].mean(0)`. Step 2: the batched form, given. Then ask which of the two printed differences is zero and why. Reason: 1a/1b are copy-paste, 1d is Hugging Face plumbing (Hard, Low value), and SCHEDULE lists TODO-1 at 9 min as one of the largest items. The masking idea is the one thing worth the time.

4. **Make 03 TODO-4 (LoRA step) given and fix the T4/bfloat16 contradiction.** Replace it with a prediction of the fine-tuned AUROC against the probe (0.77) and the CNN (0.84), and a parameter count from W + BA. Make the setup cell, section 6 text and slide 3.3 agree on what a T4 runs, and print it. Reason: the TODO is five lines copied from the slide; the contradiction will produce questions in the room.

5. **Fix the brittle checks and the missing fallbacks.** (a) 03 TODO-3 test: drop the `abs(auc_shuf_cnn - 0.66) <= 0.03` assertion, keep the comparison with the recomputed reference, so an uploaded CNN does not fail correct code. (b) 01 TODO-3c comment: say the test cell's AUROC is lower because it trains on 5,000 sequences. (c) Add "if this fails, skip it: nothing later needs it" to 01 TODO-3a/4a/5a and 02 TODO-1a-1c. (d) Give 02 TODO-2 a `def probe_auc(...)` stub like every other TODO. (e) Remove the second `#@title` line from the two part 2 Stuck cells. Reason: these cost students minutes for reasons that have nothing to do with the content.

## 7. Cut, merge, make given, or "predict, then run"

| TODO | Action |
|---|---|
| 01-1a | Keep; rewrite the letter-not-in-ALPHABET wording |
| 01-1b | Make given (merge into the fast encoder cell) |
| 01-2 | Make given; "predict the shapes and 410 / 2,491 parameters, then run" |
| 01-3a | Keep; add "predict the loss before (ln 2) and after" |
| 01-3b | Make given (repeats 3a) |
| 01-3c | Make given, or leave only the `roc_auc_score` line plus "why test_y?" |
| 01-4a, 4b | Make given; make choosing and justifying `MY_FILTER` (bound vs unbound logo) the exercise |
| 01-5a | Make given; ask "did one copy raise the score?" |
| 01-5b | Keep; remove "jumps at copy 4"; predict before the plot |
| 02-1a, 1b | Make given; one question each about the printout |
| 02-1c | Keep as the core TODO; one-sequence version first; predict which difference is 0 |
| 02-1d | Make given |
| 02-2 | Keep, or make given with a planted train-set scoring bug to find; predict train vs test AUROC |
| 03-1a + 1b | Merge into one function |
| 03-1c | Keep the line; predict the AUROC of two numbers first; remove "≈ 0.66" from the markdown |
| 03-2a | Keep |
| 03-2b | Make given; predict more or less than 0.66 |
| 03-3a | Make given (it is `count_words(seq, 2)`) |
| 03-3b | Keep; remove the expected values; predict which model drops; relax the test |
| 03-4 | Make given; predict the fine-tuned AUROC; count the LoRA parameters |

After these changes each part has 3 or 4 coding TODOs (Part 1: 1a, 3a, 5b; Part 2: 1c, 2; Part 3: 1, 2a, 3b) and 3 or 4 prediction or interpretation steps, which fits about 20 minutes of notebook.
