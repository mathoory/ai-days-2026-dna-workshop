# Applications of AI in Biotechnology and Genomics: design decisions

## 1. The audience problem, from first principles

**What we can assume:** Python and linear algebra. Nothing else, per the published abstract.

**Who will actually come:** AI Days attendees who skipped the wet-lab hackathon running the
same day. That probably means mostly CS/DS students (undergrad and grad), plus some biotech
students and staff. So the room is **bimodal**:

| | knows | doesn't know |
|---|---|---|
| **coders** (CS/DS) | train/test, neural nets, maybe how LLMs are used | what a binding site is, where labels come from, why DNA data lies |
| **biologists** | genes, RNA, controls, experimental design | how a model is trained or evaluated |

**The trap** is that "foundation models for DNA" turns into an LLM lecture with "DNA" pasted
on top. The coders have seen that lecture, and the biologists can't follow it.

**The way out** is to teach what *neither* group has: **how to think about learning from
biological sequence.** A coder knows a baseline, a biologist knows a control, and the workshop
shows they are the same idea. That bridge is the added value. Everything else is tooling.

## 2. What participants should leave with (the five ideas)

1. **A sequence is the input; an experiment is the label, and a person chose the negatives.**
   *(Coders don't know this.)*
2. **DNA has no words.** Turning it into numbers (one-hot, k-mers, tokens) is a design choice
   with biological consequences: motifs, two strands, long range.
3. **Models take shortcuts, and in genomics the classic shortcut is composition** (G+C, U,
   length, location). A high AUROC is not evidence of biology.
4. **Interrogate a model with experiments.** Planting, shuffling and matched negatives are the
   in-silico versions of controls that a wet lab already uses.
5. **Pretrained genome models:** frozen, they mostly encode composition. Fine-tuned, they can beat a task
   model (NT-500M LoRA 0.87 vs CNN 0.84 at equal labels). You test whether they help *your* question.

The abstract's line is the thread: *training is four lines of code; whether it learned anything
real comes down to how you test it.*

## 3. Bridging the two groups (design devices)

- **One pipeline, every slot.** *Question → Data → Encode → Model → Train → Evaluate →
  Interrogate.* Each slot opens with "you are here" on this strip. Items 15–16 of the brief
  become the spine instead of a separate lecture.
- **The ML ↔ lab dictionary.** One slide early on maps each ML word to its lab twin:
  - test set ↔ independent replicate
  - baseline ↔ negative control
  - shuffle test ↔ scrambled-sequence control
  - matched negatives ↔ matched background
  - AUROC ↔ the ROC curve of a diagnostic test (which a Medical Diagnostics audience already knows)

  Both halves of the room recognise half of every row.
- **Buddy pairs.** At the start, everyone sits with someone from "the other side". The biologist
  reads the biology and the coder reads the error messages. With 40–100 people this beats
  any amount of slide explanation, and it replaces a full introduction round (100 × 20 s = 33 min).
- **Notebooks carry the code burden.** The given cells do the heavy lifting, the TODOs are short,
  and the AI assistant helps whoever is weaker in Python.
- **Depth lives in the speaker notes**, as "if asked" answers for advanced participants. The slide
  stays at the common denominator.

## 4. Include / dodge

| Topic | Verdict | Where | How |
|---|---|---|---|
| Why AGO2 binding matters | **include** | 1 | one graphic: AGO2 + microRNA turns genes down; >60% of human genes are microRNA targets; disease variants; RNA drugs |
| Why predict what you can measure | **include** | 1 | CLIP measures one cell type and condition; a model reaches unmeasured transcripts, variants and designs |
| Where labels come from (CLIP, chosen negatives) | **include** | 1 | genome track: peaks = positives, "who chose the negatives?" |
| Central dogma | **include, one graphic** | 1 | DNA → RNA → protein, and where AGO2 acts |
| CLIP protocol, microRNA biogenesis, RISC detail | **dodge** | — | one sentence in the notes |
| ML in one picture (model, parameters, loss, training) | **include** | 1 | one loop diagram |
| ML timeline | **include, 1 slide** | 1 | reuse tut0 `timeline.png` |
| Train / validation / test + DNA leakage (similar sequences) | **include** | 1 | split bar + "split by gene, not by row" |
| AUROC | **include** | 1 | two score distributions + ROC |
| Backprop, optimizers, losses beyond "how wrong" | **dodge** | — | "nudge the weights downhill" |
| One-hot, convolution as motif detector | **include** | 2 | own graphics |
| Pooling, depth, dropout | **dodge** | 2 | "these layers combine detections" |
| Logos, planting, shuffled control | **include** | 3 | |
| microRNA seed pairing | **include as bonus** | 3 | one graphic + planting bars |
| Traditional ML vs foundation models | **include, 1 slide** | 4 | summary of KAUST lecture 1a (six slides → one picture) |
| Anatomy: tokenizer → embedding → backbone → head | **include** | 4 | the backbone stays a black box |
| Attention, Q/K/V, positional encoding, transformer internals | **dodge** | — | the backbone is "a function from tokens to vectors" |
| DNA is not English (no spaces, two strands, long range, mostly non-coding) | **include** | 4 | the domain twist that keeps this from being an LLM lecture |
| Tokenization (single base / k-mer / BPE) | **include** | 4 | real tokens from the models we use |
| Embedding (lookup → vector → average) | **include** | 4 | |
| One backbone, many heads | **include** | 4 | KAUST 1b slide 9, redrawn with DNA examples |
| Pretraining MLM vs AR | **include, 1 slide** | 4 | which model uses which |
| Model landscape (size vs year) | **include, 1 slide** | 4 | including "biggest ≠ best" |
| Ways to adapt: probe / partial / full / LoRA | **include, 1 slide** | 4 | frozen vs trained, cost vs data |
| LoRA mechanics | **include, 1 slide** | 7 | W frozen + small B·A; rank math dodged |
| Quantization, FSDP, distributed training, MoE, perplexity | **dodge** | — | |
| Hugging Face (what it is + the 3 lines of code) | **include, 2 slides** | 5 | KAUST lecture 2 slide 26, as an annotated code card |
| Working with an agent (prompt → paste → run → check) | **include** | 5 | |
| Composition, word counts, shuffle test | **include** | 6 | |
| Matched negatives, before/after, label curve | **include** | 7 | |
| When a foundation model is worth it | **include** | 7 + 8 | |
| From question to model (pipeline with the DNA trap at each step) | **include** | 8 | |
| Survey | **include** | 8 | QR placeholder |

## 5. Slot structure (8 × 30 min)

| # | Time | Who | Deck | Pipeline stage |
|---|---|---|---|---|
| 1 | 10:00 | **Mattan** (opening, 8 min) + **Roee** (22 min) | `01_opening_and_basics` | Question, Data |
| 2 | 10:30 | Mattan | `02_build_cnn` | Encode, Model, Train |
| 3 | 11:00 | Mattan | `03_look_inside` | Interrogate |
| 4 | 11:30 | Roee | `04_foundation_models` | Encode, Model (pretrained) |
| — | 12:00 | | break | |
| 5 | 12:15 | Mattan | `05_pretrained_model` | Model, Train, Evaluate |
| 6 | 12:45 | Mattan | `06_break_it` | Interrogate |
| 7 | 13:15 | Mattan | `07_fix_it` | Data, Evaluate |
| 8 | 13:40 | Roee | `08_question_to_model` | the whole pipeline |

Roee's decks (parts of 1, all of 4 and 8) are **drafts for him to adapt**. Their speaker notes
say what each slide is for, so he can swap in his own material.

## 6. Slide rules

- **Visible text:** a numbered title plus at most ~10 words on screen (the seminar median is
  ~8). Everything else goes in the speaker notes as skimmable bullets with a **bold lead**.
- **One idea per slide, carried by one figure.** Figures are drawn by `slides/figures/*.py`,
  in the course palette:
  - green = data
  - blue = CNN
  - purple = foundation model
  - gold = head / probe
  - red = shortcut / warning
- **Numbers on slides are measured,** from the executed notebooks and `scripts/spike/`.


## 7. Revision 2026-10-12 (after two audits; see AUDIT_FIXES.md)
- **Dropped as forced or slogan-like:**
  - the ML↔lab dictionary slide (the analogies now live in the speaker notes, where each control runs)
  - "DNA is not English" (its one relevant point, that DNA has no word boundaries, moved onto the tokenization slide, shown with real tokenizers)
  - Venn, agent loop, word-card slides
- **Every concept slide now shows a real object from our data or models:**
  - filter weights and a real dot product
  - activations and the network's real shapes
  - a filter learning
  - real tokenizers, the real sequence → vector pipeline, real next-token predictions
  - embeddings coloured by G+C, score histograms under the shuffle
- **New:** a goals slide (tut0 style, YES/NO) and "reading a result" (topic 17).
- **AI-assistant cells are gone.** Each TODO says what to build and what to check; using an assistant is the default, not a special cell.
- **The ending changed with the data:** frozen embeddings ≈ composition; fine-tuned pretrained models beat the CNN in a fair 3-seed comparison.
