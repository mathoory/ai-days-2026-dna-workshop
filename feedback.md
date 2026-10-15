# Lessons for the next run of this workshop

AI Days, Technion, 15.10.2026. Three parts (CNN, DNA foundation models, shortcuts and controls), 150 minutes of
content plus 30 minutes of breaks after Roee's 50-minute opening. Written right after the day, for whoever builds
the next version (probably me, with Claude).

## 1. Decide early, build late

- **Fix the audience before the content.** The content scope changed four times (basecaller, gene expression LR/DNN,
  CNN plus foundation models) before we knew the audience was CS/DS grad students who already know LR and DNNs.
  Ask the organisers who registered, then pick the arc.
- **Run the spike first.** One day of measuring on the HPC (timings, baselines, shortcut tests) decided more than a
  week of discussion. Keep doing it: every claim on a slide should come from a number we measured, with 3 seeds.
- **Write the punchline down before any slide.** "Frozen embeddings ≈ composition, fine-tuning beats the CNN, few
  labels is where pretraining pays" carried all three parts. It only became clear after the head-to-head run.
- **Lock the schedule once and propagate it everywhere** (website, agenda slide, notebooks, SCHEDULE.md). The day
  changed shape several times and each change had to be fixed in four places. Final shape that worked:
  Roee 50, break 10, Part 1 40 + break + 25, Part 2 50, break 10, Part 3 35 incl. feedback, end on a break.
- **Show "Tutorial X min / Hands-on Y min" per part** on the website instead of clock times.

## 2. Content

- The audience wants **intuition from real objects** (real sequences, real filters, real weights), not a coding
  class and not slogans. Cut anything that sounds like a pitch ("DNA is not English", lab dictionaries, box
  cartoons).
- **Define every term and acronym on the slide where it first appears** (CNN, AUROC, RBP, BPE, LoRA). Then keep one
  notation for the whole deck. Mixing g_θ ≈ f with f_θ, or θ with w, confused people.
- **Name features by what they are.** "G+C fraction" next to a "+" in an equation read as addition; "GC fraction",
  "U fraction" and p̂ = σ(w₁·GC + w₂·U + b) read cleanly.
- **Worked numbers beat symbols.** The CNN template with one sequence followed through it in numbers (and the
  logistic equation with real values) was clearer than any diagram alone.
- **Only show a result live if a participant can reproduce it in the notebook.** Full-data results trained offline
  are fine, but say so. No slides that copy notebook outputs.
- **LoRA did not fit in the time.** It was cut from notebook 3 and from the deck. Plan Part 3 without it, or give it
  a part of its own.
- **Show the biology once, clearly**: one honest RBP schematic (protein, mRNA, binding site) beats a 3D structure.

## 3. Slides and figures

- **Every figure as large as possible.** The single most repeated piece of feedback (20+ times) was whitespace.
  Root cause: plots were reshaped automatically but hand-drawn diagrams kept the shape they were first drawn at, so
  they came out too wide and left an empty band under them. The fix that worked, in this order:
  1. `python scripts/fit_measured.py --aspects`: measures every slide in a headless browser and writes the
     ideal width/height of each figure to `slides/figures/aspects.json`.
  2. Draw every figure, plots and diagrams alike, at that aspect (the save hook in `deckstyle.py` does it for
     plots; diagrams must be laid out for it by hand: bigger text, wrap long rows, stack).
  3. `python scripts/fit_measured.py` sets each image height, then `--check` must report nothing.
  Do this from day one, not after the feedback.
- **Fold text into the figure** (equations, captions, labels) instead of bullets above it; the slide then has one
  object to size.
- **Colour rules** (keep them): letters in BASE colours; training blue, held-out orange; CNN green, frozen blue,
  fine-tuned purple; weights orange-white-blue; learned weights vs numbers from the data in two distinct colours.
  No black-and-white plots.
- **One idea per click.** Use `.r-stack` frames for click-through figures and make the frames the same size
  (`trim_group`).
- **Style that looks professional**: flat neutral blocks, one arrow weight, colour only where it means something,
  no pastel rounded cards with grey subtitles, no drop shadows, no emoji.
- **Do not version-stamp links** (`style.css?v=...`). It broke every slide in the viewer on the morning of the
  workshop. If a slide looks stale, hard refresh (Ctrl+Shift+R).
- **Slide numbers**: feedback came as the browser counter (c/t, including the title and TOC slides), not the
  section numbers. Map between them before editing.

## 4. Notebooks

- **Merge "run only" cells.** Several given cells in a row is noise; one ▶ cell per step.
- **Tests must check results, not implementation.** The TODO-3 test spied on `roc_auc_score` and rejected a correct
  `sklearn.metrics.roc_auc_score` call during the workshop. Compare the returned value with our own computation.
- **Plain test messages**: "✅ TODO-n test passed" and failures that start with ❌ and say what to fix. No cheerful
  emoji text.
- **Don't make students type boilerplate** they cannot reason about (dtype, tensor conversion); give it.
- **Run every solution notebook end to end on the HPC after every change**, and the student version up to the first
  TODO. Then test once on a real Colab T4 before the day.
- **Rejoin files** (download at the end of a notebook, upload at the next) let people who fell behind continue.

## 5. Tooling and process

- **Publishing**: only through `scripts/publish_public.sh` (public branch, speaker notes stripped, no solutions, no
  timing notebook). Never push `main` of the source repo.
- **Figures**: `conda run -n hw1-00660121` with `CONDA_REPORT_ERRORS=false ... < /dev/null`, otherwise conda hangs.
  Windows sometimes locks a PNG mid-write; just rerun.
- **Training-based figures are not deterministic** (the filter shown on 4.3 changed each run). Pin seeds and GPU
  determinism, or keep notes generic.
- **Two Claude sessions on one repo need a claim**: one owns a deck and its figure code, the other stays out, and
  each tells the other what it committed.
- **Feedback rounds go faster as one list per deck**, applied in order, then checked on the live site. Applying a
  lesson from one deck to the others immediately saves a round.
- **Leave buffer.** The last fixes were still going in on the morning of the workshop.

## 6. For next time, still open

- Measure how long each part actually took and how many people finished each TODO; fold that into the timing.
- Collect the feedback form results and add them here.
- Decide whether the Hugging Face loading slide (6.1) and the stage strips need a new layout.
