# Slot 8 (Roee): the wrap-up should land the arc

What the participants have seen by 13:40, all on the AGO2 CLIP-seq data (held-out AUROC):

| Slot | What they did | What they found |
|---|---|---|
| 2 | built a CNN from scratch | 0.86 |
| 3 | read motifs off it, planted them | the logos look like composition; planting shows a real microRNA rule (miR-15/16) |
| 5 | an AI assistant built a genome-pretrained model + probe | 0.78, competitive; a 30× bigger model is no better (0.79) |
| 6 | tried to break both | 2 numbers (G+C, U) already give 0.66; on shuffled sequences the pretrained model barely notices (0.77) and the CNN collapses (0.66) |
| 7 | fixed the data (composition-matched negatives), fine-tuned | the free points disappear; with plenty of labels the CNN still wins (0.84 vs 0.81). **With 100 labels the pretrained model wins (0.68 vs 0.54)** |

## The takeaway to say out loud

**A foundation model is a tool for a situation, not a default.** It earns its place when:
- there are few labels (a new cell type, a rare disease, a pilot)
- there are no labels at all (zero-shot variant-effect scoring)
- the organism has no labelled data
- you need to design new sequences

On a well-measured task with plenty of labels, a small supervised model is a strong baseline.

**The abstract's line, now earned.** "Training a model takes four lines of code; working out
whether it learned anything real is the hard part." The agent wrote the code in seconds. Choosing
the negatives, the shuffle test, the control and the label-count experiment was the humans' job.

**A question to leave them with:** high-throughput assays make labels cheap for the question you
measured, in the cell type you measured it in. How many of your own questions are like that?
