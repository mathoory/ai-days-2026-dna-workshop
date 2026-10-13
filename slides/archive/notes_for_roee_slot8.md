# Slot 8 brief for Roee (13:40–14:00, 20 minutes)

What the participants have seen by 13:40, all on the AGO2 CLIP-seq data (GraphProt benchmark: bound
sequences are CLIP-seq sites, unbound sequences are random positions in the same genes). Numbers are
held-out AUROC.

| Slot | What they did | What they found |
|---|---|---|
| 2 | built a CNN (92,000 training sequences) | 0.86 on the 1,000 held-out sequences |
| 3 | read logos, inserted motifs and seed sites | logos mostly reflect composition; one seed site (miR-15/16, 0.53 vs 0.31 shuffled) clearly raises the score; the CNN only responds around letters 150–190, where the measured site sits |
| 5 | probed two pretrained models (12,000 labels) | Mistral-DNA 0.78, NT-500M 0.79; 3 minutes of pretraining from random weights: 0.55 → 0.72 |
| 6 | simple baselines and a dinucleotide shuffle | G+C and U alone 0.66, 5-letter word counts 0.74; shuffled: CNN 0.86 → 0.66, probe 0.78 → 0.77 |
| 7 | composition-matched data, LoRA, label curve | G+C/U 0.64 → 0.51, word counts 0.76 → 0.73, probe 0.81 → 0.77, CNN 0.84 → 0.84; their LoRA Mistral-DNA ≈ 0.81 |

Offline results shown in Slot 7 (on matched data):
- Fine-tuned NT-500M (LoRA, 3 seeds) vs CNN: 0.87 vs 0.84 with 20,000 labels; 0.75 vs 0.66 with 500 labels.
- Frozen pretrained models beat the CNN below about 2,500 labels; above about 5,000 the CNN is better.
- The fine-tuned models have not been shuffle-tested.

## Deck 08, slide by slide
1. **1.1 The day in eight results**: one row per result; read two or three rows aloud, not all.
2. **1.2 Ten pretrained models**: frozen models cluster at 0.72–0.79 regardless of size; fine-tuning helps some architectures (NT-500M, DNABERT-2 to 0.87) and not others (HyenaDNA drops).
3. **1.3 Five points to take home**: each point has its number from the day.
4. **2.1 When to use a pretrained model**: the label thresholds are from this task only. Variant scoring without labels was not tested today; published results for regulatory variants are mixed (Tang and Koo 2024).
5. **2.2 From a question to a model**: the procedure of the day, in order, for their own problems.
6. **2.3 Feedback** (Mattan) and **2.4 Credits**.

Things to avoid saying: that the G+C difference is purely an artefact (part may be biology of where
AGO2 binds), that frozen models "only" encode composition (they keep 0.77 on matched data), or that
fine-tuning "wins" in general (one benchmark, one test set).
