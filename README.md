# Applications of AI in Biotechnology and Genomics

Hands-on notebooks and slides for the AI Days 2026 workshop at the Technion (15.10.2026), Prof. Roee Amit and Mattan Hoory.

**Workshop page: [mathoory.github.io/ai-days-2026-dna-workshop](https://mathoory.github.io/ai-days-2026-dna-workshop/)**

Open a notebook in Colab, then *Runtime → Change runtime type → T4 GPU*, and run from the top. The data downloads automatically.

| Time | Part | | |
|---|---|---|---|
| 11:00 | 1. Machine learning and CNNs | [Open in Colab](https://colab.research.google.com/github/mathoory/ai-days-2026-dna-workshop/blob/public/notebooks/01_cnn.ipynb) | [Slides](https://mathoory.github.io/ai-days-2026-dna-workshop/slides/01_cnn.html) |
| 12:25 | 2. DNA foundation models | [Open in Colab](https://colab.research.google.com/github/mathoory/ai-days-2026-dna-workshop/blob/public/notebooks/02_foundation_models.ipynb) | [Slides](https://mathoory.github.io/ai-days-2026-dna-workshop/slides/02_foundation_models.html) |
| 13:25 | 3. Shortcuts, controls and fine-tuning | [Open in Colab](https://colab.research.google.com/github/mathoory/ai-days-2026-dna-workshop/blob/public/notebooks/03_shortcuts_and_controls.ipynb) | [Slides](https://mathoory.github.io/ai-days-2026-dna-workshop/slides/03_shortcuts_and_controls.html) |

Data: [huggingface.co/datasets/yup8/ai-days-2026-ago2](https://huggingface.co/datasets/yup8/ai-days-2026-ago2) (AGO2 CLIP-seq from the GraphProt benchmark, Maticzka et al. 2014, as used by Pan and Shen 2018, *iDeepE*).

Feedback: [forms.gle/NfE6ZHj98ri3WYmG6](https://forms.gle/NfE6ZHj98ri3WYmG6)

## How the page is built

`site/build.py` builds the page from `notebooks/` and `slides/` in this branch. `.github/workflows/pages.yml` runs it on every push
to `public` and deploys it to GitHub Pages, so new notebooks or decks appear on the page within a minute or two.
