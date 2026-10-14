"""Build the workshop page into _site/ from what is in this branch (run by .github/workflows/pages.yml).

    python site/build.py

Uses only the standard library. A notebook or deck that is not in the branch yet is shown as
"not yet available" instead of a broken link.
"""
import html
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_site"
REPO = "https://github.com/mathoory/ai-days-2026-dna-workshop"
COLAB = "https://colab.research.google.com/github/mathoory/ai-days-2026-dna-workshop/blob/public/notebooks/{}.ipynb"
FORM = "https://forms.gle/NfE6ZHj98ri3WYmG6"
LAB = "https://roee-amit.technion.ac.il/"
TITLE = "Applications of AI in Biotechnology and Genomics"

# (start, end, kind, title, who, stem); kind: talk, part, break
SCHEDULE = [
    ("10:00", "10:50", "talk", "Opening lecture", "Prof. Roee Amit", None),
    ("10:50", "11:00", "break", "Break", "", None),
    ("11:00", "11:40", "part", "Part 1. Machine learning and CNNs", "Mattan Hoory", "01_cnn"),
    ("11:40", "11:50", "break", "Break", "", None),
    ("11:50", "12:30", "part", "Part 2. DNA foundation models", "Mattan Hoory", "02_foundation_models"),
    ("12:30", "12:40", "break", "Break", "", None),
    ("12:40", "13:20", "part", "Part 3. Shortcuts, controls and fine-tuning", "Mattan Hoory", "03_shortcuts_and_controls"),
    ("13:20", "13:30", "talk", "Wrap-up and feedback", "Mattan Hoory", None),
]
SUMMARY = {
    "01_cnn": "Build a CNN that predicts AGO2 binding from RNA sequence, then read motifs off its filters and test them.",
    "02_foundation_models": "Turn sequences into vectors with a pretrained DNA language model and train a linear probe on them.",
    "03_shortcuts_and_controls": "Measure what composition alone explains, run a shuffle test and a matched control, and fine-tune with LoRA.",
}
GITHUB_MARK = ('<svg viewBox="0 0 16 16" width="22" height="22" aria-hidden="true"><path fill="currentColor" d="M8 0C3.58 0 0 '
               '3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23'
               '-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78'
               '-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 '
               '1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25'
               '.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>')


def link(href, text, cls="btn", download=False):
    d = " download" if download else ""
    ext = "" if href.startswith(("notebooks/", "slides/")) else ' target="_blank" rel="noopener"'
    return f'<a class="{cls}" href="{html.escape(href)}"{d}{ext}>{html.escape(text)}</a>'


def row(start, end, kind, title, who, stem):
    t = f'<div class="time">{start}<span>{end}</span></div>'
    if kind == "break":
        return f'<li class="slot break">{t}<div class="what">{title}</div></li>'
    acts = ""
    if stem:
        nb, deck = ROOT / "notebooks" / f"{stem}.ipynb", ROOT / "slides" / f"{stem}.html"
        a = []
        a.append(link(COLAB.format(stem), "Open in Colab", "btn primary") if nb.exists() else '<span class="btn off">Notebook not yet available</span>')
        if nb.exists():
            a.append(link(f"notebooks/{stem}.ipynb", "Download notebook", download=True))
        a.append(link(f"slides/{stem}.html", "Slides") if deck.exists() else '<span class="btn off">Slides not yet available</span>')
        acts = f'<div class="actions">{"".join(a)}</div>'
    desc = f'<p class="desc">{html.escape(SUMMARY[stem])}</p>' if stem else ""
    who = f'<div class="who">{html.escape(who)}</div>' if who else ""
    return f'<li class="slot {kind}">{t}<div class="what"><h3>{html.escape(title)}</h3>{who}{desc}{acts}</div></li>'


def logos():
    out = []
    for name, alt, href in [("technion", "Technion", "https://www.technion.ac.il"),
                            ("faculty", "Faculty of Biotechnology and Food Engineering", "https://biotech.technion.ac.il")]:
        for ext in ("svg", "png"):
            if (ROOT / "site" / "assets" / f"{name}.{ext}").exists():
                out.append(f'<a href="{href}" target="_blank" rel="noopener"><img src="assets/{name}.{ext}" alt="{alt}"></a>')
                break
    return "".join(out)


def page():
    built = datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M UTC")
    slots = "\n".join(row(*s) for s in SCHEDULE)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{TITLE} | Technion AI Days 2026</title>
<meta name="description" content="Notebooks and slides of the AI Days 2026 workshop at the Technion.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Roboto+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="assets/style.css">
</head>
<body>
<header class="top">
  <div class="wrap top-in">
    <div class="logos">{logos()}</div>
    <a class="gh" href="{REPO}" target="_blank" rel="noopener">{GITHUB_MARK}<span>GitHub</span></a>
  </div>
</header>
<section class="hero">
  <div class="wrap">
    <p class="kicker">Technion AI Days 2026 &nbsp;/&nbsp; Workshop</p>
    <h1>{TITLE}</h1>
    <p class="meta">Thursday 15 October 2026, 10:00 to 13:30 &nbsp;/&nbsp; Faculty of Biotechnology and Food Engineering, Technion</p>
    <p class="meta">Prof. Roee Amit &nbsp;and&nbsp; Mattan Hoory</p>
    <p class="host">Hosted by the <a href="{LAB}" target="_blank" rel="noopener">Roee Amit Lab</a></p>
  </div>
</section>
<main class="wrap">
  <section>
    <h2>Schedule</h2>
    <ol class="schedule">
{slots}
    </ol>
  </section>
  <section class="lab">
    <h2>The host lab</h2>
    <div class="lab-in">
      <div>
        <h3>Roee Amit Lab</h3>
        <p class="lab-sub">The Synthetic Biology Laboratory for the Decipherment of Genomic Codes,
          Faculty of Biotechnology and Food Engineering, Technion</p>
        <p>The lab studies the regulatory codes in the genome beyond the genetic code. It designs libraries of
          tens of thousands of synthetic DNA sequences, measures them in bacteria, yeast and mammalian cells,
          and fits models of the encoding rules to the results.</p>
      </div>
      <a class="lab-link" href="{LAB}" target="_blank" rel="noopener">roee-amit.technion.ac.il<span>→</span></a>
    </div>
  </section>
  <section class="feedback">
    <div>
      <h2>Feedback</h2>
      <p>{link(FORM, "Open the feedback form", "btn primary")}</p>
    </div>
    <img class="qr" src="assets/qr_feedback.svg" alt="QR code for the feedback form">
  </section>
</main>
<footer class="foot">
  <div class="wrap foot-in">
    <a class="gh" href="{REPO}" target="_blank" rel="noopener">{GITHUB_MARK}<span>mathoory/ai-days-2026-dna-workshop</span></a>
    <p>Data: AGO2 CLIP-seq from the GraphProt benchmark (Maticzka et al. 2014), as used by Pan and Shen (2018).</p>
    <p>Pipeline ideas from the KAUST Genomic Foundation Models workshop (I. Ullah, 2026):
      <a href="https://github.com/bioinfo-kaust/gfm-workshop" target="_blank" rel="noopener">github.com/bioinfo-kaust/gfm-workshop</a></p>
    <p class="built">Updated {built}</p>
  </div>
</footer>
</body>
</html>
"""


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    shutil.copytree(ROOT / "site" / "assets", OUT / "assets", ignore=shutil.ignore_patterns("*.py"))
    for d in ("notebooks", "slides"):
        if (ROOT / d).exists():
            shutil.copytree(ROOT / d, OUT / d)
    (OUT / "index.html").write_text(page(), encoding="utf-8")
    (OUT / ".nojekyll").write_text("")
    print("built", OUT)


if __name__ == "__main__":
    main()
