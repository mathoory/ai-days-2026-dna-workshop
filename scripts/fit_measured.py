"""Fit every figure in the decks to the space its slide leaves, measured in a real browser.

Needs the decks served at http://127.0.0.1:8765/ (python -m http.server 8765 in slides/) and rendered.
    python scripts/fit_measured.py [deck-stem ...]          # measure and rewrite the heights in the .qmd
    python scripts/fit_measured.py --check [deck-stem ...]  # only report slides whose content overflows

For each slide (all fragments shown) the page reports the bottom of its visible content and, for every image,
its size, the width it may use (its column, or the slide) and whether it sits in an r-stack. Full-width images
on a slide share the free height; images in columns grow until their column is full. The new height is
written into the image's `{...}` attributes (height="Npx", replacing any width or height).
"""
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

SLIDES = Path(__file__).resolve().parents[1] / "slides"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
BOTTOM, MAX_W, MIN_STEP = 850, 1440, 8

MEASURE_JS = r"""
<script>
(function () {
  function loaded(sl) {      // reveal loads a slide's images only when it is shown; wait for them
    return Promise.all([].map.call(sl.querySelectorAll('img'), function (im) {
      if (im.complete && im.naturalWidth) return Promise.resolve();
      return new Promise(function (res) { im.addEventListener('load', res); im.addEventListener('error', res);
                                          setTimeout(res, 4000); });
    }));
  }
  async function go() {
    document.querySelectorAll('.fragment').forEach(function (f) { f.classList.add('visible'); });
    var out = [], n = Reveal.getTotalSlides();
    for (var i = 0; i < n; i++) {
      Reveal.slide(i);
      document.querySelectorAll('.fragment').forEach(function (f) { f.classList.add('visible'); });
      await loaded(Reveal.getCurrentSlide());
      await new Promise(function (res) { setTimeout(res, 150); });
      var sl = Reveal.getCurrentSlide(), r = sl.getBoundingClientRect(), s = Reveal.getScale(), bottom = 0;
      sl.querySelectorAll('img,p,li,table,h2,pre,.MathJax,.katex-display,mjx-container,span').forEach(function (e) {
        if (e.closest('aside')) return;
        var b = e.getBoundingClientRect(); if (!b.width) return;
        bottom = Math.max(bottom, (b.bottom - r.top) / s);
      });
      var imgs = [].map.call(sl.querySelectorAll('img'), function (im) {
        var b = im.getBoundingClientRect(), col = im.closest('.column');
        return {src: im.getAttribute('src'), h: b.height / s, w: b.width / s,
                col: col ? col.getBoundingClientRect().width / s : null, stack: !!im.closest('.r-stack')};
      }).filter(function (o) { return o.src && o.src.indexOf('logo') < 0 && o.h > 0; });
      out.push({i: i, title: (sl.querySelector('h1,h2') || {}).textContent || '', bottom: bottom, imgs: imgs});
    }
    fetch('http://127.0.0.1:8766/', {method: 'POST', mode: 'no-cors', body: JSON.stringify(out)});
  }
  function wait() {
    if (window.Reveal && Reveal.isReady && Reveal.isReady()) setTimeout(go, 1500); else setTimeout(wait, 200);
  }
  wait();
})();
</script>
"""


def measure(stem):
    """Open the deck headless with a measuring script; the page POSTs its numbers to a listener on :8766."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    got = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            got["body"] = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8")
            self.send_response(200); self.end_headers()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 8766), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    src = SLIDES / f"{stem}.html"
    tmp = SLIDES / f"_measure_{stem}.html"
    tmp.write_text(src.read_text(encoding="utf-8").replace("</body>", MEASURE_JS + "</body>"), encoding="utf-8")
    prof = SLIDES.parent / ".measure_profile"
    try:
        subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--user-data-dir={prof}",
                        "--window-size=1600,900", f"http://127.0.0.1:8765/{tmp.name}"], capture_output=True)
        for _ in range(240):
            if "body" in got:
                break
            time.sleep(0.5)
        if "body" not in got:
            raise SystemExit(f"no measurement for {stem}")
        return json.loads(got["body"])
    finally:
        srv.shutdown()
        tmp.unlink(missing_ok=True)
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        "Get-CimInstance Win32_Process -Filter \"Name='msedge.exe'\" | Where-Object { $_.CommandLine "
                        "-like '*.measure_profile*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force "
                        "-ErrorAction SilentlyContinue }"], capture_output=True)
        time.sleep(1)
        shutil.rmtree(prof, ignore_errors=True)


def plan(slides):
    """New height per image occurrence, in slide order: list of (src, new_h or None)."""
    out = []
    for sl in slides:
        free = BOTTOM - sl["bottom"]
        imgs = sl["imgs"]
        full = [im for im in imgs if im["col"] is None]
        # full-width images share the free height; an r-stack counts once
        blocks, seen_stack = [], False
        for im in full:
            if im["stack"]:
                if not seen_stack:
                    blocks.append(im); seen_stack = True
            else:
                blocks.append(im)
        H = sum(b["h"] for b in blocks)
        g_full = None
        if blocks and H > 0:
            g_v = (H + free) / H
            g_w = min(MAX_W / im["w"] for im in full)
            g_full = min(g_v, g_w)
        for im in imgs:
            if im["col"] is None:
                g = g_full
            else:
                g = min((im["h"] + free) / im["h"], im["col"] / im["w"])
            new = int(im["h"] * g) if g else None
            out.append((im["src"], new if new and abs(new - im["h"]) >= MIN_STEP else None, sl["title"]))
    return out


def apply(stem, changes):
    p = SLIDES / f"{stem}.qmd"
    t = p.read_text(encoding="utf-8")
    pos = 0
    n = 0
    for src, new, title in changes:
        pat = re.compile(r"!\[[^\]]*\]\(" + re.escape(src) + r"\)\{([^}]*)\}")
        m = pat.search(t, pos)
        if not m:
            continue
        pos = m.end()
        if new is None:
            continue
        attrs = re.sub(r'\s*(height|width)="[^"]*"', "", m.group(1)).strip()
        attrs = (attrs + f' height="{new}px"').strip()
        t = t[:m.start(1)] + attrs + t[m.end(1):]
        pos = m.start(1) + len(attrs) + 1
        n += 1
        print(f"  {title[:44]:44s} {src.split('/')[-1]:28s} -> {new}px")
    p.write_text(t, encoding="utf-8")
    return n


def render(stem):
    subprocess.run(["quarto", "render", f"{stem}.qmd"], cwd=SLIDES, capture_output=True)


def aspects(slides):
    """Ideal width / height per image: the width it may use over the height its slide leaves for it."""
    out = {}
    for sl in slides:
        free = BOTTOM - sl["bottom"]
        full = [im for im in sl["imgs"] if im["col"] is None]
        blocks, seen = [], False
        for im in full:
            if im["stack"]:
                if not seen:
                    blocks.append(im); seen = True
            else:
                blocks.append(im)
        H = sum(b["h"] for b in blocks) or 1
        for im in sl["imgs"]:
            stem = im["src"].split("/")[-1].rsplit(".", 1)[0]
            if im["col"] is None:
                h_avail = im["h"] * (H + free) / H
                out[stem] = round(MAX_W / h_avail, 3)
            else:
                out[stem] = round(im["col"] / (im["h"] + free), 3)
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    check = "--check" in args
    if "--aspects" in args:
        table = {}
        for stem in [a_ for a_ in args if not a_.startswith("--")] or [q.stem for q in sorted(SLIDES.glob("0*.qmd"))]:
            table.update(aspects(measure(stem)))
        path = SLIDES / "figures" / "aspects.json"
        path.write_text(json.dumps(table, indent=1, sort_keys=True), encoding="utf-8")
        print("wrote", path, len(table), "figures")
        raise SystemExit
    stems = [a for a in args if not a.startswith("--")] or [q.stem for q in sorted(SLIDES.glob("0*.qmd"))]
    for stem in stems:
        if check:
            for sl in measure(stem):
                if sl["bottom"] > BOTTOM + 15 or any(im["w"] > MAX_W + 40 for im in sl["imgs"]):
                    print(f"{stem}: slide {sl['i']} {sl['title'][:50]} bottom {sl['bottom']:.0f}")
            continue
        for rnd in range(2):
            print(f"== {stem}, pass {rnd + 1}")
            n = apply(stem, plan(measure(stem)))
            render(stem)
            if not n:
                break
