"""After every render: stamp style.css and every figure in the decks with a content hash (?v=...), so a browser
never shows a stale stylesheet or an old version of a figure that kept its file name."""
import hashlib
import re
from pathlib import Path

SLIDES = Path(__file__).resolve().parents[1] / "slides"


def stamp(m):
    attr, path = m.group(1), m.group(2)
    f = SLIDES / path
    if not f.is_file():
        return m.group(0)
    v = hashlib.md5(f.read_bytes()).hexdigest()[:8]
    return f'{attr}="{path}?v={v}"'


for html in SLIDES.glob("0*.html"):
    t = html.read_text(encoding="utf-8")
    u = re.sub(r'(href|src)="((?:style\.css|assets/[^"?]+\.(?:png|jpg|jpeg|svg)))(?:\?v=[0-9a-f]+)?"', stamp, t)
    if u != t:
        html.write_text(u, encoding="utf-8")
