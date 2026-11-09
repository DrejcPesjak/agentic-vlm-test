"""Re-theme the blog figures for the dark site at drejcpesjak.github.io/website.

Reads the light SVGs in docs/blog/assets, swaps the palette for the site's
Tailwind grays with the red-400 accent, and writes SVG and PNG copies to
docs/blog/assets-dark. The light figures are never touched.

Run from the project root with: uv run python scripts/build_blog_dark.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIGHT = ROOT / "docs/blog/assets"
DARK = ROOT / "docs/blog/assets-dark"

# Light palette -> site palette (Tailwind v2 grays, red-400 accent, amber as second accent).
PALETTE = {
    "#f4f1e9": "#111827",  # canvas -> gray-900
    "white": "#1f2937",  # cards -> gray-800
    "#d2dcd9": "#374151",  # card stroke -> gray-700
    "#193339": "#f9fafb",  # ink -> near white (the dark LM box becomes a light block)
    "#c5d9d4": "#4b5563",  # subtitles inside that block -> gray-600
    "#5d7276": "#9ca3af",  # muted -> gray-400
    "#007e73": "#f87171",  # teal accent -> red-400
    "#bd4d37": "#fbbf24",  # rust accent -> amber-400
    "#e1f0eb": "#3f1d1d",  # teal tint box -> dark red tint
    "#83aaa0": "#b91c1c",  # its stroke -> red-700
    "#fff0e6": "#3a2e12",  # rust tint box -> dark amber tint
    "#d6b197": "#b45309",  # its stroke -> amber-700
    "#9cc8ba": "#7dd3fc",  # image patches, photo-like: sky, grass, sand, cloud
    "#b6d9d2": "#86efac",
    "#d9b58a": "#fde68a",
    "#789d91": "#cbd5e1",
    "#889b9e": "#4b5563",  # results: single view bars -> gray-600
    "#aec5c6": "#9ca3af",  # results: repeat bars -> gray-400
    "#e1e5df": "#1f2937",  # results: bar track -> gray-800
    "#e5ebe6": "#1f2937",  # results: note box -> gray-800
}


PASS = "#4ade80"  # green-400
FAIL = "#f87171"  # red-400
LABEL = "#9ca3af"  # gray-400, same as the FULL IMAGE label

# Per-text overrides applied after the palette swap: panel titles go regular gray,
# answers are coloured by whether the model was right.
TEXT_COLORS = {
    "THE MODEL’S CHOSEN CROP": LABEL,
    "CROP 1": LABEL,
    "CROP 2": LABEL,
    "CROP 3": LABEL,
    "U": FAIL,
    "i": PASS,
    "Single-view answer o": FAIL,
    "Answer after three crops y": PASS,
    "Circled y is visible": PASS,
}


def retheme(svg: str) -> str:
    def swap(match: re.Match) -> str:
        attr, value = match.group(1), match.group(2)
        return f'{attr}="{PALETTE.get(value.lower(), value)}"'

    svg = re.sub(r'\b(fill|stroke)="([^"]+)"', swap, svg)

    def recolor(match: re.Match) -> str:
        head, content = match.group(1), match.group(2)
        color = TEXT_COLORS.get(content)
        if color:
            head = re.sub(r'fill="[^"]+"', f'fill="{color}"', head)
        return f"{head}>{content}</text>"

    return re.sub(r"(<text [^>]*)>([^<]*)</text>", recolor, svg)


def render(paths: list[Path]) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sync_playwright = None
    if sync_playwright:
        try:
            with sync_playwright() as playwright:
                chrome = shutil.which("google-chrome")
                browser = (
                    playwright.chromium.launch(executable_path=chrome)
                    if chrome
                    else playwright.chromium.launch()
                )
                page = browser.new_page(viewport={"width": 1400, "height": 1000})
                for path in paths:
                    page.goto(path.as_uri())
                    page.locator("svg").screenshot(path=str(path.with_suffix(".png")))
                browser.close()
            return
        except Exception as error:  # fall through to ImageMagick
            print(f"playwright unavailable ({error.__class__.__name__}), using convert")
    converter = shutil.which("convert")
    assert converter, "Need playwright with Chromium or ImageMagick convert"
    for path in paths:
        subprocess.run(
            [converter, "-density", "144", str(path), str(path.with_suffix(".png"))],
            check=True,
        )


def main() -> None:
    DARK.mkdir(exist_ok=True)
    written = []
    for source in sorted(LIGHT.glob("*.svg")):
        target = DARK / source.name
        target.write_text(retheme(source.read_text()))
        written.append(target)
    render(written)
    print({"dark_figures": len(written), "output": str(DARK)})


if __name__ == "__main__":
    main()
