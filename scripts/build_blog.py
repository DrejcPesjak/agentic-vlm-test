"""Build original SVG blog figures from saved evaluation evidence, without API calls.

Run from the project root with .venv/bin/python scripts/build_blog.py.
PNG rendering is a separate browser step so the SVG sources remain editable.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import re
import shutil
import subprocess

from PIL import Image
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOG = ROOT / "docs/blog"
ASSETS = BLOG / "assets"
RUN = ROOT / "artifacts/evaluation"
INK = "#193339"
MUTED = "#5d7276"
TEAL = "#007e73"
RUST = "#bd4d37"
BG = "#f4f1e9"


def text(x, y, value, size=24, fill=INK, weight=400, anchor="start"):
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
        f'font-weight="{weight}" text-anchor="{anchor}">{html.escape(str(value))}</text>'
    )


def rect(x, y, width, height, fill="white", stroke="none", radius=16):
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
        f'rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
    )


def arrow(x1, y1, x2, y2, color=TEAL):
    import math

    angle = math.atan2(y2 - y1, x2 - x1)
    ux, uy = math.cos(angle), math.sin(angle)
    bx, by = x2 - 12 * ux, y2 - 12 * uy
    return (
        f'<path d="M {x1} {y1} L {bx} {by}" fill="none" stroke="{color}" stroke-width="3"/>'
        f'<polygon points="{x2},{y2} {bx - 6 * uy},{by + 6 * ux} '
        f'{bx + 6 * uy},{by - 6 * ux}" fill="{color}"/>'
    )


def picture(path, x, y, width, height):
    # Set the actual image dimensions explicitly. Some SVG rasterizers ignore
    # preserveAspectRatio and otherwise stretch images to their container.
    with Image.open(path) as source:
        source_width, source_height = source.size
    scale = min(width / source_width, height / source_height)
    shown_width, shown_height = source_width * scale, source_height * scale
    x += (width - shown_width) / 2
    y += (height - shown_height) / 2
    data = base64.b64encode(path.read_bytes()).decode()
    return (
        f'<image x="{x}" y="{y}" width="{shown_width}" height="{shown_height}" '
        f'href="data:image/png;base64,{data}"/>'
    )


def box(x, y, width, title, subtitle, fill="white"):
    title_size = 27 if len(title) <= 12 else 21 if len(title) <= 17 else 18
    subtitle_size = 20 if len(subtitle) <= 21 else 16
    return (
        rect(x, y, width, 108, fill, "#d2dcd9")
        + text(
            x + width / 2,
            y + 44,
            title,
            title_size,
            weight=650,
            anchor="middle",
        )
        + text(x + width / 2, y + 78, subtitle, subtitle_size, MUTED, anchor="middle")
    )


def save(name, height, body, description):
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="{height}" '
        f'viewBox="0 0 1400 {height}" role="img" aria-labelledby="title desc">'
        f'<title id="title">{html.escape(name.replace("-", " "))}</title>'
        f'<desc id="desc">{html.escape(description)}</desc>'
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" '
        'refY="4" orient="auto"><path d="M 0 0 L 8 4 L 0 8" fill="#007e73"/>'
        "</marker></defs>"
        f'<rect width="1400" height="{height}" fill="{BG}"/>'
        '<g font-family="Arial, sans-serif">' + body + "</g></svg>"
    )
    (ASSETS / f"{name}.svg").write_text(svg)


def load_episode(eid):
    directory = RUN / "episodes" / eid
    return directory, json.loads((directory / "episode.json").read_text())


def build_figures():
    ASSETS.mkdir(parents=True, exist_ok=True)
    directory, episode = load_episode("70c118cf37c2e168c6e3f5cc")
    assert episode["groundtruth"] == "i" and episode["correct"]
    body = text(45, 65, "Which letter is circled?", 44, weight=700)
    body += text(45, 105, "A real GPT-5.4 example from our evaluation", 24, MUTED)
    body += rect(40, 145, 600, 630) + rect(760, 145, 600, 630)
    body += text(65, 185, "FULL IMAGE", 22, MUTED, 700)
    body += text(785, 185, "THE MODEL’S CHOSEN CROP", 22, TEAL, 700)
    body += picture(directory / "view-1.png", 65, 205, 550, 480)
    # Image uses preserveAspectRatio meet, so compute its actual centered position.
    size = 480
    x0, y0 = 100, 205
    left, top, right, bottom = episode["views"][1]["box"]
    sx, sy = size / episode["width"], size / episode["height"]
    body += rect(
        x0 + left * sx, y0 + top * sy, (right - left) * sx, (bottom - top) * sy, "none", TEAL, 0
    )
    body += picture(directory / "view-2.png", 785, 205, 550, 480)
    body += arrow(665, 440, 735, 440)
    body += text(65, 728, "Single-view answer", 23, MUTED)
    body += text(400, 730, "U", 44, RUST, 700)
    body += text(785, 728, "Crop-run answer", 23, MUTED)
    body += text(1110, 730, "i", 44, TEAL, 700)
    body += text(
        45,
        817,
        "Correct answer i   •   Separate runs   •   Original pixels, no generated detail",
        22,
        MUTED,
    )
    save(
        "poster-example",
        850,
        body,
        "Original image and exact returned crop. "
        "Single-view answer U is wrong. Crop-run answer i is correct.",
    )

    # Show the tasks at a useful size, without mixing in crops yet.
    task_examples = [
        ("Line Plot Intersections", "0a34f90f40b19cd23f410b80", "Count crossings"),
        ("Circled Letter", "08d78f2baa52c16551613cd8", "Read the circled letter"),
        ("Subway Connections", "6223c1270c235fc44ac3c71a", "Count routes between stations"),
    ]
    body = text(55, 70, "Three tasks. Answers we can check.", 42, weight=700)
    descriptions = [
        ("Count the crossings", "Where do the red and blue lines intersect?"),
        ("Read the circled letter", "Which character sits inside the red ring?"),
        ("Trace the subway routes", "How many routes connect the named stations?"),
    ]
    for index, ((_, episode_id, _), (heading, description)) in enumerate(zip(task_examples, descriptions, strict=True)):
        directory, _ = load_episode(episode_id)
        y = 125 + index * 410
        body += rect(40, y, 1320, 380)
        body += text(80, y + 60, str(index + 1).zfill(2), 24, TEAL, 700)
        body += text(80, y + 115, heading, 32, weight=700)
        body += text(80, y + 165, description, 21, MUTED)
        image_name = "view-2.png" if index == 1 else "view-1.png"
        body += picture(directory / image_name, 700, y + 20, 620, 340)
        if index == 1:
            body += text(1010, y + 355, "Enlarged detail from the test image", 18, MUTED, anchor="middle")
    save("task-gallery", 1390, body, "Three benchmark tasks. The letter detail is enlarged for readability. All images preserve their original proportions.")

    body = text(55, 70, "Pixels and words take different paths", 42, weight=700)
    body += text(55, 112, "A simplified view of the Gemma 4 design", 24, MUTED)
    # The two streams stay spacious. The image patches and embedding vectors
    # are visual elements rather than more boxes with long labels.
    body += text(65, 185, "IMAGE", 20, TEAL, 700)
    for row in range(3):
        for col in range(4):
            body += rect(65 + col * 44, 215 + row * 44, 38, 38,
                         ["#9cc8ba", "#b6d9d2", "#d9b58a", "#789d91"][(row + col) % 4], radius=3)
    body += text(65, 385, "Split into patches", 22, MUTED)
    body += arrow(255, 278, 345, 278)
    body += rect(345, 195, 350, 185, "#e1f0eb", "#83aaa0")
    body += text(520, 245, "Vision encoder", 30, weight=700, anchor="middle")
    body += text(520, 287, "Encode patches", 22, anchor="middle")
    body += text(520, 323, "Pool and project", 22, anchor="middle")
    body += arrow(695, 278, 770, 278)
    for i, h in enumerate([64, 105, 80, 120, 72, 96]):
        body += rect(780 + i * 30, 338 - h, 20, h, TEAL, radius=3)
    body += text(865, 385, "Patch embeddings", 22, TEAL, anchor="middle")
    body += text(65, 505, "QUESTION", 20, RUST, 700)
    body += text(65, 565, "Which letter", 27, weight=650)
    body += text(65, 605, "is circled?", 27, weight=650)
    body += arrow(255, 570, 345, 570, RUST)
    body += rect(345, 485, 350, 185, "#fff0e6", "#d6b197")
    body += text(520, 535, "Text embeddings", 29, weight=700, anchor="middle")
    body += text(520, 577, "Split into tokens", 22, anchor="middle")
    body += text(520, 613, "Look up word vectors", 22, anchor="middle")
    body += arrow(695, 570, 770, 570, RUST)
    for i, h in enumerate([90, 65, 115, 85, 100, 72]):
        body += rect(780 + i * 30, 630 - h, 20, h, RUST, radius=3)
    body += text(865, 680, "Word embeddings", 22, RUST, anchor="middle")
    body += arrow(970, 278, 1040, 278)
    body += arrow(970, 570, 1040, 570, RUST)
    body += rect(1040, 195, 245, 475, "#193339", radius=20)
    body += text(1162, 392, "Language", 32, "white", 700, "middle")
    body += text(1162, 437, "model", 32, "white", 700, "middle")
    body += text(1162, 490, "Reasons over", 21, "#c5d9d4", anchor="middle")
    body += text(1162, 522, "both streams", 21, "#c5d9d4", anchor="middle")
    body += arrow(1285, 430, 1370, 430)
    body += text(1370, 400, "Answer", 20, anchor="end")
    body += text(65, 770, "The vision encoder builds its representation without using the question.", 26, weight=600)
    save("vlm-architecture", 825, body, "Image patches and text tokens become embeddings. Both streams enter the language model from the left.")

    body = text(55, 70, "One crop run, step by step", 44, weight=700)
    body += text(55, 112, "The same VLM is called again with the crop added to its conversation", 24, MUTED)
    rows = [
        (180, "1", "Question + full image", "First model call", "Crop coordinates"),
        (430, "2", "Coordinates + original pixels", "Crop tool", "New image"),
        (680, "3", "Same chat + returned crop", "Next model call", "Answer or crop again"),
    ]
    for y, number, incoming, action, outgoing in rows:
        body += text(60, y + 68, number, 42, TEAL, 700)
        body += rect(135, y, 410, 135, "white", "#d2dcd9")
        body += text(340, y + 75, incoming, 25, anchor="middle")
        body += arrow(545, y + 67, 620, y + 67)
        if number == "2":
            body += f'<path d="M 650 {y} H 920 L 950 {y+67} L 920 {y+135} H 650 L 620 {y+67} Z" fill="#fff0e6" stroke="#bd4d37" stroke-width="2"/>'
        else:
            body += rect(620, y, 330, 135, INK)
        body += text(785, y + 60, action, 27, RUST if number == "2" else "white", 700, "middle")
        body += text(785, y + 97, "Cuts pixels" if number == "2" else "Same VLM", 21, RUST if number == "2" else "#c5d9d4", anchor="middle")
        body += arrow(950, y + 67, 1025, y + 67)
        body += text(1045, y + 75, outgoing, 25, weight=600)
    body += text(135, 365, "The model chooses the rectangle. The tool receives that request.", 23, MUTED)
    body += text(135, 615, "The returned crop is appended as a tool result in the same chat.", 23, MUTED)
    body += text(135, 890, "If the model requests another crop, repeat steps 2 and 3. Maximum five image views.", 24, TEAL, 600)
    save("crop-loop", 940, body, "Three chronological steps. The model requests coordinates, a tool cuts original pixels, and the same model receives the crop with its conversation history.")

    directory, episode = load_episode("2a6072bf1dc2f15de21c494d")
    assert len(episode["views"]) == 4 and episode["groundtruth"] == "y"
    body = text(45, 70, "One crop was not always enough", 44, weight=700)
    body += text(45, 112, "GPT-4o chose these three regions in order", 25, MUTED)
    for i, view in enumerate(episode["views"][1:]):
        x = 40 + 450 * i
        body += rect(x, 155, 420, 470)
        body += text(x + 22, 200, f"CROP {i + 1}", 22, TEAL, 700)
        body += picture(directory / view["image"], x + 15, 225, 390, 310)
        label = "Circle outside this crop" if i < 2 else "Circled y is visible"
        body += text(x + 22, 585, label, 24, MUTED if i < 2 else TEAL, 600)
    body += text(45, 685, "Single-view answer o", 27, RUST, 600)
    body += text(510, 685, "Answer after three crops y", 27, TEAL, 600)
    body += text(
        45,
        738,
        "Actual returned images   •   Captions describe the pixels, not hidden model reasoning",
        22,
        MUTED,
    )
    save(
        "three-crops",
        775,
        body,
        "Three recorded crops. The first two miss the red circle. "
        "The third contains the circled y and the model answers y.",
    )

    stats = json.loads((RUN / "analysis.json").read_text())
    body = text(45, 70, "A bigger gain for GPT-5.4", 44, weight=700)
    body += text(45, 110, "Correct answers out of 60 images   •   One run per condition", 25, MUTED)
    colors = {"single": "#889b9e", "repeat": "#aec5c6", "crop": TEAL}
    labels = {"single": "Single view", "repeat": "Repeat enabled", "crop": "Required crop"}
    for i, model in enumerate(["gpt-5.4-2026-03-05", "gpt-4o-2024-08-06"]):
        x = 45 + i * 690
        body += text(x, 180, "GPT-5.4" if i == 0 else "GPT-4o", 34, weight=700)
        rows = {
            r["condition"]: r
            for r in stats["groups"]
            if r["model"] == model and r["task"] == "All tasks"
        }
        for j, condition in enumerate(["single", "repeat", "crop"]):
            y = 225 + j * 115
            row = rows[condition]
            body += text(x, y, labels[condition], 25, MUTED)
            body += rect(x, y + 18, 500, 44, "#e1e5df", radius=5)
            body += rect(x, y + 18, 500 * row["accuracy"], 44, colors[condition], radius=5)
            body += text(x + 525, y + 50, f"{row['correct']}/60", 30, weight=700)
    body += rect(40, 600, 1320, 110, "#e5ebe6")
    body += text(
        65, 644, "Every repeat-enabled agent stopped after its first view.", 27, weight=600
    )
    body += text(
        65, 681, "The view limit was equal. The actual number of views was not.", 24, MUTED
    )
    save(
        "results",
        750,
        body,
        "GPT-5.4 scored 44, 47, and 54 out of 60. GPT-4o scored 34, 33, and 36. "
        "The repeat-enabled agents all answered from one view.",
    )
    episodes = [json.loads(p.read_text()) for p in (RUN / "episodes").glob("*/episode.json")]
    provenance = {
        "run": "artifacts/evaluation",
        "protocol": 2,
        "poster_episode": "70c118cf37c2e168c6e3f5cc",
        "multi_crop_episode": "2a6072bf1dc2f15de21c494d",
        "dataset": "https://huggingface.co/datasets/XAI/vlmsareblind",
        "diagram_reference": "https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4",
        "architecture_diagram": "Original simplified redraw. Does not reuse the guide's image.",
        "task_examples": {
            "line_intersections": "0a34f90f40b19cd23f410b80",
            "circled_letter": "08d78f2baa52c16551613cd8",
            "subway_connections": "6223c1270c235fc44ac3c71a",
        },
        "crop_counts": {},
    }
    for model in sorted({e["model"] for e in episodes}):
        counts = Counter(
            sum(v["box"] is not None for v in e["views"])
            for e in episodes
            if e["model"] == model and e["condition"] == "crop"
        )
        provenance["crop_counts"][model] = dict(sorted(counts.items()))
    (ASSETS / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def inline(value):
    value = html.escape(value)
    value = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', value)
    value = re.sub(r"`([^`]+)`", r"<code>\1</code>", value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", value)
    value = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", value)
    return value


def build_preview():
    """Render the small Markdown subset used by this article, without a runtime dependency."""
    source = (BLOG / "post.md").read_text()
    prose = re.sub(r"\]\([^)]*\)", "]", source)
    assert not any(c in prose for c in ("—", ";", ":")), "Forbidden punctuation in article prose"
    body = []
    for block in source.strip().split("\n\n"):
        if block.startswith("# "):
            body.append('<p class="eyebrow">A SMALL EXPERIMENT IN VISUAL ATTENTION</p>')
            body.append("<h1>" + inline(block[2:]) + "</h1>")
            body.append('<p class="meta">by <a href="https://github.com/DrejcPesjak">Drejc Pesjak</a> · About 7 minutes · 60 images · 2 models</p>')
        elif block.startswith("## "):
            body.append("<h2>" + inline(block[3:]) + "</h2>")
        elif block.startswith("!["):
            match = re.fullmatch(r"!\[([^\]]+)\]\(([^)]+)\)", block)
            assert match
            alt, path = match.groups()
            body.append(
                f'<figure><a href="{path}"><img src="{path}" '
                f'alt="{html.escape(alt)}" loading="lazy"></a></figure>'
            )
        elif block.startswith("> "):
            rows = []
            for line in block.splitlines():
                match = re.fullmatch(r"> \*\*([^*]+)\*\* (.+)", line)
                if match:
                    who, said = match.groups()
                    kind = {"You": "you", "Model": "model"}.get(who, "sees")
                    rows.append(
                        f'<div class="row {kind}"><span class="who">{html.escape(who)}</span>'
                        f"<span>{inline(said)}</span></div>"
                    )
            body.append('<div class="chat">' + "".join(rows) + "</div>")
        elif block.startswith("- "):
            body.append(
                "<ul>"
                + "".join("<li>" + inline(line[2:]) + "</li>" for line in block.splitlines())
                + "</ul>"
            )
        else:
            body.append("<p>" + inline(block.replace("\n", " ")) + "</p>")
    css = """*{box-sizing:border-box}body{margin:0;background:#faf8f2;color:#193339;
font-family:Georgia,serif}article{max-width:880px;padding:64px 30px 80px;margin:auto}
h1{font-family:Arial,sans-serif;font-size:clamp(38px,5vw,60px);line-height:1.08;
letter-spacing:-1.5px;margin:20px 0}h2{font-family:Arial,sans-serif;font-size:29px;
line-height:1.2;margin:52px 0 18px}p,li{font-size:20px;line-height:1.8}li{padding-left:5px}
a{color:#007e73;text-underline-offset:4px}strong{color:#142e33}figure{margin:28px -40px}
figure img{width:100%;display:block;border-radius:10px}.eyebrow,.meta{font-family:Arial,sans-serif;
font-size:12px;letter-spacing:1.7px;color:#5d7276}.meta{letter-spacing:0;font-size:15px}
figure+p:has(>em){font-size:16px;color:#5d7276;line-height:1.6}footer{margin-top:60px;
border-top:1px solid #d5dfda;padding-top:20px;font:14px Arial,sans-serif}
.chat{margin:26px 0;border:1px solid #d5dfda;border-radius:12px;background:#fff;overflow:hidden;
font-family:Arial,sans-serif}.chat .row{display:flex;gap:18px;align-items:baseline;padding:14px 20px;
font-size:17px;line-height:1.5}.chat .row+.row{border-top:1px solid #e6ece8}.chat .who{flex:0 0 150px;
font-size:12px;letter-spacing:1.2px;text-transform:uppercase;color:#5d7276}.chat .model{font-weight:700}
.chat .sees{background:#f1f6f3}code{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:15px;
background:#dcebe6;border:1px solid #b9d3cb;border-radius:5px;padding:2px 7px;margin-right:3px}
@media(max-width:700px){.chat .row{flex-direction:column;gap:4px}.chat .who{flex:none}article{padding:30px 22px}p,li{font-size:18px}
figure{margin:25px -10px}h2{font-size:26px}h1{letter-spacing:-1px}}
"""
    result = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta name="description" content="A VLM compresses the image before it sees your question. Letting it crop and look again took GPT-5.4 from 73% to 90% on 60 benchmark images.">'
        '<meta name="author" content="Drejc Pesjak">'
        '<meta property="og:type" content="article">'
        '<meta property="og:title" content="Why VLMs are built wrong and why agentic VLMs are the solution">'
        '<meta property="og:description" content="A VLM compresses the image before it sees your question. Letting it crop and look again took GPT-5.4 from 73% to 90% on 60 benchmark images.">'
        '<meta property="og:url" content="https://drejcpesjak.github.io/agentic-vlm-test/blog/">'
        '<meta property="og:image" content="https://drejcpesjak.github.io/agentic-vlm-test/blog/assets/emerald-sky.webp">'
        '<meta name="twitter:card" content="summary_large_image">'
        "<title>Why VLMs are built wrong</title><style>"
        + css
        + "</style></head><body><article>"
        + "\n".join(body)
        + '<footer><a href="https://github.com/DrejcPesjak/agentic-vlm-test">Code on GitHub</a> · '
        '<a href="post.md">Markdown source</a> · '
        '<a href="assets/provenance.json">Figure sources</a></footer></article></body></html>'
    )
    (BLOG / "index.html").write_text(result)
    print(
        {
            "article_words": len(re.findall(r"\b[\w’-]+\b", prose)),
            "preview": str(BLOG / "index.html"),
            "svg_figures": 6,
        }
    )


def render_pngs():
    converter = shutil.which("convert")
    if converter:
        for path in sorted(ASSETS.glob("*.svg")):
            subprocess.run(
                [converter, "-background", "none", str(path), str(path.with_suffix(".png"))],
                check=True,
            )
        return

    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        chrome = shutil.which("google-chrome")
        browser = playwright.chromium.launch(executable_path=chrome) if chrome else (
            playwright.chromium.launch()
        )
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        for path in sorted(ASSETS.glob("*.svg")):
            page.goto(path.as_uri())
            page.locator("svg").screenshot(path=str(path.with_suffix(".png")))
        browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true", help="Also render PNGs with Chromium")
    args = parser.parse_args()
    build_figures()
    if args.render:
        render_pngs()
    build_preview()
