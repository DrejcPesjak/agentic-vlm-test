"""Static, API-free replay exports and small-sample descriptive statistics."""

from __future__ import annotations

import csv
import math
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .core import CONDITIONS, read_json, write_json

COLORS = {"single": "#8a93ad", "repeat": "#528cff", "crop": "#12a68c"}


def wilson(successes: int, n: int) -> list[float] | None:
    if not n:
        return None
    z = 1.96
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0, center - half), min(1, center + half)]


def aggregate(episodes: list[dict]) -> dict:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    pairs: dict[tuple, dict] = defaultdict(dict)
    for episode in episodes:
        for task in ("All tasks", episode["task"]):
            groups[(episode["model"], episode["condition"], task)].append(episode)
        pairs[(episode["model"], episode["sample_id"])][episode["condition"]] = episode
    rows = []
    for (model, condition, task), group in sorted(groups.items()):
        n = len(group)
        completed = sum(e["status"] == "completed" for e in group)
        correct = sum(bool(e["correct"]) for e in group)
        rows.append(
            {
                "model": model,
                "condition": condition,
                "task": task,
                "n": n,
                "completed": completed,
                "correct": correct,
                "accuracy": correct / n,
                "completion_rate": completed / n,
                "accuracy_completed": correct / completed if completed else None,
                "wilson_95": wilson(correct, n),
                "mean_views": sum(len(e.get("views", [])) for e in group) / n,
                **{
                    key: sum(e.get(key, 0) for e in group)
                    for key in (
                        "input_tokens",
                        "output_tokens",
                        "cached_tokens",
                        "reasoning_tokens",
                        "latency_seconds",
                        "nominal_cost",
                    )
                },
            }
        )
    comparisons: dict[tuple, dict] = {}
    for (model, _), conditions in pairs.items():
        for baseline in ("single", "repeat"):
            key = (model, baseline)
            row = comparisons.setdefault(
                key,
                {
                    "model": model,
                    "baseline": baseline,
                    "paired_completed": 0,
                    "improvements": 0,
                    "regressions": 0,
                    "ties": 0,
                },
            )
            if not all(
                c in conditions and conditions[c]["status"] == "completed"
                for c in (baseline, "crop")
            ):
                continue
            row["paired_completed"] += 1
            delta = int(conditions["crop"]["correct"]) - int(conditions[baseline]["correct"])
            row["improvements" if delta > 0 else "regressions" if delta < 0 else "ties"] += 1
    return {
        "groups": rows,
        "paired": list(comparisons.values()),
        "note": "Exploratory, one run per image/condition. Accuracy counts failures as incorrect; "
        "completed-only accuracy is also provided. Wilson intervals are descriptive, "
        "not evidence from repeated model runs. Paired counts exclude incomplete pairs.",
    }


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def frame(group: list[dict], root: Path, round_number: int) -> Image.Image:
    canvas = Image.new("RGB", (1440, 840), "#f5f7fb")
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 20), "CAN A CLOSER LOOK CHANGE THE ANSWER?", font=font(27), fill="#17233a")
    draw.text(
        (28, 62),
        f"{group[0]['model']} | View {round_number} | {group[0]['task']}",
        font=font(19),
        fill="#526078",
    )
    question = group[0]["question"]
    import textwrap

    draw.multiline_text(
        (28, 100), "\n".join(textwrap.wrap(question, 110)), font=font(18), fill="#17233a", spacing=4
    )
    for i, condition in enumerate(CONDITIONS):
        x = 20 + i * 475
        episode = next((e for e in group if e["condition"] == condition), None)
        draw.rounded_rectangle((x, 175, x + 460, 775), radius=16, fill="white")
        draw.text(
            (x + 18, 191),
            {"single": "One look", "repeat": "Look again", "crop": "Choose a crop"}[condition],
            font=font(23),
            fill=COLORS[condition],
        )
        if not episode or not episode.get("views"):
            draw.text((x + 18, 245), "No recorded view", font=font(20), fill="#526078")
            continue
        view = episode["views"][min(round_number, len(episode["views"])) - 1]
        with Image.open(root / view["image"]) as image:
            fitted = ImageOps.contain(image.convert("RGB"), (420, 355))
            canvas.paste(fitted, (x + 20 + (420 - fitted.width) // 2, 242))
        # Original thumbnail + actual model-selected rectangle.
        first = episode["views"][0]
        with Image.open(root / first["image"]) as original:
            thumb = ImageOps.contain(original.convert("RGB"), (160, 115))
            tx, ty = x + 20, 610
            canvas.paste(thumb, (tx, ty))
            if view["box"]:
                left, top, right, bottom = view["box"]
                sx, sy = thumb.width / original.width, thumb.height / original.height
                draw.rectangle(
                    (tx + left * sx, ty + top * sy, tx + right * sx, ty + bottom * sy),
                    outline="#12a68c",
                    width=3,
                )
        done = round_number >= len(episode["views"])
        label = f"Answer: {episode.get('answer')}" if done else "Inspecting…"
        draw.text((x + 200, 620), label, font=font(20), fill="#17233a")
        if done:
            outcome = (
                ("Correct" if episode["correct"] else "Incorrect")
                if (episode["status"] == "completed")
                else episode["status"]
            )
            draw.text((x + 200, 655), outcome, font=font(20), fill="#526078")
    draw.text(
        (28, 800),
        f"Ground truth (evaluator only): {group[0].get('groundtruth', '?')}"
        "   •   Source: VLMs Are Blind / XAI",
        font=font(17),
        fill="#526078",
    )
    return canvas


def summary_figure(stats: dict, target: Path) -> None:
    rows = [r for r in stats["groups"] if r["task"] == "All tasks"]
    image = Image.new("RGB", (1100, 140 + 64 * len(rows)), "#f5f7fb")
    draw = ImageDraw.Draw(image)
    draw.text((25, 20), "Visual accuracy · all attempted episodes", font=font(27), fill="#17233a")
    draw.text((25, 60), "Whiskers: descriptive 95% Wilson intervals", font=font(18), fill="#526078")
    for i, row in enumerate(rows):
        y = 110 + i * 64
        draw.text((25, y), f"{row['model']} / {row['condition']}", font=font(18), fill="#17233a")
        draw.rectangle(
            (360, y, 360 + max(1, 500 * row["accuracy"]), y + 28), fill=COLORS[row["condition"]]
        )
        low, high = row["wilson_95"]
        draw.line((360 + 500 * low, y + 14, 360 + 500 * high, y + 14), fill="#17233a", width=2)
        draw.text(
            (880, y),
            f"{row['correct']}/{row['n']} ({row['accuracy']:.0%})",
            font=font(18),
            fill="#17233a",
        )
    image.save(target)


def build_report(run: Path, target: Path) -> dict:
    paths = sorted((run / "episodes").glob("*/episode.json"))
    if not paths:
        raise ValueError("No episodes available to report")
    if target.resolve() == run.resolve() or run.resolve() in target.resolve().parents:
        raise ValueError("Report directory must be separate from run artifacts")
    target.mkdir(parents=True, exist_ok=True)
    episodes = []
    for path in paths:
        episode = read_json(path)
        for view in episode.get("views", []):
            source = path.parent / view["image"]
            relative = Path("assets") / episode["id"] / source.name
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            view["image"] = relative.as_posix()
        episodes.append(episode)
    stats = aggregate(episodes)
    frozen = read_json(run / "config.json")
    data = {
        "episodes": episodes,
        "stats": stats,
        "config": frozen,
        "split": frozen["config"]["split"],
        "exports": [],
    }
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for episode in episodes:
        groups[(episode["model"], episode["sample_id"])].append(episode)
    export_dir = target / "exports"
    export_dir.mkdir(exist_ok=True)
    for (_, sid), group in groups.items():
        model = group[0]["model"]
        count = max((len(e.get("views", [])) for e in group), default=1) or 1
        frames = [frame(group, target, r) for r in range(1, count + 1)]
        stem = f"{model}-{sid}"
        frames[-1].save(export_dir / f"{stem}.png")
        frames[0].save(
            export_dir / f"{stem}.gif",
            save_all=True,
            append_images=frames[1:],
            duration=1400,
            loop=0,
        )
        data["exports"].append(
            {
                "model": model,
                "sample_id": sid,
                "png": f"exports/{stem}.png",
                "gif": f"exports/{stem}.gif",
            }
        )
    summary_figure(stats, target / "summary.png")
    write_json(target / "results.json", data)
    write_json(target / "summary.json", stats)
    with (target / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stats["groups"][0]))
        writer.writeheader()
        writer.writerows(stats["groups"])
    # External JS, not inline HTML: prevents dataset/model text becoming executable markup.
    (target / "data.js").write_text(
        "window.EXPERIMENT = " + (target / "results.json").read_text().strip() + ";\n"
    )
    for source in (Path(__file__).parent / "static").iterdir():
        shutil.copyfile(source, target / source.name)
    (target / ".nojekyll").touch()
    return {"episodes": len(episodes), "report": str(target / "index.html")}
