"""Pinned benchmark download and deterministic selection; labels never enter AgentInput."""

from __future__ import annotations

import hashlib
import io
import json
import random
from collections import defaultdict
from pathlib import Path

import httpx
import pyarrow.parquet as pq
from PIL import Image

from .core import AgentInput, fingerprint, read_json, write_json
from .vision import sanitize_image

REVISION = "8c9b35b2b1e5f718c29f683306a8885b32be9c66"
DATASET_URL = f"https://huggingface.co/datasets/XAI/vlmsareblind/resolve/{REVISION}"
TASKS = ("Line Plot Intersections", "Circled Letter", "Subway Connections")


def download(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    with httpx.stream(
        "GET",
        DATASET_URL + "/data/valid-00000-of-00001.parquet",
        follow_redirects=True,
        timeout=120,
    ) as response:
        response.raise_for_status()
        with partial.open("wb") as target:
            for block in response.iter_bytes():
                target.write(block)
    partial.replace(path)


def pixel_hash(data: bytes) -> str:
    with Image.open(io.BytesIO(data)) as image:
        rgb = image.convert("RGB")
        return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def select_rows(rows: list[dict], per_task: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Balance labels by round-robin; deduplicate by both identity and pixels globally."""
    rng = random.Random(seed)
    unique: dict[str, list[dict]] = defaultdict(list)
    seen_pixels: set[str] = set()
    seen_ids: set[str] = set()
    for row in rows:
        if row["task"] not in TASKS:
            continue
        digest = pixel_hash(row["image"]["bytes"])
        metadata = json.loads(row["metadata"])
        identity = str(metadata.get("image_id", digest))
        if digest in seen_pixels or identity in seen_ids:
            continue
        seen_pixels.add(digest)
        seen_ids.add(identity)
        unique[row["task"]].append({**row, "pixel_hash": digest, "source_id": identity})
    pilot: list[dict] = []
    evaluation: list[dict] = []
    for task in TASKS:
        candidates = sorted(unique[task], key=lambda row: row["pixel_hash"])
        rng.shuffle(candidates)
        # One shared pilot from line intersections, chosen independently of model performance.
        if task == TASKS[0] and candidates:
            pilot.append(candidates.pop())
        buckets: dict[str, list[dict]] = defaultdict(list)
        for row in candidates:
            buckets[str(row["groundtruth"])].append(row)
        keys = sorted(buckets)
        rng.shuffle(keys)
        chosen: list[dict] = []
        while len(chosen) < per_task:
            available = [key for key in keys if buckets[key]]
            if not available:
                raise ValueError(f"Not enough unique images for {task}: {len(chosen)}")
            for key in available:
                if len(chosen) < per_task:
                    chosen.append(buckets[key].pop())
        evaluation.extend(chosen)
    if not pilot:
        raise ValueError(f"Missing pilot task; available tasks: {sorted(unique)}")
    return pilot, evaluation


def prepare(root: Path, per_task: int = 20, seed: int = 42) -> dict:
    if per_task < 1:
        raise ValueError("per-task must be positive")
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if (manifest["seed"], manifest["per_task"], manifest["revision"]) != (
            seed,
            per_task,
            REVISION,
        ):
            raise ValueError("Existing dataset has different settings; choose a new directory")
        return manifest
    parquet = root / "source.parquet"
    download(parquet)
    rows = pq.read_table(parquet).to_pylist()
    pilot, evaluation = select_rows(rows, per_task, seed)
    manifest = {
        "dataset": "XAI/vlmsareblind",
        "revision": REVISION,
        "license": "MIT",
        "seed": seed,
        "per_task": per_task,
        "samples": [],
    }
    labels = {}
    for split, selection in (("pilot", pilot), ("evaluation", evaluation)):
        for row in selection:
            sid = row["pixel_hash"][:20]
            image, width, height = sanitize_image(row["image"]["bytes"])
            path = root / "images" / f"{sid}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(image)
            manifest["samples"].append(
                {
                    "id": sid,
                    "task": row["task"],
                    "split": split,
                    "question": row["prompt"],
                    "image": f"images/{sid}.png",
                    "width": width,
                    "height": height,
                    "pixel_hash": row["pixel_hash"],
                }
            )
            labels[sid] = {"answer": str(row["groundtruth"]), "source_id": row["source_id"]}
    manifest["fingerprint"] = fingerprint(manifest)
    write_json(root / "labels.json", labels)
    write_json(manifest_path, manifest)
    return manifest


def load_agent_input(root: Path, sample: dict) -> AgentInput:
    data, width, height = sanitize_image((root / sample["image"]).read_bytes())
    if pixel_hash(data) != sample["pixel_hash"]:
        raise ValueError("Image does not match frozen manifest")
    return AgentInput(question=sample["question"], image=data, width=width, height=height)
