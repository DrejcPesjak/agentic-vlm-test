"""Shared records and safe serialization. AgentInput intentionally has no label fields."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONDITIONS = ("single", "repeat", "crop")
PROTOCOL_VERSION = 2


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class AgentInput:
    question: str
    image: bytes
    width: int
    height: int


@dataclass(frozen=True)
class Config:
    models: tuple[str, ...]
    split: str = "pilot"
    max_views: int = 5
    max_output_tokens: int = 1200
    max_action_errors: int = 2
    detail: str = "high"
    crop_long_side: int = 1024

    @classmethod
    def load(cls, path: Path) -> Config:
        raw = read_json(path)
        raw["models"] = tuple(raw["models"])
        cfg = cls(**raw)
        if not cfg.models or len(set(cfg.models)) != len(cfg.models):
            raise ValueError("Specify distinct models")
        if cfg.split not in ("pilot", "evaluation"):
            raise ValueError("Invalid split")
        if not 2 <= cfg.max_views <= 5 or not 1 <= cfg.max_output_tokens <= 4096:
            raise ValueError("Views must be 2..5 for mandatory cropping; output tokens 1..4096")
        if cfg.detail != "high" or cfg.crop_long_side != 1024:
            raise ValueError("This protocol fixes high detail and 1024px crops")
        if not 0 <= cfg.max_action_errors <= 2:
            raise ValueError("Action errors must be 0..2")
        return cfg
