"""Persistent conservative reservations, including ambiguous failed requests."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from filelock import FileLock

from .core import read_json, write_json

# USD per million tokens; standard pricing verified 2026-11-05 from official model pages.
# Cached input is deliberately charged at the full rate in our nominal ledger.
PRICES = {
    "gpt-5.4": (2.5, 15.0),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5.4-mini": (0.75, 4.5),
}


class BudgetExceeded(RuntimeError):
    pass


def model_family(model: str) -> str:
    for name in sorted(PRICES, key=len, reverse=True):
        if model == name or model.startswith(name + "-20"):
            return name
    raise ValueError(f"Unpriced model: {model}")


def cost(model: str, input_tokens: int, output_tokens: int) -> float:
    family = model_family(model)
    inp, out = PRICES[family]
    if family == "gpt-5.4" and input_tokens > 272_000:
        inp *= 2
        out *= 1.5
    return (input_tokens * inp + output_tokens * out) / 1_000_000


def reserve_amount(payload: dict) -> float:
    """Overestimate text by UTF-8 bytes and high-detail images by worst-case model budget.

    Full/crop images are bounded by the runner. Strip base64 before text estimation.
    Include generous framing slack. GPT-5.4 uses long-context rates for reservations.
    """
    images = 0

    def strip(value: Any) -> Any:
        nonlocal images
        if isinstance(value, dict):
            if value.get("type") == "input_image":
                images += 1
                return {"type": "input_image"}
            return {k: strip(v) for k, v in value.items()}
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value

    text_bytes = len(json.dumps(strip(payload), ensure_ascii=False).encode())
    family = model_family(payload["model"])
    # 4o-mini: at most 8 high-detail tiles (plus base) < 50k tokens.
    image_bound = 50_000 if family == "gpt-4o-mini" else 10_000
    input_bound = text_bytes + images * image_bound + 8192
    inp, out = PRICES[family]
    if family == "gpt-5.4":
        inp *= 2
        out *= 1.5
    return (input_bound * inp + payload["max_output_tokens"] * out) / 1_000_000


class Ledger:
    def __init__(self, path: Path, limit: float = 10.0):
        if not 0 < limit <= 10:
            raise ValueError("Budget must be greater than zero and at most $10")
        self.path = path
        self.limit = limit
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(path) + ".lock")

    def _read(self) -> dict:
        return read_json(self.path) if self.path.exists() else {"limit": self.limit, "entries": []}

    def reserve(self, amount: float, model: str) -> str:
        with self.lock:
            data = self._read()
            limit = min(self.limit, data["limit"])
            if sum(e["amount"] for e in data["entries"]) + amount > limit:
                raise BudgetExceeded(f"Next request would exceed ${limit:.2f} nominal budget")
            rid = uuid.uuid4().hex
            data["limit"] = limit
            data["entries"].append(
                {"id": rid, "model": model, "amount": amount, "state": "reserved"}
            )
            write_json(self.path, data)
            return rid

    def settle(self, rid: str, amount: float) -> None:
        with self.lock:
            data = self._read()
            entry = next(e for e in data["entries"] if e["id"] == rid)
            bound = entry["amount"]
            entry.update(amount=amount, state="settled")
            write_json(self.path, data)
            if amount > bound + 1e-9:
                raise BudgetExceeded("Actual cost exceeded reservation; stop and review pricing")

    def summary(self) -> dict:
        with self.lock:
            data = self._read()
            return {
                "limit": min(self.limit, data["limit"]),
                "nominal_spend": sum(
                    e["amount"] for e in data["entries"] if e["state"] == "settled"
                ),
                "reserved_unknown": sum(
                    e["amount"] for e in data["entries"] if e["state"] != "settled"
                ),
                "requests": len(data["entries"]),
            }
