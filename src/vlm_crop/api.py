"""Only the Responses endpoint is used; no hosted tools or SDK retries."""

from __future__ import annotations

import time
from pathlib import Path

from openai import APIConnectionError, APIStatusError, OpenAI

from .budget import Ledger, cost, reserve_amount
from .core import write_json


class APIClient:
    def __init__(self, ledger: Ledger):
        self.ledger = ledger
        self.client = OpenAI(max_retries=0, timeout=90.0)

    def call(self, payload: dict, directory: Path, index: int) -> dict:
        for attempt in range(2):
            rid = self.ledger.reserve(reserve_amount(payload), payload["model"])
            started = time.monotonic()
            try:
                response = self.client.responses.create(**payload).model_dump(mode="json")
            except (APIConnectionError, APIStatusError) as exc:
                status = getattr(exc, "status_code", None)
                # No response usage: retain the reservation, even if probably unbilled.
                write_json(
                    directory / f"error-{index}-{attempt}.json",
                    {"type": type(exc).__name__, "status": status, "reservation": rid},
                )
                if attempt == 0 and (status in (429, 500, 502, 503, 504)):
                    time.sleep(1)
                    continue
                raise RuntimeError(
                    f"API call failed: {type(exc).__name__}, status={status}"
                ) from None
            elapsed = time.monotonic() - started
            # Save before reconciliation so a crash never loses returned output/usage.
            record = {"response": response, "latency_seconds": elapsed, "reservation": rid}
            write_json(directory / f"response-{index}.json", record)
            usage = response.get("usage")
            if not usage:
                raise RuntimeError("API response omitted usage; reservation retained")
            amount = cost(payload["model"], usage["input_tokens"], usage["output_tokens"])
            self.ledger.settle(rid, amount)
            return {**record, "nominal_cost": amount}
        raise AssertionError("Unreachable")
