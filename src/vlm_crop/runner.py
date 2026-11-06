"""Evaluator orchestration: labels are loaded only after agent episodes finish."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from filelock import FileLock

from .budget import PRICES, BudgetExceeded, Ledger, model_family
from .core import CONDITIONS, PROTOCOL_VERSION, Config, fingerprint, read_json, write_json
from .dataset import load_agent_input
from .engine import Caller, instructions_for, run_episode, tools_for


def normalize(answer: str) -> str:
    return answer.strip().strip("{}\"' ").casefold()


def execute(data: Path, output: Path, cfg: Config, caller: Caller, ledger: Ledger) -> dict:
    for model in cfg.models:
        model_family(model)
    manifest = read_json(data / "manifest.json")
    content = {k: v for k, v in manifest.items() if k != "fingerprint"}
    if fingerprint(content) != manifest["fingerprint"]:
        raise ValueError("Manifest changed after preparation")
    samples = [s for s in manifest["samples"] if s["split"] == cfg.split]
    if not samples:
        raise ValueError("No samples in selected split")
    signature = {
        "config": asdict(cfg),
        "manifest": manifest["fingerprint"],
        "instructions": {c: instructions_for(c) for c in CONDITIONS},
        "tools": {c: tools_for(c) for c in CONDITIONS},
        "prices": PRICES,
        "protocol_version": PROTOCOL_VERSION,
    }
    digest = fingerprint(signature)
    output.mkdir(parents=True, exist_ok=True)
    # Serialize all runs sharing a ledger; API reservations also have a separate atomic lock.
    with FileLock(str(ledger.path) + ".run.lock", timeout=0):
        frozen = output / "config.json"
        if frozen.exists() and read_json(frozen)["fingerprint"] != digest:
            raise ValueError("Run settings changed; use a new run directory (same budget ledger)")
        if not frozen.exists():
            sources = {p.name: p.read_text() for p in Path(__file__).parent.glob("*.py")}
            write_json(
                frozen,
                {**signature, "fingerprint": digest, "source_fingerprint": fingerprint(sources)},
            )
        summary = {"completed": 0, "errors": 0, "skipped": 0, "existing_errors": 0}
        for sample in samples:
            agent = load_agent_input(data, sample)
            for model in cfg.models:
                for condition in CONDITIONS:
                    eid = fingerprint([sample["id"], model, condition])[:24]
                    directory = output / "episodes" / eid
                    final_path = directory / "episode.json"
                    if final_path.exists():
                        summary["skipped"] += 1
                        if read_json(final_path)["status"] != "completed":
                            summary["existing_errors"] += 1
                        continue
                    base = {
                        "id": eid,
                        "sample_id": sample["id"],
                        "task": sample["task"],
                        "question": sample["question"],
                        "split": cfg.split,
                        "width": agent.width,
                        "height": agent.height,
                        "model": model,
                        "condition": condition,
                    }
                    if (directory / "started.json").exists():
                        # An interrupted request can have been billed. Never repeat silently.
                        progress = (
                            read_json(directory / "progress.json")
                            if (directory / "progress.json").exists()
                            else {}
                        )
                        write_json(
                            final_path,
                            {
                                **base,
                                **progress,
                                "status": "interrupted",
                                "answer": None,
                                "correct": False,
                            },
                        )
                        summary["errors"] += 1
                        continue
                    write_json(directory / "started.json", {"run": digest})
                    stop = False
                    try:
                        episode = run_episode(agent, condition, model, cfg, caller, directory)
                    except Exception as exc:
                        progress_path = directory / "progress.json"
                        episode = read_json(progress_path) if progress_path.exists() else {}
                        episode.update(status="error", error=str(exc), answer=None)
                        stop = isinstance(exc, BudgetExceeded)
                    # Only the evaluator touches answers; never pass this record to run_episode.
                    truth = read_json(data / "labels.json")[sample["id"]]["answer"]
                    correct = episode.get("status") == "completed" and normalize(
                        episode.get("answer") or ""
                    ) == normalize(truth)
                    write_json(
                        final_path, {**base, **episode, "groundtruth": truth, "correct": correct}
                    )
                    summary["completed" if episode.get("status") == "completed" else "errors"] += 1
                    print(
                        f"{model} / {condition} / {sample['id']}: "
                        f"{episode.get('status')} ({len(episode.get('views', []))} views)",
                        flush=True,
                    )
                    if stop:
                        return {**summary, "budget": ledger.summary(), "stopped": "budget"}
        return {**summary, "budget": ledger.summary()}
