from __future__ import annotations

import copy
import io
import json
from unittest.mock import MagicMock

import httpx
import pytest
from openai import RateLimitError
from PIL import Image, PngImagePlugin

from vlm_crop.api import APIClient
from vlm_crop.budget import BudgetExceeded, Ledger, cost, reserve_amount
from vlm_crop.core import AgentInput, Config, fingerprint, read_json, write_json
from vlm_crop.dataset import TASKS, pixel_hash, select_rows
from vlm_crop.engine import parse_answer, run_episode
from vlm_crop.report import aggregate, build_report, wilson
from vlm_crop.runner import execute, normalize
from vlm_crop.vision import crop_image, encode_png, sanitize_image


def pixels(color="white", size=(100, 80)):
    return encode_png(Image.new("RGB", size, color))


def response(answer="1", call=None):
    output = (
        [
            {
                "type": "function_call",
                "name": call[0],
                "arguments": json.dumps(call[1]),
                "call_id": "call_test",
            }
        ]
        if call
        else [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": json.dumps({"answer": answer})}],
            }
        ]
    )
    return {
        "status": "completed",
        "model": "fake",
        "output": output,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 20,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


class FakeCaller:
    def __init__(self, outputs=None, exhaust=False):
        self.outputs = list(outputs or [])
        self.payloads = []
        self.exhaust = exhaust

    def call(self, payload, directory, index):
        self.payloads.append(copy.deepcopy(payload))
        if self.outputs:
            output = self.outputs.pop(0)
        elif payload.get("tool_choice") or (self.exhaust and payload["tools"]):
            crop = any(t["name"] == "crop" for t in payload["tools"])
            output = response(
                call=("crop", {"box": [10, 10, 50, 30]}) if crop else ("view_full", {})
            )
        else:
            output = response()
        return {"response": output, "latency_seconds": 0.1, "nominal_cost": 0.001}


@pytest.fixture
def agent():
    return AgentInput("How many intersections?", pixels(), 100, 80)


def test_sanitize_removes_metadata():
    image = Image.new("RGB", (10, 10), "red")
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("answer", "SECRET_LABEL")
    output = io.BytesIO()
    image.save(output, format="PNG", pnginfo=metadata)
    clean, width, height = sanitize_image(output.getvalue())
    assert b"SECRET_LABEL" not in clean
    assert (width, height) == (10, 10)
    assert Image.open(io.BytesIO(clean)).info == {}


@pytest.mark.parametrize(
    "box",
    [
        [0, 0, 0, 1],
        [-1, 0, 20, 20],
        [0, 0, 101, 10],
        [0, 0, 2.5, 10],
        [False, 0, 10, 10],
        [0, 1, 2],
        [0, 0, 10, float("nan")],
    ],
)
def test_invalid_crop(box):
    with pytest.raises(ValueError):
        crop_image(pixels(), box)


def test_crop_geometry_and_resize():
    image = Image.new("RGB", (100, 80), "blue")
    image.paste("red", (10, 10, 50, 30))
    cropped = Image.open(io.BytesIO(crop_image(encode_png(image), [10, 10, 50, 30])))
    assert cropped.size == (1024, 512)
    assert cropped.getpixel((0, 0)) == (255, 0, 0)
    assert cropped.getpixel((1023, 511)) == (255, 0, 0)


@pytest.mark.parametrize("condition,expected", [("single", 1), ("repeat", 5), ("crop", 5)])
def test_view_limits_and_finalization(tmp_path, agent, condition, expected):
    caller = FakeCaller(exhaust=True)
    episode = run_episode(agent, condition, "gpt-4o", Config(("gpt-4o",)), caller, tmp_path)
    assert episode["status"] == "completed"
    assert len(episode["views"]) == expected
    assert len(caller.payloads) == expected
    assert caller.payloads[-1]["tools"] == []
    assert all(p["parallel_tool_calls"] is False for p in caller.payloads)
    if condition == "repeat":
        assert all(
            (tmp_path / view["image"]).read_bytes() == agent.image for view in episode["views"]
        )
    if condition == "crop":
        # Repeated identical boxes always crop ORIGINAL pixels, not the previous view.
        assert len({(tmp_path / v["image"]).read_bytes() for v in episode["views"][1:]}) == 1


@pytest.mark.parametrize("condition", ["single", "repeat", "crop"])
def test_early_answer(tmp_path, agent, condition):
    episode = run_episode(agent, condition, "gpt-5.4", Config(("gpt-5.4",)), FakeCaller(), tmp_path)
    assert len(episode["views"]) == (2 if condition == "crop" else 1)
    assert episode["answer"] == "1"


def test_invalid_actions_are_bounded(tmp_path, agent):
    caller = FakeCaller([response(call=("crop", {"box": [-1, 0, 10, 10]}))] * 2)
    episode = run_episode(agent, "crop", "gpt-4o", Config(("gpt-4o",)), caller, tmp_path)
    assert len(episode["views"]) == 1
    assert len(caller.payloads) == 2
    assert episode["status"] == "protocol_error"
    assert episode["answer"] is None
    assert sum("error" in t for t in episode["turns"]) == 2


def test_parallel_calls_rejected(tmp_path, agent):
    output = response(call=("view_full", {}))
    output["output"] *= 2
    episode = run_episode(
        agent, "repeat", "gpt-4o", Config(("gpt-4o",)), FakeCaller([output]), tmp_path
    )
    assert episode["status"] == "protocol_error"
    assert len(episode["views"]) == 1


def test_incomplete_response_not_scored(tmp_path, agent):
    output = response()
    output["status"] = "incomplete"
    episode = run_episode(
        agent, "single", "gpt-4o", Config(("gpt-4o",)), FakeCaller([output]), tmp_path
    )
    assert episode["status"] == "incomplete" and episode["answer"] is None


@pytest.mark.parametrize(
    "text", ['{"answer":1}', '{"answer":"1","extra":"x"}', "The answer is 1", "[]"]
)
def test_strict_answer(text):
    assert parse_answer(text) is None


def test_normalization():
    assert normalize(" {A} ") == "a"
    assert normalize("11") != normalize("1")


def test_budget_persists_and_blocks(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = Ledger(path, 1)
    rid = ledger.reserve(0.7, "gpt-4o")
    with pytest.raises(BudgetExceeded):
        Ledger(path).reserve(0.4, "gpt-4o")
    ledger.settle(rid, 0.1)
    Ledger(path).reserve(0.8, "gpt-4o")
    assert Ledger(path).summary()["reserved_unknown"] == 0.8
    assert Ledger(path).summary()["limit"] == 1


def test_reservation_includes_images_and_output():
    from vlm_crop.vision import image_content

    payload = {
        "model": "gpt-4o-mini",
        "max_output_tokens": 1200,
        "input": [{"role": "user", "content": [image_content(pixels())]}],
    }
    one = reserve_amount(payload)
    payload["input"] *= 5
    assert reserve_amount(payload) > one
    assert one > cost("gpt-4o-mini", 48169, 1200)
    assert cost("gpt-5.4", 300_000, 1000) == 1.5225


def test_transient_retry_keeps_uncertain_reservation(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-key")
    monkeypatch.setattr("vlm_crop.api.time.sleep", lambda _: None)
    ledger = Ledger(tmp_path / "ledger.json")
    client = APIClient(ledger)
    error = RateLimitError(
        "rate",
        response=httpx.Response(429, request=httpx.Request("POST", "https://example.com")),
        body=None,
    )
    successful = MagicMock()
    successful.model_dump.return_value = response()
    client.client.responses.create = MagicMock(side_effect=[error, successful])
    client.call({"model": "gpt-4o", "max_output_tokens": 100, "input": []}, tmp_path, 0)
    assert ledger.summary()["requests"] == 2
    assert ledger.summary()["reserved_unknown"] > 0
    assert ledger.summary()["nominal_spend"] > 0


def rows():
    result = []
    for task_index, task in enumerate(TASKS):
        for i in range(5):
            result.append(
                {
                    "task": task,
                    "image": {"bytes": pixels((i * 40, task_index * 80, 20))},
                    "metadata": json.dumps(
                        {"image_id": f"{task_index}-{i}", "answer": "LEAK_SENTINEL"}
                    ),
                    "prompt": "Count?",
                    "groundtruth": str(i % 2),
                }
            )
    return result


def test_deterministic_selection_and_no_duplicate_images():
    source = rows()
    source += [source[0]]
    p, e = select_rows(source, 3, 42)
    p2, e2 = select_rows(source, 3, 42)
    assert [r["pixel_hash"] for r in e] == [r["pixel_hash"] for r in e2]
    assert p == p2
    assert len(e) == 9
    assert len({r["pixel_hash"] for r in p + e}) == 10
    assert {r["pixel_hash"] for r in p}.isdisjoint(r["pixel_hash"] for r in e)


def make_dataset(root):
    root.mkdir()
    image = pixels()
    (root / "image.png").write_bytes(image)
    manifest = {
        "samples": [
            {
                "id": "opaque",
                "task": TASKS[0],
                "split": "pilot",
                "question": "Count intersections",
                "image": "image.png",
                "width": 100,
                "height": 80,
                "pixel_hash": pixel_hash(image),
            }
        ]
    }
    manifest["fingerprint"] = fingerprint(manifest)
    write_json(root / "manifest.json", manifest)
    write_json(
        root / "labels.json",
        {"opaque": {"answer": "LABEL_SENTINEL", "metadata": "METADATA_SENTINEL"}},
    )


def test_runner_leakage_resume_and_report(tmp_path):
    data = tmp_path / "data"
    make_dataset(data)
    cfg = Config(("gpt-4o",))
    caller = FakeCaller(exhaust=True)
    ledger = Ledger(tmp_path / "budget.json")
    run = tmp_path / "run"
    assert execute(data, run, cfg, caller, ledger)["completed"] == 3
    outgoing = json.dumps(caller.payloads)
    assert "LABEL_SENTINEL" not in outgoing
    assert "METADATA_SENTINEL" not in outgoing
    assert "groundtruth" not in outgoing
    assert "image.png" not in outgoing
    count = len(caller.payloads)
    assert execute(data, run, cfg, caller, ledger)["skipped"] == 3
    assert len(caller.payloads) == count
    with pytest.raises(ValueError, match="settings changed"):
        execute(data, run, Config(("gpt-4o",), max_views=3), caller, ledger)
    target = tmp_path / "report"
    build_report(run, target)
    exported = read_json(target / "results.json")
    assert len(exported["episodes"]) == 3
    assert all(not e["correct"] for e in exported["episodes"])
    assert (target / "index.html").exists()
    for view in exported["episodes"][-1]["views"]:
        assert (target / view["image"]).exists()
    assert len(list((target / "exports").glob("*.png"))) == 1
    gif = Image.open(next((target / "exports").glob("*.gif")))
    assert gif.n_frames == 5


def test_interrupted_episode_not_reissued(tmp_path):
    data = tmp_path / "data"
    make_dataset(data)
    run = tmp_path / "run"
    eid = fingerprint(["opaque", "gpt-4o", "single"])[:24]
    write_json(run / "episodes" / eid / "started.json", {})
    caller = FakeCaller()
    summary = execute(data, run, Config(("gpt-4o",)), caller, Ledger(tmp_path / "budget"))
    assert summary["errors"] == 1
    assert len(caller.payloads) == 3


def test_paired_statistics_and_failures():
    episodes = [
        {
            "model": "x",
            "task": "t",
            "sample_id": "s",
            "condition": c,
            "status": "completed",
            "correct": c == "crop",
            "views": [],
        }
        for c in ("single", "repeat", "crop")
    ]
    stats = aggregate(episodes)
    assert all(p["improvements"] == 1 for p in stats["paired"])
    episodes[-1]["status"] = "error"
    episodes[-1]["correct"] = False
    stats = aggregate(episodes)
    assert all(p["paired_completed"] == 0 for p in stats["paired"])
    assert wilson(0, 0) is None
    low, high = wilson(1, 1)
    assert low < 0.5 and high == 1


def test_budget_rejects_before_api_request(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-key")
    ledger = Ledger(tmp_path / "budget.json", 0.001)
    client = APIClient(ledger)
    mock = MagicMock()
    client.client.responses.create = mock
    with pytest.raises(BudgetExceeded):
        client.call({"model": "gpt-5.4", "max_output_tokens": 1200, "input": []}, tmp_path, 0)
    mock.assert_not_called()
    assert ledger.summary()["requests"] == 0


def test_malformed_final_answers_stop_after_two_corrections(tmp_path, agent):
    malformed = response()
    malformed["output"][0]["content"][0]["text"] = "not json"
    caller = FakeCaller([malformed] * 10)
    result = run_episode(agent, "single", "gpt-4o", Config(("gpt-4o",)), caller, tmp_path)
    assert result["status"] == "protocol_error"
    assert len(caller.payloads) == 3


def test_zero_error_budget_does_not_disable_valid_tools(tmp_path, agent):
    caller = FakeCaller(exhaust=True)
    result = run_episode(
        agent, "crop", "gpt-4o", Config(("gpt-4o",), max_action_errors=0), caller, tmp_path
    )
    assert result["status"] == "completed"
    assert len(result["views"]) == 5


def test_crop_instructions_and_first_action_are_mandatory(tmp_path, agent):
    caller = FakeCaller()
    result = run_episode(agent, "crop", "gpt-4o", Config(("gpt-4o",)), caller, tmp_path)
    first, second = caller.payloads
    assert "You MUST call the crop tool" in first["instructions"]
    assert '{"box": [left, top, right, bottom]}' in first["instructions"]
    assert first["tool_choice"] == {"type": "function", "name": "crop"}
    assert "tool_choice" not in second
    assert result["views"][1]["box"] == [10, 10, 50, 30]
    assert result["status"] == "completed"


def test_answer_before_crop_is_not_accepted(tmp_path, agent):
    result = run_episode(
        agent, "crop", "gpt-4o", Config(("gpt-4o",)), FakeCaller([response()]), tmp_path
    )
    assert result["status"] == "protocol_error"
    assert result["answer"] is None


@pytest.mark.parametrize("call", [("view_full", {}), ("crop", {"box": [0, 0, 100, 80]})])
def test_full_image_does_not_satisfy_mandatory_crop(tmp_path, agent, call):
    caller = FakeCaller([response(call=call)])
    result = run_episode(agent, "crop", "gpt-4o", Config(("gpt-4o",)), caller, tmp_path)
    assert result["turns"][0]["error"]
    assert caller.payloads[1]["tool_choice"]["name"] == "crop"
    assert result["status"] == "completed"
    assert len(result["views"]) == 2


def test_mandatory_crop_needs_two_views(tmp_path, agent):
    with pytest.raises(ValueError, match="at least two views"):
        run_episode(
            agent, "crop", "gpt-4o", Config(("gpt-4o",), max_views=1), FakeCaller(), tmp_path
        )
