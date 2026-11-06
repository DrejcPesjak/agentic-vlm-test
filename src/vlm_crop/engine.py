"""Bounded visual episodes. This module has no evaluator/label access."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from .core import CONDITIONS, AgentInput, Config, write_json
from .vision import crop_image, image_content

INSTRUCTIONS = """Answer the visual question using only the supplied images.
Return your final answer as a JSON object with exactly one string field: {"answer":"..."}.
The answer must be just the requested number or letter, without explanation.
If viewing tools are available, you may request another view when useful, or answer now.
Choose each view yourself. Coordinates always refer to the ORIGINAL image, with origin at
its top-left; right and bottom bounds are exclusive. Previous views remain in context.
No tool can tell you whether your answer is correct. Request only one view per turn."""

CROP_INSTRUCTIONS = """MANDATORY CROP REQUIREMENT:
You MUST call the crop tool and inspect at least one successful crop before giving your
final answer, even if you think you already know the answer from the full image.
Your first viewing action MUST be crop. Calling view_full does NOT satisfy this requirement.
Choose the region yourself based on the question and visible image details. Select a
region smaller than the full image; do not submit the full-image bounding box.
Call crop with JSON arguments {"box": [left, top, right, bottom]}.
All four coordinates must be integers in ORIGINAL-image pixels: x increases to the right,
y increases downward, and (0, 0) is the top-left corner. The bounds must satisfy
0 <= left < right <= original width and 0 <= top < bottom <= original height.
Right and bottom are exclusive. Every crop is taken from the original image, not from
the previous crop. The returned crop is resized to 1024 pixels on its longest side,
preserving aspect ratio. An invalid crop does not satisfy the requirement.
After inspecting a successful crop, you may answer or request more crops/full-image
views within the remaining view budget. You still choose all crop coordinates yourself."""


def instructions_for(condition: str) -> str:
    if condition == "crop":
        base = INSTRUCTIONS.replace(
            "If viewing tools are available, you may request another view when useful, "
            "or answer now.",
            "Follow the mandatory crop requirement below before answering.",
        )
        return base + "\n\n" + CROP_INSTRUCTIONS
    return INSTRUCTIONS


class Caller(Protocol):
    def call(self, payload: dict, directory: Path, index: int) -> dict: ...


def tools_for(condition: str) -> list[dict]:
    if condition == "single":
        return []
    tools = [
        {
            "type": "function",
            "name": "view_full",
            "strict": True,
            "description": "View the original full image again; consumes one view.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        }
    ]
    if condition == "crop":
        tools.append(
            {
                "type": "function",
                "name": "crop",
                "strict": True,
                "description": "View an original-image crop enlarged to 1024px longest side. "
                "Consumes one view. box=[left,top,right,bottom], integer pixels.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "box": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 4,
                            "maxItems": 4,
                        }
                    },
                    "required": ["box"],
                    "additionalProperties": False,
                },
            }
        )
    return tools


def parse_answer(text: str) -> str | None:
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(value, dict) and set(value) == {"answer"} and isinstance(value["answer"], str):
        return value["answer"]
    return None


def run_episode(
    agent: AgentInput, condition: str, model: str, cfg: Config, caller: Caller, directory: Path
) -> dict:
    if condition not in CONDITIONS:
        raise ValueError("Unknown condition")
    if condition == "crop" and cfg.max_views < 2:
        raise ValueError("Mandatory cropping requires at least two views")
    if max(agent.width, agent.height) > 6000 or min(agent.width, agent.height) < 1:
        raise ValueError("Image dimensions outside supported experiment bounds")
    directory.mkdir(parents=True, exist_ok=True)
    maximum = 1 if condition == "single" else cfg.max_views
    original = directory / "view-1.png"
    original.write_bytes(agent.image)
    views = [{"round": 1, "image": original.name, "box": None}]
    history = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": f"{agent.question}\nOriginal dimensions: "
                    f"{agent.width} x {agent.height}. View 1 of {maximum}.",
                },
                image_content(agent.image),
            ],
        }
    ]
    result: dict = {
        "condition": condition,
        "model": model,
        "views": views,
        "turns": [],
        "answer": None,
        "status": "running",
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "latency_seconds": 0.0,
        "nominal_cost": 0.0,
    }
    errors = 0
    try:
        for turn in range(maximum + cfg.max_action_errors + 1):
            needs_crop = condition == "crop" and not any(v["box"] is not None for v in views)
            final_only = len(views) >= maximum or (errors > 0 and errors >= cfg.max_action_errors)
            if needs_crop and final_only:
                result.update(status="protocol_error", error="Mandatory crop was not completed")
                break
            payload = {
                "model": model,
                "instructions": instructions_for(condition),
                "input": history,
                "tools": [] if final_only else tools_for(condition),
                "parallel_tool_calls": False,
                "store": False,
                "max_output_tokens": cfg.max_output_tokens,
            }
            if needs_crop:
                payload["tool_choice"] = {"type": "function", "name": "crop"}
            if model.startswith("gpt-5"):
                payload["reasoning"] = {"effort": "none"}
            else:
                payload["temperature"] = 0
            if final_only:
                payload["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": "answer",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                            "required": ["answer"],
                            "additionalProperties": False,
                        },
                    }
                }
            write_json(directory / f"request-{turn}.json", payload)
            record = caller.call(payload, directory, turn)
            response = record["response"]
            usage = response.get("usage", {})
            for key in ("input_tokens", "output_tokens"):
                result[key] += usage.get(key, 0)
            result["cached_tokens"] += (usage.get("input_tokens_details") or {}).get(
                "cached_tokens", 0
            )
            result["reasoning_tokens"] += (usage.get("output_tokens_details") or {}).get(
                "reasoning_tokens", 0
            )
            result["latency_seconds"] += record["latency_seconds"]
            result["nominal_cost"] += record["nominal_cost"]
            output = response.get("output", [])
            result["resolved_model"] = response.get("model", model)
            calls = [item for item in output if item["type"] == "function_call"]
            text = "".join(
                part.get("text", "")
                for item in output
                if item["type"] == "message"
                for part in item.get("content", [])
                if part["type"] == "output_text"
            )
            event = {
                "view_round": len(views),
                "text": text,
                "calls": [{k: c[k] for k in ("name", "arguments")} for c in calls],
                "usage": usage,
            }
            result["turns"].append(event)
            history.extend(output)
            if response.get("status") != "completed":
                result["status"] = "incomplete"
                break
            if not calls:
                if needs_crop:
                    result.update(
                        status="protocol_error", error="Answer given before mandatory crop"
                    )
                    break
                answer = parse_answer(text)
                if answer is not None:
                    result.update(answer=answer, status="completed")
                    break
                errors += 1
                if errors > cfg.max_action_errors:
                    result["status"] = "protocol_error"
                    break
                history.append({"role": "user", "content": "Return only the required answer JSON."})
                continue
            if final_only or len(calls) != 1:
                result["status"] = "protocol_error"
                break
            call = calls[0]
            try:
                arguments = json.loads(call["arguments"])
                box = None
                if call["name"] == "view_full" and arguments == {} and not needs_crop:
                    pixels = agent.image
                elif call["name"] == "crop" and condition == "crop":
                    if not isinstance(arguments, dict) or set(arguments) != {"box"}:
                        raise ValueError("crop accepts only box")
                    box = arguments["box"]
                    pixels = crop_image(agent.image, box, cfg.crop_long_side)
                    if box == [0, 0, agent.width, agent.height]:
                        raise ValueError("Select a region smaller than the full image")
                else:
                    raise ValueError("Unknown tool or invalid arguments")
            except (ValueError, TypeError, KeyError) as exc:
                errors += 1
                event["error"] = str(exc)
                history.append(
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": f"Invalid view request: {exc}. No view consumed.",
                    }
                )
                continue
            number = len(views) + 1
            name = f"view-{number}.png"
            (directory / name).write_bytes(pixels)
            views.append({"round": number, "image": name, "box": box})
            history.append(
                {
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": [
                        {
                            "type": "input_text",
                            "text": f"View {number} of {maximum}. Original-coordinate crop: {box}. "
                            + ("No views remain; answer now." if number == maximum else ""),
                        },
                        image_content(pixels),
                    ],
                }
            )
            write_json(directory / "progress.json", result)
        if result["status"] == "running":
            result["status"] = "protocol_error"
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
        write_json(directory / "progress.json", result)
        raise
    finally:
        write_json(directory / "progress.json", result)
    return result
