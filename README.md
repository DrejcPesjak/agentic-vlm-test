# Does model-directed cropping improve visual accuracy?

A small, reproducible experiment comparing **one full-image view**, **repeated full-image views**, and **model-selected crops**. Includes a Python runner, a pinned subset of *VLMs Are Blind*, offline tests, budget accounting, and a static replay viewer with PNG/GIF exports.

## Completed evaluation

The 60-image mandatory-crop evaluation is complete: 360/360 episodes succeeded. GPT-5.4 scored 54/60 with cropping versus 44/60 single-view; GPT-4o scored 36/60 versus 34/60. Every repeat-enabled agent answered after one view, so this is not an equal-actual-view comparison. See [full results and limitations](docs/evaluation-results.md), or open `site-output-evaluation/index.html` for the replay.

## Setup

Python 3.10+ and [uv](https://docs.astral.sh/uv/) are required. Dependencies are locked in `uv.lock`.

```bash
uv sync --locked
# Put OPENAI_API_KEY=... in .env (ignored by Git).
uv run vlm-crop prepare
```

Preparation downloads about 84 MB and freezes 60 evaluation images (20 each: line intersections, circled letters, subway connections), plus one separate pilot image. Dataset revision: `8c9b35b2b1e5f718c29f683306a8885b32be9c66`. Selection uses seed 42, cycles through answer classes, and deduplicates both source identity and pixels. The manifest is immutable on resume; changing selection settings requires a new data directory.

## Run a small pilot

```bash
uv run vlm-crop run --config configs/pilot.json --output artifacts/pilot-required-crop
uv run vlm-crop budget
uv run vlm-crop report --run artifacts/pilot-required-crop --output site-output-required-crop
```

The pilot uses one shared image across GPT-5.4, GPT-4o, and GPT-4o-mini, in all three conditions. It checks API access, tool/image handling, and artifacts. It cannot establish comparative model quality or a cropping benefit. The viewer labels pilot results accordingly.

Open `site-output-required-crop/index.html` directly, or serve it:

```bash
python3 -m http.server 8000 --directory site-output-required-crop --bind 127.0.0.1
```

Then visit <http://localhost:8000>. The initial optional-crop pilot remains in `artifacts/pilot` and `site-output`. Protocol v2 requires a crop before answering; use a new run directory to avoid mixing protocols.

## Full evaluation (explicit separate command)

```bash
uv run vlm-crop run --config configs/evaluation.json --output artifacts/evaluation
uv run vlm-crop report --run artifacts/evaluation --output site-output-evaluation
```

The default evaluation selects GPT-5.4 and GPT-4o, fixed before evaluating the held-out 60 images: 360 episodes total. Change the config **before starting** if using a different supported model. The one-image pilot is only a functionality/difficulty check; do not choose models or examples based on a cropping advantage in the evaluation set.

The same command resumes by skipping existing episode records, including errors. Changed settings are rejected for an existing run directory. A process interrupted after an API request may have been charged: that episode is marked interrupted rather than silently repeated. To explicitly rerun failed work, use a new run directory; the shared ledger still applies. There is no automatic replacement of unavailable models; `gpt-5.4-mini` is a supported, priced fallback that must be recorded in the config.

## Experimental protocol

- Initial full image counts as view 1. Repeat/crop agents can request up to four further views. The crop agent MUST successfully inspect at least one crop before answering; the repeat agent may answer immediately. Single-view gets one view.
- One tool call per turn. The repeat agent has `view_full`; the crop agent has `view_full` and `crop`. The system instructions explain cropping and require it even when the answer appears obvious. Until a valid crop succeeds, API tool choice forces `crop`; an early answer or exhausted crop retries fails the episode. No other tools are available.
- The model chooses integer `[left, top, right, bottom]` coordinates in the **original** image. Right/bottom bounds are exclusive. Invalid rectangles and the full-image bounding box are rejected, not clamped. The model chooses a proper subregion itself.
- Crop pixels always come from the original, then resize with Lanczos interpolation to a 1,024px longest side, preserving aspect ratio. Full images retain original dimensions. API-side image processing is separate.
- All images use `detail=high`. Conversation history is retained. The view budget counts newly supplied images, not repeated image tokens in history.
- GPT-5 models use `reasoning.effort=none`; GPT-4o models use `temperature=0`. Requested and resolved model IDs, prompts, tool schemas, settings, and source fingerprints are recorded. Snapshots can be specified in configs.
- Final answers use a strict JSON schema on the last allowed view; earlier answers are validated against the same shape. Labels are compared after trimming whitespace/braces/quotes and ignoring letter case. Invalid/incomplete outputs are failures.
- At most two malformed-action/format corrections. Error corrections do not grant extra views. SDK retries are disabled; one retry is allowed for explicit 429/5xx responses. Network ambiguity stops the episode.

The original benchmark questions sometimes request braces; our higher-priority instructions require an answer JSON wrapper consistently across all conditions. Normalization accepts braces inside its string value.

## Ground truth isolation

Only `AgentInput(question, image_pixels, width, height)` enters the episode engine. Images are reconstructed from RGB pixels to remove metadata. Opaque IDs replace source filenames. Labels live in a separate evaluator file; source metadata and labels never enter API messages or tool responses. Models have no filesystem, browser, code execution, or grading tool. Tests inspect actual outbound payloads for sentinel labels and metadata.

The developer and offline evaluator can access labels. Published replay results intentionally show ground truth after evaluation and label it as evaluator-only information. This is an API-input boundary, not an operating-system sandbox for the developer. Public benchmark pretraining contamination cannot be ruled out.

## Budget and artifacts

The CLI shares `artifacts/budget.json` across all runs in this checkout, with a **$10 nominal ceiling**. Keep this file when cleaning results; deleting it resets accounting. A file lock serializes runs and atomic writes preserve ledger updates. Do not run separate copies with independent ledgers against the same experiment budget.

Before each request, the runner reserves a conservative input/output cost bound, including images, retained history, schema overhead, and capped output. It reconciles with returned usage. Cached input is charged at full input price for conservative nominal accounting. Unknown charges after errors/crashes retain their reservation; they are never silently released. Free daily usage is **not assumed** and actual billing may be lower. Prices are standard API model-page prices checked on 2026-11-05; recheck them before future runs. The ledger is an application-level control, not an account-wide billing limit. The configured key was stated to expire on 2026-11-12.

Each run stores:

- Frozen configuration, protocol and source fingerprints.
- Exact requests (including image payloads), responses, usage, and timings.
- Sanitized full-image/crop PNGs, crop coordinates, answers, and scores.
- Error records, progress, and resumable episode completion records.

Raw artifacts and `.env` are gitignored. Reports export only replay records and images, not raw API requests or credentials. Reports include accuracy by task/model/condition, descriptive 95% Wilson intervals, completion rates, completed-only accuracy, paired improvements/regressions, tokens, latency, and nominal cost. Failures count as incorrect in headline accuracy; missing pairs are excluded from paired comparisons. One run per condition does not measure stochastic stability.

## Blog viewer / GitHub Pages

The generated site is static, requires no API key or server backend, works under a repository subpath, and supports direct local opening. It provides model/task/outcome selectors, a synchronized round slider, replay playback, original-image crop overlays, and recorded messages/actions. It shows observable outputs, never invented hidden reasoning.

`exports/` contains a comparison PNG and replay GIF for each model/image; `summary.png` is the aggregate chart. JSON and CSV exports are linked in the viewer. Copy the generated directory to the eventual GitHub Pages publishing source when ready. Generating the report does not publish or commit anything.

## Tests

```bash
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
node --check src/vlm_crop/static/viewer.js
```

Optional browser checks verify static subpath hosting, replay, crop overlays, exports, filtering, and mobile overflow:

```bash
uv run playwright install chromium
VLM_BROWSER_TESTS=1 uv run pytest tests/test_browser.py -q
```

Tests use generated fixtures and mocked API responses; they incur no API cost. Only the explicit `run` command calls OpenAI.

## Attribution

Benchmark: [Vision Language Models Are Blind](https://vlmsareblind.github.io/), Pooyan Rahmanzadehgervi, Logan Bolton, Mohammad Reza Taesiri, and Anh Totti Nguyen, ACCV 2024. [Original code](https://github.com/anguyen8/vision-llms-are-blind), [dataset](https://huggingface.co/datasets/XAI/vlmsareblind) (dataset card declares MIT). This project implements its own evaluation harness. Preserve dataset attribution when publishing examples.
