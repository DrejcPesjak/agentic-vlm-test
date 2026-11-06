# Implementation validation — 2026-11-05

## Automated checks

- 33 offline tests: deterministic sampling and deduplication, label isolation, metadata removal, original-image crop geometry and resizing, view limits, early answers, malformed and parallel actions, final-answer parsing, interrupted-run handling, resume behavior, conservative budget reservations, transient-error retries, scoring, and report/image exports.
- 1 Chromium integration test: static hosting under a repository subpath, synchronized replay, original-image crop overlays, export links, outcome filters, mobile overflow, and browser errors.
- Ruff lint/format checks, mypy on application code, and JavaScript syntax validation.
- Actual generated pilot viewer inspected on desktop and mobile. No JavaScript errors or horizontal mobile overflow. Checked local file opening as well as HTTP subpath hosting.
- Generated report and source scanned against local key values without printing those values; no matches found.

## Live mini test

Prepared the pinned benchmark: 60 held-out evaluation images and one independent pilot image, seed 42. Inspected representative images from all three tasks.

Ran nine pilot episodes: the same line-intersection image, three models, and three conditions. All nine completed without API errors. Ground truth was zero intersections.

| Model | Single view | Repeated full view | Crop-enabled |
| --- | --- | --- | --- |
| GPT-5.4 | Correct, 1 view | Correct, 1 view | Correct, 1 view |
| GPT-4o | Incorrect, 1 view | Incorrect, 1 view | Incorrect, 1 view |
| GPT-4o-mini | Incorrect, 1 view | Correct, 5 views | Correct, 5 views |

The crop-enabled mini agent requested the **full image** on every additional view. This is not evidence of a cropping improvement. It demonstrates the agent's option to revisit the full image and that the harness preserves the selected actions honestly.

A separate, unscored two-view API integration check forced the first tool name to `crop` while leaving coordinates to the model. The model chose `[0, 0, 1152, 1152]` (the full extent), and the tool returned a 1024×1024 image. The follow-up response completed successfully. This verifies the live crop-tool/image-response path; localized crop geometry is covered by deterministic offline tests. This check is stored separately under `artifacts/crop-smoke/` and excluded from the experimental report.

The pilot used 17 API requests; the separate tool check used 2. Total nominal usage cost: **$0.15246685**, with zero unresolved reservations, against the persistent $10 limit. This does not assert actual billing; free usage may reduce it.

Re-running the pilot command skipped all nine episodes without new requests. No full evaluation, deployment, or commits were performed.

## Artifacts

- `artifacts/data/manifest.json`: frozen 61-image selection.
- `artifacts/pilot/`: exact pilot requests, responses, image views, and scored episodes.
- `artifacts/crop-smoke/`: explicitly forced integration check, not experimental evidence.
- `artifacts/budget.json`: shared cost ledger; retain when cleaning outputs.
- `site-output/index.html`: static pilot replay, JSON/CSV statistics, PNG/GIF exports.
- `artifacts/viewer-desktop.png` and `artifacts/viewer-mobile.png`: inspected screenshots.

The evaluation configuration pins the pilot-resolved snapshots `gpt-5.4-2026-03-05` and `gpt-4o-2024-08-06`. They retain the planned flagship comparison; the pilot is too small to rank models. The 60-image evaluation is ready but has not been executed.

## Protocol v2: mandatory crop instruction

The crop condition now receives system instructions requiring a successful crop before answering, explaining original-image coordinates, bounds, exclusive right/bottom edges, and resizing. API tool choice forces the crop tool until a valid crop succeeds. The model still chooses its region. Full-image boxes do not satisfy the requirement; premature answers and exhausted crop retries fail the episode. The other conditions retain their previous instructions.

The frozen run signature now includes per-condition system instructions and protocol version 2. Existing optional-crop pilot results remain protocol v1 and cannot be resumed under v2. The static viewer derives its methodology label from each run's saved version. README commands use a new pilot output directory for v2.

Validation: 38 offline tests passed, with lint, type, format, and JavaScript syntax checks passing. No new live API calls were made for this prompt update; the earlier pilot numbers above still describe protocol v1.
