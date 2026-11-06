# Full evaluation — mandatory-crop protocol v2

Completed 60 unique held-out images: 20 line-intersection, 20 circled-letter, and 20 subway-connection images. Two model snapshots × three conditions = **360 completed episodes**, with no failed episodes. All 120 crop episodes included at least one successful subregion crop. The pilot image was excluded.

## Overall accuracy

| Model | Single view | Repeat enabled | Mandatory crop |
| --- | ---: | ---: | ---: |
| gpt-5.4-2026-03-05 | 44/60 (73.3%) | 47/60 (78.3%) | 54/60 (90.0%) |
| gpt-4o-2024-08-06 | 34/60 (56.7%) | 33/60 (55.0%) | 36/60 (60.0%) |

## Accuracy by task

| Model | Task | Single | Repeat enabled | Crop |
| --- | --- | ---: | ---: | ---: |
| gpt-5.4-2026-03-05 | Line Plot Intersections | 20/20 | 20/20 | 20/20 |
| gpt-5.4-2026-03-05 | Circled Letter | 18/20 | 18/20 | 20/20 |
| gpt-5.4-2026-03-05 | Subway Connections | 6/20 | 9/20 | 14/20 |
| gpt-4o-2024-08-06 | Line Plot Intersections | 12/20 | 11/20 | 13/20 |
| gpt-4o-2024-08-06 | Circled Letter | 16/20 | 15/20 | 16/20 |
| gpt-4o-2024-08-06 | Subway Connections | 6/20 | 7/20 | 7/20 |

## Paired changes on the same images

| Model | Crop compared with | Wrong → correct | Correct → wrong | Unchanged |
| --- | --- | ---: | ---: | ---: |
| gpt-5.4-2026-03-05 | single | 10 | 0 | 50 |
| gpt-5.4-2026-03-05 | repeat | 9 | 2 | 49 |
| gpt-4o-2024-08-06 | single | 9 | 7 | 44 |
| gpt-4o-2024-08-06 | repeat | 8 | 5 | 47 |

## What this result supports

On this sample, the mandatory-crop workflow improved GPT-5.4 by 10/60 answers (+16.7 percentage points) over single-view and by 7/60 (+11.7 points) over repeat-enabled. GPT-4o improved by 2/60 (+3.3 points) and 3/60 (+5.0 points), respectively, with appreciable gains and regressions on individual images.

**Every repeat-enabled episode stopped after its first full-image view.** That arm had permission to revisit the image but did not exercise it. These results therefore compare a mandatory additional crop against workflows that chose to answer immediately, not cropping against an equal number of full-image revisits. The maximum viewing budget was equal, but actual views and inference steps were not.

GPT-5.4 crop episodes averaged 2.08 views; GPT-4o crop episodes averaged 2.37. Three invalid crop actions were rejected and corrected within the configured limit. No prompts, model choices, or samples were tuned during evaluation.

This is an exploratory run with only 20 images per task and one episode per image/condition. Single and repeat answers can differ even when both use one view, because tool availability/request structure differ and outputs are not guaranteed deterministic. The observed gains do not establish image compression as the causal mechanism. The public benchmark may also have appeared in model training data.

## Cost and artifacts

- Evaluation: **$2.0558675** nominal usage cost across 510 API requests.
- Total including the earlier pilot and crop-tool smoke check: **$2.20833435**, below the $10 ceiling. No unresolved reservations remain. Actual billing may be lower with free usage.
- Run: `artifacts/evaluation/` (frozen protocol/config, exact requests/responses, crops, scores, usage).
- Interactive report: `site-output-evaluation/index.html`.
- Statistics: `site-output-evaluation/summary.json`, `summary.csv`, and `summary.png`.
- Per-example comparison PNGs and replay GIFs: `site-output-evaluation/exports/`.
- The original optional-crop pilot is preserved in `artifacts/pilot/` and `site-output/`.
- No commits or deployment performed.
