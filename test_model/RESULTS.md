# Full evaluation results

All three models were evaluated on the complete 1,319 rows from
`data/unprocessed/test.jsonl`. Every prompt used the requested Chinese system
prompt. Generation was greedy with a 512-token response limit and was split
evenly across physical GPU 0 and GPU 1.

| Model | Strict format | Strict answer | Flexible answer | Full reward | Mean reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 269/1319 (20.39%) | 74/1319 (5.61%) | 1014/1319 (76.88%) | 0/1319 | 0.2618 |
| SFT final | 1316/1319 (99.77%) | 830/1319 (62.93%) | 830/1319 (62.93%) | 829/1319 (62.85%) | 1.5166 |
| GRPO step 600 | 0/1319 (0.00%) | 296/1319 (22.44%) | 992/1319 (75.21%) | 0/1319 | 0.3555 |

The mean reward is the unmodified project GSM8K reward with maximum 2.0.
Strict answer uses the reward function's `####`/boxed extraction. Flexible
answer uses the last numeric value and is reported separately so format-invalid
base responses can still be evaluated for mathematical correctness.

Base has the highest format-agnostic answer accuracy. In 940 cases its answer
was correct only after flexible extraction. SFT is the only model that reliably
learned the complete strict protocol and therefore has by far the highest
project reward. Its three format failures include two responses truncated while
still reasoning at the 512-token limit.

The GRPO adapter often emits closing `</think>` and `<answer>` tags, but greedy
responses commonly omit the opening `<think>` and/or the `####` marker. This
explains its 0% full-format rate despite 75.21% flexible answer accuracy. GRPO
was trained with stochastic rollouts, but this comparison deliberately uses the
same deterministic generation settings for all three models.

Machine-readable summaries and every raw generation are in
`test_model/results/full/`.

A newer GRPO step-800 checkpoint was subsequently evaluated on all 1,319 rows.
See `RESEARCH_REPORT.md` for the current SFT/GRPO comparison; this file retains
the original three-model step-600 run for reproducibility.
