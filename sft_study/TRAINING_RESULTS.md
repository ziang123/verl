# Training results

Model: `/media/iie/4Tb/model/Qwen2.5-3B-Instruct`

Hardware: physical GPU 0 and GPU 1 (`Tesla V100-PCIE-32GB`), FP16, two-process
DDP, `NCCL_P2P_DISABLE=1`.

Data: 7,473 training examples and 1,319 test examples. All 8,792 converted rows
score 2.0 with the current GSM8K GRPO reward before training.

## Staged checks

| Run | Final train loss | Final eval loss | Fixed generation |
| --- | ---: | ---: | --- |
| 5 steps | 0.384 | 0.459 | strict format, reward 2.0 |
| 50 steps | 0.292 | 0.377 | strict format, reward 2.0 |
| Full, 1,404 steps | 0.205 | 0.428 | strict format, reward 2.0 |

The full run completed three epochs. Its lowest periodic eval loss was 0.365 at
step 400; eval loss rose in the third epoch while training loss continued to
fall, which indicates overfitting. All periodic generations after training
started followed the strict tag/newline format, although answer correctness on
the fixed question fluctuated at intermediate checkpoints.

## Multi-example check

Twenty deterministic, evenly spaced test examples were generated greedily.

| Adapter | Token limit | Strict format | Exact answer/full reward | Mean reward |
| --- | ---: | ---: | ---: | ---: |
| `checkpoint-250` | 256 | 20/20 | 15/20 | 1.675 |
| `final` | 512 | 20/20 | 16/20 | 1.740 |

Use `outputs/qwen25-3b-lora-full/final` when inference permits the GRPO-aligned
512-token response limit. `checkpoint-250` remains a useful alternative when
shorter responses or less task-specific fitting are preferred.
