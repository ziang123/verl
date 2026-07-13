# Three-model GSM8K evaluation

This evaluates the base, SFT, and GRPO step-600 models on every row of
`data/unprocessed/test.jsonl`. The specified Chinese system prompt is added at
runtime; processed SFT data is not used as the evaluation source.

The run is deterministic greedy generation with a 512-token response limit.
Each model is sharded across physical GPU 0 and GPU 1. Results contain the
unaltered project reward, all five reward components, strict answer accuracy,
and flexible numeric accuracy for responses that do not follow the format.

Run all 1,319 examples with:

```bash
test_model/run_all.sh
```

For a short pipeline check:

```bash
MAX_SAMPLES=20 test_model/run_all.sh
```

The GRPO FSDP checkpoint was converted with:

```bash
PYTHONPATH="$PWD" conda run -n rl_study python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir outputs/checkpoints/grpo_qwen25_3b_instruct_gsm8k_lora_2gpu26/global_step_600/actor \
  --target_dir test_model/models/grpo_step_600_merged \
  --use_cpu_initialization
```

The completed full-run comparison is summarized in `RESULTS.md`; raw generations
and machine-readable summaries are under `results/full/`.

## GRPO checkpoint comparison

The step 100-700 GRPO checkpoints can be exported as adapter-only PEFT models
with:

```bash
test_model/export_all_grpo_adapters.sh
```

Run the fixed 200-row comparison on physical GPU 0 and GPU 1 with:

```bash
test_model/run_grpo_checkpoints.sh
```

`GRPO_CHECKPOINT_RESULTS.md` records the checkpoint comparison, including the
full 1,319-row finalist evaluation. `RESEARCH_REPORT.md` contains the complete
SFT/GRPO study, training-metric analysis, figures, and the DAPO/GRPO-on-SFT
follow-up design. Generated
adapters are under `models/grpo_checkpoints/`, while raw answers and summaries
are under `results/grpo_checkpoint_comparison/` and
`results/grpo_full_finalists/`.
