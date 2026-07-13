# Qwen2.5-3B-Instruct strict-format LoRA SFT

This directory trains `/media/iie/4Tb/model/Qwen2.5-3B-Instruct` on the raw
GSM8K JSONL files under `data/unprocessed`. Converted chat data is stored in
`data/sft-data`. It uses GPU 0 and GPU 1 only. Those
GPUs are V100s, so the launcher defaults to FP16 rather than unsupported BF16.
Their NCCL P2P path does not complete collectives on this host, so this launcher
also sets `NCCL_P2P_DISABLE=1` locally and uses shared-memory collectives.

The preprocessing step converts every assistant target to this exact shape:

```text
<think>
the original GSM8K reasoning
</think>
<answer>
#### the numeric final answer
</answer>
```

Every converted row must score `2.0` with
`verl.utils.reward_score.gsm8k.compute_score`; preprocessing fails closed if a
row does not satisfy the same strict format and answer checks used by GRPO.

Run preprocessing independently with:

```bash
conda run -n rl_study python sft_study/preprocess_data.py
```

Run the required 5-step, 50-step, and full stages in sequence with:

```bash
sft_study/run_staged.sh
```

For one stage, set `MAX_STEPS=5`, `MAX_STEPS=50`, or `MAX_STEPS=0` (all three
epochs) before invoking `sft_study/run_train.sh`.

Each run writes:

- `logs/<run>.log`: console training and generated responses.
- `outputs/<run>/metrics.jsonl`: train/eval losses and learning rate.
- `outputs/<run>/loss.png`: loss curve refreshed during training.
- `outputs/<run>/tensorboard/`: TensorBoard scalars and generated text.
- `outputs/<run>/generations.jsonl`: fixed before/during/after generations,
  strict-format status, extracted answer, and GRPO reward.
- `outputs/<run>/final/`: the PEFT LoRA adapter and tokenizer.

TensorBoard can be started with:

```bash
conda run -n rl_study tensorboard --logdir sft_study/outputs --port 6006
```

Evaluate a saved adapter on evenly spaced test examples with:

```bash
CUDA_VISIBLE_DEVICES=0 python sft_study/evaluate_adapter.py \
  --adapter-path sft_study/outputs/qwen25-3b-lora-full/final \
  --output-file sft_study/outputs/qwen25-3b-lora-full/evaluation.jsonl
```

The completed 3-epoch run is under `outputs/qwen25-3b-lora-full`. With the same
512-token response limit used by GRPO, its `final` adapter produced strict
format on 20/20 evenly spaced test examples, with 16/20 exact answers and full
reward. See `TRAINING_RESULTS.md` for the staged and full-run results.
