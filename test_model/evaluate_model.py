#!/usr/bin/env python3
"""Generate one model's GSM8K answers and score them with the project reward."""

from __future__ import annotations

import argparse
import json
import random
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from verl.utils.reward_score.gsm8k import (  # noqa: E402
    compute_score,
    correctness_reward,
    digit_reward,
    extract_solution,
    hard_format_reward,
    mark_reward,
    process_reward,
)

SYSTEM_PROMPT = """按照如下格式回答问题：

<think>
你的思考过程
</think>
<answer>
#### 你的最终答案
</answer>
"""

BASE_MODEL_PATH = "/media/iie/4Tb/model/Qwen2.5-3B-Instruct"
SFT_ADAPTER_PATH = ROOT_DIR / "sft_study/outputs/qwen25-3b-lora-full/final"
GRPO_ADAPTER_PATH = ROOT_DIR / "test_model/models/grpo_step_600_merged/lora_adapter"
RAW_TEST_PATH = ROOT_DIR / "data/unprocessed/test.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("base", "sft", "grpo"), required=True)
    parser.add_argument("--model-name", default=None, help="Result label; defaults to --model.")
    parser.add_argument("--base-model-path", default=BASE_MODEL_PATH)
    parser.add_argument("--sft-adapter-path", type=Path, default=SFT_ADAPTER_PATH)
    parser.add_argument("--grpo-adapter-path", type=Path, default=GRPO_ADAPTER_PATH)
    parser.add_argument("--test-file", type=Path, default=RAW_TEST_PATH)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0, help="0 evaluates the complete test file")
    parser.add_argument(
        "--sample-count",
        type=int,
        default=0,
        help="Select this many evenly spaced rows across the eligible test set; 0 disables sampling.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_number(value: str | None) -> str | None:
    if value is None:
        return None
    return value.replace(",", "").replace("$", "").strip()


def numeric_equal(left: str | None, right: str | None) -> bool:
    left = normalize_number(left)
    right = normalize_number(right)
    if left is None or right is None:
        return False
    try:
        return Decimal(left) == Decimal(right)
    except InvalidOperation:
        return left == right


def load_rows(
    path: Path,
    max_samples: int,
    sample_count: int,
    shard_index: int,
    num_shards: int,
) -> list[dict[str, Any]]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for source_index, line in enumerate(handle):
            if not line.strip():
                continue
            if max_samples > 0 and source_index >= max_samples:
                break
            row = json.loads(line)
            question = str(row["question"]).strip()
            raw_answer = str(row["answer"]).strip()
            if "\n#### " not in raw_answer:
                raise ValueError(f"Row {source_index} has no GSM8K final-answer marker")
            ground_truth = normalize_number(raw_answer.rsplit("\n#### ", maxsplit=1)[1])
            rows.append(
                {
                    "source_index": source_index,
                    "question": question,
                    "ground_truth": ground_truth,
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": question},
                    ],
                }
            )
    if sample_count > 0:
        if sample_count > len(rows):
            raise ValueError(f"--sample-count={sample_count} exceeds {len(rows)} eligible rows")
        if sample_count == 1:
            rows = [rows[0]]
        else:
            selected_indices = [round(index * (len(rows) - 1) / (sample_count - 1)) for index in range(sample_count)]
            rows = [rows[index] for index in selected_indices]
    rows = [row for selection_index, row in enumerate(rows) if selection_index % num_shards == shard_index]
    if not rows:
        raise ValueError(f"No rows assigned to shard {shard_index}/{num_shards}")
    return rows


def load_model(args: argparse.Namespace) -> tuple[Any, Any, str | None]:
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).cuda()
    adapter_path: Path | None = None
    if args.model == "sft":
        adapter_path = args.sft_adapter_path
    elif args.model == "grpo":
        adapter_path = args.grpo_adapter_path
    if adapter_path is not None:
        if not (adapter_path / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"Missing adapter weights: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer, None if adapter_path is None else str(adapter_path)


def score_response(generated: str, ground_truth: str) -> dict[str, object]:
    strict_answer = extract_solution(generated, method="strict")
    flexible_answer = extract_solution(generated, method="flexible")
    return {
        "strict_format": hard_format_reward(generated) == 0.3,
        "strict_answer": strict_answer,
        "flexible_answer": flexible_answer,
        "strict_answer_correct": numeric_equal(strict_answer, ground_truth),
        "flexible_answer_correct": numeric_equal(flexible_answer, ground_truth),
        "reward/format": hard_format_reward(generated),
        "reward/marks": mark_reward(generated),
        "reward/process": process_reward(generated),
        "reward/digit": digit_reward(generated, ground_truth),
        "reward/correctness": correctness_reward(generated, ground_truth),
        "reward/total": compute_score(generated, ground_truth),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each evaluator process requires exactly one visible CUDA GPU")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_rows(args.test_file, args.max_samples, args.sample_count, args.shard_index, args.num_shards)
    model, tokenizer, adapter_path = load_model(args)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as output_handle:
        for start in tqdm(range(0, len(rows), args.batch_size), desc=f"{args.model} shard {args.shard_index}"):
            batch = rows[start : start + args.batch_size]
            prompt_texts = [
                tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
                for row in batch
            ]
            inputs = tokenizer(prompt_texts, padding=True, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            prompt_width = inputs["input_ids"].shape[1]
            for row, token_ids in zip(batch, output_ids, strict=True):
                generated = tokenizer.decode(token_ids[prompt_width:], skip_special_tokens=True).strip()
                result = {
                    "model": args.model_name or args.model,
                    "base_model_path": args.base_model_path,
                    "adapter_path": adapter_path,
                    "source_index": row["source_index"],
                    "question": row["question"],
                    "ground_truth": row["ground_truth"],
                    "generated": generated,
                    **score_response(generated, row["ground_truth"]),
                }
                output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                output_handle.flush()


if __name__ == "__main__":
    main()
