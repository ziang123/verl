#!/usr/bin/env python3
"""Evaluate strict response formatting for a saved LoRA adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.utils.reward_score.gsm8k import compute_score, extract_solution, hard_format_reward

DEFAULT_MODEL_PATH = "/media/iie/4Tb/model/Qwen2.5-3B-Instruct"
DEFAULT_EVAL_FILE = Path(__file__).resolve().parents[1] / "data" / "sft-data" / "test.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def load_samples(path: Path, count: int) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if count <= 0 or count > len(rows):
        count = len(rows)
    if count == 1:
        return [rows[0]]
    indices = [round(index * (len(rows) - 1) / (count - 1)) for index in range(count)]
    return [rows[index] for index in indices]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Evaluation expects exactly one visible CUDA GPU")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).cuda()
    model = PeftModel.from_pretrained(base_model, args.adapter_path).eval()
    model.config.use_cache = True

    samples = load_samples(args.eval_file, args.num_samples)
    results: list[dict[str, object]] = []
    for start in range(0, len(samples), args.batch_size):
        batch = samples[start : start + args.batch_size]
        prompts = [
            tokenizer.apply_chat_template(row["messages"][:-1], tokenize=False, add_generation_prompt=True)
            for row in batch
        ]
        inputs = tokenizer(prompts, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
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
            ground_truth = str(row["ground_truth"])
            extracted = extract_solution(generated, method="strict")
            results.append(
                {
                    "question": row["question"],
                    "ground_truth": ground_truth,
                    "generated": generated,
                    "strict_format": hard_format_reward(generated) == 0.3,
                    "extracted_answer": extracted,
                    "answer_correct": extracted == ground_truth,
                    "reward": compute_score(generated, ground_truth, method="strict"),
                }
            )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    summary = {
        "adapter_path": str(args.adapter_path),
        "num_samples": len(results),
        "strict_format_rate": sum(bool(row["strict_format"]) for row in results) / len(results),
        "answer_accuracy": sum(bool(row["answer_correct"]) for row in results) / len(results),
        "full_reward_rate": sum(float(row["reward"]) == 2.0 for row in results) / len(results),
        "mean_reward": sum(float(row["reward"]) for row in results) / len(results),
        "results_file": str(args.output_file),
    }
    summary_path = args.output_file.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
