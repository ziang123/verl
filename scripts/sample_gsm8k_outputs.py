#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_gsm8k_answer(text: str, method: str = "strict") -> str | None:
    text = text[-300:]
    if method == "strict":
        matches = re.findall(r"#### (\-?[0-9\.\,]+)", text)
        return matches[-1].replace(",", "").replace("$", "") if matches else None
    matches = re.findall(r"(\-?[0-9\.\,]+)", text)
    for value in reversed(matches):
        if value not in {"", "."}:
            return value.replace(",", "")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample Qwen GSM8K outputs and show reward extraction.")
    parser.add_argument("--model", default="/media/iie/4Tb/model/Qwen2.5-Math-1.5B")
    parser.add_argument("--data", default=str(Path.home() / "data/gsm8k/train.parquet"))
    parser.add_argument("--output", default="logs/gsm8k_sample_outputs.jsonl")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    df = pd.read_parquet(args.data).head(args.num_samples)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            messages = row["prompt"]
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
            response_ids = generated[0, inputs["input_ids"].shape[1] :]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            gt = row["reward_model"]["ground_truth"]
            strict_answer = extract_gsm8k_answer(response, "strict")
            flexible_answer = extract_gsm8k_answer(response, "flexible")
            record = {
                "index": int(idx),
                "question": row["extra_info"]["question"],
                "ground_truth": gt,
                "response": response,
                "strict_extracted": strict_answer,
                "flexible_extracted": flexible_answer,
                "strict_score": 1.0 if strict_answer == gt else 0.0,
                "flexible_score": 1.0 if flexible_answer == gt else 0.0,
                "response_tokens": int(response_ids.numel()),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print("=" * 80)
            print(f"sample={idx} gt={gt} tokens={record['response_tokens']}")
            print(f"strict={strict_answer!r} flexible={flexible_answer!r}")
            print(response[:2000])

    print(f"\nSaved samples to {output_path}")


if __name__ == "__main__":
    main()
