#!/usr/bin/env python3
"""Convert raw GSM8K JSONL into strict-format chat SFT data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verl.utils.reward_score.gsm8k import compute_score, extract_solution, hard_format_reward

ROOT_DIR = Path(__file__).resolve().parents[1]


SYSTEM_PROMPT = """按照如下格式回答问题：

<think>
你的思考过程
</think>
<answer>
#### 你的最终答案
</answer>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT_DIR / "data" / "unprocessed")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "data" / "sft-data")
    return parser.parse_args()


def convert_row(row: dict[str, str], split: str, index: int) -> dict[str, object]:
    question = row.get("question", "").strip()
    raw_answer = row.get("answer", "").strip()
    if not question or not raw_answer:
        raise ValueError(f"{split}[{index}] must contain non-empty question and answer fields")
    if "\n#### " not in raw_answer:
        raise ValueError(f"{split}[{index}] answer has no final '#### ' marker")

    reasoning, ground_truth = raw_answer.rsplit("\n#### ", maxsplit=1)
    reasoning = reasoning.strip()
    ground_truth = ground_truth.replace(",", "").strip()
    response = f"<think>\n{reasoning}\n</think>\n<answer>\n#### {ground_truth}\n</answer>"

    format_reward = hard_format_reward(response)
    extracted = extract_solution(response, method="strict")
    total_reward = compute_score(response, ground_truth, method="strict")
    if format_reward != 0.3 or extracted != ground_truth or total_reward != 2.0:
        raise ValueError(
            f"{split}[{index}] failed strict reward validation: "
            f"format={format_reward}, extracted={extracted!r}, expected={ground_truth!r}, total={total_reward}"
        )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": response},
        ],
        "question": question,
        "response": response,
        "ground_truth": ground_truth,
        "split": split,
        "source_index": index,
        "strict_reward": total_reward,
    }


def convert_file(input_path: Path, output_path: Path, split: str) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {input_path}:{line_number}") from exc
            converted = convert_row(row, split=split, index=count)
            target.write(json.dumps(converted, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    args = parse_args()
    summary: dict[str, object] = {"system_prompt": SYSTEM_PROMPT, "splits": {}}
    for split in ("train", "test"):
        input_path = args.input_dir / f"{split}.jsonl"
        if not input_path.is_file():
            raise FileNotFoundError(f"Missing input file: {input_path}")
        output_path = args.output_dir / f"{split}.jsonl"
        count = convert_file(input_path, output_path, split)
        summary["splits"][split] = {"source": str(input_path), "output": str(output_path), "count": count}
        print(f"{split}: wrote {count} strict-format examples to {output_path}")

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
