#!/usr/bin/env python3
"""Combine two GPU result shards and write model-comparison summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MODEL_ORDER = ("base", "sft", "grpo")
REWARD_KEYS = (
    "reward/format",
    "reward/marks",
    "reward/process",
    "reward/digit",
    "reward/correctness",
    "reward/total",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=1319)
    return parser.parse_args()


def load_model_rows(results_dir: Path, model_name: str, expected_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob(f"{model_name}.shard-*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    rows.sort(key=lambda row: int(row["source_index"]))
    indices = [int(row["source_index"]) for row in rows]
    if len(rows) != expected_count or indices != list(range(expected_count)):
        raise ValueError(
            f"{model_name}: expected exactly indices 0..{expected_count - 1}, "
            f"got {len(rows)} rows spanning {indices[:1]}..{indices[-1:]}."
        )
    combined_path = results_dir / f"{model_name}.jsonl"
    with combined_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, object]:
    count = len(rows)
    strict_format_count = sum(bool(row["strict_format"]) for row in rows)
    strict_answer_correct_count = sum(bool(row["strict_answer_correct"]) for row in rows)
    flexible_answer_correct_count = sum(bool(row["flexible_answer_correct"]) for row in rows)
    full_reward_count = sum(float(row["reward/total"]) == 2.0 for row in rows)
    return {
        "model": rows[0]["model"],
        "count": count,
        "strict_format_count": strict_format_count,
        "strict_format_rate": strict_format_count / count,
        "strict_answer_correct_count": strict_answer_correct_count,
        "strict_answer_accuracy": strict_answer_correct_count / count,
        "flexible_answer_correct_count": flexible_answer_correct_count,
        "flexible_answer_accuracy": flexible_answer_correct_count / count,
        "flexible_only_correct_count": sum(
            bool(row["flexible_answer_correct"]) and not bool(row["strict_answer_correct"]) for row in rows
        ),
        "full_reward_count": full_reward_count,
        "full_reward_rate": full_reward_count / count,
        "mean_rewards": {
            key: sum(float(row[key]) for row in rows) / count
            for key in REWARD_KEYS
        },
    }


def write_markdown(summaries: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Three-model GSM8K evaluation",
        "",
        "All 1,319 raw test questions use the same system prompt and greedy generation with a 512-token limit.",
        "The mean reward uses the project's strict five-component GSM8K reward without modification.",
        "Flexible accuracy extracts the last numeric answer for format-invalid responses, "
        "primarily for the base model.",
        "",
        "| Model | Strict format | Strict accuracy | Flexible accuracy | Full reward | Mean reward |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['model']} | {summary['strict_format_count']}/{summary['count']} "
            f"({summary['strict_format_rate']:.2%}) | "
            f"{summary['strict_answer_correct_count']}/{summary['count']} "
            f"({summary['strict_answer_accuracy']:.2%}) | "
            f"{summary['flexible_answer_correct_count']}/{summary['count']} "
            f"({summary['flexible_answer_accuracy']:.2%}) | "
            f"{summary['full_reward_count']}/{summary['count']} ({summary['full_reward_rate']:.2%}) | "
            f"{summary['mean_rewards']['reward/total']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Mean reward components",
            "",
            "| Model | Format | Marks | Process | Digit | Correctness | Total |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for summary in summaries:
        rewards = summary["mean_rewards"]
        lines.append(
            f"| {summary['model']} | {rewards['reward/format']:.4f} | {rewards['reward/marks']:.4f} | "
            f"{rewards['reward/process']:.4f} | {rewards['reward/digit']:.4f} | "
            f"{rewards['reward/correctness']:.4f} | {rewards['reward/total']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    summaries = []
    for model_name in MODEL_ORDER:
        rows = load_model_rows(args.results_dir, model_name, args.expected_count)
        summary = summarize(rows)
        summaries.append(summary)
        with (args.results_dir / f"{model_name}.summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    with (args.results_dir / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    write_markdown(summaries, args.results_dir / "comparison.md")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
