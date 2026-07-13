#!/usr/bin/env python3
"""Aggregate fixed-subset results for GRPO checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from summarize_results import summarize

STEPS = (100, 200, 300, 400, 500, 600, 700, 800)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=200)
    parser.add_argument("--training-summary", type=Path, default=None)
    parser.add_argument("--steps", type=int, nargs="+", default=list(STEPS))
    return parser.parse_args()


def load_rows(results_dir: Path, step: int, expected_count: int) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(results_dir.glob(f"grpo_step_{step}.shard-*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    rows.sort(key=lambda row: int(row["source_index"]))
    if len(rows) != expected_count or len({int(row["source_index"]) for row in rows}) != expected_count:
        raise ValueError(f"step {step}: expected {expected_count} unique rows, got {len(rows)}")
    with (results_dir / f"grpo_step_{step}.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def load_training_buckets(path: Path | None) -> dict[int, dict[str, str]]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {int(row["step_end"]): row for row in rows}


def write_markdown(summaries: list[dict[str, Any]], path: Path, expected_count: int) -> None:
    lines = [
        "# GRPO checkpoint comparison",
        "",
        f"Offline evaluation uses the same {expected_count} GSM8K test rows, system prompt, greedy decoding, and",
        "512-token response limit for every checkpoint.",
        "",
        "| Step | Mean reward | Strict format | Strict accuracy | Flexible accuracy | Full reward |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['step']} | {summary['mean_rewards']['reward/total']:.4f} | "
            f"{summary['strict_format_rate']:.2%} | {summary['strict_answer_accuracy']:.2%} | "
            f"{summary['flexible_answer_accuracy']:.2%} | {summary['full_reward_rate']:.2%} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    training = load_training_buckets(args.training_summary)
    summaries = []
    reference_indices = None
    for step in args.steps:
        rows = load_rows(args.results_dir, step, args.expected_count)
        indices = [int(row["source_index"]) for row in rows]
        if reference_indices is None:
            reference_indices = indices
        elif indices != reference_indices:
            raise ValueError(f"step {step} did not use the same fixed test rows")
        summary = summarize(rows)
        summary["step"] = step
        if step in training:
            summary["training_rollout_bucket"] = training[step]
        summaries.append(summary)
    with (args.results_dir / "checkpoint_comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    write_markdown(summaries, args.results_dir / "checkpoint_comparison.md", args.expected_count)
    ranked = sorted(summaries, key=lambda row: float(row["mean_rewards"]["reward/total"]), reverse=True)
    print(json.dumps(ranked, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
