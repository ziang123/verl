#!/usr/bin/env python3
"""Summarize GRPO reward and format metrics over fixed step intervals."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from verl.utils.reward_score import gsm8k  # noqa: E402

SAMPLES_PER_PROMPT = 4


@dataclass
class RolloutRecord:
    step: int
    prompt: str
    ground_truth: str
    correct: bool
    extractable_numeric: bool
    strict_format: bool
    complete_think: bool
    complete_answer: bool


@dataclass
class Summary:
    step_start: int
    step_end: int
    num_steps: int
    num_rollouts: int
    num_prompt_groups: int
    incomplete_prompt_groups: int
    average_reward: float
    numeric_answer_accuracy: float
    extractable_numeric_ratio: float
    strict_format_ratio: float
    complete_think_ratio: float
    complete_answer_ratio: float
    all_4_correct_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate GRPO metrics from a metrics CSV and rollout JSONL files.")
    parser.add_argument("metrics_csv", type=Path, help="CSV produced by scripts/plot_training_metrics.py.")
    parser.add_argument("rollout_dir", type=Path, help="Directory containing <step>.jsonl rollout files.")
    parser.add_argument("--interval", type=int, default=100, help="Number of training steps per bucket.")
    parser.add_argument("--output", type=Path, default=None, help="Optionally write the summaries as CSV.")
    parser.add_argument(
        "--complete-only",
        action="store_true",
        help="Omit the final bucket when it contains fewer than --interval steps.",
    )
    return parser.parse_args()


def load_metric_rewards(path: Path) -> dict[int, float]:
    rewards: dict[int, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"step", "critic/score/mean"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")

        for row_number, row in enumerate(reader, start=2):
            try:
                step = int(float(row["step"]))
                reward = float(row["critic/score/mean"])
            except (TypeError, ValueError):
                print(f"Warning: skipping invalid metrics row {row_number}", file=sys.stderr)
                continue
            rewards[step] = reward
    return rewards


def has_complete_think(output: str) -> bool:
    return output.count("<think>\n") == 1 and output.count("\n</think>\n") == 1


def has_complete_answer(output: str) -> bool:
    return output.count("<answer>\n") == 1 and output.count("\n</answer>") == 1


def load_rollouts(path: Path) -> list[RolloutRecord]:
    records: list[RolloutRecord] = []
    rollout_files = sorted(
        (candidate for candidate in path.glob("*.jsonl") if candidate.stem.isdigit()),
        key=lambda candidate: int(candidate.stem),
    )
    for rollout_file in rollout_files:
        file_step = int(rollout_file.stem)
        with rollout_file.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                    output = str(row["output"])
                    ground_truth = str(row["gts"])
                    step = int(row.get("step", file_step))
                    prompt = str(row["input"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    print(f"Warning: skipping invalid rollout row {rollout_file}:{line_number}", file=sys.stderr)
                    continue

                if step != file_step:
                    raise ValueError(f"Step mismatch in {rollout_file}:{line_number}: found {step}")

                records.append(
                    RolloutRecord(
                        step=step,
                        prompt=prompt,
                        ground_truth=ground_truth,
                        correct=gsm8k.correctness_reward(output, ground_truth) > 0,
                        extractable_numeric=gsm8k.digit_reward(output) > 0,
                        strict_format=gsm8k.hard_format_reward(output) > 0,
                        complete_think=has_complete_think(output),
                        complete_answer=has_complete_answer(output),
                    )
                )
    return records


def mean(values: list[float | bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(
    metric_rewards: dict[int, float],
    rollout_records: list[RolloutRecord],
    interval: int,
    complete_only: bool,
) -> list[Summary]:
    if interval <= 0:
        raise ValueError("--interval must be positive")

    records_by_step: dict[int, list[RolloutRecord]] = defaultdict(list)
    for record in rollout_records:
        records_by_step[record.step].append(record)

    common_steps = sorted(set(metric_rewards).intersection(records_by_step))
    if not common_steps:
        raise ValueError("No common steps were found in the metrics CSV and rollout directory")

    summaries: list[Summary] = []
    max_step = common_steps[-1]
    for step_start in range(1, max_step + 1, interval):
        nominal_end = step_start + interval - 1
        steps = [step for step in common_steps if step_start <= step <= nominal_end]
        if not steps:
            continue
        if complete_only and len(steps) < interval:
            continue

        bucket_records = [record for step in steps for record in records_by_step[step]]
        prompt_groups: dict[tuple[int, str, str], list[RolloutRecord]] = defaultdict(list)
        for record in bucket_records:
            prompt_groups[(record.step, record.prompt, record.ground_truth)].append(record)

        complete_groups = [group for group in prompt_groups.values() if len(group) == SAMPLES_PER_PROMPT]
        incomplete_groups = len(prompt_groups) - len(complete_groups)

        summaries.append(
            Summary(
                step_start=steps[0],
                step_end=steps[-1],
                num_steps=len(steps),
                num_rollouts=len(bucket_records),
                num_prompt_groups=len(complete_groups),
                incomplete_prompt_groups=incomplete_groups,
                average_reward=mean([metric_rewards[step] for step in steps]),
                numeric_answer_accuracy=mean([record.correct for record in bucket_records]),
                extractable_numeric_ratio=mean([record.extractable_numeric for record in bucket_records]),
                strict_format_ratio=mean([record.strict_format for record in bucket_records]),
                complete_think_ratio=mean([record.complete_think for record in bucket_records]),
                complete_answer_ratio=mean([record.complete_answer for record in bucket_records]),
                all_4_correct_ratio=mean([all(record.correct for record in group) for group in complete_groups]),
            )
        )
    return summaries


def print_table(summaries: list[Summary]) -> None:
    headers = [
        "steps",
        "rollouts",
        "prompts",
        "avg_reward",
        "answer_acc",
        "numeric",
        "strict_fmt",
        "think",
        "answer",
        "all_4_correct",
    ]
    rows = []
    for summary in summaries:
        rows.append(
            [
                f"{summary.step_start}-{summary.step_end}",
                str(summary.num_rollouts),
                str(summary.num_prompt_groups),
                f"{summary.average_reward:.4f}",
                f"{summary.numeric_answer_accuracy:.2%}",
                f"{summary.extractable_numeric_ratio:.2%}",
                f"{summary.strict_format_ratio:.2%}",
                f"{summary.complete_think_ratio:.2%}",
                f"{summary.complete_answer_ratio:.2%}",
                f"{summary.all_4_correct_ratio:.2%}",
            ]
        )

    widths = [max(len(headers[index]), *(len(row[index]) for row in rows)) for index in range(len(headers))]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))

    incomplete = sum(summary.incomplete_prompt_groups for summary in summaries)
    if incomplete:
        print(f"Warning: excluded {incomplete} prompt groups that did not contain exactly 4 rollouts.", file=sys.stderr)


def write_csv(summaries: list[Summary], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(summary) for summary in summaries]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    metric_rewards = load_metric_rewards(args.metrics_csv)
    rollout_records = load_rollouts(args.rollout_dir)
    summaries = summarize(metric_rewards, rollout_records, args.interval, args.complete_only)
    print_table(summaries)
    if args.output is not None:
        write_csv(summaries, args.output)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
