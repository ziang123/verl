#!/usr/bin/env python3
"""Build reproducible tables and figures for the SFT/GRPO research report."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

METRIC_COLUMNS = (
    "critic/score/mean",
    "actor/entropy",
    "actor/kl_loss",
    "actor/pg_loss",
    "actor/loss",
    "actor/grad_norm",
    "actor/ppo_kl",
    "actor/pg_clipfrac",
    "response_length/mean",
    "response_length/clip_ratio",
    "perf/throughput",
    "perf/time_per_step",
    "training/rollout_actor_probs_pearson_corr",
    "training/rollout_probs_diff_mean",
    "rollout_corr/k3_kl",
    "rollout_corr/ppl_ratio",
    "actor/perf/cpu_memory_used_gb",
    "actor/perf/max_memory_allocated_gb",
    "actor/perf/max_memory_reserved_gb",
    "perf/mfu/actor",
    "perf/mfu/actor_infer",
    "timing_s/gen",
    "timing_s/old_log_prob",
    "timing_s/ref",
    "timing_s/update_actor",
    "timing_s/update_weights",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--rollout-summary", type=Path, required=True)
    parser.add_argument("--base-summary", type=Path, required=True)
    parser.add_argument("--sft-summary", type=Path, required=True)
    parser.add_argument("--grpo-comparison", type=Path, required=True)
    parser.add_argument("--base-results", type=Path, required=True)
    parser.add_argument("--sft-results", type=Path, required=True)
    parser.add_argument("--grpo-results", type=Path, required=True)
    parser.add_argument("--grpo-step", type=int, default=800)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=100)
    parser.add_argument("--rolling-window", type=int, default=25)
    return parser.parse_args()


def load_metrics(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"step", *METRIC_COLUMNS}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        for raw in reader:
            try:
                row = {"step": float(raw["step"])}
                row.update({column: float(raw[column]) for column in METRIC_COLUMNS})
            except (TypeError, ValueError):
                continue
            rows.append(row)
    rows.sort(key=lambda row: row["step"])
    if not rows:
        raise ValueError(f"No complete metric rows found in {path}")
    return rows


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def bucket_metrics(rows: list[dict[str, float]], interval: int) -> list[dict[str, float]]:
    buckets: dict[int, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        bucket_start = ((int(row["step"]) - 1) // interval) * interval + 1
        buckets[bucket_start].append(row)
    summaries = []
    for start, bucket in sorted(buckets.items()):
        summary = {
            "step_start": float(int(bucket[0]["step"])),
            "step_end": float(int(bucket[-1]["step"])),
            "num_steps": float(len(bucket)),
        }
        for column in METRIC_COLUMNS:
            summary[column] = float(np.mean([row[column] for row in bucket]))
        summaries.append(summary)
    return summaries


def write_bucket_csv(rows: list[dict[str, float]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rolling(values: list[float], window: int) -> np.ndarray:
    result = np.empty(len(values), dtype=float)
    for index in range(len(values)):
        result[index] = np.mean(values[max(0, index - window + 1) : index + 1])
    return result


def plot_series(
    axis: Any,
    steps: list[float],
    rows: list[dict[str, float]],
    columns: list[tuple[str, str]],
    window: int,
    title: str,
) -> None:
    for column, label in columns:
        values = [row[column] for row in rows]
        axis.plot(steps, values, alpha=0.12, linewidth=0.7)
        axis.plot(steps, rolling(values, window), linewidth=1.8, label=label)
    axis.set_title(title)
    axis.set_xlabel("Training step")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)


def save_optimization_plot(rows: list[dict[str, float]], output: Path, window: int) -> None:
    steps = [row["step"] for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    plot_series(axes[0, 0], steps, rows, [("critic/score/mean", "reward")], window, "Rollout reward")
    plot_series(
        axes[0, 1],
        steps,
        rows,
        [("actor/entropy", "entropy"), ("actor/kl_loss", "reference KL loss")],
        window,
        "Exploration and reference divergence",
    )
    plot_series(
        axes[1, 0],
        steps,
        rows,
        [("actor/pg_loss", "policy gradient loss"), ("actor/loss", "total actor loss")],
        window,
        "Actor objective",
    )
    plot_series(axes[1, 1], steps, rows, [("actor/grad_norm", "gradient norm")], window, "Gradient norm")
    figure.savefig(output, dpi=170)
    plt.close(figure)


def save_efficiency_plot(rows: list[dict[str, float]], output: Path, window: int) -> None:
    steps = [row["step"] for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    plot_series(axes[0, 0], steps, rows, [("response_length/mean", "mean tokens")], window, "Response length")
    plot_series(
        axes[0, 1], steps, rows, [("response_length/clip_ratio", "clip ratio")], window, "Response clipping"
    )
    plot_series(axes[1, 0], steps, rows, [("perf/throughput", "tokens/s")], window, "Throughput")
    plot_series(axes[1, 1], steps, rows, [("perf/time_per_step", "seconds")], window, "Step time")
    figure.savefig(output, dpi=170)
    plt.close(figure)


def save_consistency_plot(rows: list[dict[str, float]], output: Path, window: int) -> None:
    steps = [row["step"] for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    plot_series(
        axes[0, 0],
        steps,
        rows,
        [("training/rollout_actor_probs_pearson_corr", "probability correlation")],
        window,
        "Rollout/training probability correlation",
    )
    plot_series(
        axes[0, 1],
        steps,
        rows,
        [("training/rollout_probs_diff_mean", "mean probability difference")],
        window,
        "Rollout/training probability difference",
    )
    plot_series(axes[1, 0], steps, rows, [("rollout_corr/k3_kl", "K3 KL")], window, "Rollout/training KL")
    plot_series(axes[1, 1], steps, rows, [("rollout_corr/ppl_ratio", "PPL ratio")], window, "PPL ratio")
    figure.savefig(output, dpi=170)
    plt.close(figure)


def save_resource_plot(rows: list[dict[str, float]], output: Path, window: int) -> None:
    steps = [row["step"] for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    plot_series(
        axes[0, 0],
        steps,
        rows,
        [("actor/perf/cpu_memory_used_gb", "CPU memory")],
        window,
        "Driver CPU memory (GB)",
    )
    plot_series(
        axes[0, 1],
        steps,
        rows,
        [
            ("actor/perf/max_memory_allocated_gb", "allocated"),
            ("actor/perf/max_memory_reserved_gb", "reserved"),
        ],
        window,
        "Actor GPU memory (GB)",
    )
    plot_series(
        axes[1, 0],
        steps,
        rows,
        [("perf/mfu/actor", "actor"), ("perf/mfu/actor_infer", "inference")],
        window,
        "Model FLOPs utilization",
    )
    plot_series(
        axes[1, 1],
        steps,
        rows,
        [
            ("timing_s/gen", "generation"),
            ("timing_s/old_log_prob", "old log prob"),
            ("timing_s/ref", "reference"),
            ("timing_s/update_actor", "actor update"),
            ("timing_s/update_weights", "weight sync"),
        ],
        window,
        "Step timing breakdown (seconds)",
    )
    figure.savefig(output, dpi=170)
    plt.close(figure)


def save_rollout_plot(rows: list[dict[str, str]], output: Path) -> None:
    x = [(int(row["step_start"]) + int(row["step_end"])) / 2 for row in rows]
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    axes[0].plot(x, [float(row["average_reward"]) for row in rows], marker="o", label="mean reward")
    axes[0].set_ylabel("Reward")
    axes[0].grid(alpha=0.2)
    accuracy_axis = axes[0].twinx()
    for column, label in (
        ("numeric_answer_accuracy", "answer accuracy"),
        ("extractable_numeric_ratio", "numeric extractability"),
        ("all_4_correct_ratio", "all 4 correct"),
        ("mixed_correctness_ratio", "mixed group"),
    ):
        accuracy_axis.plot(x, [float(row[column]) for row in rows], marker=".", label=label)
    accuracy_axis.set_ylabel("Ratio")
    lines = axes[0].lines + accuracy_axis.lines
    axes[0].legend(lines, [line.get_label() for line in lines], frameon=False, ncol=2)
    for column, label in (
        ("strict_format_ratio", "strict full format"),
        ("complete_think_ratio", "complete think tags"),
        ("complete_answer_ratio", "complete answer tags"),
    ):
        axes[1].plot(x, [float(row[column]) for row in rows], marker="o", label=label)
    axes[1].set_xlabel("Training step bucket midpoint")
    axes[1].set_ylabel("Ratio")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def load_model_summaries(args: argparse.Namespace) -> list[dict[str, Any]]:
    base = json.loads(args.base_summary.read_text(encoding="utf-8"))
    sft = json.loads(args.sft_summary.read_text(encoding="utf-8"))
    grpo_candidates = json.loads(args.grpo_comparison.read_text(encoding="utf-8"))
    grpo = next((row for row in grpo_candidates if int(row["step"]) == args.grpo_step), None)
    if grpo is None:
        raise ValueError(f"No step {args.grpo_step} result in {args.grpo_comparison}")
    return [base, sft, grpo]


def save_model_plot(summaries: list[dict[str, Any]], output: Path) -> None:
    labels = ["Base", "SFT", f"GRPO-{summaries[2]['step']}"]
    metrics = (
        ("Mean reward / 2", [float(row["mean_rewards"]["reward/total"]) / 2 for row in summaries]),
        ("Strict format", [float(row["strict_format_rate"]) for row in summaries]),
        ("Strict accuracy", [float(row["strict_answer_accuracy"]) for row in summaries]),
        ("Flexible accuracy", [float(row["flexible_answer_accuracy"]) for row in summaries]),
        ("Full reward", [float(row["full_reward_rate"]) for row in summaries]),
    )
    x = np.arange(len(labels))
    width = 0.15
    figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for index, (name, values) in enumerate(metrics):
        axis.bar(x + (index - 2) * width, values, width, label=name)
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1.04)
    axis.set_ylabel("Rate (mean reward normalized by maximum 2.0)")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=3)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def write_snapshot(rows: list[dict[str, float]], models: list[dict[str, Any]], path: Path) -> None:
    snapshot = {
        "metric_step_start": int(rows[0]["step"]),
        "metric_step_end": int(rows[-1]["step"]),
        "metric_row_count": len(rows),
        "ppo_kl_nonzero_steps": sum(row["actor/ppo_kl"] != 0 for row in rows),
        "pg_clipfrac_nonzero_steps": sum(row["actor/pg_clipfrac"] != 0 for row in rows),
        "models": models,
    }
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def format_diagnostics(paths: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    diagnostics = []
    for model, path in paths:
        rows = load_jsonl(path)
        outputs = [str(row["generated"]) for row in rows]
        count = len(rows)
        if count == 0:
            raise ValueError(f"No results found in {path}")
        diagnostics.append(
            {
                "model": model,
                "count": count,
                "starts_with_think_rate": sum(output.startswith("<think>\n") for output in outputs) / count,
                "complete_think_rate": sum(
                    output.count("<think>\n") == 1 and output.count("\n</think>\n") == 1 for output in outputs
                )
                / count,
                "complete_answer_rate": sum(
                    output.count("<answer>\n") == 1 and output.count("\n</answer>") == 1 for output in outputs
                )
                / count,
                "ends_with_answer_rate": sum(output.endswith("</answer>") for output in outputs) / count,
                "strict_marker_rate": sum(row.get("strict_answer") is not None for row in rows) / count,
                "strict_format_rate": sum(bool(row["strict_format"]) for row in rows) / count,
                "mean_output_characters": float(np.mean([len(output) for output in outputs])),
            }
        )
    return diagnostics


def write_diagnostics(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_comparison(sft_path: Path, grpo_path: Path) -> dict[str, Any]:
    sft = {int(row["source_index"]): row for row in load_jsonl(sft_path)}
    grpo = {int(row["source_index"]): row for row in load_jsonl(grpo_path)}
    if sft.keys() != grpo.keys():
        raise ValueError("SFT and GRPO results do not contain the same source indices")
    indices = sorted(sft)
    for index in indices:
        if (sft[index]["question"], sft[index]["ground_truth"]) != (
            grpo[index]["question"],
            grpo[index]["ground_truth"],
        ):
            raise ValueError(f"SFT and GRPO source data differ at index {index}")

    def difference(metric: str) -> dict[str, float]:
        values = np.array([float(grpo[index][metric]) - float(sft[index][metric]) for index in indices])
        mean = float(np.mean(values))
        margin = float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values)))
        return {"mean": mean, "ci95_low": mean - margin, "ci95_high": mean + margin}

    sft_correct = [bool(sft[index]["flexible_answer_correct"]) for index in indices]
    grpo_correct = [bool(grpo[index]["flexible_answer_correct"]) for index in indices]
    return {
        "count": len(indices),
        "both_flexible_correct": sum(left and right for left, right in zip(sft_correct, grpo_correct, strict=True)),
        "sft_only_flexible_correct": sum(
            left and not right for left, right in zip(sft_correct, grpo_correct, strict=True)
        ),
        "grpo_only_flexible_correct": sum(
            right and not left for left, right in zip(sft_correct, grpo_correct, strict=True)
        ),
        "neither_flexible_correct": sum(
            not left and not right for left, right in zip(sft_correct, grpo_correct, strict=True)
        ),
        "grpo_minus_sft": {
            "mean_reward": difference("reward/total"),
            "strict_format_rate": difference("strict_format"),
            "strict_answer_accuracy": difference("strict_answer_correct"),
            "flexible_answer_accuracy": difference("flexible_answer_correct"),
        },
    }


def main() -> None:
    args = parse_args()
    if args.interval < 1 or args.rolling_window < 1:
        raise ValueError("--interval and --rolling-window must be positive")
    data_dir = args.output_dir / "data"
    figure_dir = args.output_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(args.metrics_csv)
    rollout_summary = load_csv(args.rollout_summary)
    model_summaries = load_model_summaries(args)
    write_bucket_csv(bucket_metrics(metrics, args.interval), data_dir / "grpo_metrics_by_100_steps.csv")
    write_snapshot(metrics, model_summaries, data_dir / "research_snapshot.json")
    write_diagnostics(
        format_diagnostics(
            [
                ("Base", args.base_results),
                ("SFT", args.sft_results),
                (f"GRPO-{args.grpo_step}", args.grpo_results),
            ]
        ),
        data_dir / "model_format_diagnostics.csv",
    )
    (data_dir / "sft_grpo_paired_comparison.json").write_text(
        json.dumps(paired_comparison(args.sft_results, args.grpo_results), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    save_optimization_plot(metrics, figure_dir / "grpo_optimization.png", args.rolling_window)
    save_efficiency_plot(metrics, figure_dir / "grpo_efficiency.png", args.rolling_window)
    save_consistency_plot(metrics, figure_dir / "grpo_policy_consistency.png", args.rolling_window)
    save_resource_plot(metrics, figure_dir / "grpo_resources.png", args.rolling_window)
    save_rollout_plot(rollout_summary, figure_dir / "grpo_rollout_trends.png")
    save_model_plot(model_summaries, figure_dir / "model_comparison.png")
    print(f"Wrote report assets through step {int(metrics[-1]['step'])} to {args.output_dir}")


if __name__ == "__main__":
    main()
