#!/usr/bin/env python3
"""Plot selected metrics from a verl console training log."""

import argparse
import math
import re
import time
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
STEP_RE = re.compile(r"step:(\d+)")
NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
METRIC_RE = re.compile(rf" - ([A-Za-z0-9_./-]+):(?:np\.(?:float64|float32|int64|int32)\()?({NUMBER_RE})")

DEFAULT_METRICS = [
    "actor/loss",
    "actor/pg_loss",
    "actor/kl_loss",
    "actor/grad_norm",
    "critic/rewards/mean",
    "critic/score/mean",
    "response_length/clip_ratio",
]


def parse_log(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = ANSI_RE.sub("", raw_line)
            step_match = STEP_RE.search(line)
            if not step_match:
                continue

            row: dict[str, float] = {"step": int(step_match.group(1))}
            for key, value in METRIC_RE.findall(line):
                try:
                    parsed = float(value)
                except ValueError:
                    continue
                if math.isfinite(parsed):
                    row[key] = parsed
            if len(row) > 1:
                rows.append(row)
    return rows


def write_csv(rows: list[dict[str, float]], output_csv: Path) -> None:
    keys = ["step"] + sorted({key for row in rows for key in row if key != "step"})
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8") as handle:
        handle.write(",".join(keys) + "\n")
        for row in rows:
            handle.write(",".join("" if key not in row else str(row[key]) for key in keys) + "\n")


def plot(rows: list[dict[str, float]], metrics: list[str], output_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    available = [metric for metric in metrics if any(metric in row for row in rows)]
    if not available:
        raise RuntimeError("None of the requested metrics were found in the log.")

    fig, axes = plt.subplots(len(available), 1, figsize=(11, 2.6 * len(available)), sharex=True)
    if len(available) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available):
        xs = [row["step"] for row in rows if metric in row]
        ys = [row[metric] for row in rows if metric in row]
        ax.plot(xs, ys, linewidth=1.6)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("training step")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def render_once(
    log_file: Path,
    output_png: Path,
    output_csv: Path,
    metrics: list[str],
    allow_empty: bool = False,
) -> int | None:
    rows = parse_log(log_file)
    if not rows:
        if allow_empty:
            return None
        raise SystemExit(f"No metric rows found in {log_file}")

    write_csv(rows, output_csv)
    plot(rows, metrics, output_png)
    return int(rows[-1]["step"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot metrics from a verl console training log.")
    parser.add_argument("log_file", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    parser.add_argument("--watch", action="store_true", help="Keep refreshing outputs while the log grows.")
    parser.add_argument("--interval", type=float, default=30.0, help="Refresh interval in seconds for --watch.")
    args = parser.parse_args()

    output_png = args.output or args.log_file.with_suffix(".loss.png")
    output_csv = args.csv or args.log_file.with_suffix(".metrics.csv")

    if args.watch:
        last_step = None
        while True:
            try:
                step = render_once(args.log_file, output_png, output_csv, args.metrics, allow_empty=True)
                if step is not None and step != last_step:
                    print(f"Wrote {output_png} through step {step}", flush=True)
                    print(f"Wrote {output_csv}", flush=True)
                    last_step = step
            except Exception as exc:  # keep the watcher alive while training starts
                print(f"Plot update failed: {exc}", flush=True)
            time.sleep(max(args.interval, 1.0))

    step = render_once(args.log_file, output_png, output_csv, args.metrics)
    print(f"Wrote {output_png} through step {step}")
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
