from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import pandas as pd


METHOD_ORDER = ["diayn", "dads", "lsd_delta"]
CONDITION_ORDER = ["clean", "noise-med", "delay-3", "noise-med-delay-3", "encoded-clean"]
DEFAULT_METRIC = "best_single_option_policy/return-average"


def _find_manifests(root_dir: str) -> List[str]:
    manifests: List[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        if "robustness_manifest.json" in filenames:
            manifests.append(os.path.join(dirpath, "robustness_manifest.json"))
    return sorted(manifests)


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _derive_condition(manifest: Dict[str, object]) -> str:
    robustness = manifest.get("robustness", {})
    if not isinstance(robustness, dict):
        return "unknown"

    encoder_type = str(robustness.get("encoder_type", "identity")).strip().lower()
    noise_std = float(robustness.get("noise_std") or 0.0)
    delay_steps = int(robustness.get("delay_steps") or 0)
    encoded = encoder_type not in {"", "none", "identity"}

    if not encoded and noise_std == 0.0 and delay_steps == 0:
        return "clean"
    if encoded and noise_std == 0.0 and delay_steps == 0:
        return "encoded-clean"
    if encoded and noise_std > 0.0 and delay_steps == 0:
        if abs(noise_std - 0.075) < 1e-12:
            return "encoded-noise-med"
        return "encoded-noise"
    if encoded and noise_std == 0.0 and delay_steps == 3:
        return "encoded-delay-3"
    if encoded and delay_steps > 0 and noise_std == 0.0:
        return f"encoded-delay-{delay_steps}"
    if encoded and noise_std > 0.0 and delay_steps > 0:
        return f"encoded-noise-delay-{delay_steps}"
    if not encoded and noise_std > 0.0 and delay_steps == 0:
        if abs(noise_std - 0.075) < 1e-12:
            return "noise-med"
        return "noise"
    if not encoded and noise_std > 0.0 and delay_steps > 0:
        if abs(noise_std - 0.075) < 1e-12:
            return f"noise-med-delay-{delay_steps}"
        return f"noise-delay-{delay_steps}"
    if not encoded and noise_std == 0.0 and delay_steps > 0:
        return f"delay-{delay_steps}"
    return "unknown"


def _sort_key(value: str, preferred: Iterable[str]) -> tuple[int, str]:
    preferred = list(preferred)
    if value in preferred:
        return (preferred.index(value), value)
    return (len(preferred), value)


def _safe_name(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_.-]+", "_", value)
    return value.strip("_")


def load_progress(
    root_dir: str,
    env_filter: str | None = None,
    include_run_substrings: List[str] | None = None,
    exclude_run_substrings: List[str] | None = None,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    include_run_substrings = include_run_substrings or []
    exclude_run_substrings = exclude_run_substrings or []
    for manifest_path in _find_manifests(root_dir):
        manifest = _load_json(manifest_path)
        environment = str(manifest.get("environment", ""))
        if env_filter and environment != env_filter:
            continue

        run_dir = os.path.dirname(manifest_path)
        if include_run_substrings and not any(token in run_dir for token in include_run_substrings):
            continue
        if exclude_run_substrings and any(token in run_dir for token in exclude_run_substrings):
            continue

        progress_path = os.path.join(run_dir, "progress.csv")
        if not os.path.exists(progress_path):
            continue

        frame = pd.read_csv(progress_path)
        if frame.empty:
            continue

        frame["method"] = manifest.get("method")
        frame["environment"] = environment
        frame["seed"] = manifest.get("seed")
        frame["condition"] = _derive_condition(manifest)
        frame["run_dir"] = run_dir
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def plot_learning_curves(df: pd.DataFrame, metric: str, output_dir: str) -> List[str]:
    if metric not in df.columns:
        available = ", ".join(sorted(str(col) for col in df.columns if "return" in str(col)))
        raise ValueError(f"Metric {metric!r} not found. Return-like metrics: {available}")

    os.makedirs(output_dir, exist_ok=True)
    written: List[str] = []
    conditions = sorted(df["condition"].dropna().unique(), key=lambda c: _sort_key(c, CONDITION_ORDER))
    methods = sorted(df["method"].dropna().unique(), key=lambda m: _sort_key(m, METHOD_ORDER))

    for condition in conditions:
        subset = df[df["condition"] == condition]
        if subset.empty:
            continue

        plt.figure(figsize=(8, 5))
        for method in methods:
            method_df = subset[subset["method"] == method]
            if method_df.empty:
                continue

            grouped = method_df.groupby("epoch", as_index=False)[metric].mean()
            plt.plot(grouped["epoch"], grouped[metric], label=method)

        plt.title(f"HalfCheetah {condition}: {metric}")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        plt.legend()
        path = os.path.join(output_dir, f"learning_curve_{_safe_name(condition)}_{_safe_name(metric)}.png")
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        written.append(path)

    return written


def plot_final_metric(df: pd.DataFrame, metric: str, output_dir: str) -> str:
    final_rows = (
        df.sort_values("epoch")
        .groupby(["condition", "method", "seed"], as_index=False)
        .tail(1)
    )
    summary = (
        final_rows.groupby(["condition", "method"], as_index=False)[metric]
        .mean()
        .sort_values(
            ["condition", "method"],
            key=lambda col: col.map(lambda x: _sort_key(str(x), CONDITION_ORDER if col.name == "condition" else METHOD_ORDER)),
        )
    )

    labels = [f"{row.condition}\n{row.method}" for row in summary.itertuples()]
    plt.figure(figsize=(max(8, len(labels) * 0.75), 5))
    plt.bar(labels, summary[metric])
    plt.ylabel(metric)
    plt.title(f"Final HalfCheetah metric: {metric}")
    plt.xticks(rotation=35, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"final_metric_{_safe_name(metric)}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="code/unified_skill_discovery/logs/local/half-cheetah")
    parser.add_argument("--env", default="HalfCheetah-v4")
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--output-dir", default="outputs/figures")
    parser.add_argument(
        "--include-run-substring",
        action="append",
        default=[],
        help="Only include runs whose directory contains this substring. Can be repeated.",
    )
    parser.add_argument(
        "--exclude-run-substring",
        action="append",
        default=[],
        help="Exclude runs whose directory contains this substring. Can be repeated.",
    )
    args = parser.parse_args()

    df = load_progress(
        args.root,
        env_filter=args.env,
        include_run_substrings=args.include_run_substring,
        exclude_run_substrings=args.exclude_run_substring,
    )
    if df.empty:
        raise SystemExit(f"No progress rows found under {args.root}")

    curve_paths = plot_learning_curves(df, args.metric, args.output_dir)
    final_path = plot_final_metric(df, args.metric, args.output_dir)
    for path in curve_paths + [final_path]:
        print(path)


if __name__ == "__main__":
    main()
