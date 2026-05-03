from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = ["diayn", "dads", "lsd_delta"]
CONDITION_ORDER = ["clean", "noise-med", "delay-3", "noise-med-delay-3", "encoded-clean"]
METHOD_LABELS = {
    "diayn": "DIAYN",
    "dads": "DADS-style",
    "lsd_delta": "LSD-delta",
}
METRICS = {
    "return": ("skill_heatmap_return.png", "Mean environment return"),
    "x_displacement": ("skill_heatmap_x_displacement.png", "Mean x displacement"),
    "action_energy": ("skill_heatmap_action_energy.png", "Mean squared action"),
    "mean_x_velocity": ("skill_heatmap_mean_x_velocity.png", "Mean x velocity"),
}


def _float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _sort_key(label: tuple[str, str]) -> tuple[int, int]:
    method, condition = label
    method_i = METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)
    condition_i = CONDITION_ORDER.index(condition) if condition in CONDITION_ORDER else len(CONDITION_ORDER)
    return method_i, condition_i


def _aggregate_by_skill(
    rows: list[dict[str, str]],
    metric: str,
) -> tuple[list[str], list[int], np.ndarray]:
    grouped: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        skill = _float(row.get("skill_id"))
        value = _float(row.get(metric))
        if skill is None or value is None:
            continue
        grouped[
            (
                row.get("method", ""),
                row.get("condition", ""),
                row.get("seed", ""),
                int(skill),
            )
        ].append(value)

    per_seed: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for (method, condition, _seed, skill), values in grouped.items():
        per_seed[(method, condition, skill)].append(mean(values))

    labels = sorted({(method, condition) for method, condition, _skill in per_seed}, key=_sort_key)
    skills = sorted({skill for _method, _condition, skill in per_seed})
    matrix = np.full((len(labels), len(skills)), np.nan)

    for row_i, (method, condition) in enumerate(labels):
        for col_i, skill in enumerate(skills):
            values = per_seed.get((method, condition, skill), [])
            if values:
                matrix[row_i, col_i] = mean(values)

    y_labels = [
        f"{METHOD_LABELS.get(method, method)} / {condition}"
        for method, condition in labels
    ]
    return y_labels, skills, matrix


def _plot_heatmap(rows: list[dict[str, str]], metric: str, output_path: Path, title: str) -> None:
    y_labels, skills, matrix = _aggregate_by_skill(rows, metric)
    if matrix.size == 0:
        raise ValueError(f"No values found for metric {metric!r}.")

    fig_height = max(5.5, 0.38 * len(y_labels) + 1.5)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.set_ylabel(title, rotation=270, labelpad=16)

    ax.set_title(title)
    ax.set_xlabel("Skill id")
    ax.set_ylabel("Objective / condition")
    ax.set_xticks(np.arange(len(skills)), labels=[str(skill) for skill in skills])
    ax.set_yticks(np.arange(len(y_labels)), labels=y_labels)
    ax.tick_params(axis="y", labelsize=8)

    for row_i in range(matrix.shape[0]):
        for col_i in range(matrix.shape[1]):
            value = matrix[row_i, col_i]
            if not np.isnan(value):
                ax.text(col_i, row_i, f"{value:.1f}", ha="center", va="center", fontsize=6, color="white")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    rows = _read_rows(args.input)
    for metric, (filename, title) in METRICS.items():
        _plot_heatmap(rows, metric, args.output_dir / filename, title)


if __name__ == "__main__":
    main()
