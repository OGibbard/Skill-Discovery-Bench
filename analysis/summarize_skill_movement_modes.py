from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from statistics import mean, stdev
from typing import Dict, Iterable, List


METHOD_ORDER = ["diayn", "dads", "lsd_delta"]
CONDITION_ORDER = ["clean", "noise-med", "delay-3", "noise-med-delay-3", "encoded-clean"]
METHOD_LABELS = {
    "diayn": r"\texttt{diayn}",
    "dads": r"\texttt{dads}",
    "lsd_delta": r"\texttt{lsd\_delta}",
}
SUMMARY_METRICS = [
    "forward_skill_count",
    "backward_skill_count",
    "stationary_skill_count",
    "active_skill_count",
    "mean_abs_x_displacement",
    "x_displacement_span",
    "mean_abs_x_velocity",
    "mean_action_energy",
    "movement_per_action_energy",
]


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


def _read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fmt(value: object, digits: int = 2) -> str:
    numeric = _float(value)
    if numeric is None:
        return ""
    return f"{numeric:.{digits}f}"


def _fmt_latex(value: object, digits: int = 1) -> str:
    numeric = _float(value)
    if numeric is None:
        return "--"
    return f"{numeric:.{digits}f}"


def _summarize_run(rows: List[Dict[str, str]], stationary_threshold: float) -> Dict[str, object]:
    first = rows[0]
    by_skill: dict[str, list[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_skill[str(row.get("skill_id", ""))].append(row)

    displacements = []
    velocities = []
    energies = []
    for skill_rows in by_skill.values():
        skill_displacements = [
            value for row in skill_rows
            if (value := _float(row.get("x_displacement"))) is not None
        ]
        skill_velocities = [
            value for row in skill_rows
            if (value := _float(row.get("mean_x_velocity"))) is not None
        ]
        skill_energies = [
            value for row in skill_rows
            if (value := _float(row.get("action_energy"))) is not None
        ]
        if skill_displacements:
            displacements.append(mean(skill_displacements))
        if skill_velocities:
            velocities.append(mean(skill_velocities))
        if skill_energies:
            energies.append(mean(skill_energies))

    forward = sum(1 for value in displacements if value > stationary_threshold)
    backward = sum(1 for value in displacements if value < -stationary_threshold)
    stationary = sum(1 for value in displacements if abs(value) <= stationary_threshold)
    active = forward + backward
    mean_abs_displacement = mean(abs(value) for value in displacements) if displacements else 0.0
    displacement_span = (max(displacements) - min(displacements)) if displacements else 0.0
    mean_energy = mean(energies) if energies else 0.0

    return {
        "method": first["method"],
        "condition": first["condition"],
        "seed": first["seed"],
        "n_skills": len(displacements),
        "forward_skill_count": forward,
        "backward_skill_count": backward,
        "stationary_skill_count": stationary,
        "active_skill_count": active,
        "mean_abs_x_displacement": mean_abs_displacement,
        "x_displacement_span": displacement_span,
        "mean_abs_x_velocity": mean(abs(value) for value in velocities) if velocities else 0.0,
        "mean_action_energy": mean_energy,
        "movement_per_action_energy": mean_abs_displacement / (mean_energy + 1e-8),
    }


def summarize_runs(rows: Iterable[Dict[str, str]], stationary_threshold: float) -> List[Dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("method", ""), row.get("condition", ""), row.get("seed", ""))].append(row)

    out = []
    for _, group in grouped.items():
        out.append(_summarize_run(group, stationary_threshold))
    return sorted(out, key=lambda row: (str(row["method"]), str(row["condition"]), int(str(row["seed"]))))


def summarize_across_seeds(run_rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: dict[tuple[str, str], list[Dict[str, object]]] = defaultdict(list)
    for row in run_rows:
        grouped[(str(row["method"]), str(row["condition"]))].append(row)

    summaries: List[Dict[str, object]] = []
    for (method, condition), group in grouped.items():
        seeds = sorted(str(row["seed"]) for row in group)
        out: Dict[str, object] = {
            "method": method,
            "condition": condition,
            "n_seeds": len(seeds),
            "seeds": "|".join(seeds),
        }
        for metric in SUMMARY_METRICS:
            values = []
            for row in group:
                value = _float(row.get(metric))
                if value is not None:
                    values.append(value)
            out[f"{metric}_mean"] = mean(values) if values else ""
            out[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        summaries.append(out)

    def sort_key(row: Dict[str, object]) -> tuple[int, int]:
        method = str(row["method"])
        condition = str(row["condition"])
        method_i = METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)
        condition_i = CONDITION_ORDER.index(condition) if condition in CONDITION_ORDER else len(CONDITION_ORDER)
        return (method_i, condition_i)

    return sorted(summaries, key=sort_key)


def _write_csv(path: str, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    fieldnames = sorted({key for row in rows for key in row})
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_latex(path: str, rows: Iterable[Dict[str, object]], stationary_threshold: float) -> None:
    rows = list(rows)
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Post-hoc movement-mode coverage from all-skill rollouts. Each skill is first averaged over its evaluation rollouts, then classified as forward, backward, or stationary using a displacement threshold of "
        + f"$\\pm {stationary_threshold:g}$"
        + r" over a 1000-step rollout. Movement per action energy is mean absolute displacement divided by mean squared action magnitude.}",
        r"  \label{tab:skill_movement_modes}",
        r"  \begin{tabular}{llrrrrr}",
        r"    \toprule",
        r"    Objective & Condition & Forward & Backward & Stationary & $|x|$ disp. & Move/energy \\",
        r"    \midrule",
    ]
    for row in rows:
        method = METHOD_LABELS.get(str(row["method"]), str(row["method"]))
        condition = str(row["condition"])
        lines.append(
            "    "
            + f"{method} & \\texttt{{{condition}}} & "
            + f"{_fmt_latex(row['forward_skill_count_mean'])} & "
            + f"{_fmt_latex(row['backward_skill_count_mean'])} & "
            + f"{_fmt_latex(row['stationary_skill_count_mean'])} & "
            + f"{_fmt_latex(row['mean_abs_x_displacement_mean'])} & "
            + f"{_fmt_latex(row['movement_per_action_energy_mean'], digits=2)} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Per-skill rollout CSV from evaluate_skill_repertoire.py.")
    parser.add_argument("--output-run-csv", required=True)
    parser.add_argument("--output-summary-csv", required=True)
    parser.add_argument("--output-tex", required=True)
    parser.add_argument("--stationary-threshold", type=float, default=1.0)
    args = parser.parse_args()

    run_rows = summarize_runs(_read_rows(args.input), args.stationary_threshold)
    summary_rows = summarize_across_seeds(run_rows)
    _write_csv(args.output_run_csv, run_rows)
    _write_csv(args.output_summary_csv, summary_rows)
    _write_latex(args.output_tex, summary_rows, args.stationary_threshold)
    print(f"Wrote {len(run_rows)} run-level movement summaries to {args.output_run_csv}")
    print(f"Wrote {len(summary_rows)} seed-averaged movement summaries to {args.output_summary_csv}")
    print(f"Wrote LaTeX movement-mode table to {args.output_tex}")


if __name__ == "__main__":
    main()
