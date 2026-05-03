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
    "behavior_diversity",
    "active_skill_count",
    "x_displacement_span",
    "mean_abs_x_displacement",
    "intrinsic_reward_avg",
    "skill_objective_loss",
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


def _key(row: Dict[str, object]) -> tuple[str, str, str]:
    return (str(row.get("method", "")), str(row.get("condition", "")), str(row.get("seed", "")))


def _aggregate_lookup(rows: Iterable[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    out = {}
    for row in rows:
        run_dir = row.get("run_dir", "")
        if run_dir:
            out[run_dir] = row
    return out


def _summary_lookup(rows: Iterable[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    out = {}
    for row in rows:
        run_dir = row.get("run_dir", "")
        if run_dir:
            out[run_dir] = row
    return out


def _movement_lookup(rows: Iterable[Dict[str, str]]) -> Dict[tuple[str, str, str], Dict[str, str]]:
    return {_key(row): row for row in rows}


def build_run_rows(
    degradation_rows: Iterable[Dict[str, str]],
    aggregate_rows: Iterable[Dict[str, str]],
    skill_rows: Iterable[Dict[str, str]],
    movement_rows: Iterable[Dict[str, str]],
) -> List[Dict[str, object]]:
    aggregate_by_run = _aggregate_lookup(aggregate_rows)
    skill_by_run = _summary_lookup(skill_rows)
    movement_by_key = _movement_lookup(movement_rows)

    out: List[Dict[str, object]] = []
    for row in degradation_rows:
        if row.get("run_complete") != "True":
            continue
        method = row.get("method", "")
        run_dir = row.get("run_dir", "")
        agg = aggregate_by_run.get(run_dir, {})
        skill = skill_by_run.get(run_dir, {})
        movement = movement_by_key.get(_key(row), {})
        prefix = f"{method}_progress__"

        out.append(
            {
                "method": method,
                "condition": row.get("condition", ""),
                "seed": row.get("seed", ""),
                "run_dir": run_dir,
                "best_skill_return": _float(row.get("metric_value")),
                "behavior_diversity": _float(skill.get("behavior_pairwise_distance_standardized")),
                "active_skill_count": _float(movement.get("active_skill_count")),
                "x_displacement_span": _float(movement.get("x_displacement_span")),
                "mean_abs_x_displacement": _float(movement.get("mean_abs_x_displacement")),
                "intrinsic_reward_avg": _float(agg.get(prefix + "intrinsic-reward-avg")),
                "skill_objective_loss": _float(agg.get(prefix + "skill-objective-loss")),
            }
        )
    return out


def summarize_across_seeds(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: dict[tuple[str, str], list[Dict[str, object]]] = defaultdict(list)
    for row in rows:
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

    clean_by_method = {
        str(row["method"]): row
        for row in summaries
        if str(row["condition"]) == "clean"
    }
    for row in summaries:
        clean = clean_by_method.get(str(row["method"]), {})
        for metric in SUMMARY_METRICS:
            value = _float(row.get(f"{metric}_mean"))
            clean_value = _float(clean.get(f"{metric}_mean"))
            delta = "" if value is None or clean_value is None else value - clean_value
            row[f"{metric}_delta_from_clean"] = delta

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


def _fmt(value: object, digits: int = 2) -> str:
    numeric = _float(value)
    if numeric is None:
        return "--"
    return f"{numeric:.{digits}f}"


def _write_latex(path: str, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Objective-aligned benefit metrics. Diversity is standardized pairwise behavioural distance between skills; active skills count non-stationary forward/backward skills; $x$ span measures directional coverage; intrinsic signal is the final logged intrinsic reward average for the method's own objective.}",
        r"  \label{tab:intended_benefit_metrics}",
        r"  \begin{tabular}{llrrrr}",
        r"    \toprule",
        r"    Objective & Condition & Diversity & Active & $x$ span & Intrinsic signal \\",
        r"    \midrule",
    ]
    for row in rows:
        method = METHOD_LABELS.get(str(row["method"]), str(row["method"]))
        condition = str(row["condition"])
        lines.append(
            "    "
            + f"{method} & \\texttt{{{condition}}} & "
            + f"{_fmt(row.get('behavior_diversity_mean'))} & "
            + f"{_fmt(row.get('active_skill_count_mean'), digits=1)} & "
            + f"{_fmt(row.get('x_displacement_span_mean'), digits=1)} & "
            + f"{_fmt(row.get('intrinsic_reward_avg_mean'))} \\\\"
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
    parser.add_argument("--degradation-csv", required=True)
    parser.add_argument("--aggregate-csv", required=True)
    parser.add_argument("--skill-summary-csv", required=True)
    parser.add_argument("--movement-run-csv", required=True)
    parser.add_argument("--output-run-csv", required=True)
    parser.add_argument("--output-summary-csv", required=True)
    parser.add_argument("--output-tex", required=True)
    args = parser.parse_args()

    run_rows = build_run_rows(
        _read_rows(args.degradation_csv),
        _read_rows(args.aggregate_csv),
        _read_rows(args.skill_summary_csv),
        _read_rows(args.movement_run_csv),
    )
    summary_rows = summarize_across_seeds(run_rows)
    _write_csv(args.output_run_csv, run_rows)
    _write_csv(args.output_summary_csv, summary_rows)
    _write_latex(args.output_tex, summary_rows)
    print(f"Wrote {len(run_rows)} run-level intended-benefit rows to {args.output_run_csv}")
    print(f"Wrote {len(summary_rows)} seed-averaged intended-benefit rows to {args.output_summary_csv}")
    print(f"Wrote LaTeX intended-benefit table to {args.output_tex}")


if __name__ == "__main__":
    main()
