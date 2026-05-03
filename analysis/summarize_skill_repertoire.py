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
    "best_skill_return",
    "mean_skill_return",
    "median_skill_return",
    "worst_skill_return",
    "std_skill_return",
    "skill_return_spread",
    "behavior_pairwise_distance_standardized",
]


def _float(value: str) -> float | None:
    text = (value or "").strip()
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


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def _fmt_latex(value: object) -> str:
    numeric = _float(str(value))
    if numeric is None:
        return "--"
    return f"{numeric:.1f}"


def summarize(rows: Iterable[Dict[str, str]]) -> List[Dict[str, object]]:
    grouped: dict[tuple[str, str], list[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("method", ""), row.get("condition", ""))].append(row)

    summaries: List[Dict[str, object]] = []
    for (method, condition), group in grouped.items():
        seeds = sorted({row.get("seed", "") for row in group})
        out: Dict[str, object] = {
            "method": method,
            "condition": condition,
            "n_seeds": len(seeds),
            "seeds": "|".join(seeds),
            "snapshot_epoch_min": min(int(float(row["snapshot_epoch"])) for row in group),
            "snapshot_epoch_max": max(int(float(row["snapshot_epoch"])) for row in group),
        }
        for metric in SUMMARY_METRICS:
            values = [v for row in group if (v := _float(row.get(metric, ""))) is not None]
            out[f"{metric}_mean"] = _fmt(mean(values)) if values else ""
            out[f"{metric}_std"] = _fmt(stdev(values)) if len(values) > 1 else "0"
        summaries.append(out)

    def sort_key(row: Dict[str, object]) -> tuple[int, int]:
        method = str(row["method"])
        condition = str(row["condition"])
        method_i = METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)
        condition_i = CONDITION_ORDER.index(condition) if condition in CONDITION_ORDER else len(CONDITION_ORDER)
        return (method_i, condition_i)

    return sorted(summaries, key=sort_key)


def write_csv(path: str, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    fieldnames = [
        "method",
        "condition",
        "n_seeds",
        "seeds",
        "snapshot_epoch_min",
        "snapshot_epoch_max",
    ]
    for metric in SUMMARY_METRICS:
        fieldnames += [f"{metric}_mean", f"{metric}_std"]

    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_latex(path: str, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Post-hoc skill-repertoire metrics from the latest saved checkpoint, \texttt{itr\_200}. Values are means over seeds. Best and mean returns evaluate skill usefulness beyond the single best training metric; spread and diversity summarize behavioural variation across skills.}",
        r"  \label{tab:skill_repertoire_summary}",
        r"  \begin{tabular}{llrrrr}",
        r"    \toprule",
        r"    Objective & Condition & Best return & Mean return & Return spread & Diversity \\",
        r"    \midrule",
    ]
    for row in rows:
        method = METHOD_LABELS.get(str(row["method"]), str(row["method"]))
        condition = str(row["condition"])
        lines.append(
            "    "
            + f"{method} & \\texttt{{{condition}}} & "
            + f"{_fmt_latex(row['best_skill_return_mean'])} & "
            + f"{_fmt_latex(row['mean_skill_return_mean'])} & "
            + f"{_fmt_latex(row['skill_return_spread_mean'])} & "
            + f"{_fmt_latex(row['behavior_pairwise_distance_standardized_mean'])} \\\\"
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
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-tex", required=True)
    args = parser.parse_args()

    rows = summarize(_read_rows(args.input))
    write_csv(args.output_csv, rows)
    write_latex(args.output_tex, rows)
    print(f"Wrote {len(rows)} skill-repertoire summary rows to {args.output_csv}")
    print(f"Wrote LaTeX skill-repertoire table to {args.output_tex}")


if __name__ == "__main__":
    main()
