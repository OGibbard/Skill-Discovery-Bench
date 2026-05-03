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


def _fmt_latex(value: str) -> str:
    numeric = _float(value)
    if numeric is None:
        return "--"
    return f"{numeric:.1f}"


def summarize(rows: Iterable[Dict[str, str]]) -> List[Dict[str, object]]:
    grouped: dict[tuple[str, str], list[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("run_complete") != "True":
            continue
        grouped[(row.get("method", ""), row.get("condition", ""))].append(row)

    summaries: List[Dict[str, object]] = []
    for (method, condition), group in grouped.items():
        metric_values = [v for row in group if (v := _float(row.get("metric_value", ""))) is not None]
        delta_values = [v for row in group if (v := _float(row.get("delta_from_clean", ""))) is not None]
        seeds = sorted({row.get("seed", "") for row in group})
        summaries.append(
            {
                "method": method,
                "condition": condition,
                "n_seeds": len(seeds),
                "seeds": "|".join(seeds),
                "metric_mean": _fmt(mean(metric_values)) if metric_values else "",
                "metric_std": _fmt(stdev(metric_values)) if len(metric_values) > 1 else "0",
                "delta_from_clean_mean": _fmt(mean(delta_values)) if delta_values else "",
                "delta_from_clean_std": _fmt(stdev(delta_values)) if len(delta_values) > 1 else "0",
            }
        )

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
        "metric_mean",
        "metric_std",
        "delta_from_clean_mean",
        "delta_from_clean_std",
    ]
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
    seed_counts = sorted({int(row["n_seeds"]) for row in rows if str(row.get("n_seeds", "")).isdigit()})
    seed_count = seed_counts[-1] if seed_counts else 0
    seed_label = f"{seed_count}-seed" if seed_count else "multi-seed"
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        rf"  \caption{{{seed_label.capitalize()} HalfCheetah corrected-scope summary. Values are mean best single-option policy return over completed seeds; $\Delta$ is the mean clean-relative change for each objective.}}",
        r"  \label{tab:seed_degradation_summary}",
        r"  \begin{tabular}{llrrr}",
        r"    \toprule",
        r"    Objective & Condition & Seeds & Return mean & $\Delta$ mean \\",
        r"    \midrule",
    ]
    for row in rows:
        method = METHOD_LABELS.get(str(row["method"]), str(row["method"]))
        condition = str(row["condition"])
        lines.append(
            "    "
            + f"{method} & \\texttt{{{condition}}} & {row['n_seeds']} & "
            + f"{_fmt_latex(str(row['metric_mean']))} & {_fmt_latex(str(row['delta_from_clean_mean']))} \\\\"
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
    print(f"Wrote {len(rows)} mean-summary rows to {args.output_csv}")
    print(f"Wrote LaTeX mean-summary table to {args.output_tex}")


if __name__ == "__main__":
    main()
