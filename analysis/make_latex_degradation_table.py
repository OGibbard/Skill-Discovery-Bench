from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List


CONDITION_LABELS = {
    "clean": "clean",
    "noise-med": "noise-med",
    "delay-3": "delay-3",
    "encoded-clean": "encoded-clean",
}
METHOD_LABELS = {
    "diayn": r"\texttt{diayn}",
    "dads": r"\texttt{dads}",
    "lsd_delta": r"\texttt{lsd\_delta}",
}
METHOD_ORDER = ["diayn", "dads", "lsd_delta"]
CONDITION_ORDER = ["clean", "noise-med", "delay-3", "encoded-clean"]


def _float_or_none(value: str) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _format_float(value: str) -> str:
    numeric = _float_or_none(value)
    if numeric is None:
        return "--"
    return f"{numeric:.1f}"


def _sort_key(row: Dict[str, str]) -> tuple:
    method = row.get("method", "")
    condition = row.get("condition", "")
    method_i = METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)
    condition_i = (
        CONDITION_ORDER.index(condition)
        if condition in CONDITION_ORDER
        else len(CONDITION_ORDER)
    )
    return (method_i, condition_i)


def render(rows: List[Dict[str, str]], *, complete_only: bool) -> str:
    selected = []
    for row in rows:
        if complete_only and row.get("run_complete") != "True":
            continue
        selected.append(row)
    selected.sort(key=_sort_key)

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Seed-0 HalfCheetah clean-relative degradation summary for completed corrected-scope runs. Values are best single-option policy return; $\Delta$ is relative to each method's clean seed-0 run.}",
        r"  \label{tab:seed0_degradation_live}",
        r"  \begin{tabular}{llrrr}",
        r"    \toprule",
        r"    Objective & Condition & Return & $\Delta$ vs clean & Epoch \\",
        r"    \midrule",
    ]

    for row in selected:
        method = METHOD_LABELS.get(row.get("method", ""), row.get("method", ""))
        condition = CONDITION_LABELS.get(row.get("condition", ""), row.get("condition", ""))
        metric = _format_float(row.get("metric_value", ""))
        delta = _format_float(row.get("delta_from_clean", ""))
        epoch = row.get("epoch", "--") or "--"
        if row.get("run_complete") != "True":
            condition = f"{condition} (partial)"
        lines.append(f"    {method} & \\texttt{{{condition}}} & {metric} & {delta} & {epoch} \\\\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-partial", action="store_true")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(render(rows, complete_only=not args.include_partial))
    print(f"Wrote LaTeX table to {args.output}")


if __name__ == "__main__":
    main()
