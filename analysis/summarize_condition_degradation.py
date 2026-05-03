from __future__ import annotations

import argparse
import csv
import math
import os
from typing import Dict, Iterable, List


DEFAULT_METRIC = "best_single_option_policy/return-average"
MAIN_CONDITIONS = ["clean", "noise-med", "delay-3", "noise-med-delay-3", "encoded-clean"]
METHOD_ORDER = ["diayn", "dads", "lsd_delta"]


def _coerce_float(value) -> float | None:
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


def _coerce_int(value) -> int | None:
    numeric = _coerce_float(value)
    if numeric is None:
        return None
    return int(numeric)


def _read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _metric_key(method: str, metric: str) -> str:
    return f"{method}_progress__{metric}"


def _sort_key(row: Dict[str, object]) -> tuple:
    method = str(row.get("method", ""))
    condition = str(row.get("condition", ""))
    method_i = METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)
    condition_i = (
        MAIN_CONDITIONS.index(condition)
        if condition in MAIN_CONDITIONS
        else len(MAIN_CONDITIONS)
    )
    return (str(row.get("environment", "")), int(row.get("seed", 0)), method_i, condition_i)


def summarize(
    rows: Iterable[Dict[str, str]],
    *,
    metric: str,
    include_run_substrings: List[str],
    conditions: List[str],
) -> List[Dict[str, object]]:
    selected: List[Dict[str, object]] = []
    for row in rows:
        run_dir = row.get("run_dir", "")
        if include_run_substrings and not any(token in run_dir for token in include_run_substrings):
            continue
        condition = row.get("condition", "")
        if condition not in conditions:
            continue
        method = row.get("method", "")
        value = _coerce_float(row.get(_metric_key(method, metric)))
        epoch = _coerce_int(row.get(f"{method}_progress__epoch"))
        pool_size = _coerce_int(row.get(f"{method}_progress__pool-size"))
        selected.append(
            {
                "method": method,
                "environment": row.get("environment", ""),
                "condition": condition,
                "seed": _coerce_int(row.get("seed")),
                "metric": metric,
                "metric_value": value,
                "epoch": epoch,
                "pool_size": pool_size,
                "run_complete": bool(epoch is not None and epoch >= 249),
                "run_dir": run_dir,
            }
        )

    clean_by_key: Dict[tuple, float] = {}
    for row in selected:
        if row["condition"] != "clean":
            continue
        value = row["metric_value"]
        if value is None:
            continue
        clean_by_key[(row["method"], row["environment"], row["seed"])] = float(value)

    summaries: List[Dict[str, object]] = []
    for row in selected:
        clean_value = clean_by_key.get((row["method"], row["environment"], row["seed"]))
        value = row["metric_value"]
        delta = None
        ratio = None
        if value is not None and clean_value is not None:
            delta = float(value) - clean_value
            if abs(clean_value) > 1e-12:
                ratio = float(value) / clean_value
        out = dict(row)
        out["clean_metric_value"] = clean_value
        out["delta_from_clean"] = delta
        out["ratio_to_clean"] = ratio
        summaries.append(out)

    return sorted(summaries, key=_sort_key)


def _write_csv(path: str, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    fieldnames = [
        "method",
        "environment",
        "condition",
        "seed",
        "metric",
        "metric_value",
        "clean_metric_value",
        "delta_from_clean",
        "ratio_to_clean",
        "epoch",
        "pool_size",
        "run_complete",
        "run_dir",
    ]
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", required=True, help="Input aggregate CSV.")
    parser.add_argument("--output", required=True, help="Output degradation CSV.")
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument(
        "--include-run-substring",
        action="append",
        default=[],
        help="Only include aggregate rows whose run_dir contains this token. Repeatable.",
    )
    parser.add_argument(
        "--condition",
        action="append",
        default=[],
        help="Condition to include. Defaults to the corrected main conditions.",
    )
    args = parser.parse_args()

    rows = _read_rows(args.aggregate)
    summaries = summarize(
        rows,
        metric=args.metric,
        include_run_substrings=args.include_run_substring,
        conditions=args.condition or MAIN_CONDITIONS,
    )
    _write_csv(args.output, summaries)
    print(f"Wrote {len(summaries)} degradation rows to {args.output}")


if __name__ == "__main__":
    main()
