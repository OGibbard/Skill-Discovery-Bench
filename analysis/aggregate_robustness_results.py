from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, Iterable, List, Tuple


def _coerce_value(value):
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    text = str(value).strip()
    if text == "":
        return None
    try:
        if any(ch in text for ch in [".", "e", "E"]):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _flatten(prefix: str, payload, out: Dict[str, object]) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            new_prefix = f"{prefix}__{key}" if prefix else str(key)
            _flatten(new_prefix, value, out)
        return
    out[prefix] = payload


def _find_manifests(root_dir: str) -> List[str]:
    manifests = []
    for dirpath, _, filenames in os.walk(root_dir):
        if "robustness_manifest.json" in filenames:
            manifests.append(os.path.join(dirpath, "robustness_manifest.json"))
    manifests.sort()
    return manifests


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_last_csv_row(path: str) -> Dict[str, object]:
    if not os.path.exists(path):
        return {}
    last_row = None
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row:
                last_row = row
    if last_row is None:
        return {}
    return {key: _coerce_value(value) for key, value in last_row.items()}


def _extract_progress_metrics(prefix: str, run_dir: str) -> Dict[str, object]:
    metrics = {}
    _flatten(prefix, _read_last_csv_row(os.path.join(run_dir, "progress.csv")), metrics)
    return metrics


def _extract_diayn_metrics(run_dir: str) -> Dict[str, object]:
    return _extract_progress_metrics("diayn_progress", run_dir)


def _extract_unified_skill_metrics(method: str, run_dir: str) -> Dict[str, object]:
    return _extract_progress_metrics(f"{method}_progress", run_dir)


def _extract_lsd_metrics(run_dir: str) -> Dict[str, object]:
    metrics = {}
    _flatten("lsd_train", _read_last_csv_row(os.path.join(run_dir, "progress.csv")), metrics)
    _flatten("lsd_eval", _read_last_csv_row(os.path.join(run_dir, "progress_eval.csv")), metrics)
    return metrics


def _extract_dads_metrics(run_dir: str) -> Dict[str, object]:
    event_files: List[str] = []
    for dirpath, _, filenames in os.walk(run_dir):
        for filename in filenames:
            if "tfevents" in filename:
                event_files.append(os.path.join(dirpath, filename))

    if not event_files:
        return {}

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {"dads_metrics_error": "tensorboard backend not available"}

    latest_by_tag: Dict[str, Tuple[int, float]] = {}
    for event_file in sorted(event_files):
        accumulator = EventAccumulator(event_file, size_guidance={"scalars": 0})
        try:
            accumulator.Reload()
        except Exception as exc:
            return {"dads_metrics_error": f"failed to read {event_file}: {exc}"}
        for tag in accumulator.Tags().get("scalars", []):
            scalars = accumulator.Scalars(tag)
            if not scalars:
                continue
            last_event = scalars[-1]
            previous = latest_by_tag.get(tag)
            if previous is None or last_event.step >= previous[0]:
                latest_by_tag[tag] = (last_event.step, last_event.value)

    metrics = {}
    for tag, (_, value) in latest_by_tag.items():
        sanitized_tag = tag.replace("/", "__")
        metrics[f"dads_scalar__{sanitized_tag}"] = value
    return metrics


def _extract_metrics(method: str, run_dir: str) -> Dict[str, object]:
    if method in {"diayn", "lsd_delta"}:
        return _extract_unified_skill_metrics(method, run_dir)
    if method == "dads":
        metrics = _extract_unified_skill_metrics(method, run_dir)
        if metrics:
            return metrics
        return _extract_dads_metrics(run_dir)
    if method == "lsd":
        return _extract_lsd_metrics(run_dir)
    return {}


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


def aggregate_runs(root_dir: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for manifest_path in _find_manifests(root_dir):
        manifest = _load_json(manifest_path)
        run_dir = os.path.dirname(manifest_path)
        row: Dict[str, object] = {
            "manifest_path": manifest_path,
            "run_dir": run_dir,
            "method": manifest.get("method"),
            "environment": manifest.get("environment"),
            "seed": manifest.get("seed"),
            "condition": _derive_condition(manifest),
            "created_at_utc": manifest.get("created_at_utc"),
        }
        _flatten("robustness", manifest.get("robustness", {}), row)
        _flatten("extra", manifest.get("extra", {}), row)
        row.update(_extract_metrics(str(manifest.get("method")), run_dir))
        rows.append(row)
    return rows


def _write_csv(path: str, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("No robustness manifests found to aggregate.")

    fieldnames = sorted({key for row in rows for key in row.keys()})
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
    default_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser.add_argument("--root", type=str, default=default_root,
                        help="Directory to search recursively for robustness manifests.")
    parser.add_argument("--output", type=str, default=os.path.join(default_root, "robustness_summary.csv"),
                        help="Output CSV path.")
    args = parser.parse_args()

    rows = aggregate_runs(args.root)
    _write_csv(args.output, rows)
    print(f"Wrote {len(rows)} aggregated runs to {args.output}")


if __name__ == "__main__":
    main()
