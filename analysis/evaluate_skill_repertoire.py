from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from contextlib import nullcontext
from itertools import combinations
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, Iterable, List

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNIFIED_ROOT = PROJECT_ROOT / "code" / "unified_skill_discovery"
if str(UNIFIED_ROOT) not in sys.path:
    sys.path.insert(0, str(UNIFIED_ROOT))

import joblib
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from rllab.misc.ext import set_seed
from sac.misc.sampler import rollout
from sac.policies.hierarchical_policy import FixedOptionPolicy


MAIN_CONDITIONS = {"clean", "noise-med", "delay-3", "noise-med-delay-3", "encoded-clean"}
SNAPSHOT_RE = re.compile(r"itr_(\d+)\.pkl$")
_RECORD_VIDEO_PATCHED = False


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


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _latest_snapshot(run_dir: Path) -> tuple[Path | None, int | None]:
    best_path: Path | None = None
    best_epoch: int | None = None
    for path in run_dir.glob("itr_*.pkl"):
        match = SNAPSHOT_RE.match(path.name)
        if match is None:
            continue
        epoch = int(match.group(1))
        if best_epoch is None or epoch > best_epoch:
            best_epoch = epoch
            best_path = path
    return best_path, best_epoch


def _disable_recording_wrappers(env) -> None:
    inner = getattr(env, "env", None)
    if inner is None:
        return
    if inner.__class__.__name__ == "RecordVideo" and hasattr(inner, "env"):
        close = getattr(inner, "close_video_recorder", None)
        if callable(close):
            close()
        env.env = inner.env
        if hasattr(env, "monitoring"):
            env.monitoring = False


def _patch_record_video_wrapper() -> None:
    global _RECORD_VIDEO_PATCHED
    if _RECORD_VIDEO_PATCHED:
        return
    try:
        import gymnasium as gym
        import gymnasium.wrappers.rendering as rendering
    except Exception:
        return

    class NoOpRecordVideo(gym.Wrapper):
        def __init__(self, env, *args, **kwargs):
            super().__init__(env)

        def close_video_recorder(self) -> None:
            return None

    gym.wrappers.RecordVideo = NoOpRecordVideo
    rendering.RecordVideo = NoOpRecordVideo
    _RECORD_VIDEO_PATCHED = True


def _infer_num_skills(policy, env) -> int:
    policy_obs_dim = getattr(policy, "_Ds", None)
    if policy_obs_dim is None:
        policy_obs_dim = policy.env_spec.observation_space.flat_dim
    obs_dim = env.spec.observation_space.flat_dim
    num_skills = int(policy_obs_dim) - int(obs_dim)
    if num_skills <= 0:
        raise ValueError(
            f"Could not infer positive skill count from policy_obs_dim={policy_obs_dim} "
            f"and obs_dim={obs_dim}"
        )
    return num_skills


def _path_metric(path: Dict[str, np.ndarray], skill_id: int, rollout_id: int) -> Dict[str, object]:
    rewards = np.asarray(path["rewards"], dtype=float)
    actions = np.asarray(path["actions"], dtype=float)
    observations = np.asarray(path["observations"], dtype=float)
    env_infos = path.get("env_infos", {}) or {}

    x_position = np.asarray(env_infos.get("x_position", []), dtype=float)
    x_velocity = np.asarray(env_infos.get("x_velocity", []), dtype=float)
    final_x = float(x_position[-1]) if x_position.size else ""
    start_x = float(x_position[0]) if x_position.size else 0.0
    x_displacement = float(final_x - start_x) if x_position.size else ""

    return {
        "skill_id": skill_id,
        "rollout_id": rollout_id,
        "return": float(np.sum(rewards)),
        "episode_length": int(len(rewards)),
        "final_x_position": final_x,
        "x_displacement": x_displacement,
        "mean_x_velocity": float(np.mean(x_velocity)) if x_velocity.size else "",
        "mean_abs_action": float(np.mean(np.abs(actions))) if actions.size else "",
        "action_energy": float(np.mean(np.square(actions))) if actions.size else "",
        "final_obs_norm": float(np.linalg.norm(observations[-1])) if observations.size else "",
        "mean_obs_norm": float(np.mean(np.linalg.norm(observations, axis=1))) if observations.size else "",
    }


def _pairwise_distance(features: np.ndarray, *, standardize: bool) -> float:
    if features.shape[0] < 2:
        return 0.0
    values = features.astype(float)
    if standardize:
        scale = np.std(values, axis=0)
        keep = scale > 1e-12
        if not np.any(keep):
            return 0.0
        values = (values[:, keep] - np.mean(values[:, keep], axis=0)) / scale[keep]
    distances = [
        float(np.linalg.norm(values[i] - values[j]))
        for i, j in combinations(range(values.shape[0]), 2)
    ]
    return float(mean(distances)) if distances else 0.0


def _summarize_run(base: Dict[str, object], skill_rows: List[Dict[str, object]]) -> Dict[str, object]:
    by_skill: dict[int, list[Dict[str, object]]] = {}
    for row in skill_rows:
        by_skill.setdefault(int(row["skill_id"]), []).append(row)

    skill_returns: List[float] = []
    skill_features: List[List[float]] = []
    best_skill_id: int | None = None
    best_skill_return = float("-inf")

    for skill_id in sorted(by_skill):
        rows = by_skill[skill_id]
        returns = [float(row["return"]) for row in rows]
        skill_return = mean(returns)
        skill_returns.append(skill_return)
        if skill_return > best_skill_return:
            best_skill_return = skill_return
            best_skill_id = skill_id

        def feature_mean(name: str) -> float:
            values = [_float(row.get(name)) for row in rows]
            values = [value for value in values if value is not None]
            return mean(values) if values else 0.0

        skill_features.append(
            [
                feature_mean("x_displacement"),
                feature_mean("mean_x_velocity"),
                feature_mean("mean_abs_action"),
                feature_mean("final_obs_norm"),
                feature_mean("mean_obs_norm"),
            ]
        )

    feature_array = np.asarray(skill_features, dtype=float)
    return_std = stdev(skill_returns) if len(skill_returns) > 1 else 0.0
    summary = dict(base)
    summary.update(
        {
            "n_skills": len(skill_returns),
            "n_rollouts_per_skill": len(skill_rows) // max(1, len(skill_returns)),
            "best_skill_id": best_skill_id,
            "best_skill_return": best_skill_return,
            "mean_skill_return": mean(skill_returns) if skill_returns else "",
            "median_skill_return": median(skill_returns) if skill_returns else "",
            "worst_skill_return": min(skill_returns) if skill_returns else "",
            "std_skill_return": return_std,
            "skill_return_spread": (max(skill_returns) - min(skill_returns)) if skill_returns else "",
            "behavior_pairwise_distance_raw": _pairwise_distance(feature_array, standardize=False),
            "behavior_pairwise_distance_standardized": _pairwise_distance(feature_array, standardize=True),
        }
    )
    return summary


def _write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _eligible_runs(rows: Iterable[Dict[str, str]], include_substrings: list[str]) -> List[Dict[str, str]]:
    selected = []
    for row in rows:
        run_dir = row.get("run_dir", "")
        if row.get("run_complete") != "True":
            continue
        if row.get("condition") not in MAIN_CONDITIONS:
            continue
        if include_substrings and not any(token in run_dir for token in include_substrings):
            continue
        selected.append(row)
    selected.sort(key=lambda r: (int(r["seed"]), r["method"], r["condition"]))
    return selected


def evaluate_run(
    row: Dict[str, str],
    *,
    max_path_length: int,
    n_rollouts: int,
    deterministic: bool,
) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    run_dir = Path(row["run_dir"])
    snapshot_path, snapshot_epoch = _latest_snapshot(run_dir)
    if snapshot_path is None:
        raise FileNotFoundError(f"No itr_*.pkl snapshot found in {run_dir}")

    tf.reset_default_graph()
    with tf.Session():
        _patch_record_video_wrapper()
        data = joblib.load(snapshot_path)
        policy = data["policy"]
        env = data["env"]
        _disable_recording_wrappers(env)
        num_skills = _infer_num_skills(policy, env)
        base = {
            "method": row["method"],
            "condition": row["condition"],
            "seed": row["seed"],
            "run_dir": str(run_dir),
            "snapshot_path": str(snapshot_path),
            "snapshot_epoch": snapshot_epoch,
            "training_final_epoch": row.get("epoch", ""),
            "training_final_best_skill_return": row.get("metric_value", ""),
        }

        context = policy.deterministic(deterministic) if hasattr(policy, "deterministic") else nullcontext()
        skill_rows: List[Dict[str, object]] = []
        with context:
            for skill_id in range(num_skills):
                fixed_policy = FixedOptionPolicy(policy, num_skills, skill_id)
                for rollout_id in range(n_rollouts):
                    path = rollout(env, fixed_policy, max_path_length)
                    metric = _path_metric(path, skill_id, rollout_id)
                    metric.update(base)
                    skill_rows.append(metric)

        return skill_rows, _summarize_run(base, skill_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degradation-csv", required=True, type=Path)
    parser.add_argument("--output-skill-csv", required=True, type=Path)
    parser.add_argument("--output-summary-csv", required=True, type=Path)
    parser.add_argument("--max-path-length", type=int, default=1000)
    parser.add_argument("--n-rollouts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--deterministic", type=int, default=1)
    parser.add_argument("--include-run-substring", action="append", default=[])
    args = parser.parse_args()

    set_seed(args.seed)
    np.random.seed(args.seed)

    runs = _eligible_runs(_read_rows(args.degradation_csv), args.include_run_substring)
    if args.max_runs is not None:
        runs = runs[: args.max_runs]
    if not runs:
        raise ValueError("No eligible completed main runs found.")

    all_skill_rows: List[Dict[str, object]] = []
    summaries: List[Dict[str, object]] = []
    for index, row in enumerate(runs, start=1):
        print(
            f"[{index}/{len(runs)}] evaluating seed={row['seed']} "
            f"method={row['method']} condition={row['condition']}",
            flush=True,
        )
        skill_rows, summary = evaluate_run(
            row,
            max_path_length=args.max_path_length,
            n_rollouts=args.n_rollouts,
            deterministic=bool(args.deterministic),
        )
        all_skill_rows.extend(skill_rows)
        summaries.append(summary)

    _write_csv(args.output_skill_csv, all_skill_rows)
    _write_csv(args.output_summary_csv, summaries)
    print(f"Wrote {len(all_skill_rows)} skill rollout rows to {args.output_skill_csv}")
    print(f"Wrote {len(summaries)} run summary rows to {args.output_summary_csv}")


if __name__ == "__main__":
    main()
