from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import datetime as _dt
import json
import math
import os
from typing import Any, Dict, Iterable

import numpy as np


def _canonical_encoder_type(encoder_type: str) -> str:
    encoder_type = (encoder_type or "identity").strip().lower()
    if encoder_type in {"none", "identity"}:
        return "identity"
    if encoder_type in {"linear", "random_linear"}:
        return "linear"
    if encoder_type in {"mlp", "random_mlp"}:
        return "mlp"
    raise ValueError(f"Unsupported encoder_type: {encoder_type}")


@dataclass(frozen=True)
class ObservationCorruptionConfig:

    noise_std: float = 0.0
    delay_steps: int = 0
    encoder_type: str = "identity"
    encoder_dim: int = 0
    encoder_hidden_dim: int = 0
    keep_first_dims: int = 0
    seed: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "encoder_type", _canonical_encoder_type(self.encoder_type))
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")
        if self.delay_steps < 0:
            raise ValueError("delay_steps must be non-negative")
        if self.encoder_dim < 0:
            raise ValueError("encoder_dim must be non-negative")
        if self.encoder_hidden_dim < 0:
            raise ValueError("encoder_hidden_dim must be non-negative")
        if self.keep_first_dims < 0:
            raise ValueError("keep_first_dims must be non-negative")

    def enabled(self) -> bool:
        return (
            self.noise_std > 0
            or self.delay_steps > 0
            or self.encoder_type != "identity"
        )

    def output_dim(self, input_dim: int) -> int:
        input_dim = int(input_dim)
        if self.keep_first_dims > input_dim:
            raise ValueError(
                f"keep_first_dims ({self.keep_first_dims}) exceeds input_dim ({input_dim})"
            )
        suffix_dim = input_dim - self.keep_first_dims
        if self.encoder_type == "identity":
            encoded_suffix_dim = suffix_dim
        else:
            encoded_suffix_dim = self.encoder_dim or suffix_dim
        return self.keep_first_dims + encoded_suffix_dim

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_observation_corruption_config(
    *,
    noise_std: float = 0.0,
    delay_steps: int = 0,
    encoder_type: str = "identity",
    encoder_dim: int = 0,
    encoder_hidden_dim: int = 0,
    keep_first_dims: int = 0,
    seed: int | None = None,
) -> ObservationCorruptionConfig:
    return ObservationCorruptionConfig(
        noise_std=float(noise_std),
        delay_steps=int(delay_steps),
        encoder_type=encoder_type,
        encoder_dim=int(encoder_dim),
        encoder_hidden_dim=int(encoder_hidden_dim),
        keep_first_dims=int(keep_first_dims),
        seed=seed if seed is None else int(seed),
    )


class ObservationCorruptor:

    def __init__(self, input_dim: int, config: ObservationCorruptionConfig):
        self.input_dim = int(input_dim)
        self.config = config
        self.keep_first_dims = int(config.keep_first_dims)
        if self.keep_first_dims > self.input_dim:
            raise ValueError(
                f"keep_first_dims ({self.keep_first_dims}) exceeds input_dim ({self.input_dim})"
            )

        self._suffix_input_dim = self.input_dim - self.keep_first_dims
        self.output_dim = int(config.output_dim(self.input_dim))
        self._suffix_output_dim = self.output_dim - self.keep_first_dims
        self._rng = np.random.RandomState(config.seed)
        self._delay_buffer: deque[np.ndarray] = deque(maxlen=config.delay_steps + 1)

        self._encoder_params: Dict[str, np.ndarray] = {}
        if config.encoder_type == "identity":
            if config.encoder_dim not in (0, self._suffix_input_dim):
                raise ValueError(
                    "encoder_dim is only supported with encoder_type=linear or mlp"
                )
        elif config.encoder_type == "linear":
            scale = 1.0 / math.sqrt(max(1, self._suffix_input_dim))
            self._encoder_params["weight"] = (
                self._rng.randn(self._suffix_output_dim, self._suffix_input_dim) * scale
            ).astype(np.float32)
            self._encoder_params["bias"] = np.zeros(self._suffix_output_dim, dtype=np.float32)
        elif config.encoder_type == "mlp":
            hidden_dim = config.encoder_hidden_dim or max(
                self._suffix_input_dim,
                self._suffix_output_dim,
            )
            scale1 = 1.0 / math.sqrt(max(1, self._suffix_input_dim))
            scale2 = 1.0 / math.sqrt(max(1, hidden_dim))
            self._encoder_params["w1"] = (
                self._rng.randn(hidden_dim, self._suffix_input_dim) * scale1
            ).astype(np.float32)
            self._encoder_params["b1"] = np.zeros(hidden_dim, dtype=np.float32)
            self._encoder_params["w2"] = (
                self._rng.randn(self._suffix_output_dim, hidden_dim) * scale2
            ).astype(np.float32)
            self._encoder_params["b2"] = np.zeros(self._suffix_output_dim, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported encoder_type: {config.encoder_type}")

    def _flatten(self, observation: np.ndarray | Iterable[float]) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        if obs.shape[0] != self.input_dim:
            raise ValueError(
                f"Expected flattened observation dim {self.input_dim}, got {obs.shape[0]}"
            )
        return obs

    def _encode_suffix(self, suffix: np.ndarray) -> np.ndarray:
        if self.config.encoder_type == "identity":
            return suffix
        if self.config.encoder_type == "linear":
            return self._encoder_params["weight"].dot(suffix) + self._encoder_params["bias"]

        hidden = np.tanh(self._encoder_params["w1"].dot(suffix) + self._encoder_params["b1"])
        return self._encoder_params["w2"].dot(hidden) + self._encoder_params["b2"]

    def _measurement(self, observation: np.ndarray | Iterable[float]) -> np.ndarray:
        flat_obs = self._flatten(observation)
        prefix = flat_obs[: self.keep_first_dims]
        suffix = flat_obs[self.keep_first_dims :]
        encoded_suffix = self._encode_suffix(suffix)
        measurement = np.concatenate([prefix, encoded_suffix], axis=0).astype(np.float32)
        if self.config.noise_std > 0:
            measurement = measurement + self._rng.normal(
                loc=0.0,
                scale=self.config.noise_std,
                size=measurement.shape,
            ).astype(np.float32)
        return measurement

    def reset(self, observation: np.ndarray | Iterable[float]) -> np.ndarray:
        measurement = self._measurement(observation)
        self._delay_buffer.clear()
        for _ in range(self.config.delay_steps + 1):
            self._delay_buffer.append(measurement.copy())
        return self._delay_buffer[0].copy()

    def step(self, observation: np.ndarray | Iterable[float]) -> np.ndarray:
        measurement = self._measurement(observation)
        self._delay_buffer.append(measurement.copy())
        if not self._delay_buffer:
            return measurement
        return self._delay_buffer[0].copy()


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def save_robustness_manifest(
    *,
    log_dir: str,
    method: str,
    environment: str,
    seed: int | None,
    config: ObservationCorruptionConfig,
    extra: Dict[str, Any] | None = None,
) -> str:
    os.makedirs(log_dir, exist_ok=True)
    payload: Dict[str, Any] = {
        "created_at_utc": _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "method": method,
        "environment": environment,
        "seed": seed,
        "robustness": config.to_dict(),
    }
    if extra:
        payload["extra"] = _jsonify(extra)

    manifest_path = os.path.join(log_dir, "robustness_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(_jsonify(payload), f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest_path
