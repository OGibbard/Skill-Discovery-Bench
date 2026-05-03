from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CODE_DIR = os.path.join(PROJECT_ROOT, "code")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from dissertation.robustness_core import (
    ObservationCorruptor,
    build_observation_corruption_config,
)


RAW_LABELS = [
    "z",
    "angle",
    "bthigh",
    "bshin",
    "bfoot",
    "fthigh",
    "fshin",
    "ffoot",
    "x_vel",
    "z_vel",
    "ang_vel",
    "bthigh_vel",
    "bshin_vel",
    "bfoot_vel",
    "fthigh_vel",
    "fshin_vel",
    "ffoot_vel",
]


def _sample_halfcheetah_observations(
    *,
    env_id: str,
    episodes: int,
    max_steps: int,
    seed: int,
) -> np.ndarray:
    import gymnasium as gym

    env = gym.make(env_id)
    observations = []
    rng = np.random.default_rng(seed)

    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            observations.append(np.asarray(obs, dtype=np.float32))
            for _ in range(max_steps):
                action = rng.uniform(
                    low=env.action_space.low,
                    high=env.action_space.high,
                    size=env.action_space.shape,
                ).astype(env.action_space.dtype)
                obs, _, terminated, truncated, _ = env.step(action)
                observations.append(np.asarray(obs, dtype=np.float32))
                if terminated or truncated:
                    break
    finally:
        env.close()

    return np.asarray(observations, dtype=np.float32)


def _encode_observations(
    observations: np.ndarray,
    *,
    encoder_dim: int,
    hidden_dim: int,
    seed: int,
) -> np.ndarray:
    config = build_observation_corruption_config(
        noise_std=0.0,
        delay_steps=0,
        encoder_type="mlp",
        encoder_dim=encoder_dim,
        encoder_hidden_dim=hidden_dim,
        seed=seed,
    )
    corruptor = ObservationCorruptor(input_dim=observations.shape[1], config=config)
    return np.asarray([corruptor.reset(obs) for obs in observations], dtype=np.float32)


def _standardize(values: np.ndarray) -> np.ndarray:
    return (values - values.mean(axis=0, keepdims=True)) / (
        values.std(axis=0, keepdims=True) + 1e-8
    )


def _pca_2d(values: np.ndarray) -> np.ndarray:
    values = _standardize(values)
    _, _, vh = np.linalg.svd(values, full_matrices=False)
    return values @ vh[:2].T


def _plot_schematic(ax: plt.Axes, raw_dim: int, hidden_dim: int, encoded_dim: int) -> None:
    ax.axis("off")
    boxes = [
        (0.04, 0.32, 0.18, 0.36, f"Raw MuJoCo state\n{raw_dim} dimensions"),
        (0.31, 0.32, 0.22, 0.36, f"Fixed random MLP encoder\ntanh hidden layer, {hidden_dim} units"),
        (0.62, 0.32, 0.18, 0.36, f"Compressed observation\n{encoded_dim} dimensions"),
        (0.88, 0.32, 0.10, 0.36, "Policy\ninput"),
    ]
    for x, y, w, h, label in boxes:
        ax.add_patch(
            plt.Rectangle(
                (x, y),
                w,
                h,
                facecolor="#edf3f8",
                edgecolor="#30475e",
                linewidth=1.5,
            )
        )
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9.5)

    for x1, x2 in [(0.22, 0.31), (0.53, 0.62), (0.80, 0.88)]:
        ax.annotate(
            "",
            xy=(x2, 0.50),
            xytext=(x1, 0.50),
            arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#30475e"},
        )

    ax.text(
        0.50,
        0.12,
        "Noise and delay are applied after this encoder in corrupted conditions.",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#555555",
    )


def _heatmap_panel(
    ax: plt.Axes,
    values: np.ndarray,
    title: str,
    y_labels: list[str],
    max_samples: int,
) -> None:
    subset = values[:max_samples].T
    subset = _standardize(subset.T).T
    image = ax.imshow(subset, aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    ax.set_title(title, pad=10)
    ax.set_xlabel("Sampled timestep")
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.tick_params(axis="y", pad=2)
    return image


def make_figure(
    raw_obs: np.ndarray,
    encoded_obs: np.ndarray,
    output: str,
    *,
    hidden_dim: int,
    max_heatmap_samples: int,
) -> None:
    raw_pca = _pca_2d(raw_obs)
    encoded_pca = _pca_2d(encoded_obs)
    timesteps = np.arange(len(raw_obs))

    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[0.65, 1.35, 1.15], hspace=0.18, wspace=0.14)

    ax_schematic = fig.add_subplot(grid[0, :])
    _plot_schematic(ax_schematic, raw_obs.shape[1], hidden_dim, encoded_obs.shape[1])

    ax_raw_heat = fig.add_subplot(grid[1, 0])
    raw_image = _heatmap_panel(
        ax_raw_heat,
        raw_obs,
        "Raw HalfCheetah observations, standardized",
        RAW_LABELS[: raw_obs.shape[1]],
        max_heatmap_samples,
    )

    ax_enc_heat = fig.add_subplot(grid[1, 1])
    enc_labels = [f"enc_{idx}" for idx in range(encoded_obs.shape[1])]
    enc_image = _heatmap_panel(
        ax_enc_heat,
        encoded_obs,
        "Encoded sensor observations, standardized",
        enc_labels,
        max_heatmap_samples,
    )

    cbar = fig.colorbar(enc_image, ax=[ax_raw_heat, ax_enc_heat], fraction=0.02, pad=0.015)
    cbar.set_label("Standard deviations from dimension mean")
    raw_image.set_clim(enc_image.get_clim())

    ax_raw_pca = fig.add_subplot(grid[2, 0])
    scatter = ax_raw_pca.scatter(
        raw_pca[:, 0],
        raw_pca[:, 1],
        c=timesteps,
        cmap="viridis",
        s=6,
        alpha=0.75,
    )
    ax_raw_pca.set_title("Raw observations projected to 2D by PCA")
    ax_raw_pca.set_xlabel("PC1")
    ax_raw_pca.set_ylabel("PC2")
    ax_raw_pca.grid(True, alpha=0.25)

    ax_enc_pca = fig.add_subplot(grid[2, 1])
    ax_enc_pca.scatter(
        encoded_pca[:, 0],
        encoded_pca[:, 1],
        c=timesteps,
        cmap="viridis",
        s=6,
        alpha=0.75,
    )
    ax_enc_pca.set_title("Encoded observations projected to 2D by PCA")
    ax_enc_pca.set_xlabel("PC1")
    ax_enc_pca.set_ylabel("PC2")
    ax_enc_pca.grid(True, alpha=0.25)

    cbar = fig.colorbar(scatter, ax=[ax_raw_pca, ax_enc_pca], fraction=0.02, pad=0.015)
    cbar.set_label("Sample order")

    fig.suptitle(
        "Sensor Encoding Visualization: HalfCheetah 17D State to 8D Fixed MLP Observation",
        fontsize=15,
    )
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="HalfCheetah-v4")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--encoder-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--max-heatmap-samples", type=int, default=350)
    parser.add_argument(
        "--output",
        default="outputs/figures/sensor_encoding_halfcheetah.png",
    )
    args = parser.parse_args()

    raw_obs = _sample_halfcheetah_observations(
        env_id=args.env_id,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    encoded_obs = _encode_observations(
        raw_obs,
        encoder_dim=args.encoder_dim,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
    )
    make_figure(
        raw_obs,
        encoded_obs,
        args.output,
        hidden_dim=args.hidden_dim,
        max_heatmap_samples=args.max_heatmap_samples,
    )
    print(args.output)
    print(f"sampled_observations={raw_obs.shape[0]}")
    print(f"raw_dim={raw_obs.shape[1]}")
    print(f"encoded_dim={encoded_obs.shape[1]}")


if __name__ == "__main__":
    main()
