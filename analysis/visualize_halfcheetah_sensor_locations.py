from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch


SENSOR_GROUPS = [
    {
        "name": "Torso / root state",
        "xy": (0.0, 0.55),
        "color": "#2f6f9f",
        "text_xy": (-1.25, 0.85),
        "channels": [
            "obs[0] root z",
            "obs[1] torso pitch",
            "obs[8] x velocity",
            "obs[9] z velocity",
            "obs[10] pitch velocity",
        ],
    },
    {
        "name": "Back thigh hinge",
        "xy": (-0.50, 0.33),
        "color": "#c75b39",
        "text_xy": (-1.35, -0.18),
        "channels": ["obs[2] angle", "obs[11] angular velocity"],
    },
    {
        "name": "Back shin hinge",
        "xy": (-0.80, -0.15),
        "color": "#d9972f",
        "text_xy": (-1.35, -0.74),
        "channels": ["obs[3] angle", "obs[12] angular velocity"],
    },
    {
        "name": "Back foot hinge",
        "xy": (-0.45, -0.55),
        "color": "#729b42",
        "text_xy": (-1.35, -1.30),
        "channels": ["obs[4] angle", "obs[13] angular velocity"],
    },
    {
        "name": "Front thigh hinge",
        "xy": (0.55, 0.33),
        "color": "#8d5fbf",
        "text_xy": (1.35, -0.18),
        "channels": ["obs[5] angle", "obs[14] angular velocity"],
    },
    {
        "name": "Front shin hinge",
        "xy": (0.83, -0.15),
        "color": "#4f9a8b",
        "text_xy": (1.35, -0.74),
        "channels": ["obs[6] angle", "obs[15] angular velocity"],
    },
    {
        "name": "Front foot hinge",
        "xy": (0.48, -0.55),
        "color": "#a84b7a",
        "text_xy": (1.35, -1.30),
        "channels": ["obs[7] angle", "obs[16] angular velocity"],
    },
]


def _draw_segment(ax, start, end, *, linewidth=13, color="#525252") -> None:
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color=color,
        linewidth=linewidth,
        solid_capstyle="round",
        zorder=1,
    )


def _draw_body(ax) -> None:

    torso_a = (-0.70, 0.55)
    torso_b = (0.75, 0.55)
    bthigh = (-0.50, 0.33)
    bshin = (-0.80, -0.15)
    bfoot = (-0.45, -0.55)
    fthigh = (0.55, 0.33)
    fshin = (0.83, -0.15)
    ffoot = (0.48, -0.55)

    _draw_segment(ax, torso_a, torso_b, linewidth=18, color="#5a5f64")
    _draw_segment(ax, (-0.48, 0.50), bthigh, linewidth=10)
    _draw_segment(ax, bthigh, bshin, linewidth=10)
    _draw_segment(ax, bshin, bfoot, linewidth=10)
    _draw_segment(ax, (0.50, 0.50), fthigh, linewidth=10)
    _draw_segment(ax, fthigh, fshin, linewidth=10)
    _draw_segment(ax, fshin, ffoot, linewidth=10)

    for xy in [torso_a, torso_b, bthigh, bshin, bfoot, fthigh, fshin, ffoot]:
        ax.add_patch(Circle(xy, 0.055, facecolor="#f3f3f3", edgecolor="#333333", linewidth=1.0, zorder=2))

    ax.text(0.0, 0.78, "torso", ha="center", va="center", fontsize=10, color="#333333")
    ax.text(-0.95, 0.20, "back leg", ha="center", va="center", fontsize=9, color="#555555", rotation=60)
    ax.text(0.98, 0.20, "front leg", ha="center", va="center", fontsize=9, color="#555555", rotation=-60)


def _draw_sensor_callout(ax, group: dict) -> None:
    x, y = group["xy"]
    tx, ty = group["text_xy"]
    color = group["color"]

    ax.add_patch(Circle((x, y), 0.105, facecolor=color, edgecolor="white", linewidth=1.6, zorder=4))
    ax.add_patch(Circle((x, y), 0.135, facecolor="none", edgecolor=color, linewidth=1.4, zorder=3, alpha=0.55))

    box_text = group["name"] + "\n" + "\n".join(group["channels"])
    ha = "left" if tx > x else "right"
    box_width = 1.34 if "Torso" not in group["name"] else 1.62
    box_height = 0.35 if len(group["channels"]) == 2 else 0.62
    box_x = tx if ha == "left" else tx - box_width
    box_y = ty - box_height / 2
    box = FancyBboxPatch(
        (box_x, box_y),
        box_width,
        box_height,
        boxstyle="round,pad=0.035,rounding_size=0.035",
        facecolor="#ffffff",
        edgecolor=color,
        linewidth=1.4,
        zorder=5,
    )
    ax.add_patch(box)
    ax.text(
        box_x + (0.05 if ha == "left" else box_width - 0.05),
        ty,
        box_text,
        ha=ha,
        va="center",
        fontsize=8,
        color="#222222",
        zorder=6,
        linespacing=1.25,
    )
    anchor_x = box_x if ha == "right" else box_x + box_width
    ax.annotate(
        "",
        xy=(x, y),
        xytext=(anchor_x, ty),
        arrowprops={"arrowstyle": "->", "lw": 1.1, "color": color},
        zorder=4,
    )


def make_figure(output: str) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    ax.set_facecolor("#f7f5ef")
    _draw_body(ax)

    for group in SENSOR_GROUPS:
        _draw_sensor_callout(ax, group)

    ax.text(
        0.0,
        1.72,
        "Physical Interpretation of HalfCheetah Observation Channels",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="#222222",
    )
    ax.text(
        0.0,
        1.50,
        "Joint angle and velocity sensors on each leg, plus torso/root height, pitch, and velocity estimates",
        ha="center",
        va="center",
        fontsize=11,
        color="#444444",
    )

    ax.set_xlim(-2.90, 2.90)
    ax.set_ylim(-1.95, 1.90)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="outputs/figures/halfcheetah_sensor_locations.png",
    )
    args = parser.parse_args()
    make_figure(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
