"""Rebuild manuscript figures from the released aggregate CSVs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


INK = "#263238"
BLUE = "#23609c"
ORANGE = "#c2782f"
GOLD = "#b59231"
GREY = "#77838b"
LIGHT = "#d7dde1"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save(fig, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(target.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def box(ax, xy, wh, label, color=BLUE):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                           facecolor="white", edgecolor=color, linewidth=1.6)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            color=INK, fontsize=10, linespacing=1.2)


def arrow(ax, start, end, color=GREY):
    ax.annotate("", xy=end, xytext=start,
                arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.4,
                            "shrinkA": 0, "shrinkB": 0})


def architecture(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 4.3))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 4.7)
    ax.axis("off")
    box(ax, (.3, 2.65), (1.75, .9), "Synthetic thermal\nplant", BLUE)
    box(ax, (2.5, 2.65), (1.75, .9), "Sparse probes +\ncausal observer", BLUE)
    box(ax, (4.7, 2.65), (1.75, .9), "CEM-MPC / PID\ncontroller", BLUE)
    box(ax, (6.9, 2.65), (1.9, .9), "Command checks +\none-cycle hold", BLUE)
    box(ax, (4.7, .55), (1.75, .9), "ROM / FNO / PINO\ncandidate predictor", ORANGE)
    box(ax, (7.0, .55), (1.7, .9), "Independent PID\nfallback", GOLD)
    arrow(ax, (2.05, 3.10), (2.5, 3.10))
    arrow(ax, (4.25, 3.10), (4.7, 3.10))
    arrow(ax, (6.45, 3.10), (6.9, 3.10))
    arrow(ax, (5.575, 1.45), (5.575, 2.65), ORANGE)
    arrow(ax, (7.85, 1.45), (7.85, 2.65), GOLD)
    ax.plot([8.80, 10.50, 10.50, 1.17, 1.17],
            [3.10, 3.10, 4.22, 4.22, 3.58], color=GREY, lw=1.4)
    arrow(ax, (1.17, 3.95), (1.17, 3.58))
    ax.text(9.5, 3.24, "applied power", ha="center", fontsize=9, color=GREY)
    ax.text(5.8, 4.33, "next measured state", ha="center", fontsize=9, color=GREY)
    ax.text(.3, .12, "Scoring plant state stays behind the observation boundary.",
            fontsize=9, color=GREY)
    save(fig, output / "architecture")


def control_differences(analysis: Path, output: Path) -> None:
    rows = read(analysis / "scenario_metrics.csv")
    values = {(r["scenario_id"], r["controller"]): float(r["seed_mean"])
              for r in rows if r["metric"] == "rmse_K"}
    scenarios = [f"N{i:02d}" for i in range(1, 7)] + [f"S{i:02d}" for i in range(1, 5)] + ["F01", "F02"]
    x = np.arange(len(scenarios))
    fig, ax = plt.subplots(figsize=(10.7, 4.2))
    for label, comparator, color, marker, offset in (
        ("PINO − data-only FNO", "B2_DATA_NO_MPC", BLUE, "o", -.11),
        ("PINO − physical ROM", "B1_ROM_CEM", ORANGE, "s", .11),
    ):
        diff = [values[(sid, "B3_PINO_MPC")] - values[(sid, comparator)] for sid in scenarios]
        ax.scatter(x + offset, diff, marker=marker, s=37, color=color, label=label, zorder=3)
    ax.axhline(0, color=INK, lw=1)
    ax.axvline(5.5, color=LIGHT, lw=1)
    ax.axvline(9.5, color=LIGHT, lw=1)
    ax.set_xticks(x, scenarios)
    ax.set_ylabel("Paired episode field RMSE difference (K)", color=INK)
    ax.set_ylim(-1, 21)
    ax.set_xlim(-.6, 11.6)
    ax.text(2.5, 20.4, "Nominal (6)", ha="center", fontsize=9, color=GREY)
    ax.text(7.5, 20.4, "Material shift (4)", ha="center", fontsize=9, color=GREY)
    ax.text(10.5, 20.4, "Sensor stress (2)", ha="center", fontsize=9, color=GREY)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.13),
              ncol=2, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=LIGHT, lw=.7)
    fig.tight_layout(rect=(0, .08, 1, 1))
    save(fig, output / "control_paired_rmse")


def prediction(analysis: Path, output: Path) -> None:
    rows = read(analysis / "prediction_trajectory_metrics.csv")
    group = {}
    for r in rows:
        key = (int(r["trajectory_seed"]), r["variant"])
        group.setdefault(key, []).append(float(r["field_rmse_K"]))
    seeds = sorted({seed for seed, _ in group})
    names = ["ROM", "data_only", "physics_informed"]
    labels = ["Physical ROM", "Data-only FNO", "PINO"]
    colors = [GREY, BLUE, ORANGE]
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for seed in seeds:
        y = [float(np.mean(group[(seed, name)])) for name in names]
        ax.plot(range(3), y, color=LIGHT, lw=1, zorder=1)
        for j, value in enumerate(y):
            ax.scatter(j, value, s=18, facecolor="white", edgecolor=colors[j],
                       linewidth=1.1, zorder=2)
    for j, name in enumerate(names):
        mean = float(np.mean([np.mean(group[(seed, name)]) for seed in seeds]))
        ax.scatter(j, mean, marker="D", s=68, color=colors[j], zorder=4)
    ax.set_xticks(range(3), labels)
    ax.set_ylabel("12-step field forecast RMSE (K)", color=INK)
    ax.set_ylim(0, 1.65)
    ax.set_xlim(-.35, 2.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=LIGHT, lw=.7)
    fig.tight_layout()
    save(fig, output / "prediction_field_rmse")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    architecture(args.output)
    control_differences(args.analysis_dir, args.output)
    prediction(args.analysis_dir, args.output)
    print(f"Saved three figures as PDF and PNG under {args.output}")
