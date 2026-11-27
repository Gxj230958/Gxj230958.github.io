"""Plot per-tier device coverage from the benchmark campaign's task-results.md."""

import argparse
import re
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    totals = Counter()
    complete = Counter()
    for line in args.records.read_text().splitlines():
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        match = re.fullmatch(r"(L[123])_\d+_\w+", cells[0])
        if match:
            tier = match[1]
            totals[tier] += 1
            complete[tier] += cells[4] == "complete_verified"

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 13,
        "text.color": "#302c27",
        "xtick.color": "#514839",
        "axes.labelcolor": "#302c27",
    })
    fig, ax = plt.subplots(figsize=(6.5, 3.25), facecolor="#fcfaf5")
    fig.subplots_adjust(left=.25, right=.80, top=.95, bottom=.28)
    ax.set_facecolor("#fcfaf5")
    labels = ["Single op", "Fused op", "Multi-stage"]
    for row, (tier, label) in enumerate(zip(("L1", "L2", "L3"), labels)):
        passes, tasks = complete[tier], totals[tier]
        ax.barh(row, 100, height=.36, color="#e7dfd0", zorder=1)
        ax.barh(row, 100 * passes / tasks, height=.36, color="#a34b2b", zorder=2)
        ax.text(-.41, row-.18, tier, transform=ax.get_yaxis_transform(),
                fontweight="bold", va="center", fontsize=16)
        ax.text(-.41, row+.20, label, transform=ax.get_yaxis_transform(),
                va="center", fontsize=15)
        ax.text(1.05, row, f"{passes} / {tasks}", transform=ax.get_yaxis_transform(),
                va="center", fontweight="bold", fontsize=16)
    ax.set(xlim=(0, 100), ylim=(2.53, -.47), yticks=[],
           xticks=[0, 25, 50, 75, 100], xlabel="Tasks with full device coverage (%)")
    ax.xaxis.labelpad = 8
    ax.xaxis.label.set_size(15)
    ax.tick_params(axis="x", labelsize=14, length=3, pad=4)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#aa9c87")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "device-results.svg", metadata={"Date": None})
    fig.savefig(args.output_dir / "device-results.pdf")
    fig.savefig(args.output_dir / "device-results.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
