"""Draw the TP2 pause and communication figure and typeset the rollout table."""

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import median

from plot_project import (
    BLUE,
    INK,
    MUTED,
    ORANGE,
    RULE,
    TEAL,
    box,
    canvas,
    export,
    label,
    plt,
    records,
)

MODES = (
    ("stream-all", "Full broadcast", BLUE),
    ("rank-lanes", "Row shards", ORANGE),
    ("rank-columns", "Row + column", TEAL),
)


def activity(intervals):
    """Partition the sender clock into disjoint spans with one or multiple active calls."""
    events = defaultdict(int)
    for row in intervals:
        start = float(row["start_us"])
        events[start] += 1
        events[start + float(row["duration_us"])] -= 1
    spans = {1: [], 2: []}
    depth, last = 0, 0.0
    for time, change in sorted(events.items()):
        if depth and time > last:
            spans[min(depth, 2)].append((last / 1000, (time - last) / 1000))
        depth += change
        last = time
    return spans


def latency_and_trace(out):
    fig, ax = canvas(400, 398)
    label(
        ax, 8, 15, "Parallel shard transfers shorten the update pause", 17.6, bold=True
    )
    label(ax, 9, 46, "A", 17, BLUE, bold=True)
    label(ax, 32, 46, "Complete update pause", 17, bold=True)
    label(ax, 11, 69, "Five session medians", 13.5, MUTED)
    ax.plot(157, 69, "D", ms=4, color=MUTED)
    label(ax, 166, 69, "Middle of five", 13.5, MUTED)
    label(ax, 390, 69, "Median", 13.5, MUTED, "right")
    left, right = 111, 321

    def pause_x(ms):
        return left + (ms - 500) / 180 * (right - left)

    rows = [r for r in records("acceleration-sessions.csv") if r["tp"] == "2"]
    for t in (500, 550, 600, 650):
        x = pause_x(t)
        ax.plot([x, x], [80, 180], color="#DCE3E5", lw=0.65)
        label(ax, x, 192, str(t), 12, MUTED, "center")
    for (mode, name, color), y in zip(MODES, (93, 129, 165)):
        sample = sorted(
            (r for r in rows if r["mode"] == mode), key=lambda r: int(r["session"])
        )
        values = [float(r["pause_ms"]) for r in sample]
        middle = median(values)
        label(ax, 10, y, name, 14, INK)
        ax.plot([left, right], [y, y], color="#DCE3E5", lw=0.7, zorder=0)
        offsets = [(i - (len(values) - 1) / 2) * 5 for i in range(len(values))]
        for ms, dy in zip(values, offsets):
            ax.plot(
                pause_x(ms),
                y + dy,
                "D" if ms == middle else "o",
                ms=4.2,
                mfc=color if ms == middle else "white",
                mec=color,
                mew=0.95,
            )
        label(ax, 390, y, f"{middle:.1f}", 16, color, "right", True)
    label(ax, (left + right) / 2, 210, "Update pause (ms)", 14, INK, "center")
    ax.plot([9, 391], [228, 228], color=RULE, lw=0.8)

    label(ax, 9, 251, "B", 17, TEAL, bold=True)
    label(ax, 32, 251, "Sender HCCL activity", 17, bold=True)
    single, concurrent = "#E7AE5E", "#006B60"
    box(ax, 12, 268, 12, 9, single, single)
    label(ax, 30, 272, "One active call", 13.5)
    box(ax, 151, 268, 12, 9, concurrent, concurrent)
    label(ax, 169, 272, "Two or more", 13.5)
    label(ax, 390, 272, "Overlap", 13.5, MUTED, "right")
    left, right = 72, 315
    scale = (right - left) / 460
    intervals = records("sender-hccl-relative.csv")
    for y, run in [(302, "2"), (329, "3")]:
        spans = activity(r for r in intervals if r["round"] == run)
        label(ax, 10, y, f"Run {run}", 13.5)
        for depth, color in [(1, single), (2, concurrent)]:
            bars = [(left + a * scale, b * scale) for a, b in spans[depth]]
            ax.broken_barh(
                bars, (y - 6, 12), facecolors=color, linewidths=0, antialiaseds=False
            )
        overlap = sum(span for _, span in spans[2])
        label(ax, 390, y, f"{overlap:.1f} ms", 14.5, TEAL, "right", True)
    ax.plot([left, right], [348, 348], color=MUTED, lw=0.75)
    for t in (0, 100, 200, 300, 400):
        x = left + t * scale
        ax.plot([x, x], [348, 352], color=MUTED, lw=0.75)
        label(ax, x, 364, str(t), 12, MUTED, "center")
    label(
        ax,
        (left + right) / 2,
        385,
        "Time since first communication call (ms)",
        14,
        INK,
        "center",
    )
    export(fig, out, "parallel-handoff")


def rollout_table():
    rows = [r for r in records("rollout-sessions.csv") if r["tp"] == "2"]
    table = [
        "| Update cadence | Broadcast | Shards | Paired gain |",
        "|:---|---:|---:|---:|",
    ]
    for cadence, name in [
        ("1", "Every batch"),
        ("4", "Every four batches"),
        ("0", "Fixed policy"),
    ]:
        rates = []
        for mode in ("stream-all", "rank-columns"):
            rates.append(
                {
                    int(r["session"]): float(r["tok_s"])
                    for r in rows
                    if r["mode"] == mode and r["batches"] == cadence
                }
            )
        baseline, shards = rates
        if baseline.keys() != shards.keys() or len(baseline) != 5:
            raise ValueError(f"Incomplete session pairs at cadence {cadence}")
        gain = "Control"
        if cadence != "0":
            changes = [(shards[k] / baseline[k] - 1) * 100 for k in baseline]
            gain = (
                f"**+{median(changes):.1f}%** ({min(changes):.1f}–{max(changes):.1f}%)"
            )
        table.append(
            f"| {name} | {median(baseline.values()):.1f} | {median(shards.values()):.1f} | {gain} |"
        )
    return "\n".join(table) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("figures"))
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"font.family": "STIXGeneral", "pdf.fonttype": 42, "svg.fonttype": "path"}
    )
    latency_and_trace(out)
    (out / "rollout-table.md").write_text(rollout_table())


if __name__ == "__main__":
    main()
