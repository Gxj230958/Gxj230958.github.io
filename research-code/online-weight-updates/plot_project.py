"""Draw the project card, receiver diagram and comparison panels from the public tables."""

import argparse
import csv
from math import ceil
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parent
MIB = 2**20
INK = "#172B39"
MUTED = "#475D6A"
BLUE = "#006DA8"
ORANGE = "#A9470A"
TEAL = "#006E5C"
PALE_BLUE = "#DCEEF8"
PALE_ORANGE = "#FCE9D6"
PALE_TEAL = "#D6F0E8"
RULE = "#B8C5CA"


def records(name):
    with (ROOT / "results" / name).open(newline="") as f:
        return list(csv.DictReader(f))


def canvas(width, height):
    fig = plt.figure(figsize=(width / 72, height / 72))
    ax = fig.add_axes([0, 0, 1, 1])
    frame(ax, width, height)
    return fig, ax


def frame(ax, width, height):
    ax.set(xlim=(0, width), ylim=(height, 0), aspect="equal")
    ax.axis("off")


def label(ax, x, y, text, size, color=INK, ha="left", bold=False):
    return ax.text(
        x,
        y,
        text,
        fontsize=size,
        color=color,
        ha=ha,
        va="center",
        weight="bold" if bold else "normal",
    )


def box(ax, x, y, w, h, fill, edge, hatch=None):
    ax.add_patch(Rectangle((x, y), w, h, fc=fill, ec=edge, lw=0.85, hatch=hatch))


def export(fig, out, stem):
    for kind in ("svg", "pdf"):
        path = out / f"{stem}.{kind}"
        fig.savefig(path, facecolor="white")
        if kind == "svg":
            lines = path.read_text().splitlines()
            path.write_text("\n".join(line.rstrip() for line in lines) + "\n")
    plt.close(fig)


def cover(out, end):
    fig, ax = canvas(252, 168)
    native, stream = end
    gain = int(stream["largest_observed_pass_tokens"]) - int(
        native["largest_observed_pass_tokens"]
    )
    label(ax, 11, 29, f"+{gain:,}", 34, TEAL, bold=True)
    label(ax, 12, 57, "KV tokens at the update limit", 17)
    ax.plot([12, 240], [75, 75], color=RULE, lw=0.8)
    for y, row, name, symbol, color in [
        (103, native, "Per-parameter", "o", ORANGE),
        (139, stream, "Vocabulary, K=1", "s", TEAL),
    ]:
        ax.plot(17, y, marker=symbol, ms=5.6, color=color)
        label(ax, 28, y, name, 14.7)
        label(
            ax,
            240,
            y,
            f"{int(row['largest_observed_pass_tokens']):,}",
            17,
            color,
            "right",
            True,
        )
    export(fig, out, "cover")


def receiver(out, slot_mib, tensor_mib):
    fig, ax = canvas(400, 347)
    fraction = slot_mib / tensor_mib
    focus = 4

    def strip(x, y, width, height, fill, edge, divided=False):
        box(ax, x, y, width, height, fill, edge)
        if divided:
            for n in range(1, ceil(tensor_mib / slot_mib)):
                xx = x + n * fraction * width
                ax.plot([xx, xx], [y, y + height], color=edge, lw=0.8)
        start = x + focus * fraction * width
        box(ax, start, y, fraction * width, height, PALE_BLUE, BLUE, "////")
        return start + fraction * width / 2

    def route(points, color, weight):
        xs, ys = zip(*points[:-1])
        ax.plot(xs, ys, color=color, lw=weight, solid_capstyle="butt")
        ax.add_patch(
            FancyArrowPatch(
                points[-2],
                points[-1],
                arrowstyle="-|>",
                shrinkA=0,
                shrinkB=0.5,
                mutation_scale=11,
                color=color,
                lw=weight,
            )
        )

    label(ax, 8, 18, "Where does a received weight fragment go?", 20, bold=True)
    label(ax, 12, 53, "Trainer", 16, bold=True)
    label(ax, 388, 53, f"{tensor_mib:,.1f} MiB vocabulary matrix", 14, MUTED, "right")
    source = strip(13, 68, 374, 22, "#EDF2F4", MUTED, divided=True)
    route([(source, 91), (source, 123)], BLUE, 1.6)
    label(ax, source + 12, 106, "HCCL broadcast", 13.5, BLUE)
    box(ax, source - 23, 124, 46, 22, PALE_BLUE, BLUE, "////")
    label(ax, source - 32, 135, "Receive slot", 16, INK, "right", True)
    label(ax, source + 32, 135, f"≤ {slot_mib:g} MiB", 14, MUTED)

    left, right = 94, 302
    ax.plot([source, source], [146, 162], color=INK, lw=1.3)
    route([(source, 162), (left, 162), (left, 178)], MUTED, 1.3)
    route([(source, 162), (right, 162), (right, 178)], TEAL, 2.0)
    label(ax, left, 191, "Reassemble, then copy", 15.5, MUTED, "center", True)
    label(ax, right, 191, "Write to the final offset", 15.5, TEAL, "center", True)
    temp = strip(16, 207, 157, 21, "#F6EEE4", "#896845")
    label(ax, left, 243, f"{tensor_mib:,.1f} MiB temporary", 14, "#79532E", "center")
    label(ax, right, 218, "Load-plan byte range", 14, TEAL, "center")
    # The copy installs the assembled input. The direct write targets one range.
    dest = strip(16, 287, 368, 24, PALE_TEAL, TEAL)
    route([(temp, 254), (temp, 271), (temp, 286)], MUTED, 1.3)
    route([(right, 230), (right, 264), (dest, 264), (dest, 286)], TEAL, 2.0)
    label(ax, 16, 329, "Live model weight", 16, bold=True)
    label(ax, 385, 329, "Existing device storage", 14, MUTED, "right")
    export(fig, out, "receiver")


PW, PH = 240, 187


def draw_memory(ax, peaks):
    frame(ax, PW, PH)
    label(ax, 7, 13, "A   Receiver allocation", 16.5, bold=True)
    label(ax, 7, 35, "Whole-cycle peak above baseline (MiB)", 12.4, MUTED)
    left, right, maximum = 8, 229, 1400
    for y, val, name, color, pattern in [
        (60, peaks[0], "Bounded reassembly", ORANGE, "//"),
        (106, peaks[1], "Vocabulary streaming", TEAL, None),
    ]:
        label(ax, left, y, name, 13.6)
        label(ax, right, y, f"{val:,.1f}", 13.8, color, "right", True)
        box(
            ax,
            left,
            y + 10,
            (right - left) * val / maximum,
            12,
            PALE_ORANGE if pattern else TEAL,
            color,
            pattern,
        )
    ax.plot([left, right], [144, 144], color=MUTED, lw=0.75)
    for value in [0, 700, 1400]:
        x = left + (right - left) * value / maximum
        ax.plot([x, x], [144, 148], color=MUTED, lw=0.75)
        label(
            ax,
            x,
            159,
            f"{value:,}",
            11.7,
            MUTED,
            "left" if value == 0 else "right" if value == maximum else "center",
        )
    label(
        ax,
        8,
        179,
        f"{peaks[0] - peaks[1]:,.0f} MiB less with two slots",
        13.2,
        TEAL,
        bold=True,
    )


def draw_capacity(ax, end):
    frame(ax, PW, PH)
    label(ax, 7, 13, "B   Update boundary", 16.5, bold=True)
    label(ax, 7, 35, "Allocated KV blocks · 128 tokens / block", 12.4, MUTED)
    label(ax, 122, 61, "Last pass", 12.4, MUTED, "center")
    label(ax, 197, 61, "Next OOM", 12.4, MUTED, "center")
    for y, name, row in [
        (86, "Per-parameter", end[0]),
        (125, "Vocabulary, K=1", end[1]),
    ]:
        label(ax, 7, y, name, 12.4)
        for x, field, mark, color in [
            (101, "largest_observed_pass_blocks", "o", TEAL),
            (176, "first_observed_oom_blocks", "x", ORANGE),
        ]:
            ax.plot(x, y, marker=mark, ms=4.8, mew=1.2, color=color)
            label(ax, x + 7, y, f"{int(row[field]):,}", 14.2, color, bold=True)
    ax.plot([8, 229], [145, 145], color=RULE, lw=0.75)
    gain = int(end[1]["largest_observed_pass_blocks"]) - int(
        end[0]["largest_observed_pass_blocks"]
    )
    label(ax, 8, 179, f"{gain:,} more blocks with one slot", 13.2, TEAL, bold=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("figures"))
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "svg.fonttype": "path",
            "pdf.fonttype": 42,
            "hatch.linewidth": 0.6,
            "savefig.dpi": 360,
        }
    )
    capacities = {r["receiver"]: r for r in records("capacity.csv")}
    end = [capacities["per-parameter"], capacities["vocabulary K=1"]]
    sessions = records("matched-peaks.csv")
    peaks = [
        median(int(r["cycle_peak_bytes"]) / MIB for r in sessions if r["mode"] == mode)
        for mode in ("bounded", "stream")
    ]
    tensors = {
        r["tensor"]: int(r["nbytes"]) / MIB for r in records("receiver-layout.csv")
    }
    cover(out, end)
    receiver(out, int(sessions[0]["bucket_mib"]), tensors["model.embed_tokens.weight"])
    for stem, draw, data in [
        ("receiver-peak", draw_memory, peaks),
        ("capacity-boundary", draw_capacity, end),
    ]:
        fig, ax = canvas(PW, PH)
        draw(ax, data)
        export(fig, out, stem)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=((2 * PW + 12) / 72, PH / 72),
        gridspec_kw={"wspace": 12 / PW, "left": 0, "right": 1, "top": 1, "bottom": 0},
    )
    draw_memory(axes[0], peaks)
    draw_capacity(axes[1], end)
    export(fig, out, "measured-results")


if __name__ == "__main__":
    main()
