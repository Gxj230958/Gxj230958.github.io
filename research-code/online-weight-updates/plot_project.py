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


def arrow(ax, x1, x2, y, color=MUTED):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y), (x2, y), arrowstyle="-|>", mutation_scale=10, lw=1.05, color=color
        )
    )


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
    fig, ax = canvas(420, 284)
    label(ax, 10, 13, "One vocabulary matrix, identical HCCL broadcasts", 17, bold=True)
    # Draw each on-wire fragment to scale, including the short final fragment.
    left, width, focus = 11, 233, 2
    fraction = slot_mib / tensor_mib
    for idx in range(ceil(tensor_mib / slot_mib)):
        offset = idx * fraction
        box(
            ax,
            left + offset * width,
            34,
            min(fraction, 1 - offset) * width,
            18,
            BLUE if idx == focus else PALE_BLUE,
            BLUE,
        )
    label(ax, 250, 43, f"≤ {slot_mib:g} MiB / broadcast", 14, BLUE)
    label(ax, 11, 65, f"{tensor_mib:,.1f} MiB BF16 input embedding", 13, MUTED)
    ax.plot([10, 410], [80, 80], color=RULE, lw=0.8)

    for y, name, assemble in [
        (98, "Upstream PR", True),
        (200, "Independent study", False),
    ]:
        label(ax, 10, y, name, 17, bold=True)
        middle = y + 27
        box(ax, 13, middle - 7, 60, 16, PALE_BLUE, BLUE)
        box(ax, 15, middle - 5, 56, 12, BLUE, BLUE)
        label(ax, 43, y + 53, "Receive slot", 13.5, BLUE, "center")

        if assemble:
            arrow(ax, 76, 142, middle)
            box(ax, 144, middle - 8, 101, 18, PALE_ORANGE, ORANGE, "//")
            box(
                ax,
                144 + focus * fraction * 101,
                middle - 8,
                fraction * 101,
                18,
                BLUE,
                BLUE,
            )
            arrow(ax, 248, 300, middle)
            label(ax, 195, y + 53, "Reassemble input", 13.5, ORANGE, "center")
            label(
                ax,
                195,
                y + 71,
                f"{tensor_mib:,.1f} MiB temporary",
                13,
                ORANGE,
                "center",
            )
        else:
            arrow(ax, 76, 300, middle, TEAL)
            label(ax, 188, y + 53, "Load-plan byte range", 13.5, TEAL, "center")

        box(ax, 302, middle - 8, 103, 18, PALE_TEAL, TEAL)
        box(
            ax, 302 + focus * fraction * 103, middle - 8, fraction * 103, 18, BLUE, BLUE
        )
        label(ax, 354, y + 53, "Live weight", 13.5, TEAL, "center")
    ax.plot([10, 410], [182, 182], color=RULE, lw=0.8)
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
