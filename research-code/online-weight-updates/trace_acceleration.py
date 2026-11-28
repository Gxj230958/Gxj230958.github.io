"""Summarize sender and receiver activity in separate full-update profiles."""

import argparse
import csv
import json
from decimal import Decimal
from pathlib import Path

from profile_summary import elapsed


def summarize(file):
    with file.open() as f:
        kernels = list(csv.DictReader(f))
    streams = []
    moves = []
    for kernel in kernels:
        start = float(kernel["Start Time(us)"])
        span = (start, start + float(kernel["Duration(us)"]))
        if kernel["Accelerator Core"] == "COMMUNICATION":
            streams.append(span)
        elif "TensorMove" in kernel["Name"]:
            moves.append(span)
    if not streams:
        raise ValueError(f"No HCCL activity in {file}")
    timeline = []
    for start, end in streams:
        timeline.extend([(start, 1), (end, -1)])
    depth = 0
    previous = 0
    overlap = 0
    for when, change in sorted(timeline):
        if depth >= 2:
            overlap += when - previous
        previous = when
        depth += change
    device = int(kernels[0]["Device_id"])
    return {
        "device": device,
        "round": int(file.parents[2].name.split("-")[1]),
        "hccl_calls": len(streams),
        "hccl_union_ms": elapsed(streams),
        "concurrent_hccl_ms": overlap / 1000,
        "copy_union_ms": elapsed(moves),
        "copy_hccl_overlap_ms": elapsed(streams)
        + elapsed(moves)
        - elapsed(streams + moves),
    }


def export_sender_intervals(files, output):
    """Write only per-round relative HCCL kernel times from the TP2 sender."""
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["round", "start_us", "duration_us"])
        for file in files:
            with file.open() as source:
                calls = sorted(
                    (Decimal(r["Start Time(us)"]), Decimal(r["Duration(us)"]))
                    for r in csv.DictReader(source)
                    if r["Accelerator Core"] == "COMMUNICATION"
                )
            origin = calls[0][0]
            run = int(file.parents[2].name.split("-")[1])
            writer.writerows(
                (run, start - origin, duration) for start, duration in calls
            )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    result = {}
    for mode, workers in [("stream-all", 1), ("both-direct", 1), ("rank-columns", 2)]:
        result[mode] = {}
        for side in ["sender", "receiver"]:
            root = args.directory / mode
            if side == "sender":
                root = root / "sender"
            files = sorted(
                root.glob("round-*/*/ASCEND_PROFILER_OUTPUT/kernel_details.csv")
            )
            if len(files) != (2 if side == "sender" else workers * 2):
                raise ValueError(f"Incomplete {side} export for {mode}")
            if mode == "rank-columns" and side == "sender":
                export_sender_intervals(
                    files, args.output.with_name("sender-hccl-relative.csv")
                )
            result[mode][side] = [summarize(f) for f in files]
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
