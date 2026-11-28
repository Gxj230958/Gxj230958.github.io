"""Reduce receiver timelines to unions of observed event intervals per update."""

import argparse
import csv
import json
from pathlib import Path


def merge(intervals):
    result = []
    for a, b in sorted(intervals):
        if result and a <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], b))
        else:
            result.append((a, b))
    return result


def elapsed(events):
    return sum(b - a for a, b in merge(events)) / 1000


def summarize(directory):
    out = []
    for file in sorted(
        directory.glob("round-*/*/ASCEND_PROFILER_OUTPUT/kernel_details.csv")
    ):
        comm, copy = [], []
        with file.open() as f:
            for r in csv.DictReader(f):
                start = float(r["Start Time(us)"])
                interval = (start, start + float(r["Duration(us)"]))
                if r["Accelerator Core"] == "COMMUNICATION":
                    comm.append(interval)
                elif "TensorMove" in r["Name"]:
                    copy.append(interval)
        trace = json.loads(file.with_name("trace_view.json").read_text())
        trace = trace["traceEvents"] if isinstance(trace, dict) else trace
        enqueue, wait = [], []
        for e in trace:
            if e.get("ph") != "X":
                continue
            interval = (float(e["ts"]), float(e["ts"]) + float(e["dur"]))
            if e.get("cat") == "enqueue":
                enqueue.append(interval)
            if e.get("name") in ("EVENT WAIT", "Notify Wait"):
                wait.append(interval)
        overlap = (
            sum(
                max(0, min(b, d) - max(a, c))
                for a, b in merge(comm)
                for c, d in merge(copy)
            )
            / 1000
        )
        if not comm or not copy:
            raise ValueError(f"Missing communication or TensorMove events in {file}")
        out.append(
            {
                "round": int(file.parents[2].name.split("-")[1]),
                "hccl_events": len(comm),
                "copy_events": len(copy),
                "host_enqueue_ms": elapsed(enqueue),
                "device_wait_ms": elapsed(wait),
                "hccl_ms": elapsed(comm),
                "copy_ms": elapsed(copy),
                "overlap_ms": overlap,
            }
        )
    if not out:
        raise ValueError(f"No receiver trace found in {directory}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.write_text(json.dumps(summarize(a.directory), indent=2) + "\n")
