"""Compare paired, independently initialized weight-update and rollout sessions."""

import argparse
import csv
import json
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parent
LABELS = {
    "stream-all": "All-eligible baseline",
    "both-direct": "Direct broadcast",
    "rank-lanes": "Parallel rank links",
    "rank-batch": "Batched rank sends",
    "rank-shards": "Serial rank sends",
    "rank-columns": "Row + column shards",
}


def session(record, name):
    short = [r for r in record["rounds"] if not r["warmup"]]
    base = {
        "session": name,
        "cards": record["cards"],
        "mode": record["settings"]["mode"],
        "tp": record["settings"]["tp"],
        "blocks": record["blocks"],
    }
    row = dict(
        base,
        pause_ms=median(r["pause_ms"] for r in short),
        p95_ms=np.percentile([r["pause_ms"] for r in short], 95),
        drain_ms=median(r["drain_ms"] for r in short),
        transfer_ms=median(r["transfer_ms"] for r in short),
        resume_ms=median(r["resume_ms"] for r in short),
        sender_mib=max(r["sender_peak_bytes"] for r in short) / 2**20,
        receiver_mib=max(max(r["extra_peak"]) for r in short) / 2**20,
        short_tok_s=1000
        * sum(r["tokens"] for r in short)
        / sum(r["cycle_ms"] for r in short),
    )
    roll = []
    for k in (1, 4, 0):
        vals = [r for r in record["rollouts"] if r["batches_per_update"] == k]
        assert {r["version"] for r in vals} == {"instruct", "base"}
        assert all(
            r["cold_sentinel_match"] and r["tokens"] == 2048 * (k or 2) for r in vals
        )
        duration = sum(r["cycle_ms"] for r in vals)
        roll.append(
            dict(
                base,
                batches=k,
                tok_s=1000 * sum(r["tokens"] for r in vals) / duration,
                first_token_ms=median(r["first_token_ms"] for r in vals),
                pause_fraction=sum(r["pause_ms"] for r in vals) / duration,
                divergent_sequences=sum(r["divergent_sequences"] for r in vals),
                sequences=sum(r["tokens"] // 128 for r in vals),
            )
        )
    return row, roll


def table(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare(rows, key, baseline, candidate, decrease=False):
    paired = []
    for i in range(1, 6):
        a, b = (
            next(r[key] for r in rows if r["mode"] == m and r["session"] == i)
            for m in (baseline, candidate)
        )
        paired.append(100 * (1 - b / a if decrease else b / a - 1))
    return {
        "median": median(paired),
        "low": min(paired),
        "high": max(paired),
        "pairs": paired,
    }


def make_figures(rows, roll, comparisons, dest, wires):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for path in (ROOT / "fonts").glob("*.otf"):
        font_manager.fontManager.addfont(path)
    plt.rcParams.update(
        {
            "font.family": "Latin Modern Roman"
            if list((ROOT / "fonts").glob("*.otf"))
            else "STIXGeneral",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "path",
        }
    )
    blue, orange, green = "#0072B2", "#BE6500", "#008169"
    fig, axes = plt.subplots(
        1, 2, figsize=(8.2, 3.0), layout="constrained", sharey=True
    )
    for ax, (tp, modes) in zip(axes, comparisons.items()):
        vals = [r for r in rows if r["tp"] == tp]
        for x, m in enumerate(modes):
            rs = [r for r in vals if r["mode"] == m]
            base = 0
            for key, label, color, h in [
                ("transfer_ms", "Weight update", blue, ""),
                ("drain_ms", "Pause & drain", orange, "//"),
                ("resume_ms", "Resume", green, ".."),
            ]:
                v = median(r[key] for r in rs)
                ax.bar(
                    x,
                    v,
                    bottom=base,
                    color=color,
                    hatch=h,
                    width=0.46,
                    label=label if x == 0 else None,
                )
                base += v
            ax.text(
                x, base + 12, f"{median(r['pause_ms'] for r in rs):.0f} ms", ha="center"
            )
        w = wires[tp]
        ax.axhline(w, ls="--", color="#42494f", lw=1.1, label="Broadcast only")
        ax.set_xticks([0, 1], ["Baseline", LABELS[modes[1]].replace(" ", "\n", 1)])
        ax.set_title(f"TP{tp} · {tp + 1} NPUs", loc="left", weight="bold")
        ax.set_ylim(0, 1050)
        ax.set_xlim(-0.55, 1.55)
    axes[0].set_ylabel("Update pause (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    for ext in ("svg", "pdf"):
        fig.savefig(dest / f"updatephases.{ext}", dpi=360)
    fig.legends[0].remove()
    focus = axes[1]
    focus.tick_params(labelleft=True)
    focus.set_ylabel("Update pause (ms)")
    focus.legend(loc="upper right", fontsize=7, ncol=2, frameon=False)
    fig.canvas.draw()
    box = (
        focus.get_tightbbox(fig.canvas.get_renderer())
        .transformed(fig.dpi_scale_trans.inverted())
        .padded(0.05)
    )
    fig.savefig(dest / "updatephases_tp2.svg", bbox_inches=box)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0), layout="constrained")
    for ax, (tp, modes) in zip(axes, comparisons.items()):
        for m, off, color, mark in zip(
            modes, [-0.10, 0.10], [blue, orange], ["o", "s"]
        ):
            for x, k in enumerate([1, 4, 0]):
                ys = [
                    r["tok_s"]
                    for r in roll
                    if r["mode"] == m and r["tp"] == tp and r["batches"] == k
                ]
                mid = median(ys)
                ax.errorbar(
                    x + off,
                    mid,
                    yerr=[[mid - min(ys)], [max(ys) - mid]],
                    color=color,
                    marker=mark,
                    capsize=3,
                    ls="none",
                    label=LABELS[m] if x == 0 else None,
                )
        ax.set_title(f"TP{tp} · 16 concurrent sequences", loc="left", weight="bold")
        ax.set_xticks([0, 1, 2], ["Every batch", "Every 4", "Fixed policy"])
        ax.set_xlim(-0.45, 2.45)
        ymax = max(r["tok_s"] for r in roll if r["mode"] in modes and r["tp"] == tp)
        ax.set_ylim(0, 1.20 * ymax)
        for x, k in enumerate((1, 4)):
            selected = [r for r in roll if r["tp"] == tp and r["batches"] == k]
            effect = compare(selected, "tok_s", *modes)["median"]
            ax.text(
                x,
                1.06 * ymax,
                f"{effect:+.1f}%",
                ha="center",
                fontsize=9,
                color="#3a4147",
            )
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    axes[0].set_ylabel("Completed output tokens / s")
    for ext in ("svg", "pdf"):
        fig.savefig(dest / f"rolloutfrequency.{ext}", dpi=360)
    focus = axes[1]
    focus.set_ylabel("Completed output tokens / s")
    fig.canvas.draw()
    box = (
        focus.get_tightbbox(fig.canvas.get_renderer())
        .transformed(fig.dpi_scale_trans.inverted())
        .padded(0.05)
    )
    fig.savefig(dest / "rolloutfrequency_tp2.svg", bbox_inches=box)
    plt.close(fig)

    baseline, candidate = comparisons[2]
    paired = compare(
        [r for r in rows if r["tp"] == 2], "pause_ms", baseline, candidate, True
    )
    modes = [baseline, candidate]
    med = [
        median(r["pause_ms"] for r in rows if r["mode"] == m and r["tp"] == 2)
        for m in modes
    ]
    fig = plt.figure(figsize=(3.5, 2.33), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(
        0.05,
        0.93,
        f"{paired['median']:.1f}% shorter",
        va="top",
        size=23,
        color="#006E5E",
        weight="bold",
    )
    ax.text(0.05, 0.70, "TP2 weight-update pause", size=14, color="#172D3B")
    for y, v, label, col in [
        (0.45, med[0], "Broadcast", blue),
        (0.18, med[1], "Rank shards", green),
    ]:
        ax.text(0.05, y + 0.055, label, fontsize=11, color="#172D3B")
        ax.text(
            0.95,
            y + 0.055,
            f"{v:.0f} ms",
            fontsize=11,
            ha="right",
            weight="bold",
            color=col,
        )
        ax.barh(y, v / max(med) * 0.90, left=0.05, height=0.045, color=col)
    for ext in ("svg", "pdf"):
        fig.savefig(dest / f"rolloutcover.{ext}", dpi=360)
    plt.close(fig)
    for name in (
        "updatephases",
        "rolloutfrequency",
        "updatephases_tp2",
        "rolloutfrequency_tp2",
        "rolloutcover",
    ):
        path = dest / (name + ".svg")
        path.write_text(
            "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
        )


def read_numeric_table(path):
    with path.open() as f:
        return [
            {k: (v if k in ("mode", "cards") else float(v)) for k, v in row.items()}
            for row in csv.DictReader(f)
        ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "acceleration")
    ap.add_argument("--output", type=Path, default=ROOT / "figures")
    ap.add_argument(
        "--from-tables", type=Path, help="Redraw from exported session and wire CSVs"
    )
    args = ap.parse_args()
    if args.from_tables:
        t = args.from_tables
        args.output.mkdir(parents=True, exist_ok=True)
        wires = {
            r["tp"]: r["median_ms"]
            for r in read_numeric_table(t / "wire-reference.csv")
        }
        make_figures(
            read_numeric_table(t / "acceleration-sessions.csv"),
            read_numeric_table(t / "rollout-sessions.csv"),
            {1: ("stream-all", "both-direct"), 2: ("stream-all", "rank-columns")},
            args.output,
            wires,
        )
        return
    rows, roll, summary, comparisons = [], [], {}, {}
    for tp in (1, 2):
        jobs = json.loads(
            (ROOT / f"configs/acceleration-formal-tp{tp}.json").read_text()
        )
        if tp == 2:
            jobs += json.loads(
                (ROOT / "configs/acceleration-formal-columns.json").read_text()
            )
        methods = []
        for _, tag, _ in jobs:
            name = tag.rsplit("/", 1)[1]
            d = json.loads((args.data / (name + ".json")).read_text())
            assert d["status"] == "passed", name
            if d["settings"]["mode"] not in methods:
                methods.append(d["settings"]["mode"])
            s, r = session(d, int(name.rsplit("-s", 1)[1]))
            rows.append(s)
            roll.extend(r)
        assert len(jobs) == 5 * len(methods) and methods[0] == "stream-all"
        comparisons[tp] = [methods[0], methods[-1]]
        rs = [r for r in rows if r["tp"] == tp]
        out = {}
        for candidate in methods[1:]:
            result = {}
            for metric in ["pause_ms", "p95_ms", "short_tok_s"]:
                result[metric] = compare(
                    rs, metric, methods[0], candidate, decrease=metric != "short_tok_s"
                )
            for k in (1, 4, 0):
                selected = [r for r in roll if r["tp"] == tp and r["batches"] == k]
                result[f"throughput_{k}"] = compare(
                    selected, "tok_s", methods[0], candidate
                )
                result[f"baseline_update_fraction_{k}"] = median(
                    r["pause_fraction"] for r in selected if r["mode"] == methods[0]
                )
            out[candidate] = result
        summary[tp] = out
    args.output.mkdir(parents=True, exist_ok=True)
    table(args.output / "acceleration-sessions.csv", rows)
    table(args.output / "rollout-sessions.csv", roll)
    (args.output / "acceleration-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    controls = []
    for name in (
        "ablation-send-direct",
        "ablation-receive-direct",
        "ablation-both-direct",
        "cal-tp1-all",
        "cal-tp1-native",
        "cal-tp1-threshold",
        "cal-tp2-all",
        "cal-tp2-native",
        "cal-tp2-threshold",
        "layout-rank-lanes-16",
        "layout-rank-lanes-128",
        "layout-rank-columns-16",
        "layout-rank-columns-128",
    ):
        d = json.loads((args.data / (name + ".json")).read_text())
        assert d["status"] == "passed", name
        rs = [r for r in d["rounds"] if not r["warmup"]]
        controls.append(
            {
                "experiment": name,
                "placement": d["cards"],
                "cycles": len(rs),
                "median_pause_ms": median(r["pause_ms"] for r in rs),
            }
        )
    table(args.output / "transport-controls.csv", controls)
    caps = []
    for p in sorted(args.data.glob("capacity-*.json")):
        d = json.loads(p.read_text())
        caps.append(
            {
                "probe": p.stem,
                "mode": d["settings"]["mode"],
                "placement": d["cards"],
                "kv_blocks": d["blocks"],
                "status": d["status"],
                "phase": d["phase"],
                "completed_cycles": len(d["rounds"]),
            }
        )
    table(args.output / "capacity-rechecks.csv", caps)
    wires = []
    for tp in (1, 2):
        d = json.loads((args.data / f"cal-tp{tp}-native.json").read_text())["wire"]
        wires.append(
            {
                "tp": tp,
                "checkpoint_bytes": d["checkpoint_bytes"],
                "messages": d["messages"],
                "median_ms": median(d["ms"]),
                "min_ms": min(d["ms"]),
                "max_ms": max(d["ms"]),
            }
        )
    table(args.output / "wire-reference.csv", wires)
    make_figures(
        rows, roll, comparisons, args.output, {r["tp"]: r["median_ms"] for r in wires}
    )


if __name__ == "__main__":
    main()
