"""Session admission for BF16 Qwen2, calibrated on 7B TP1 (milliseconds per update)."""

import math
from pathlib import Path

from inventory import inventory
from streaming import VOCAB

MIB = 2**20
# Second update in each TP1 preflight, after the first reload plan has been built.
CALIBRATION_MS = {
    ("stream", 2, 128): 902,
    ("unpacked", 1, 128): 956,
    ("stream", 1, 128): 1018,
}


def predict_pause_ms(calibration_ms, weight_bytes):
    # The pinned AsyncLLM wait barrier sleeps for 20 ms in every transaction.
    return 20 + (calibration_ms - 20) * weight_bytes / 15_231_233_024


def select(checkpoint: Path, budget):
    tensors, _ = inventory(checkpoint)
    sizes = {name: value["nbytes"] for name, value in tensors.items()}
    candidates = []
    for (mode, slots, bucket), latency in CALIBRATION_MS.items():
        if mode == "unpacked":
            # The first embedding can overlap the next Q projection on the receiver.
            peak = max(
                max(sizes.values()),
                sizes["model.embed_tokens.weight"]
                + sizes["model.layers.0.self_attn.q_proj.weight"],
            )
            threshold = 0
        else:
            room = budget - slots * bucket * MIB
            selected = {name for name in VOCAB if sizes.get(name, 0) > room}
            peak = slots * bucket * MIB + max(
                (v for n, v in sizes.items() if n not in selected), default=0
            )
            threshold = max(0, math.floor(room) + 1)
            if not selected:
                # The no-direct-write case uses the native bounded path.
                mode = "bounded"
                latency = 955 if slots == 2 else 1025
        candidates.append(
            {
                "mode": mode,
                "slots": slots,
                "bucket_mib": bucket,
                "predicted_peak_bytes": peak,
                "threshold_bytes": threshold,
                "calibration_ms": latency,
                "predicted_pause_ms": predict_pause_ms(latency, sum(sizes.values())),
            }
        )
    feasible = [c for c in candidates if c["predicted_peak_bytes"] <= budget]
    if not feasible:
        # Keep a smaller legal communication geometry as a capacity fallback.
        remainder = max(
            (size for name, size in sizes.items() if name not in VOCAB), default=0
        )
        for bucket in (64, 32):
            if remainder + bucket * MIB <= budget:
                choice = {
                    "mode": "stream",
                    "slots": 1,
                    "bucket_mib": bucket,
                    "predicted_peak_bytes": remainder + bucket * MIB,
                    "threshold_bytes": 0,
                    "calibration_ms": None,
                    "predicted_pause_ms": None,
                }
                candidates.append(choice)
                return choice, candidates
        raise ValueError(f"No candidate fits update budget {budget / MIB:.1f} MiB")
    return min(feasible, key=lambda c: c["predicted_pause_ms"]), candidates
