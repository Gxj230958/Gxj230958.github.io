"""Derive packing sizes from safetensors headers and the Qwen2 checkpoint config."""

import argparse
import json
import struct
from pathlib import Path


def inventory(path):
    config = json.loads((path / "config.json").read_text())
    index_file = path / "model.safetensors.index.json"
    index = (
        json.loads(index_file.read_text())["weight_map"]
        if index_file.exists()
        else None
    )
    files = sorted(set(index.values())) if index else ["model.safetensors"]
    tensors = {}
    for file in files:
        with (path / file).open("rb") as source:
            size = struct.unpack("<Q", source.read(8))[0]
            header = json.loads(source.read(size))
        for name, entry in header.items():
            if name != "__metadata__":
                start, end = entry["data_offsets"]
                tensors[name] = {
                    "shape": entry["shape"],
                    "dtype": entry["dtype"],
                    "nbytes": end - start,
                }
    if index:
        assert set(tensors) == set(index)
    assert all(t["dtype"] == "BF16" for t in tensors.values())
    dim = (
        config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"]
    )
    token_bytes = (
        2 * 2 * config["num_hidden_layers"] * config["num_key_value_heads"] * dim
    )
    return tensors, token_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("instruct", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    tensors, kv_bytes = inventory(args.base)
    assert inventory(args.instruct) == (tensors, kv_bytes)
    first = sorted(tensors.items(), key=lambda t: -t[1]["nbytes"])[:3]
    report = {
        "models": [args.base.name, args.instruct.name],
        "parameters": len(tensors),
        "weight_bytes": sum(t["nbytes"] for t in tensors.values()),
        "largest": first,
        "kv_bytes_per_token_tp1": kv_bytes,
        "checkpoint_tensors": tensors,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
