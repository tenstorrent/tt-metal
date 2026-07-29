# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Query TT-Metal's empirical NoC estimator for the row fan-out transfer groups."""

import argparse
import json
import os
import subprocess
from pathlib import Path

VARIANTS = ("unicast", "mcast")
_REPO_ROOT = Path(__file__).resolve().parents[5]


def _positive(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _find_binary(explicit=None):
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if build_dir := os.environ.get("TT_METAL_BUILD_DIR"):
        candidates.append(Path(build_dir) / "test/tt_metal/noc_estimate")
    for build_dir in sorted(_REPO_ROOT.glob("build*")):
        candidates.append(build_dir / "test/tt_metal/noc_estimate")

    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError(
        "noc_estimate is not built; configure TT-Metal with --build-metal-tests, "
        "build target noc_estimate, then pass --binary or set TT_METAL_BUILD_DIR"
    )


def estimate(binary, *, variant, row_width, payload_bytes, num_writes, arch, aiclk_mhz):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if row_width < 2:
        raise ValueError("row_width must be at least 2")
    if payload_bytes % num_writes:
        raise ValueError("num_writes must divide payload_bytes")

    peers = row_width - 1
    transaction_count = num_writes * peers if variant == "unicast" else num_writes
    transactions_per_barrier = peers if variant == "unicast" else 1
    mechanism = "UNICAST" if variant == "unicast" else "MULTICAST"
    command = [
        str(binary),
        "--arch",
        arch,
        "--mechanism",
        mechanism,
        "--pattern",
        "ONE_TO_ROW",
        "--memory",
        "L1",
        "--num-transactions",
        str(transaction_count),
        "--num-transactions-per-barrier",
        str(transactions_per_barrier),
        "--transaction-size-bytes",
        str(payload_bytes // num_writes),
        "--num-subordinates",
        str(peers),
        "--same-axis=true",
        "--loopback=false",
        "--noc-index",
        "0",
        "--aiclk-mhz",
        str(aiclk_mhz),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def format_model(binary, *, row_widths, payload_bytes_values, num_writes_values, arch, aiclk_mhz):
    lines = [
        "# TT-Metal NoC-estimator model",
        "",
        "Transfer-only estimate for one sender round. Kernel dispatch, rotating-sender ordering,",
        "and semaphore synchronization are intentionally outside this model.",
        "Row-pattern coverage is sparse; unsupported fan-outs use the estimator's nearest measured point.",
        "",
        "| Cores/row | Payload | Dispatches | Bytes/write | Unicast ns | Mcast ns | Model pick |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row_width in row_widths:
        for payload_bytes in payload_bytes_values:
            for num_writes in num_writes_values:
                estimates = {
                    variant: estimate(
                        binary,
                        variant=variant,
                        row_width=row_width,
                        payload_bytes=payload_bytes,
                        num_writes=num_writes,
                        arch=arch,
                        aiclk_mhz=aiclk_mhz,
                    )
                    for variant in VARIANTS
                }
                winner = min(VARIANTS, key=lambda variant: estimates[variant]["latency_ns"])
                lines.append(
                    f"| {row_width} | {payload_bytes} B | {num_writes} | "
                    f"{payload_bytes // num_writes} B | {estimates['unicast']['latency_ns']:.1f} | "
                    f"{estimates['mcast']['latency_ns']:.1f} | {winner} |"
                )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.row_unicast_vs_mcast.noc_model")
    parser.add_argument("--binary", help="path to the built noc_estimate executable")
    parser.add_argument("--row-width", nargs="+", type=_positive, default=[2])
    parser.add_argument("--payload-bytes", nargs="+", type=_positive, default=[2048])
    parser.add_argument("--num-writes", nargs="+", type=_positive, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--arch", choices=("WORMHOLE_B0", "BLACKHOLE"), default="WORMHOLE_B0")
    parser.add_argument("--aiclk-mhz", type=float, default=1000.0)
    args = parser.parse_args()

    binary = _find_binary(args.binary)
    print(
        format_model(
            binary,
            row_widths=args.row_width,
            payload_bytes_values=args.payload_bytes,
            num_writes_values=args.num_writes,
            arch=args.arch,
            aiclk_mhz=args.aiclk_mhz,
        ),
        end="",
    )


if __name__ == "__main__":
    main()
