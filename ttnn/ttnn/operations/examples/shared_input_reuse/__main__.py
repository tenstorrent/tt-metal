# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""CLI: measure per-core DRAM read vs read-once+multicast of a shared input on your own shape.

    # streamed shared input (chunk-by-chunk multicast with a per-chunk handshake)
    python -m ttnn.operations.examples.shared_input_reuse [--chunk-rows 16] [--d-cols 4] [--chunks 19] [--trials 10]

    # resident window (one-shot multicast of a small w every Tensix core keeps in L1, out = x * w)
    python -m ttnn.operations.examples.shared_input_reuse --window [--cores 16,32,64,full] \\
        [--window-tiles 1,8,32,128,256] [--stream-mb 16,64] [--window-chunk 32] [--repeats 3] [--trials 3]

Runs the matching test through scripts/run_safe_pytest.sh (device lock, in-process profiler, post-run reset) and
prints the table. Run from the repo root with the env active. The streamed mode rewrites report.md unless
SIR_REPORT points elsewhere; the window mode only prints.
"""

import argparse
import os
import subprocess
import sys

_TESTS = "tests/ttnn/unit_tests/operations/examples/test_shared_input_reuse.py"


def main():
    ap = argparse.ArgumentParser(prog="python -m ttnn.operations.examples.shared_input_reuse")
    ap.add_argument("--chunk-rows", type=int, default=16, help="streamed: tile-rows per chunk. Default 16.")
    ap.add_argument("--d-cols", type=int, default=4, help="streamed: input width in tiles (tile-cols). Default 4.")
    ap.add_argument("--chunks", type=int, default=19, help="streamed: number of chunks. Default 19.")
    ap.add_argument("--trials", type=int, default=None, help="profiled rounds (median). Default 10 / 3 (window).")
    ap.add_argument("--window", action="store_true", help="measure the resident-window one-shot broadcast instead.")
    ap.add_argument("--cores", default="16,32,64,full", help="window: Tensix-core counts ('full' = whole grid).")
    ap.add_argument("--window-tiles", default="1,8,32,128,256", help="window: w width in tiles (2048 B each).")
    ap.add_argument("--stream-mb", default="16,64", help="window: MiB of x (the per-row stream; out is the same).")
    ap.add_argument("--window-chunk", type=int, default=32, help="window: tiles per DRAM batch / multicast.")
    ap.add_argument(
        "--variants",
        default="per_core,oneshot_mcast,lockstep_mcast",
        help="window: variants to compare; the first is the baseline.",
    )
    ap.add_argument("--repeats", type=int, default=3, help="window: DRAM placements per point (median).")
    args = ap.parse_args()

    env = dict(os.environ)
    env.setdefault("TT_METAL_LOGGER_LEVEL", "error")
    if args.window:
        env.update(
            SIR_WIN_CORES=args.cores,
            SIR_WIN_TILES=args.window_tiles,
            SIR_WIN_STREAM_MB=args.stream_mb,
            SIR_WIN_CHUNK=str(args.window_chunk),
            SIR_WIN_VARIANTS=args.variants,
            SIR_WIN_REPEATS=str(args.repeats),
            SIR_TRIALS=str(args.trials or 3),
        )
        test = f"{_TESTS}::test_shared_input_reuse_window_sweep"
        print(
            f"[shared_input_reuse] window: cores={args.cores} window_tiles={args.window_tiles} "
            f"stream_mb={args.stream_mb} chunk={args.window_chunk} variants={args.variants}"
        )
    else:
        env.update(
            SIR_CHUNK_ROWS=str(args.chunk_rows),
            SIR_D_COLS=str(args.d_cols),
            SIR_CHUNKS=str(args.chunks),
            SIR_TRIALS=str(args.trials or 10),
        )
        test = f"{_TESTS}::test_shared_input_reuse_device_perf"
        print(f"[shared_input_reuse] chunk_rows={args.chunk_rows} d_cols={args.d_cols} chunks={args.chunks}")
    cmd = ["scripts/run_safe_pytest.sh", "--run-all", test]
    print(f"[shared_input_reuse] {' '.join(cmd)}")
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
