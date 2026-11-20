# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated Q-pack bench (bench/bench_qpack.cpp): one core, compute only, fp32 DEST -> fp32 accumulator rows.

    QPB_BENCH=1 QPB_REPS=256 QPB_ROWS=640 bench_run.sh <tag>

Variants (layout, shape, l1acc): zst / fst x 4x1 (one 4-tile run) / 1x4 (four 1-tile runs, one per plane) x overwrite /
L1 accumulation. Perf only (DEST holds garbage): DEVICE KERNEL DURATION of the 2nd dispatch per variant (bench_report.py).
"""

import json
import os
import socket
from pathlib import Path

import pytest

import ttnn

HERE = Path(__file__).parent
KERNEL = HERE / "bench" / "bench_qpack.cpp"
RESULTS = HERE / "results_bench.jsonl"
LAYOUTS = {"zst": 0, "fst": 1, "fst2": 2, "fst3": 3}
SHAPES = {"4x1": 0, "1x4": 1, "side": 2}


def _core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def _variants():
    out = []
    for lay in os.environ.get("QPB_LAYOUTS", "zst,fst").split(","):
        for sh in os.environ.get("QPB_SHAPES", "4x1,1x4").split(","):
            for acc in [int(a) for a in os.environ.get("QPB_ACC", "0,1").split(",")]:
                out.append((lay, sh, acc))
    return out


@pytest.mark.skipif(os.environ.get("QPB_BENCH") != "1", reason="bench: run with QPB_BENCH=1")
def test_qpack_bench(device):
    import torch

    reps = int(os.environ.get("QPB_REPS", "256"))
    rows = int(os.environ.get("QPB_ROWS", "640"))
    plane = rows * 128
    n_planes = 4
    tag = os.environ.get("QPB_TAG", "")
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(_core(), [4 * rows, 32], ttnn.ShardOrientation.ROW_MAJOR),
    )
    acc = ttnn.from_torch(
        torch.zeros(4 * rows, 32), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem
    )
    check = os.environ.get("QPB_CHECK") == "1"
    if check:
        reps = 1
        TILE = 32
        r_ = torch.arange(TILE).view(TILE, 1).float()
        c_ = torch.arange(TILE).view(1, TILE).float()
        base_tile = r_ * 32 + c_
        src = torch.cat([base_tile * (2.0**j) for j in range(4)], dim=1)  # [32, 128]: tile j = (32 r + c) 2^j
        imem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(_core(), [32, 128], ttnn.ShardOrientation.ROW_MAJOR),
        )
        tt_src = ttnn.from_torch(src, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=imem)
    for lay, sh, l1acc in _variants():
        compute = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL),
            core_ranges=_core(),
            compile_time_args=[LAYOUTS[lay], SHAPES[sh], l1acc, reps, plane, int(check)],
            defines=[],
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi,
                fp32_dest_acc_en=True,
                math_approx_mode=False,
                dst_full_sync_en=False,
            ),
        )
        cbs = [ttnn.cb_descriptor_from_sharded_tensor(0, acc)]
        if check:
            cbs.append(ttnn.cb_descriptor_from_sharded_tensor(1, tt_src))
            acc_host = torch.zeros(4 * rows, 32)
            acc = ttnn.from_torch(
                acc_host, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem
            )
            cbs[0] = ttnn.cb_descriptor_from_sharded_tensor(0, acc)
        desc = ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs)
        if check:
            ttnn.generic_op([acc, tt_src], desc)
            got = ttnn.to_torch(acc).float().flatten()
            fl = plane // 4  # floats per plane
            bad = 0
            for j in range(4):
                tile = base_tile * (2.0**j)
                for r in range(32):
                    if sh == "1x4":
                        p_, row = j, 2 + r
                    else:
                        p_, row = 0, 2 + 32 * j + r
                    if lay == "zst":
                        v = got[p_ * fl + row * 32 : p_ * fl + row * 32 + 32]
                    else:
                        lo = got[p_ * fl + row * 16 : p_ * fl + row * 16 + 16]
                        hi = got[p_ * fl + fl // 2 + row * 16 : p_ * fl + fl // 2 + row * 16 + 16]
                        v = torch.cat([lo, hi])
                    bad += int((v != tile[r]).sum())
            print(f"CHECK {lay}_{sh}: {'OK' if bad == 0 else f'{bad} bad datums'}")
            if bad and os.environ.get("QPB_DEBUG") == "1":
                # where do tile 0's rows 0, 1 (left 0..15, right 16..31) land? float offsets in plane units of 16
                for val in (0.0 + 1, 16.0, 32.0 + 1, 48.0, 2.0 * 16, 4.0 * 16):
                    idx = (got == val).nonzero().flatten()[:6].tolist()
                    print(f"  val {val}: float idx {idx} (plane floats {fl}, half {fl // 2})")
            continue
        ttnn.generic_op([acc], desc)
        ttnn.generic_op([acc], desc)
        with RESULTS.open("a") as f:
            f.write(
                json.dumps(
                    dict(
                        bench="qpack",
                        variant=f"{lay}_{sh}_{'acc' if l1acc else 'ovw'}",
                        reps=reps,
                        tiles=4 * reps,
                        rows=rows,
                        box=socket.gethostname(),
                        arch=str(device.arch()),
                        tag=tag,
                    )
                )
                + "\n"
            )
