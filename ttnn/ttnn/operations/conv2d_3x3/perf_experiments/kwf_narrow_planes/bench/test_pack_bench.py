# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated fp32 untilize-pack bench for conv2d_3x3's kw-fold accumulator (one core, compute only).

    PB_BENCH=1 PB_VARIANTS=mop2x2,mop4x1,rep4x1,zst4x1 PB_ACC=0,1 \
        scripts/run_safe_pytest.sh --run-all --import-mode=prepend \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/kwf_narrow_planes/bench/test_pack_bench.py

Per variant and L1-acc mode: a copy run (DEST = the input's 4 fp32 tiles, every section) gated bit-exact against
the expected row-major plane, then a pack-only run (no unpack / math) whose DEVICE KERNEL DURATION is the pack
cost. Perf is printed and appended to results_bench.jsonl, never asserted.
"""

import os

if os.environ.get("PB_BENCH") == "1":
    os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
    os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
    os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
    os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import json
import socket
from pathlib import Path

import pytest

import ttnn

HERE = Path(__file__).parent
KERNEL = HERE / "kernels" / "pack_bench.cpp"
RESULTS = HERE / "results_bench.jsonl"
TILE = 32
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
SECTIONS = int(os.environ.get("PB_SECTIONS", "32"))
PASSES = int(os.environ.get("PB_PASSES", "3"))

# name -> (code, sbw, sbh)
VARIANTS = {
    "mop2x2": (0, 2, 2),
    "mop1x4": (1, 4, 1),
    "mop4x1": (2, 1, 4),
    "rep4x1": (3, 1, 4),
    "rep4x1_s2": (4, 1, 4),
    "rep4x1_s4": (5, 1, 4),
    "tile": (6, 0, 0),
    "zst4x1": (7, 1, 4),
    "zst4x1_s2": (8, 1, 4),
    "zst4x1_s4": (9, 1, 4),
    "empty": (10, 0, 0),
}


def _core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def _shard_cfg(h, w):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(_core(), [h, w], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    total, found = 0.0, False
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get(_DURATION_KEY)
            if entry is not None:
                total += float(entry.duration)
                found = True
    return total if found else None


def _out_shape(sbw, sbh):
    if sbw == 0:  # tilized: 4 tiles per section stacked
        return [SECTIONS * 4 * TILE, TILE], ttnn.TILE_LAYOUT
    return [SECTIONS * sbh * TILE, sbw * TILE], ttnn.ROW_MAJOR_LAYOUT


def _descriptor(tt_in, tt_out, code, acc, copy):
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL),
        core_ranges=_core(),
        compile_time_args=[code, SECTIONS, PASSES, acc, copy],
        defines=[],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            math_approx_mode=False,
            dst_full_sync_en=False,
        ),
    )
    cb_in = ttnn.cb_descriptor_from_sharded_tensor(0, tt_in)
    cb_out = ttnn.cb_descriptor_from_sharded_tensor(16, tt_out)
    cb_out.format_descriptors = [ttnn.CBFormatDescriptor(buffer_index=16, data_format=ttnn.float32, page_size=4096)]
    return ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=[cb_in, cb_out])


@pytest.mark.skipif(os.environ.get("PB_BENCH") != "1", reason="bench: run with PB_BENCH=1")
def test_pack_bench(device):
    import torch

    box, arch = socket.gethostname(), str(device.arch())
    names = os.environ.get("PB_VARIANTS", "mop2x2,mop1x4,mop4x1,rep4x1,zst4x1,tile,empty").split(",")
    accs = [int(a) for a in os.environ.get("PB_ACC", "0,1").split(",")]
    tiles = SECTIONS * PASSES * 4
    lines = [f"kwf pack bench box={box} arch={arch} 1 core, fp32 DEST half-sync, {SECTIONS} sections x {PASSES} passes"]
    failures = []
    torch.manual_seed(3)
    src = torch.randint(-8, 9, (4 * TILE, TILE)).float()  # tf32-exact, sums exact
    tt_in = ttnn.from_torch(
        src, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_shard_cfg(4 * TILE, TILE)
    )
    t = [src[i * TILE : (i + 1) * TILE] for i in range(4)]
    for acc in accs:
        for name in names:
            code, sbw, sbh = VARIANTS[name]
            shape, layout = _out_shape(sbw, sbh)
            if code == 10:
                shape, layout = [SECTIONS * 4 * TILE, TILE], ttnn.TILE_LAYOUT
            mult = PASSES if acc else 1
            if sbw == 0:
                block = torch.cat(t, 0)
            elif sbw == 2:
                block = torch.cat([torch.cat([t[0], t[1]], 1), torch.cat([t[2], t[3]], 1)], 0)
            elif sbw == 4:
                block = torch.cat(t, 1)
            else:
                block = torch.cat(t, 0)
            want = block.repeat(SECTIONS, 1) * mult
            row = {}
            for copy in (1, 0):
                tt_out = ttnn.allocate_tensor_on_device(
                    ttnn.Shape(shape), ttnn.float32, layout, device, _shard_cfg(shape[0], shape[1])
                )
                desc = _descriptor(tt_in, tt_out, code, acc, copy)
                _read_kernel_ns(device)
                ttnn.generic_op([tt_in, tt_out], desc)
                ns = _read_kernel_ns(device)
                row["copy" if copy else "pack"] = ns
                if copy and code != 10:
                    got = ttnn.to_torch(tt_out).float().reshape(want.shape)
                    exact = bool(torch.equal(got, want))
                    row["exact"] = exact
                    if not exact:
                        bad = (got != want).nonzero()
                        failures.append(f"{name} acc{acc}: {bad.shape[0]} mismatches, first {bad[:4].tolist()}")
                        r0 = bad[0, 0].item()
                        lines.append(
                            f"    {name} acc{acc} row {r0}: got {got[r0, :8].tolist()} want {want[r0, :8].tolist()}"
                        )
                tt_out.deallocate()
            per = row["pack"] / tiles if row.get("pack") else 0
            lines.append(
                f"  acc{acc} {name:<10} pack {row.get('pack', 0):>9.0f} ns ({per:6.1f} ns/tile, {per * 1.35:6.1f} cyc/tile)"
                f"  copy {row.get('copy', 0):>9.0f} ns  {'bit-exact' if row.get('exact') else ('-' if code == 10 else 'MISMATCH')}"
            )
            with RESULTS.open("a") as f:
                f.write(
                    json.dumps(dict(variant=name, acc=acc, sections=SECTIONS, passes=PASSES, box=box, arch=arch, **row))
                    + "\n"
                )
    tt_in.deallocate()
    print("\n" + "\n".join(lines))
    assert not failures, failures
