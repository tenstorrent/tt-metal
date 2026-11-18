# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated BH fast-tilize chunking bake-off (compute only, one core, no data movement).

    FT_BENCH=1 FT_TWS=6,8 FT_VARIANTS=helper,base,balanced,mop1,balanced_mop1 \
        scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/fast_tilize_chunking/test_tilize_bench.py

bf16 row-major tap blocks [32, tw * 32] -> bf8_b tiles, fp32 DEST, LoFi, half sync (conv2d_3x3's N-slice prelude).
Every non-diag variant is gated BIT-EXACT against the "helper" variant (fast_tilize_block, the library path) on the
same device, and the helper against the host bf8_b quantization (close, not exact: the device bf16 -> bf8_b pack
rounds differently). diag_* variants are timed only. Perf is printed (DEVICE KERNEL DURATION), never asserted.
Results append to results_bench.jsonl.
"""

import os

# `import ttnn` walks every module under ttnn/operations; only touch the environment when asked.
if os.environ.get("FT_BENCH") == "1":
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
KERNEL = HERE / "kernels" / "bench_tilize.cpp"
RESULTS = HERE / "results_bench.jsonl"
TILE = 32
CB_IN, CB_OUT = 0, 16
NB = 8  # distinct resident source blocks, read round-robin
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

# name -> (split, pack_mode, diag); split 0 helper / 1 base / 2 balanced; pack_mode 0 stock / 1 mop1 / 2 stream /
# 3 noaddr (diag) / 4 hoist (+mop1) / 5 hoist (stock MOP) /
# 6 hoist3 / 7 hoist_nostall (diag bound) / 8 lean (hoist3, no MOP start_op); diag 1 NoPack / 2 PackOnly.
VARIANTS = {
    "helper": (0, 0, 0),
    "base": (1, 0, 0),
    "balanced": (2, 0, 0),
    "mop1": (1, 1, 0),
    "balanced_mop1": (2, 1, 0),
    "stream": (1, 2, 0),
    "balanced_stream": (2, 2, 0),
    "hoist": (1, 4, 0),
    "hoist_stock": (1, 5, 0),
    "balanced_hoist": (2, 4, 0),
    "hoist3": (1, 6, 0),
    "diag_hoist_nostall": (1, 7, 0),
    "lean": (1, 8, 0),
    "balanced_lean": (2, 8, 0),
    "diag_lean_packonly": (1, 8, 2),
    "diag_hoist_packonly": (1, 4, 2),
    "diag_base_nopack": (1, 1, 1),
    "diag_base_packonly": (1, 1, 2),
    "diag_bal_nopack": (2, 1, 1),
    "diag_bal_packonly": (2, 1, 2),
    "diag_stream_packonly": (1, 2, 2),
    "diag_noaddr_packonly": (1, 3, 2),
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


def _descriptor(tt_in, tt_out, tw, blocks, reps, code):
    split, pack_mode, diag = code
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL),
        core_ranges=_core(),
        compile_time_args=[split, pack_mode, diag, tw, blocks, NB, reps],
        defines=[],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            math_approx_mode=False,
            dst_full_sync_en=False,
        ),
    )
    cb_in = ttnn.cb_descriptor_from_sharded_tensor(CB_IN, tt_in)
    # One 2 KB bf16 page per tile, as conv2d_3x3's cb_im2col_rm.
    cb_in.format_descriptors = [
        ttnn.CBFormatDescriptor(buffer_index=CB_IN, data_format=ttnn.bfloat16, page_size=2 * TILE * TILE)
    ]
    cb_out = ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, tt_out)
    return ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=[cb_in, cb_out])


@pytest.mark.skipif(os.environ.get("FT_BENCH") != "1", reason="bench: run with FT_BENCH=1")
def test_tilize_bench(device):
    import torch

    box, arch = socket.gethostname(), str(device.arch())
    tws = [int(t) for t in os.environ.get("FT_TWS", "6").split(",")]
    variants = os.environ.get("FT_VARIANTS", "helper,base,balanced,mop1,balanced_mop1").split(",")
    reps = int(os.environ.get("FT_REPS", "4"))
    tiles_per_call = int(os.environ.get("FT_TILES", "432"))
    tag = os.environ.get("FT_TAG", "")
    lines = [f"fast_tilize_chunking bench box={box} arch={arch} 1 core, bf16 RM -> bf8_b, fp32 DEST half-sync, LoFi"]
    failures = []
    for tw in tws:
        blocks = max(1, tiles_per_call // tw)
        tiles = blocks * tw * reps
        torch.manual_seed(7)
        # Rows of very different magnitude so bf8_b's shared exponents vary across the tile.
        src = (torch.randn(NB * TILE, tw * TILE) * torch.exp2(torch.randint(-6, 7, (NB * TILE, 1)).float())).to(
            torch.bfloat16
        )
        tt_in = ttnn.from_torch(
            src,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=_shard_cfg(NB * TILE, tw * TILE),
        )
        expect = src.reshape(NB, TILE, tw * TILE).repeat((blocks + NB - 1) // NB, 1, 1)[:blocks]
        expect = expect.reshape(blocks * TILE, tw * TILE)
        host_q = ttnn.to_torch(ttnn.from_torch(expect, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()
        ref, base_ns = None, None
        lines.append(f"  tw={tw}: {blocks} tap blocks x {reps} calls = {tiles} tiles")
        for name in variants:
            code = VARIANTS[name]
            tt_out = ttnn.from_torch(
                torch.full((blocks * TILE, tw * TILE), float("nan")),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=_shard_cfg(blocks * TILE, tw * TILE),
            )
            desc = _descriptor(tt_in, tt_out, tw, blocks, reps, code)
            _read_kernel_ns(device)  # flush
            ttnn.generic_op([tt_in, tt_out], desc)
            ns = _read_kernel_ns(device)
            got = ttnn.to_torch(tt_out).float()
            note = ""
            if name == "helper":
                ref = got
                g, h = got.flatten(), host_q.flatten()
                pcc = torch.corrcoef(torch.stack([g, h]))[0, 1].item()
                frac = (g != h).float().mean().item()
                close = bool(torch.isfinite(g).all()) and pcc > 0.9999
                note = f"vs host bf8_b: pcc {pcc:.6f}, {frac * 100:.2f}% differ ({'ok' if close else 'FAR'})"
                if not close:
                    failures.append(f"tw{tw}/helper: far from host quantization")
            elif not name.startswith("diag"):
                exact = ref is not None and bool(torch.equal(got, ref))
                note = "bit-exact vs helper" if exact else "MISMATCH vs helper"
                if not exact:
                    failures.append(f"tw{tw}/{name}: not bit-exact vs helper")
            if name == "base":
                base_ns = ns
            speed = f"{base_ns / ns:5.3f}x" if (base_ns and ns) else "  -  "
            per_tile = ns / tiles if ns else 0
            lines.append(f"    {name:<20} {ns:>10.0f} ns  {speed} vs base  {per_tile:6.2f} ns/tile  {note}")
            with RESULTS.open("a") as f:
                f.write(
                    json.dumps(dict(tw=tw, variant=name, ns=ns, tiles=tiles, note=note, tag=tag, box=box, arch=arch))
                    + "\n"
                )
            tt_out.deallocate()
        tt_in.deallocate()
    print("\n" + "\n".join(lines))
    assert not failures, failures
