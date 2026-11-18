# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated wide-untilize bake-off for conv2d_3x3's TILE slab (compute only, one core, no data movement).

    WU_BENCH=1 WU_SHAPES=c8_r21,c6_r30 WU_VARIANTS=helper,wide \
        scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/wide_untilize/test_wide_untilize_bench.py

Every non-diag variant is gated on bit-exactness against the input (untilize of bf8_b / bf16 into bf16 is lossless),
plus (WU_TILIZE_ROWS > 0) a bit-exact tilize run right after each untilize, the op's next pack phase. diag_* variants
are wrong by construction and only timed. Perf is printed (DEVICE KERNEL DURATION, in-process profiler), never
asserted. Results append to results_bench.jsonl.
"""

import os

# `import ttnn` walks every module under ttnn/operations; only touch the environment when asked.
if os.environ.get("WU_BENCH") == "1":
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
KERNEL = HERE / "kernels" / "bench_compute.cpp"
RESULTS = HERE / "results_bench.jsonl"
TILE = 32
CB_IN, CB_OUT = 0, 16
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

# name -> (ct, slab tile-rows, untilize batch rows, M-blocks): one core's untilize work.
SHAPES = {
    "c8_r21": (8, 21, 7, 2),  # semantic res2.fuse_conv.1 (HEIGHT, 256 ch): 2 M-blocks x 21 rows, batch 7
    "c6_r30": (6, 30, 2, 1),  # aspp.convs.1 (BLOCK 11x8, 192 ch / core): slab 30, batch 2
    "c2_r27": (2, 27, 9, 1),  # res5.1 (BLOCK 8x10, 64 ch / core): slab 27, batch 9
    "c3_r30": (3, 30, 5, 1),
    "c4_r24": (4, 24, 6, 1),
    "c5_r21": (5, 21, 7, 1),
    "c7_r14": (7, 14, 7, 1),
    "c9_r21": (9, 21, 7, 1),
    "c10_r12": (10, 12, 6, 1),
    "c16_r8": (16, 8, 4, 1),
    "c17_r4": (17, 4, 2, 1),
}
# name -> variant code
VARIANTS = {
    "helper": 0,
    "base_replica": 1,
    "diag_base_nopack": 2,
    "diag_base_packonly": 3,
    "wide": 10,
    "wide_batchsync": 11,
    "wide_hybrid": 14,
    "wide_lazy": 19,
    "wide_hybrid_folded": 16,
    "diag_hybrid_folded_nopack": 17,
    "diag_hybrid_nopack": 18,
    "diag_wide_nopack": 12,
    "diag_wide_packonly": 13,
    "diag_wide_unpackonly": 15,
}
TILIZE_ROWS = int(os.environ.get("WU_TILIZE_ROWS", "0"))
TILIZE_W = int(os.environ.get("WU_TILIZE_W", "1"))
DTYPE = ttnn.bfloat16 if os.environ.get("WU_DTYPE", "bf8b") == "bf16" else ttnn.bfloat8_b


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


def _descriptor(tt_in, tt_out, ct, rows, batch, reps, code, tz=None):
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL),
        core_ranges=_core(),
        compile_time_args=[code, ct, rows, reps, batch, TILIZE_ROWS if tz else 0, TILIZE_W],
        defines=[],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            math_approx_mode=False,
            dst_full_sync_en=False,
        ),
    )
    cb_in = ttnn.cb_descriptor_from_sharded_tensor(CB_IN, tt_in)
    cb_out = ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, tt_out)
    # One 2 KB bf16 page per tile, as conv2d_3x3's cb_halo_slab.
    cb_out.format_descriptors = [
        ttnn.CBFormatDescriptor(buffer_index=CB_OUT, data_format=ttnn.bfloat16, page_size=2 * TILE * TILE)
    ]
    cbs = [cb_in, cb_out]
    if tz is not None:
        cb_rm = ttnn.cb_descriptor_from_sharded_tensor(1, tz[0])
        cb_rm.format_descriptors = [
            ttnn.CBFormatDescriptor(buffer_index=1, data_format=ttnn.bfloat16, page_size=2 * TILE * TILE)
        ]
        cbs += [cb_rm, ttnn.cb_descriptor_from_sharded_tensor(2, tz[1])]
    return ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs)


def _describe_mismatch(got, ref, lines):
    bad = (got.float() != ref.float()).nonzero()
    rows_bad = sorted(set(bad[:, 0].tolist()))
    cols_bad = sorted(set(bad[:, 1].tolist()))
    lines.append(
        f"      {bad.shape[0]} mismatches; rows {rows_bad[:8]}..({len(rows_bad)}), cols {cols_bad[:8]}..({len(cols_bad)})"
    )
    # Where did the first bad 16-datum segment come from?
    r, c = bad[0].tolist()
    c16 = (c // 16) * 16
    seg = got[r, c16 : c16 + 16].float()
    segs = ref.float().reshape(ref.shape[0], -1, 16)
    hits = (segs == seg).all(dim=2).nonzero().tolist()[:3]
    lines.append(f"      first bad (row {r}, col {c}): its 16-datum segment equals ref (row, seg) {hits}")


@pytest.mark.skipif(os.environ.get("WU_BENCH") != "1", reason="bench: run with WU_BENCH=1")
def test_wide_untilize_bench(device):
    import torch

    box, arch = socket.gethostname(), str(device.arch())
    shapes = os.environ.get("WU_SHAPES", "c8_r21").split(",")
    variants = os.environ.get("WU_VARIANTS", "helper,wide").split(",")
    tag = os.environ.get("WU_TAG", "")
    lines = [
        f"wide_untilize bench box={box} arch={arch} 1 core, fp32 DEST half-sync, dtype={DTYPE}, "
        f"tilize-after={TILIZE_ROWS}x{TILIZE_W}"
    ]
    failures = []
    for shape in shapes:
        ct, rows, batch, reps = SHAPES[shape]
        torch.manual_seed(7)
        # Rows of very different magnitude so bf8_b's shared exponents vary across the tile.
        src = torch.randn(rows * TILE, ct * TILE) * torch.exp2(torch.randint(-6, 7, (rows * TILE, 1)).float())
        tt_in = ttnn.from_torch(
            src,
            dtype=DTYPE,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=_shard_cfg(rows * TILE, ct * TILE),
        )
        ref = ttnn.to_torch(tt_in).to(torch.bfloat16)
        base_ns = None
        lines.append(f"  {shape}: ct={ct} rows={rows} batch={batch} reps={reps} ({rows * ct * reps} tiles timed)")
        for name in variants:
            code = VARIANTS[name]
            # NaN-filled, so a variant that skips writes cannot pass on a previous variant's L1 leftovers.
            tt_out = ttnn.from_torch(
                torch.full((rows * TILE, ct * TILE), float("nan"), dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=_shard_cfg(rows * TILE, ct * TILE),
            )
            tz = None
            if TILIZE_ROWS:
                tz_src = torch.randn(TILIZE_ROWS * TILE, TILIZE_W * TILE) * 3
                tz_rm = ttnn.from_torch(
                    tz_src.to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=device,
                    memory_config=_shard_cfg(TILIZE_ROWS * TILE, TILIZE_W * TILE),
                )
                tz_out = ttnn.from_torch(
                    torch.full((TILIZE_ROWS * TILE, TILIZE_W * TILE), float("nan"), dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=_shard_cfg(TILIZE_ROWS * TILE, TILIZE_W * TILE),
                )
                tz = (tz_rm, tz_out)
            desc = _descriptor(tt_in, tt_out, ct, rows, batch, reps, code, tz)
            io = [tt_in, tt_out] + ([tz[0], tz[1]] if tz else [])
            ttnn.generic_op(io, desc)  # compile + correctness run
            got = ttnn.to_torch(tt_out)
            tz_note = ""
            if tz is not None:
                tz_got = ttnn.to_torch(tz[1]).float()
                tz_exact = bool(torch.equal(tz_got, tz_src.to(torch.bfloat16).float()))
                tz_note = " tilize-after:" + ("ok" if tz_exact else "MISMATCH")
                if not tz_exact and not name.startswith("diag"):
                    failures.append(f"{shape}/{name}: tilize after untilize wrong")
            _read_kernel_ns(device)  # flush
            ttnn.generic_op(io, desc)
            ns = _read_kernel_ns(device)
            exact = bool(torch.equal(got.to(torch.bfloat16), ref))
            if not name.startswith("diag") and not exact:
                failures.append(f"{shape}/{name}: not bit-exact")
                _describe_mismatch(got.to(torch.bfloat16), ref, lines)
            if name == "helper":
                base_ns = ns
            speed = f"{base_ns / ns:5.3f}x" if (base_ns and ns) else "  -  "
            per_tile = ns / (rows * ct * reps) if ns else 0
            lines.append(
                f"    {name:<22} {ns:>10.0f} ns  {speed}  {per_tile:6.1f} ns/tile  "
                f"{'bit-exact' if exact else 'MISMATCH'}{tz_note}"
            )
            with RESULTS.open("a") as f:
                f.write(
                    json.dumps(
                        dict(
                            shape=shape,
                            variant=name,
                            ns=ns,
                            exact=exact,
                            dtype=str(DTYPE),
                            tag=tag,
                            box=box,
                            arch=arch,
                        )
                    )
                    + "\n"
                )
            tt_out.deallocate()
        tt_in.deallocate()
    print("\n" + "\n".join(lines))
    assert not failures, failures
