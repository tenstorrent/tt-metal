# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated untilize bake-off for conv2d_3x3's TILE slab (compute only, one core, no data movement).

    NU_BENCH=1 NU_SHAPES=c1_r60 NU_VARIANTS=helper,rowmop_b1,contig_b4 \
        scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/narrow_untilize/test_untilize_bench.py

Every variant is gated on bit-exactness against the bf8_b input (untilize is lossless); diag_* variants are
wrong by construction and only timed. Perf is printed (DEVICE KERNEL DURATION, in-process profiler), never
asserted. Results append to results_bench.jsonl.
"""

import os

# `import ttnn` walks every module under ttnn/operations; only touch the environment when asked.
if os.environ.get("NU_BENCH") == "1":
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

# name -> (ct, slab tile-rows, untilize batch rows): the slab one M-block untilizes.
SHAPES = {
    "c1_r60": (1, 60, 30),  # res4.0.conv2 focus: 32-ch slice, slab 60, batch 30
    "c1_r12": (1, 12, 6),
    "c2_r30": (2, 30, 15),
    "c6_r24": (6, 24, 12),
}
REPS = int(os.environ.get("NU_REPS", "4"))
# > 0: tilize this many 1-tile RM tile-rows after each untilize (the op's next pack phase) and check them.
TILIZE_ROWS = int(os.environ.get("NU_TILIZE_ROWS", "0"))
TILIZE_DTYPE = ttnn.bfloat8_b if os.environ.get("NU_TILIZE_DTYPE", "bf16") == "bf8b" else ttnn.bfloat16

# name -> (variant code, rows_per_section, diag, ct constraint)
VARIANTS = {
    "helper": (0, 1),
    "rowmop_b1": (1, 1),
    "rowmop_b2": (1, 2),
    "rowmop_b4": (1, 4),
    "contig_b1": (2, 1),
    "contig_b2": (2, 2),
    "contig_b4": (2, 4),
    "contig_b4_secsync": (4, 4),
    "contig_b2_secsync": (4, 2),
    "strided_b1": (5, 1),
    "strided_b2": (5, 2),
    "strided_b4": (5, 4),
    "diag_packonly_strided_b4": (6, 4),
    "wide_batch": (3, 1),
    "diag_nopack_b1": (10, 1),
    "diag_nopack_b4": (10, 4),
    "diag_packonly_rowmop_b1": (11, 1),
    "diag_packonly_rowmop_b4": (11, 4),
    "diag_packonly_contig_b4": (13, 4),
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


def _descriptor(tt_in, tt_out, ct, rows, batch, code, rps, tz=None):
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL),
        core_ranges=_core(),
        compile_time_args=[code, ct, rows, REPS, batch, rps, 0 if tz is None else tz[0].shape[0] // TILE],
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


@pytest.mark.skipif(os.environ.get("NU_BENCH") != "1", reason="bench: run with NU_BENCH=1")
def test_untilize_bench(device):
    import torch

    box, arch = socket.gethostname(), str(device.arch())
    shapes = os.environ.get("NU_SHAPES", "c1_r60").split(",")
    variants = os.environ.get("NU_VARIANTS", "helper,rowmop_b1,rowmop_b4,contig_b1,contig_b4").split(",")
    lines = [f"narrow_untilize bench box={box} arch={arch} 1 core, fp32 DEST half-sync, reps={REPS}"]
    failures = []
    for shape in shapes:
        ct, rows, batch = SHAPES[shape]
        torch.manual_seed(7)
        src = torch.randn(rows * TILE, ct * TILE) * 4
        tt_in = ttnn.from_torch(
            src,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=_shard_cfg(rows * TILE, ct * TILE),
        )
        ref = ttnn.to_torch(tt_in).to(torch.bfloat16)
        base_ns = None
        lines.append(f"  {shape}: ct={ct} rows={rows} batch={batch} ({rows * ct * REPS} tiles timed)")
        for name in variants:
            code, rps = VARIANTS[name]
            if (code in (1, 2, 4, 5, 6, 10, 11, 13) and ct != 1) or (code == 3 and ct < 2):
                continue
            tt_out = ttnn.allocate_tensor_on_device(
                ttnn.Shape([rows * TILE, ct * TILE]),
                ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT,
                device,
                _shard_cfg(rows * TILE, ct * TILE),
            )
            tz = None
            if TILIZE_ROWS:
                tz_src = torch.randn(TILIZE_ROWS * TILE, TILE) * 3
                tz_rm = ttnn.from_torch(
                    tz_src.to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=device,
                    memory_config=_shard_cfg(TILIZE_ROWS * TILE, TILE),
                )
                tz_out = ttnn.allocate_tensor_on_device(
                    ttnn.Shape([TILIZE_ROWS * TILE, TILE]),
                    TILIZE_DTYPE,
                    ttnn.TILE_LAYOUT,
                    device,
                    _shard_cfg(TILIZE_ROWS * TILE, TILE),
                )
                tz = (tz_rm, tz_out)
            desc = _descriptor(tt_in, tt_out, ct, rows, batch, code, rps, tz)
            io = [tt_in, tt_out] + ([tz[0], tz[1]] if tz else [])
            ttnn.generic_op(io, desc)  # compile + correctness run
            got = ttnn.to_torch(tt_out)
            if tz is not None:
                tz_got = ttnn.to_torch(tz[1]).float()
                tz_ref = ttnn.to_torch(
                    ttnn.from_torch(tz_src.to(torch.bfloat16), dtype=TILIZE_DTYPE, layout=ttnn.TILE_LAYOUT)
                ).float()
                tz_bad = (tz_got != tz_ref).reshape(-1, TILE, TILE).any(dim=2)
                tz_exact = not bool(tz_bad.any())
                if not tz_exact:
                    rows_bad = tz_bad.nonzero()[:, 1].unique().tolist()
                    failures.append(f"{shape}/{name}: tilize after untilize wrong, bad rows-in-tile {rows_bad}")
                    g3 = tz_got.reshape(-1, TILE, TILE)
                    r3 = tz_ref.reshape(-1, TILE, TILE)
                    src_rows = tz_src.to(torch.bfloat16).float().reshape(-1, TILE)
                    halves = src_rows.reshape(-1, 2, 16)  # [row, half, 16]
                    for r_i in range(16):
                        parts = []
                        for h in range(2):
                            seg = g3[0, r_i, h * 16 : (h + 1) * 16]
                            hits = (halves == seg).all(dim=2).nonzero().tolist()
                            parts.append(f"h{h}<-{hits[:2]}" + (" zeros" if bool((seg == 0).all()) else ""))
                        lines.append(f"      tile0 row {r_i}: " + "  ".join(parts))
                    for t_i in range(0):
                        for r_i in (0, 1, 8, 15, 16):
                            row = g3[t_i, r_i]
                            hits = [k for k in range(src_rows.shape[0]) if torch.equal(src_rows[k], row)]
                            lines.append(
                                f"      tile {t_i} row {r_i}: ok={torch.equal(row, r3[t_i, r_i])} "
                                f"zeros={bool((row == 0).all())} matches src rows {hits[:4]} first vals {row[:4].tolist()}"
                            )
                lines.append(f"    {name:<26} tilize-after: {'bit-exact' if tz_exact else 'MISMATCH'}")
            _read_kernel_ns(device)  # flush
            ttnn.generic_op(io, desc)
            ns = _read_kernel_ns(device)
            exact = bool(torch.equal(got.to(torch.bfloat16), ref))
            if not name.startswith("diag") and not exact:
                bad = (got.float() != ref.float()).nonzero()
                failures.append(f"{shape}/{name}: not bit-exact ({bad.shape[0]} mismatches, first {bad[:4].tolist()})")
                badrows = sorted(set(bad[:, 0].tolist()))
                by_tile = {}
                for r in badrows:
                    by_tile.setdefault(r // TILE, []).append(r % TILE)
                halves_ref = ref.float().reshape(-1, 2, 16)
                t0 = min(by_tile)
                for r_i in by_tile[t0][:20]:
                    parts = []
                    for h in range(2):
                        seg = got[t0 * TILE + r_i, h * 16 : (h + 1) * 16].float()
                        parts.append(f"h{h}<-{(halves_ref == seg).all(dim=2).nonzero().tolist()[:2]}")
                    lines.append(f"      untilize tile-row {t0} row {r_i}: " + "  ".join(parts))
                for t_i, rs in sorted(by_tile.items()):
                    gr = got[t_i * TILE + rs[0]].float()
                    hits = [k for k in range(ref.shape[0]) if torch.equal(ref[k].float(), gr)]
                    lines.append(
                        f"      untilize tile-row {t_i}: bad rows {rs[0]}..{rs[-1]} ({len(rs)}) first bad row == ref rows {hits[:3]}"
                    )
            if name == "helper":
                base_ns = ns
            speed = f"{base_ns / ns:5.3f}x" if (base_ns and ns) else "  -  "
            per_tile = ns / (rows * ct * REPS) if ns else 0
            lines.append(
                f"    {name:<26} {ns:>10.0f} ns  {speed}  {per_tile:6.1f} ns/tile  "
                f"{'bit-exact' if exact else 'MISMATCH'}"
            )
            with RESULTS.open("a") as f:
                f.write(
                    json.dumps(dict(shape=shape, variant=name, ns=ns, exact=exact, reps=REPS, box=box, arch=arch))
                    + "\n"
                )
            tt_out.deallocate()
        tt_in.deallocate()
    print("\n" + "\n".join(lines))
    assert not failures, failures
