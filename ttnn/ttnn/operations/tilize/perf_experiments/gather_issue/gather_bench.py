# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated bake-off harness for the cross-core L1 GATHER of tilize's reader.

Concept under study (ONE): the RISC-serial cost of ISSUING the gather's read
transactions — address generation + command-buffer writes — and whether it can be
made cheaper per transaction, or the transaction count reduced.

Isolation (perf-lab Invariant 1, "Tensix <-> Tensix NoC" row): no DRAM anywhere,
compute trivial (absent), multi-core, both sides L1-sharded. The program is ONE
NCRISC reader kernel per destination core and nothing else — no CB handshake, no
compute kernel, no writer — so the measured `DEVICE KERNEL DURATION [ns]` is the
gather and only the gather.

Geometry (identical to the production plan it reconstructs):
  * source      ROW_MAJOR bf16/fp32, WIDTH-sharded L1 over `src_cores`. Its page
    is ONE SHARD ROW, so a tensor row is `src_cores` pages living on `src_cores`
    DIFFERENT cores' L1 — the gather.
  * destination the reading core's own L1 (its HEIGHT shard), holding the blocks
    back to back, each block `tile_h` rows of `row_bytes` at stride `row_bytes` —
    byte-for-byte the layout the production reader builds in its input CB.
"""

import os

# In-process device profiler — all three, before the device opens.
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

from pathlib import Path


# `ttnn/` may not import torch at module scope (scripts/validate_no_global_torch_imports.py
# — the shipped package must not drag torch in). These perf-experiment benches DO need it
# for their bit-exact oracle, so the import is done inside a function scope and published
# under the module-global name, which keeps every `torch.` use below unchanged.
def _load_torch():
    global torch
    import torch


_load_torch()
import ttnn
from loguru import logger

KERNEL = str(Path(__file__).resolve().parent / "experiment_kernels" / "gather_reader.cpp")

TILE_W = 32
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

# variant id -> name. 0 is the honest baseline (the op's current approach).
VARIANTS = {0: "baseline", 1: "hoist", 2: "coalesce", 3: "scratch", 4: "strip"}


def width_shard(shape, num_cores):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (shape[-2], shape[-1] // num_cores), ttnn.ShardOrientation.ROW_MAJOR),
    )


def height_shard(shape, num_cores):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (shape[-2] // num_cores, shape[-1]), ttnn.ShardOrientation.ROW_MAJOR),
    )


def plan(shape, src_cores, dst_cores, wt_chunk, elem_bytes, tile_h=TILE_W):
    """The derived block geometry — one source for the kernel args and the golden."""
    h, w = shape[-2], shape[-1]
    page_bytes = (w // src_cores) * elem_bytes  # one source shard row
    row_bytes = wt_chunk * TILE_W * elem_bytes
    assert (w * elem_bytes) % row_bytes == 0, "W must split into whole chunks"
    shard_rows = h // dst_cores
    assert shard_rows % tile_h == 0
    return dict(
        page_bytes=page_bytes,
        row_pages=src_cores,
        row_bytes=row_bytes,
        n_chunks=(w * elem_bytes) // row_bytes,
        shard_rows=shard_rows,
        nt_h_local=shard_rows // tile_h,
        tile_h=tile_h,
        elem_bytes=elem_bytes,
    )


def variant_applicable(variant, p):
    """Structural expressibility — NOT a perf judgement."""
    if variant in (2, 3, 4):
        # Both wide-transfer variants need whole-page slices (source contiguous
        # down the block's rows). Variant 2 additionally needs the block width to
        # BE the page.
        if variant == 2:
            return p["row_bytes"] == p["page_bytes"]
        return p["row_bytes"] % p["page_bytes"] == 0
    return True


def build(input_tensor, output_tensor, *, variant, p, dst_cores):
    all_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dst_cores - 1, 0))})
    scratch_bytes = p["tile_h"] * p["row_bytes"]

    ct = [
        variant,
        p["tile_h"],
        p["row_bytes"],
        p["page_bytes"],
        p["row_pages"],
        p["n_chunks"],
    ]
    ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    rt = ttnn.RuntimeArgs()
    src_addr = input_tensor.buffer_address()
    dst_addr = output_tensor.buffer_address()
    blocks = p["nt_h_local"] * p["n_chunks"]
    for k in range(dst_cores):
        rt[k][0] = [src_addr, dst_addr, k * p["nt_h_local"], blocks]

    # Reader-private L1 scratch. Created for EVERY variant (only variant 3 uses
    # it) so the L1 map is identical across the bake-off.
    cb = ttnn.CBDescriptor(
        total_size=scratch_bytes,
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=0,
                data_format=input_tensor.dtype,
                page_size=scratch_bytes,
            )
        ],
    )
    kernel = ttnn.KernelDescriptor(
        kernel_source=KERNEL,
        core_ranges=all_cores,
        compile_time_args=ct,
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb])


def golden(torch_input, p, dst_cores, *, strip=False):
    """The destination bytes the gather must produce, as a whole-tensor view.

    `strip=False` — blocks of `tile_h` rows x `row_bytes`, row-major inside the
    block (the production input-CB layout). `strip=True` — the same blocks, but
    laid out strip-major: all `tile_h` rows of source shard column 0, then of
    column 1, ... (variant 4).
    """
    h, w = torch_input.shape[-2], torch_input.shape[-1]
    flat = torch_input.reshape(h, w)
    unit_w = (p["page_bytes"] if strip else p["row_bytes"]) // p["elem_bytes"]
    unit_w = min(unit_w, p["row_bytes"] // p["elem_bytes"])
    units = w // unit_w
    out = torch.empty_like(flat)
    for k in range(dst_cores):
        lo, hi = k * p["shard_rows"], (k + 1) * p["shard_rows"]
        sub = flat[lo:hi].reshape(p["nt_h_local"], p["tile_h"], units, unit_w)
        out[lo:hi] = sub.permute(0, 2, 1, 3).reshape(p["shard_rows"], w)
    return out.reshape(torch_input.shape)


def read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    total, found = 0.0, False
    for programs in (per_chip or {}).values():
        for program in programs:
            results = getattr(program, "program_analyses_results", None) or {}
            entry = results.get(_DURATION_KEY)
            if entry is None:
                continue
            total += float(entry.duration)
            found = True
    return total if found else None


def run(device, *, shape, dtype, src_cores, dst_cores, wt_chunk, variant, label):
    """Correctness-gate the variant, then ONE fresh measured launch.

    Device kernel duration has no warm-up transient, so a trial loop would just
    re-measure the same number N times.
    """
    elem_bytes = {ttnn.bfloat16: 2, ttnn.float32: 4}[dtype]
    p = plan(shape, src_cores, dst_cores, wt_chunk, elem_bytes)

    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    # Distinct values everywhere so a mis-addressed gather cannot alias into a
    # passing comparison.
    n = 1
    for d in shape:
        n *= d
    torch_input = (torch.arange(n, dtype=torch.float32) % 4096).reshape(shape).to(torch_dtype)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=width_shard(shape, src_cores),
    )
    tt_out = ttnn.from_torch(
        torch.zeros(shape, dtype=torch_dtype),
        dtype=dtype,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=height_shard(shape, dst_cores),
    )

    descriptor = build(tt_in, tt_out, variant=variant, p=p, dst_cores=dst_cores)

    ttnn.generic_op([tt_in, tt_out], descriptor)  # warm: compile + program cache
    ttnn.synchronize_device(device)
    read_kernel_ns(device)  # discard the warm window

    ttnn.generic_op([tt_in, tt_out], descriptor)
    ttnn.synchronize_device(device)
    ns = read_kernel_ns(device)

    got = ttnn.to_torch(tt_out)
    want = golden(torch_input, p, dst_cores, strip=(variant == 4))
    transfers = p["nt_h_local"] * p["n_chunks"] * _transfers_per_block(variant, p)
    logger.info(
        f"GATHER {label} shape={list(shape)} W{src_cores}->H{dst_cores} chunk={wt_chunk} "
        f"row_bytes={p['row_bytes']} page={p['page_bytes']} blocks/core={p['nt_h_local'] * p['n_chunks']} "
        f"xfers/core={transfers} ns={ns}"
    )
    assert torch.equal(got, want), f"{label}: gather is NOT bit-exact"
    assert ns is not None, "profiler produced no data"
    return ns


def _transfers_per_block(variant, p):
    slices = max(1, p["row_bytes"] // p["page_bytes"])
    if variant in (0, 1):
        return p["tile_h"] * slices
    if variant == 2:
        return 1
    return slices
