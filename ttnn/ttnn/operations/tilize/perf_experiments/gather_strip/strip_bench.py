# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end bake-off for the STRIP form of tilize's cross-core L1 gather.

Perf 1 measured this idea with NO compute kernel at all, so it never priced the
thing that actually blocks graduation: the strip layout changes what the input CB
slot contains (strip-major instead of row-major), which changes the tilize
contract. This bench puts the REAL pipeline back in the loop — reader + the
library `compute_kernel_lib::tilize` + the local-shard writer drain — and
correctness-gates every arm bit-exactly against torch.

Plan reconstructed: the op's `reshard` path (tilize_program_descriptor.py,
W_REGION + P_ACCESSOR in / P_LOCAL_SHARD out):

  * source      ROW_MAJOR, WIDTH-sharded L1 over `src_cores`. Its page is ONE
    SHARD ROW, so a tensor row is `src_cores` pages on `src_cores` different
    cores' L1 — the gather.
  * destination TILE, HEIGHT-sharded L1 over `dst_cores`; the output CB is
    ALIASED on the resident shard, so compute packs in place and the writer
    only drains.

Arms:
  op         the REAL op (`ttnn.tilize` through the production descriptor) — the
             honest end-to-end baseline, and the check that the reconstruction
             below is faithful.
  row        this bench's reconstruction of the op's CURRENT gather (one transfer
             per source row per page slice) + tilize<WT_CHUNK>(blocks).
  strip      one transfer per (block, source shard), strip-major slot, and
             tilize<PAGE_TILES>(blocks * slices) — the SAME library helper, the
             SAME tile sequence, at a different block width.
  strip_fine strip transfers with a per-STRIP CB handshake (finer pipeline).
"""

import os

# In-process device profiler — all three, before the device opens.
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

from pathlib import Path


# `ttnn/` may not import torch at module scope (validate_no_global_torch_imports).
def _load_torch():
    global torch
    import torch


_load_torch()
import ttnn
from loguru import logger

from ttnn.operations.tilize import tilize as tilize_op
from ttnn.operations.tilize import tilize_program_descriptor as pd

KDIR = Path(__file__).resolve().parent / "experiment_kernels"
READER = str(KDIR / "strip_reader.cpp")
COMPUTE = str(KDIR / "strip_compute.cpp")
WRITER = str(KDIR / "strip_writer.cpp")

TILE_H = 32
TILE_W = 32
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

MODES = {"row": 0, "strip": 1, "strip_fine": 2}
ARMS = ("op", "row", "strip", "strip_fine")

_ELEM = {ttnn.bfloat16: 2, ttnn.float32: 4}
_TORCH = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}


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


def plan(shape, dtype, src_cores, dst_cores, wt_chunk=None, h_in=None):
    """Block geometry — ONE source for the kernel args, the golden and the domain.

    `wt_chunk=None` takes the production host's own choice (derive_shard_blocking
    against the same L1 cap), so the `row` arm really is the op's blocking.
    """
    h, w = shape[-2], shape[-1]
    elem = _ELEM[dtype]
    in_tile_bytes = TILE_H * TILE_W * elem
    shard_wt = w // TILE_W
    if wt_chunk is None:
        # out CB is aliased (0 streaming bytes on that side) — the op's own cap.
        cap = pd.wt_cap(2, in_tile_bytes, 0)
        wt_chunk, n_chunks = pd.derive_shard_blocking(shard_wt, cap)
    else:
        assert shard_wt % wt_chunk == 0
        n_chunks = shard_wt // wt_chunk
    page_bytes = (w // src_cores) * elem
    row_bytes = wt_chunk * TILE_W * elem
    shard_rows = h // dst_cores
    assert shard_rows % TILE_H == 0
    return dict(
        h_in=h if h_in is None else h_in,
        elem=elem,
        in_tile_bytes=in_tile_bytes,
        wt_chunk=wt_chunk,
        n_chunks=n_chunks,
        shard_wt=shard_wt,
        page_bytes=page_bytes,
        page_tiles=page_bytes // (TILE_W * elem),
        row_pages=src_cores,
        row_bytes=row_bytes,
        nt_h_local=shard_rows // TILE_H,
        blocks_per_core=(shard_rows // TILE_H) * n_chunks,
        # whole-page slices: the strip form's ONLY structural precondition
        strip_ok=(row_bytes % page_bytes == 0) and (page_bytes % (TILE_W * elem) == 0),
        slices=max(1, row_bytes // page_bytes),
    )


def build(input_tensor, output_tensor, *, mode, p, dst_cores):
    all_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dst_cores - 1, 0))})
    strip = mode != 0
    # The compute/writer block width. The strip layout is consumed by the SAME
    # library helper at PAGE_TILES width over `slices` x more blocks.
    drain_wt = p["page_tiles"] if strip else p["wt_chunk"]
    compute_blocks = p["blocks_per_core"] * (p["slices"] if strip else 1)

    reader_ct = [
        mode,
        TILE_H,
        p["wt_chunk"],
        p["row_bytes"],
        p["page_bytes"],
        p["row_pages"],
        p["n_chunks"],
        p["page_tiles"],
        p["slices"],
        p["h_in"],
        p["elem"],
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    reader_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    src_addr = input_tensor.buffer_address()
    for k in range(dst_cores):
        reader_rt[k][0] = [src_addr, k * p["nt_h_local"], p["blocks_per_core"], 0]
        compute_rt[k][0] = [compute_blocks]
        writer_rt[k][0] = [compute_blocks]

    tile_descriptor = ttnn.TileDescriptor(TILE_H, TILE_W)
    out_tile_bytes = output_tensor.buffer_page_size()
    # Streaming input CB — the op's own formula: CB_DEPTH * NT_BLK * WT_CHUNK.
    cb_in = ttnn.CBDescriptor(
        total_size=2 * p["wt_chunk"] * p["in_tile_bytes"],
        core_ranges=all_cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=0,
                data_format=input_tensor.dtype,
                page_size=p["in_tile_bytes"],
                tile=tile_descriptor,
            )
        ],
    )
    # Output CB ALIASED on the resident output shard (zero-copy), as the op does.
    cb_out = ttnn.cb_descriptor_from_sharded_tensor(16, output_tensor, core_ranges=all_cores)
    cb_out.total_size = p["nt_h_local"] * p["shard_wt"] * out_tile_bytes
    cb_out.format_descriptors = [
        ttnn.CBFormatDescriptor(
            buffer_index=16,
            data_format=output_tensor.dtype,
            page_size=out_tile_bytes,
            tile=tile_descriptor,
        )
    ]

    compute_config = ttnn.ComputeConfigDescriptor()
    # PRECISION CONTRACT — copied verbatim from the production descriptor and
    # IDENTICAL across every arm: fp32 -> fp32 must be bit-exact, so Dest stays
    # fp32 and the unpacker is stopped from downgrading to tf32.
    lossless_fp32 = input_tensor.dtype == ttnn.float32 and output_tensor.dtype == ttnn.float32
    compute_config.fp32_dest_acc_en = lossless_fp32
    if lossless_fp32:
        unpack_modes = [ttnn.UnpackToDestMode.Default] * 32
        unpack_modes[0] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute_config.unpack_to_dest_mode = unpack_modes

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=READER,
            core_ranges=all_cores,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=WRITER,
            core_ranges=all_cores,
            compile_time_args=[drain_wt],
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=COMPUTE,
            core_ranges=all_cores,
            compile_time_args=[drain_wt],
            runtime_args=compute_rt,
            config=compute_config,
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[cb_out, cb_in])


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


def _make_input(shape, dtype, device, src_cores):
    n = 1
    for d in shape:
        n *= d
    # Distinct values everywhere so a mis-addressed gather cannot alias into a
    # passing comparison (bf16 holds every integer < 256 exactly; the modulus
    # keeps it exact so `torch.equal` is a real bar).
    torch_input = (torch.arange(n, dtype=torch.float32) % 251).reshape(shape).to(_TORCH[dtype])
    tt_in = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=width_shard(shape, src_cores),
    )
    return torch_input, tt_in


def run(device, *, shape, dtype, src_cores, dst_cores, arm, wt_chunk=None, h_in=None, label=""):
    """Correctness-gate the arm, then ONE fresh measured launch.

    `shape` is the (padded) TARGET; `h_in` (if given) is the source's real row
    count, i.e. the padded R_PAD path — the source tensor is [.., h_in, W] and the
    reader materializes the fill into the CB as it reads, exactly as the op does.

    Device kernel duration has no warm-up transient, so a trial loop would just
    re-measure the same number N times.
    """
    p = plan(shape, dtype, src_cores, dst_cores, wt_chunk, h_in)
    src_shape = list(shape) if h_in is None else list(shape[:-2]) + [h_in, shape[-1]]
    torch_src, tt_in = _make_input(src_shape, dtype, device, src_cores)
    out_mem = height_shard(shape, dst_cores)
    if h_in is None:
        torch_want = torch_src
    else:
        pad = torch.zeros(list(shape[:-2]) + [shape[-2] - h_in, shape[-1]], dtype=_TORCH[dtype])
        torch_want = torch.cat([torch_src, pad], dim=-2)

    if arm == "op":
        call = dict(memory_config=out_mem)
        if h_in is not None:
            call.update(output_padded_shape=list(shape), pad_value=0.0)
        tilize_op(tt_in, **call)
        ttnn.synchronize_device(device)
        read_kernel_ns(device)
        out = tilize_op(tt_in, **call)
        ttnn.synchronize_device(device)
        ns = read_kernel_ns(device)
    else:
        tt_out = ttnn.from_torch(
            torch.zeros(shape, dtype=_TORCH[dtype]),
            dtype=dtype,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=out_mem,
        )
        descriptor = build(tt_in, tt_out, mode=MODES[arm], p=p, dst_cores=dst_cores)
        ttnn.generic_op([tt_in, tt_out], descriptor)  # warm: compile + program cache
        ttnn.synchronize_device(device)
        read_kernel_ns(device)  # discard the warm window

        ttnn.generic_op([tt_in, tt_out], descriptor)
        ttnn.synchronize_device(device)
        ns = read_kernel_ns(device)
        out = tt_out

    got = ttnn.to_torch(out)
    xfers = p["blocks_per_core"] * (p["slices"] if arm != "row" else TILE_H * p["slices"])
    logger.info(
        f"STRIP {label} shape={list(shape)} h_in={p['h_in']} W{src_cores}->H{dst_cores} "
        f"wt_chunk={p['wt_chunk']} n_chunks={p['n_chunks']} page_tiles={p['page_tiles']} "
        f"slices={p['slices']} blocks/core={p['blocks_per_core']} xfers/core={xfers} ns={ns}"
    )
    assert torch.equal(got.to(torch.float32), torch_want.to(torch.float32)), f"{label}: NOT bit-exact"
    assert ns is not None, "profiler produced no data"
    return ns
