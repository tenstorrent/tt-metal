# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""``tilize`` — host planner + ProgramDescriptor.

Two dataflow paths (see ``op_design.md`` "Dataflow Strategy"):

* **Path A/C — generic** (``path="generic"``).  RM sticks are read through a
  ``TensorAccessor`` into a tile-page input CB, tilized, and written back as
  whole TILE pages through the output ``TensorAccessor``.  The work unit is a
  *chunk-block* = 32 rows x ``chunk_wt`` tile-columns; each core owns a 2D
  rectangle (contiguous tile-row range x contiguous column-chunk range), so the
  split degenerates to pure-height when height fills the grid and to pure-width
  when ``nt_h == 1``.  Covers interleaved I/O and every
  interleaved<->sharded / cross-spec-sharded combination.  When the input is
  ROW_MAJOR-sharded with ``pages_per_row > 1`` the reader switches to a raw
  strided read (the helper hard-codes one page per logical row).

* **Path B — aliased, zero-copy** (``path="alias"``).  Same-spec L1-sharded in
  and out: both CBs are built with ``cb_descriptor_from_sharded_tensor`` so the
  CB base address *is* the shard base address.  Zero NoC traffic on both sides;
  the reader degenerates to one ``cb_push_back`` and the writer to one
  ``cb_wait_front``/``cb_pop_front``.

Only two CBs in either path — tilize is a single-phase compute with no
intermediate.  Per-core CB L1 is ``depth * chunk_wt * (tile_in + tile_out)``
with ``chunk_wt <= WT_CHUNK_MAX``, i.e. bounded by a constant in ``W``.
"""

from __future__ import annotations

import os
from math import gcd, prod
from pathlib import Path

import ttnn

from ttnn.operations._op_contract import UnsupportedAxisValue

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_HW = 32

# Caps the reader transaction at 1024 B (bf16) / 2048 B (fp32) and bounds the
# per-core CB footprint independently of W.
WT_CHUNK_MAX = 16
# Conservative literal: there is no `device.l1_size_per_core()` Python binding
# on this build. Both CBs combined.
L1_CB_BUDGET_BYTES = 131072

# Fast-tilize LLK limit (`can_use_fast_tilize`: block_width_tiles < 256).
MAX_BLOCK_WIDTH_TILES = 256

CB_RM_INPUT = 0
CB_TILED_OUTPUT = 16


# ---------------------------------------------------------------------------
# Small integer helpers (ttnn.div_up / round_up / find_max_divisor are not
# bound on this build — verified).
# ---------------------------------------------------------------------------


def _div_up(a: int, b: int) -> int:
    return -(-a // b)


def _largest_divisor_le(n: int, limit: int) -> int:
    """Largest divisor of ``n`` that is <= ``limit`` (never skips 5 or 7)."""
    limit = max(1, min(limit, n))
    for d in range(limit, 0, -1):
        if n % d == 0:
            return d
    return 1


def _split_contiguous(total: int, parts: int):
    """``parts`` contiguous (start, count) ranges covering ``total`` units.

    The first ``total % parts`` partitions get one extra unit.
    """
    base, rem = divmod(total, parts)
    ranges = []
    start = 0
    for i in range(parts):
        count = base + (1 if i < rem else 0)
        ranges.append((start, count))
        start += count
    return ranges


# ---------------------------------------------------------------------------
# Shard geometry
# ---------------------------------------------------------------------------


def _shard_geometry(tensor):
    """2D-normalised shard geometry, or None when the tensor is interleaved."""
    memory_config = tensor.memory_config()
    if not memory_config.is_sharded():
        return None

    shard_spec = memory_config.shard_spec
    if shard_spec is not None:
        shard_h = int(shard_spec.shape[0])
        shard_w = int(shard_spec.shape[1])
        grid = shard_spec.grid
        orientation = shard_spec.orientation
    else:
        nd = memory_config.nd_shard_spec
        if nd is None:
            return None
        shard_shape = list(nd.shard_shape)
        shard_h = int(prod(shard_shape[:-1]))
        shard_w = int(shard_shape[-1])
        grid = nd.grid
        orientation = nd.orientation

    return {
        "h": shard_h,
        "w": shard_w,
        "grid": grid,
        "grid_key": str(grid),
        "orientation": orientation,
        "layout": memory_config.memory_layout,
        "buffer": memory_config.buffer_type,
    }


def _alias_eligible(in_geo, out_geo, folded_h: int, width: int) -> bool:
    """True iff the same-spec zero-copy path (Path B) applies."""
    if in_geo is None or out_geo is None:
        return False
    if in_geo["buffer"] != ttnn.BufferType.L1 or out_geo["buffer"] != ttnn.BufferType.L1:
        return False
    for key in ("h", "w", "orientation", "layout", "grid_key"):
        if in_geo[key] != out_geo[key]:
            return False

    shard_h, shard_w = in_geo["h"], in_geo["w"]
    if shard_h % TILE_HW or shard_w % TILE_HW:
        return False
    if folded_h % shard_h or width % shard_w:
        return False
    if (folded_h // shard_h) * (width // shard_w) != in_geo["grid"].num_cores():
        return False
    # Whole shard width is one tilize block, so it must fit the LLK limit.
    return shard_w // TILE_HW < MAX_BLOCK_WIDTH_TILES


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


def build_plan(input_tensor, output_tensor, device, *, use_multicore=True, use_double_buffer=True):
    """Evaluate the host planner once per program build.

    The tile grid is derived from the **output** tensor's padded shape — that is
    the page grid the writer addresses. A ROW_MAJOR-*sharded* input can carry
    extra padding on its last dim (its width is rounded up to a whole number of
    shard widths, e.g. logical W=160 with shard_W=96 stores a padded W=192), and
    that padding is a source *stride* concern only. Deriving the tile grid from
    the input's padded shape would invent tile columns that do not exist in the
    output and silently corrupt every page index.
    """
    out_padded = list(output_tensor.padded_shape)
    in_padded = list(input_tensor.padded_shape)

    folded_h = int(prod(out_padded[:-1]))
    width = int(out_padded[-1])
    nt_h = folded_h // TILE_HW
    wt = width // TILE_HW
    total_tiles = nt_h * wt

    # Only the last dim may differ between the two padded shapes; anything else
    # means the row fold is not the same on both sides and the plain
    # "flatten the leading dims" mapping below would not hold.
    if in_padded[:-1] != out_padded[:-1]:
        raise UnsupportedAxisValue(
            f"tilize: input padded shape {in_padded} and output padded shape "
            f"{out_padded} disagree on the leading dims — the row fold is not "
            "expressible as a single flatten"
        )
    if int(in_padded[-1]) < width:
        raise UnsupportedAxisValue(
            f"tilize: input padded width {in_padded[-1]} is narrower than the " f"output width {width}"
        )

    elem_in = input_tensor.element_size()
    tile_in = ttnn.tile_size(input_tensor.dtype)
    tile_out = ttnn.tile_size(output_tensor.dtype)
    tile_row_bytes = TILE_HW * elem_in

    in_geo = _shard_geometry(input_tensor)
    out_geo = _shard_geometry(output_tensor)

    plan = {
        "folded_h": folded_h,
        "width": width,
        "in_padded_width": int(in_padded[-1]),
        "nt_h": nt_h,
        "wt": wt,
        "total_tiles": total_tiles,
        "elem_in": elem_in,
        "tile_in": tile_in,
        "tile_out": tile_out,
        "tile_row_bytes": tile_row_bytes,
        "needs_cast": int(output_tensor.dtype != input_tensor.dtype),
    }

    # Path B is inherently multi-core (one shard per core), so an explicit
    # use_multicore=False request routes to the generic single-core path
    # instead of being refused.
    if use_multicore and _alias_eligible(in_geo, out_geo, folded_h, width):
        return _plan_alias(plan, in_geo)

    return _plan_generic(
        plan,
        input_tensor,
        device,
        in_geo,
        use_multicore=use_multicore,
        use_double_buffer=use_double_buffer,
    )


def _plan_alias(plan, geo):
    """Path B: one resident shard per core, no NoC traffic on either side."""
    shard_h, shard_w = geo["h"], geo["w"]
    chunk_wt = shard_w // TILE_HW
    num_blocks = shard_h // TILE_HW
    shard_tiles = chunk_wt * num_blocks

    grid = geo["grid"]
    cores = []
    for core_range in grid.ranges():
        cores.extend(ttnn.grid_to_cores(core_range.start, core_range.end, True))

    plan.update(
        {
            "path": "alias",
            "core_ranges": grid,
            "cores": cores,
            "chunk_wt": chunk_wt,
            "shard_tiles": shard_tiles,
            "num_blocks": num_blocks,
            "depth": 1,  # the CB *is* the shard
            "row_page_stride": 1,
            "source_page_bytes": shard_w * plan["elem_in"],
            "chunk_row_bytes": shard_w * plan["elem_in"],
            "ncores": len(cores),
            "cb_bytes_per_core": shard_tiles * (plan["tile_in"] + plan["tile_out"]),
        }
    )
    return plan


def _plan_generic(plan, input_tensor, device, in_geo, *, use_multicore, use_double_buffer):
    """Path A/C: 2D height-first rectangular split over the compute grid."""
    nt_h, wt = plan["nt_h"], plan["wt"]
    tile_in, tile_out = plan["tile_in"], plan["tile_out"]
    elem_in = plan["elem_in"]
    width = plan["width"]

    # --- source page geometry (one page == one stick of `page_bytes`) --------
    # NB: the stride is measured against the input's *padded* row, which for a
    # ROW_MAJOR-sharded input may be wider than the logical/tile row.
    in_page_bytes = input_tensor.buffer_page_size()
    in_padded_row_bytes = plan["in_padded_width"] * elem_in
    if in_padded_row_bytes % in_page_bytes:
        raise UnsupportedAxisValue(
            f"tilize: input padded row of {in_padded_row_bytes} B is not a whole " f"number of {in_page_bytes} B pages"
        )
    row_page_stride = in_padded_row_bytes // in_page_bytes

    if in_page_bytes % (TILE_HW * elem_in):
        raise UnsupportedAxisValue(
            f"tilize: input page of {in_page_bytes} B is not a whole number of " f"{TILE_HW * elem_in} B tile-columns"
        )

    # A chunk must never straddle a source page, so when a logical row spans
    # several pages the chunk width has to divide BOTH Wt (for the column split)
    # and the page width in tiles (so `byte_offset` stays inside one page).
    page_wt = in_page_bytes // (TILE_HW * elem_in)
    chunk_unit = wt if row_page_stride == 1 else gcd(wt, page_wt)

    # --- planner (op_design.md "Host planner") ------------------------------
    grid = device.compute_with_storage_grid_size()
    grid_cores = grid.x * grid.y
    max_cores = 1 if not use_multicore else min(grid_cores, plan["total_tiles"])

    bytes_per_chunk_tile = tile_in + tile_out
    # "Depth-2 only if it fits": the smallest possible depth-2 footprint is one
    # chunk tile-pair. If even that exceeds the budget, fall back to depth 1
    # rather than OOM (the ttnn.concat pattern). Decided BEFORE the chunk width
    # so `max_chunk_l1` is computed against the depth actually used; the
    # post-loop assert below is then an invariant, not a second clamp.
    depth = 2 if use_double_buffer else 1
    if depth * bytes_per_chunk_tile > L1_CB_BUDGET_BYTES:
        depth = 1
    max_chunk_l1 = max(1, L1_CB_BUDGET_BYTES // (depth * bytes_per_chunk_tile))

    n_h = min(nt_h, max_cores)
    want_chunks = _div_up(max_cores, n_h)
    max_chunk_par = max(1, wt // want_chunks)
    max_chunk = min(WT_CHUNK_MAX, max_chunk_l1, max_chunk_par)

    chunk_wt = _largest_divisor_le(chunk_unit, max_chunk)
    assert wt % chunk_wt == 0, f"chunk_wt={chunk_wt} must divide Wt={wt}"
    assert depth * chunk_wt * bytes_per_chunk_tile <= L1_CB_BUDGET_BYTES, (
        f"CB budget blown: depth={depth} chunk_wt={chunk_wt} "
        f"bytes_per_chunk_tile={bytes_per_chunk_tile} > {L1_CB_BUDGET_BYTES}"
    )
    n_chunks = wt // chunk_wt
    n_w = min(n_chunks, max(1, max_cores // n_h))
    ncores = n_h * n_w

    cores = ttnn.grid_to_cores(ncores, grid.x, grid.y, True)
    core_ranges = ttnn.num_cores_to_corerangeset(ncores, grid, True)

    row_ranges = _split_contiguous(nt_h, n_h)
    chunk_ranges = _split_contiguous(n_chunks, n_w)

    work = []
    for i in range(n_h):
        row_start, row_count = row_ranges[i]
        for j in range(n_w):
            chunk_start, chunk_count = chunk_ranges[j]
            work.append(
                {
                    "core": cores[i * n_w + j],
                    "row_start": row_start,
                    "row_count": row_count,
                    "chunk_start": chunk_start,
                    "chunk_count": chunk_count,
                }
            )

    plan.update(
        {
            "path": "generic",
            "core_ranges": core_ranges,
            "cores": cores,
            "work": work,
            "chunk_wt": chunk_wt,
            "chunk_row_bytes": chunk_wt * TILE_HW * elem_in,
            "row_page_stride": row_page_stride,
            "source_page_bytes": in_page_bytes,
            "shard_tiles": 0,
            "depth": depth,
            "n_h": n_h,
            "n_w": n_w,
            "ncores": ncores,
            "cb_bytes_per_core": depth * chunk_wt * bytes_per_chunk_tile,
        }
    )
    return plan


# ---------------------------------------------------------------------------
# ComputeConfigDescriptor
# ---------------------------------------------------------------------------

_FP32_DEST_IN = (ttnn.float32, ttnn.uint32, ttnn.int32)
_FP32_DEST_OUT = (ttnn.float32, ttnn.bfloat8_b, ttnn.uint32, ttnn.int32)


def _compute_config(in_dtype, out_dtype):
    fp32_dest_acc_en = in_dtype in _FP32_DEST_IN or out_dtype in _FP32_DEST_OUT

    config = ttnn.ComputeConfigDescriptor()
    config.fp32_dest_acc_en = fp32_dest_acc_en
    # `can_use_fast_tilize` requires !get_dst_full_sync_enabled().
    config.dst_full_sync_en = False
    if fp32_dest_acc_en:
        # Must be assigned wholesale: nanobind's bound vector copies on
        # __getitem__, so in-place element assignment is silently dropped.
        modes = [ttnn.UnpackToDestMode.Default] * 32
        modes[CB_RM_INPUT] = ttnn.UnpackToDestMode.UnpackToDestFp32
        config.unpack_to_dest_mode = modes
    return config


# ---------------------------------------------------------------------------
# CB descriptors
# ---------------------------------------------------------------------------


def _plain_cb(index, dtype, page_size, num_pages, core_ranges):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
    )


def _aliased_cb(index, tensor, page_size, num_pages, core_ranges):
    """CB whose L1 base address *is* the tensor's shard base address."""
    cb = ttnn.cb_descriptor_from_sharded_tensor(
        index, tensor, total_size=num_pages * page_size, core_ranges=core_ranges
    )
    # Read-modify-write-back: the bound vector copies on __getitem__.
    format_descriptors = cb.format_descriptors
    format_descriptors[0].page_size = page_size
    cb.format_descriptors = format_descriptors
    return cb


# ---------------------------------------------------------------------------
# ProgramDescriptor
# ---------------------------------------------------------------------------


def _ablation_flags():
    """Perf-ablation compile-time flags (/perf-measure stage attribution).

    ``TILIZE_SKIP_DM=1`` drops the noc_async_read/write payload, ``TILIZE_SKIP_COMPUTE=1``
    drops the tilize LLK; both keep every CB op, barrier and loop trip count so the
    synchronization structure — and therefore the timing structure — is unchanged.
    Output is garbage by design; only ``_bench_tilize.py`` sets these.
    """
    return (
        int(os.environ.get("TILIZE_SKIP_DM", "0")),
        int(os.environ.get("TILIZE_SKIP_COMPUTE", "0")),
    )


def create_program_descriptor(input_tensor, output_tensor, plan) -> ttnn.ProgramDescriptor:
    alias = plan["path"] == "alias"
    core_ranges = plan["core_ranges"]
    chunk_wt = plan["chunk_wt"]
    skip_dm, skip_compute = _ablation_flags()

    # ---------------- circular buffers ----------------
    if alias:
        pages = plan["shard_tiles"]
        cb_rm_input = _aliased_cb(CB_RM_INPUT, input_tensor, plan["tile_in"], pages, core_ranges)
        cb_tiled_output = _aliased_cb(CB_TILED_OUTPUT, output_tensor, plan["tile_out"], pages, core_ranges)
    else:
        pages = plan["depth"] * chunk_wt
        cb_rm_input = _plain_cb(CB_RM_INPUT, input_tensor.dtype, plan["tile_in"], pages, core_ranges)
        cb_tiled_output = _plain_cb(CB_TILED_OUTPUT, output_tensor.dtype, plan["tile_out"], pages, core_ranges)

    alias_flag = 1 if alias else 0

    # ---------------- reader ----------------
    reader_ct_args = [
        alias_flag,
        chunk_wt,
        plan["chunk_row_bytes"],
        plan["row_page_stride"],
        plan["source_page_bytes"],
        plan["shard_tiles"],
        skip_dm,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    # ---------------- writer ----------------
    writer_ct_args = [
        alias_flag,
        chunk_wt,
        plan["tile_out"],
        plan["wt"],
        plan["shard_tiles"],
        skip_dm,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # ---------------- compute ----------------
    compute_ct_args = [chunk_wt, plan["needs_cast"], skip_compute]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()

    src_addr = input_tensor.buffer_address()
    dst_addr = output_tensor.buffer_address()

    if alias:
        for core in plan["cores"]:
            reader_rt[core.x][core.y] = [src_addr]
            writer_rt[core.x][core.y] = [dst_addr]
            compute_rt[core.x][core.y] = [plan["num_blocks"]]
    else:
        for unit in plan["work"]:
            core = unit["core"]
            row_start = unit["row_start"]
            row_count = unit["row_count"]
            chunk_start = unit["chunk_start"]
            chunk_count = unit["chunk_count"]
            reader_rt[core.x][core.y] = [
                src_addr,
                row_start * TILE_HW,
                row_count * TILE_HW,
                chunk_start,
                chunk_count,
            ]
            writer_rt[core.x][core.y] = [dst_addr, row_start, row_count, chunk_start, chunk_count]
            compute_rt[core.x][core.y] = [row_count * chunk_count]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tilize_reader.cpp"),
        core_ranges=core_ranges,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tilize_writer.cpp"),
        core_ranges=core_ranges,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tilize_compute.cpp"),
        core_ranges=core_ranges,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt,
        config=_compute_config(input_tensor.dtype, output_tensor.dtype),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=[cb_rm_input, cb_tiled_output],
    )
