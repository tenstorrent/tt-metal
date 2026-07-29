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

``depth`` itself is gated (Refinement 1, lever C16): ``use_double_buffer=None``
— the public default — asks the planner for depth-2 only in the regime where it
was *measured* to pay (``depth2_pays``); ``True``/``False`` force it. See
``A0_KNEE_CORES`` and ``BANDWIDTH_KNEE_CORES`` for the measurements behind both
gates.
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

# --- A0 active-core criterion: min(grid, total_tiles, A0_KNEE_CORES) ----------
#
# master.md Part 2 A0 states the criterion as `active == min(grid, total_tiles,
# bandwidth_knee)`, and `examples/dram_saturation/report.md` measures that knee at
# ~16 cores @ 190.9 GB/s for a *large-transaction* DRAM copy (16 -> 64 buys
# +1.5 %). tilize's knee was MEASURED for tilize's own transfer shapes
# (probes/probe_009.py + probe_010.py, Refinement 1) and it is **the whole grid**:
#
#   d_tall_narrow [1,1,2048,32], forced core cap -> device ns (median of 5x10)
#     64c 3 623 | 32c 5 186 | 16c 8 580 | 8c 14 780 | 4c 27 950 | 1c 107 561
#
# i.e. latency is ~linear in tiles-per-core: capping at 16 cores is 2.4x SLOWER,
# not ~2x faster. Two measured reasons the bandwidth knee never binds here:
#   1. a W=32 bf16 ROW_MAJOR input has 64 B DRAM pages, so the reader issues 64 B
#      transactions. The NoC model puts 64 B interleaved DRAM reads at
#      0.68-1.41 B/cyc/core, i.e. 45-90 GB/s aggregate over 64 cores -- the
#      190.9 GB/s knee is UNREACHABLE for this shape at any core count. The op is
#      read-transaction-rate bound, not DRAM-bandwidth bound.
#   2. the sync/dispatch floor scales with BLOCKS PER CORE, not with core count
#      (sync_only: 64c/1blk 1 202 ns, 16c/4blk 3 079 ns, 4c/16blk 10 677 ns
#      ~= 590 + 612*blocks), so shedding cores *adds* sync cost.
#
# Keep the term in the formula (it is A0's criterion, and a future shape family
# with big transactions could re-introduce a real knee) but set it above any
# current compute grid => identity. Changing this constant requires re-running
# probes/probe_009.py.
A0_KNEE_CORES = 64

# --- C16 depth-2 default gate (master.md C16 "but only when it pays") ---------
# Depth-2 buys read/write overlap across a block boundary. Measured
# (probes/probe_010.py + the paired in-run `x_*_depth2` bench rows, 7 rounds,
# CV <= 1.2 %), it pays in exactly two situations and is dead L1 otherwise:
#
#  1. **Below the DRAM bandwidth-saturation knee** the binding resource is the
#     core's OWN NoC issue rate, so overlapping its reader and writer is a large
#     win: c_single_core depth1/depth2 = **1.321**, x_wide_short_1core = **1.360**.
#  2. **At or above the knee** DRAM aggregate bandwidth is the binding resource
#     and both depths already reach it -- but each block boundary still costs one
#     un-overlapped fill/drain, and those add up with the block count:
#       blk/core  depth1/depth2   verdict
#         1       0.995 - 1.010   depth-2 structurally inert (nothing to overlap)
#         4       0.998 / 1.005   free  (a_square, e_square_bf8b_out)
#         8       1.019 - 1.028   costs ~2 %  (e_square_fp32, e_square_fp32_to_bf16,
#                                 g_sharded_to_dram -- 3 independent regimes, same sign)
#     so depth-1 is the default only up to DEPTH1_MAX_BLOCKS_PER_CORE boundaries.
#
# NB this is narrower than the refinement's proposal ("default off once the op is
# DRAM-saturated with large per-core work"): measurement says *large* per-core
# work is precisely where the residual overlap still pays. 4 is measured free and
# 8 measured costly; 5-7 is unmeasured, so the threshold sits at the conservative
# end of that gap.
BANDWIDTH_KNEE_CORES = 16
MIN_BLOCKS_FOR_DEPTH2 = 2
DEPTH1_MAX_BLOCKS_PER_CORE = 4
# Sweep hook: probes set this to force a core cap while re-measuring the knee.
# None => A0_KNEE_CORES decides. Never set in production.
CORE_CAP_OVERRIDE = None

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


def a0_active_cores(grid_cores: int, total_tiles: int) -> int:
    """master.md A0: ``min(grid, total_tiles, bandwidth_knee)``.

    The single place the active-core count is decided for the generic path, so
    the bench / unit-test A0 assert can check the *declared* criterion instead of
    re-deriving it. See ``A0_KNEE_CORES`` for why the knee term is identity on
    this op (measured, not assumed).
    """
    cap = A0_KNEE_CORES if CORE_CAP_OVERRIDE is None else int(CORE_CAP_OVERRIDE)
    return max(1, min(grid_cores, total_tiles, cap))


def depth2_pays(ncores: int, blocks_per_core: int) -> bool:
    """C16 gate: is depth-2 worth 2x the per-core CB L1 on this plan?

    Three measured clauses (numbers in the ``BANDWIDTH_KNEE_CORES`` comment):

    1. fewer than ``MIN_BLOCKS_FOR_DEPTH2`` blocks -> **no**: there is no block
       boundary to overlap, so depth-2 cannot do anything except cost L1.
    2. below the DRAM bandwidth-saturation knee -> **yes**: the core's own NoC
       issue rate is the bound and overlapping its reader/writer is worth 1.3x.
    3. at or above the knee -> **only past ``DEPTH1_MAX_BLOCKS_PER_CORE``**: DRAM
       aggregate bandwidth is the bound, but each un-overlapped block boundary
       still costs, and beyond 4 boundaries that reaches ~2 %.
    """
    if blocks_per_core < MIN_BLOCKS_FOR_DEPTH2:
        return False
    if ncores < BANDWIDTH_KNEE_CORES:
        return True
    return blocks_per_core > DEPTH1_MAX_BLOCKS_PER_CORE


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


def build_plan(input_tensor, output_tensor, device, *, use_multicore=True, use_double_buffer=None):
    """Evaluate the host planner once per program build.

    ``use_double_buffer=None`` (the public default) means *the planner decides*:
    depth-2 only where it was measured to pay (see ``depth2_pays``). ``True`` /
    ``False`` force depth-2 / depth-1 and keep their documented meaning.

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

    chunk_cap = None
    if use_double_buffer is None:
        # C16 gate. The depth feeds the L1 chunk-width budget, which feeds the
        # 2D split, which decides ncores / blocks-per-core -- i.e. the gate's own
        # inputs. Resolve it with a depth-2 trial plan (pure host arithmetic, no
        # device work) and re-plan once at the chosen depth.
        trial = _plan_generic(dict(plan), input_tensor, device, in_geo, use_multicore=use_multicore, depth_request=2)
        if depth2_pays(trial["ncores"], trial["blocks_per_core"]):
            depth_request = 2
        else:
            depth_request = 1
            # Pin the chunk width to the depth-2 plan's, so the *only* difference
            # between the gated plan and the ungated one is that the CB has half
            # the pages. Letting the freed L1 grow the chunk instead would change
            # the reader's transaction size and the work split behind the caller's
            # back -- measured a 1.3 % LOSS on e_square_fp32 (chunk 8 -> 16) with
            # zero L1 saved. Non-regression is then structural, not just measured.
            chunk_cap = trial["chunk_wt"]
    else:
        depth_request = 2 if use_double_buffer else 1

    return _plan_generic(
        plan,
        input_tensor,
        device,
        in_geo,
        use_multicore=use_multicore,
        depth_request=depth_request,
        chunk_cap=chunk_cap,
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
            "blocks_per_core": num_blocks,
            "depth": 1,  # the CB *is* the shard; use_double_buffer is inert here
            "row_page_stride": 1,
            "source_page_bytes": shard_w * plan["elem_in"],
            "chunk_row_bytes": shard_w * plan["elem_in"],
            "ncores": len(cores),
            "cb_bytes_per_core": shard_tiles * (plan["tile_in"] + plan["tile_out"]),
        }
    )
    return plan


def _plan_generic(plan, input_tensor, device, in_geo, *, use_multicore, depth_request, chunk_cap=None):
    """Path A/C: 2D height-first rectangular split over the compute grid.

    ``chunk_cap`` pins the chunk width from a previous (depth-2) pass so the C16
    depth gate cannot change the transaction shape — see ``build_plan``.
    """
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
    if not use_multicore:
        # use_multicore=False means EXACTLY one core (the acceptance test and the
        # c_single_core bench regime depend on it) -- the A0 knee term is a G clamp
        # inside the multicore path, never a new user-visible mode.
        max_cores = 1
    else:
        max_cores = a0_active_cores(grid_cores, plan["total_tiles"])

    bytes_per_chunk_tile = tile_in + tile_out
    # "Depth-2 only if it fits": the smallest possible depth-2 footprint is one
    # chunk tile-pair. If even that exceeds the budget, fall back to depth 1
    # rather than OOM (the ttnn.concat pattern). Decided BEFORE the chunk width
    # so `max_chunk_l1` is computed against the depth actually used; the
    # post-loop assert below is then an invariant, not a second clamp.
    depth = depth_request
    if depth * bytes_per_chunk_tile > L1_CB_BUDGET_BYTES:
        depth = 1
    max_chunk_l1 = max(1, L1_CB_BUDGET_BYTES // (depth * bytes_per_chunk_tile))

    n_h = min(nt_h, max_cores)
    want_chunks = _div_up(max_cores, n_h)
    max_chunk_par = max(1, wt // want_chunks)
    max_chunk = min(WT_CHUNK_MAX, max_chunk_l1, max_chunk_par)
    if chunk_cap is not None:
        max_chunk = min(max_chunk, chunk_cap)

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
            # Busiest core's chunk-block count -- the C16 gate's "is there
            # anything to pipeline?" input, and the per-block sync cost's
            # multiplier (measured ~612 ns/block, see A0_KNEE_CORES).
            "blocks_per_core": max(u["row_count"] * u["chunk_count"] for u in work),
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
