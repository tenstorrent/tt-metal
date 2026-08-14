# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""tilize — ProgramDescriptor (CBs, kernels, args).

Blocking model (op_design.md §1): the work unit is a **block** =
1 tile-row x ``WT_CHUNK`` tile-columns. Every knob — ``CB_DEPTH``, ``NT_BLK``,
``WT_CHUNK``, ``NUM_CORES`` — is a parameter with a single source in
``derive_blocking()``; CB page counts, loop trip counts and grid sizing are all
computed *from* those knobs, never from a whole-op dimension.

Blocks are indexed W-chunk-major (``wc = b // NT_H``, ``row = b % NT_H``) so a
core's consecutive blocks share one W chunk and march linearly through the
source page ids.
"""

from __future__ import annotations

import struct
from pathlib import Path

import ttnn


KERNEL_DIR = Path(__file__).parent / "kernels"

# --- fixed hardware / library facts (not knobs) ----------------------------
DEFAULT_TILE_WIDTH = 32  # a tile is always 32 wide
NUM_CIRCULAR_BUFFERS = 32

# --- blocking-model constants (single source; every knob derives from these)-
CB_L1_BUDGET = 1_048_576  # bytes of L1 reserved for the two streaming CBs
FAST_TILIZE_MAX_W = 255  # tilize_helpers.inl:95 -> block_width_tiles < 256

# --- block factors ----------------------------------------------------------
# NT_BLK: tile-rows per reader barrier-block. The library reader
# (read_sticks_for_tilize) barriers once per tile-row, so the Phase-0 value is 1
# and raising it needs a custom reader (op_design.md lamp L3). It is a named
# knob here — never an inlined literal — so the CB formula below is already
# written against it.
NT_BLK = 1

# --- CB indices (semantic names; the numeric slot is just a buffer index) ---
CB_INPUT_STICKS = 0  # reader -> compute  (row-major sticks, tile-sized pages)
CB_OUTPUT_TILES = 16  # compute -> writer (tiled pages)

# --- reader regime selector (op_design.md §5.1) -----------------------------
R_ALIGNED = 0
R_PAD = 1

# --- placement regime selector, per side (op_design.md §5.2) ----------------
# P_LOCAL_SHARD: the CB is ALIASED on the resident L1 shard
# (ttnn.cb_descriptor_from_sharded_tensor) so tilize unpacks straight out of /
# packs straight into the shard — ZERO NoC traffic on that side. The reader
# then only publishes pages it did not fetch, and the writer only drains pages
# it did not send (the drain is kept so the CB still has exactly one consumer).
# P_ACCESSOR: TensorAccessor over interleaved DRAM/L1, or over a NON-local
# L1-sharded tensor (the cross-core L1 gather).
P_ACCESSOR = 0
P_LOCAL_SHARD = 1

# --- work-assignment mode (how a core's block index maps to geometry) -------
# W_BLOCKS: the interleaved split — a contiguous range of the global,
#           W-chunk-major block index (`wc = b // NT_H`, `row = b % NT_H`).
# W_REGION: the sharded split — the core owns the rectangular tile region of
#           its own shard, walked tile-row-major with the W chunk INNERMOST so
#           the push order matches the shard's own linear tile order (which is
#           what lets an aliased output CB take `WT_CHUNK < shard_wt`).
W_BLOCKS = 0
W_REGION = 1

# --- lever counterfactual switches ------------------------------------------
# Each applied perf lever gets an OFF arm here so its payoff can be MEASURED
# (see _bench_tilize.py `levers=dict(...)` arms and lever_ledger.json). Production
# never touches these — every entry is the ON (optimal) value.
#   w_split      0 -> pure height split, no W chunking (grid collapses on a
#                     short/wide shape). op_design.md §1.3 candidate 2.
#   row_wise     0 -> split_work_to_cores column-wise (master.md A1's trap).
#   block_write  0 -> writer barriers per TILE PAGE instead of per block
#                     (master.md B7 off).
#   page_write    0 -> each tile page written as two half-page transactions
#                     (master.md B5 off: sub-page scatter).
#   noc_split     0 -> reader and writer configs SWAPPED (reader on BRISC/NOC1,
#                     writer on NCRISC/NOC0) (master.md B9 off).
#   regime_select 0 -> always take the R_PAD reader, even on an aligned input
#                     (master.md D20 off: no compile-time specialization).
#   fp32_dest     0 -> do not enable fp32 DEST / lossless unpack on fp32->fp32
#                     (master.md F25: the knob this op gates on the dtype pair).
#   multicore     0 -> force the single-core grid regardless of use_multicore
#                     (master.md A0 off arm; the caller kwarg stays authoritative
#                     when this is 1).
#   double_buffer 0 -> force CB_DEPTH 1 regardless of use_double_buffer
#                     (master.md C16 off arm).
LEVERS = {
    "w_split": 1,
    "row_wise": 1,
    "block_write": 1,
    "page_write": 1,
    "noc_split": 1,
    "regime_select": 1,
    "fp32_dest": 1,
    "multicore": 1,
    "double_buffer": 1,
}

# --- classification ablation (perf-only; op_design.md §9.1) ------------------
# Stub a stage's PAYLOAD while keeping every CB reserve/push/wait/pop, barrier
# and loop trip count, so the duration diff attributes time to that stage.
# Production is always {0, 0}; the bench flips these. Output is wrong by design
# when either is set — never assert PCC on an ablated run.
ABLATE = {
    "compute": 0,  # 1 -> compute does the CB handshake but no tilize_block
    "dm": 0,  # 1 -> reader/writer issue no NoC transfers
}


def _prod(values):
    out = 1
    for v in values:
        out *= int(v)
    return out


def _div_up(a, b):
    return -(-a // b)


def cb_pages(cb_depth, wt_chunk):
    """Pages per streaming CB — THE single source for the CB geometry.

    A function of the three block knobs only (``CB_DEPTH``, ``NT_BLK``,
    ``WT_CHUNK``), never of ``WT`` / ``NT_H`` / any tensor dimension. Both CB
    descriptors, ``derive_blocking()``'s L1 ceiling and the depth fallback all
    read this, so turning a knob lands in exactly one place.
    """
    return cb_depth * NT_BLK * wt_chunk


def cb_bytes(cb_depth, wt_chunk, in_tile_bytes, out_tile_bytes):
    """L1 bytes held by the two streaming CBs together, from ``cb_pages()``."""
    return cb_pages(cb_depth, wt_chunk) * (in_tile_bytes + out_tile_bytes)


def wt_cap(cb_depth, in_tile_bytes, out_tile_bytes):
    """L1 ceiling on the W block factor — THE single source for that cap.

    ``in_tile_bytes`` / ``out_tile_bytes`` are the *streaming* page sizes: pass 0
    for a side whose CB is aliased on a resident shard, since that side costs no
    extra L1 (it IS the tensor). When neither side streams, only the library's
    fast-tilize width bound remains.
    """
    per_chunk_tile = cb_bytes(cb_depth, 1, in_tile_bytes, out_tile_bytes)
    if per_chunk_tile == 0:
        return FAST_TILIZE_MAX_W
    return max(1, min(FAST_TILIZE_MAX_W, CB_L1_BUDGET // per_chunk_tile))


def derive_shard_blocking(shard_wt, cap):
    """``WT_CHUNK`` for a core whose W extent is one shard (op_design.md §1.4).

    Same rule as ``derive_blocking()``: the COARSEST exact divisor of the width
    that fits the L1 cap — never the minimal unit. Returns
    ``(wt_chunk, n_chunks)`` with ``n_chunks * wt_chunk == shard_wt``, so every
    block is the same width and one compute kernel covers the core.
    """
    n_chunks = next(c for c in range(1, shard_wt + 1) if shard_wt % c == 0 and shard_wt // c <= cap)
    return shard_wt // n_chunks, n_chunks


def derive_blocking(nt_h, wt, in_tile_bytes, out_tile_bytes, num_cores, cb_depth):
    """The three block knobs — single source of truth (op_design.md §1.4).

    Returns ``(wt_chunk, n_chunks, num_blocks)``.

    * ``WT_CHUNK`` is the COARSEST chunk that fits: the whole tile-row width
      unless the L1 ceiling or the grid-fill floor forces it smaller.
    * ``n_chunks`` divides ``WT`` exactly, so every block has the same width and
      there is exactly one compute kernel (no cliff-width variant).
    * ``NT_H >= NUM_CORES`` implies ``n_chunks == 1``, i.e. the wide-shape
      machinery is inert on tall shapes (byte-identical to a pure height split).
    """
    # Bytes both CBs hold per tile-column of chunk width — read from cb_bytes()
    # via wt_cap() so the ceiling can never drift from the CB sizing below (it
    # carries the NT_BLK factor too, which a hand-written formula here would have
    # dropped) and the sharded path shares the same cap.
    cap = wt_cap(cb_depth, in_tile_bytes, out_tile_bytes)

    n_want = max(1, _div_up(num_cores, nt_h))  # grid-fill floor
    n_want = max(n_want, _div_up(wt, cap))  # L1 ceiling
    n_want = min(n_want, wt)  # can never split W finer than one tile-column

    n_chunks = next(c for c in range(n_want, wt + 1) if wt % c == 0)
    wt_chunk = wt // n_chunks
    return wt_chunk, n_chunks, nt_h * n_chunks


def shard_side_plan(tensor, padded_shape, tile_h, tile_w):
    """The per-core tile region of an L1-sharded tensor — or None when the shard
    cannot back a zero-copy CB (op_design.md §5.2 ``side_regime``).

    Returns ``{"cores", "shard_ht", "shard_wt", "regions"}`` where ``regions[i]``
    is shard *i*'s ``(tile_row0, tile_col0)`` in the folded (tile-row, tile-col)
    grid and ``cores[i]`` is the core that holds it. Both legacy 2-D and ND
    specs go through the SAME derivation: a legacy ShardSpec is exactly an ND
    spec whose shard shape has rank 2 over the folded 2-D view of the tensor,
    which ``memory_config.nd_shard_spec`` already reports.

    None (→ the accessor path) whenever the shard is not a whole number of tiles,
    does not divide the tensor evenly, or does not fold to a CONTIGUOUS band of
    tile-rows. Those are the uneven / padded shard grids a later refinement owns.
    """
    memory_config = tensor.memory_config()
    if not memory_config.is_sharded():
        return None
    if memory_config.buffer_type != ttnn.BufferType.L1:
        return None  # a DRAM shard is not resident in any core's L1
    nd_spec = getattr(memory_config, "nd_shard_spec", None)
    if nd_spec is None:
        return None

    shard = [int(d) for d in nd_spec.shard_shape]
    padded = [int(d) for d in padded_shape]
    if len(padded) < 2 or len(shard) < 2:
        return None
    if len(shard) == len(padded):
        ref = padded
    elif len(shard) == 2:
        ref = [_prod(padded[:-1]), padded[-1]]  # the folded 2-D view legacy specs describe
    else:
        return None

    # whole tiles, and an even split on every dim
    if shard[-2] % tile_h or shard[-1] % tile_w:
        return None
    if any(s <= 0 or r % s for r, s in zip(ref, shard)):
        return None
    chunks = [r // s for r, s in zip(ref, shard)]

    # A shard's folded row range must be contiguous: once a leading dim is split,
    # every inner ROW dim has to be whole (the W dim may always split).
    for d in range(len(ref) - 2):
        if chunks[d] > 1 and any(chunks[e] > 1 for e in range(d + 1, len(ref) - 1)):
            return None

    n_shards = _prod(chunks)
    regions = []
    for shard_index in range(n_shards):
        # shard index -> chunk multi-index, row-major over `chunks`
        idx, rem = [], shard_index
        for c in reversed(chunks):
            idx.append(rem % c)
            rem //= c
        idx.reverse()
        rows = 0
        for d in range(len(ref) - 1):  # every ROW dim (0 .. N-2)
            rows += idx[d] * shard[d] * _prod(ref[d + 1 : len(ref) - 1])
        if rows % tile_h:
            return None
        regions.append((rows // tile_h, (idx[-1] * shard[-1]) // tile_w))

    cores = list(ttnn.get_optimal_worker_cores_for_sharded_tensor(tensor))  # shard order
    if len(cores) != n_shards:
        return None

    return {
        "cores": cores,
        "shard_ht": _prod(shard[:-1]) // tile_h,
        "shard_wt": shard[-1] // tile_w,
        "regions": regions,
    }


def _same_placement(a, b):
    """True when two shard plans put the same tile region on the same core."""
    return (
        a["shard_ht"] == b["shard_ht"]
        and a["shard_wt"] == b["shard_wt"]
        and a["regions"] == b["regions"]
        and [(c.x, c.y) for c in a["cores"]] == [(c.x, c.y) for c in b["cores"]]
    )


def _pack_pad_word(value, dtype):
    """The fill, packed in the **input** element format, in the low bytes of a word.

    The kernel replicates it across the store width, so a sub-word element fills
    correctly (a value written once per 32-bit word is invisible at 0 and
    garbage at any other fill).
    """
    if value is None:
        return 0
    if dtype == ttnn.float32:
        return struct.unpack("<I", struct.pack("<f", float(value)))[0]
    if dtype == ttnn.bfloat16:
        bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
        # round-to-nearest-even on the truncated mantissa
        bits += 0x7FFF + ((bits >> 16) & 1)
        return (bits >> 16) & 0xFFFF
    if dtype in (ttnn.uint32, ttnn.int32):
        return int(value) & 0xFFFFFFFF
    if dtype == ttnn.uint16:
        return int(value) & 0xFFFF
    if dtype == ttnn.uint8:
        return int(value) & 0xFF
    raise ValueError(f"tilize: no pad-value packing for dtype {dtype}")


def create_program_descriptor(input_tensor, output_tensor, plan) -> ttnn.ProgramDescriptor:
    # ========== 1. TENSOR / TILE GEOMETRY =================================
    tile_h, tile_w = plan.tile_h, plan.tile_w
    elem_in = input_tensor.element_size()

    in_tile_bytes = tile_h * tile_w * elem_in  # row-major CB page (tile-sized)
    out_tile_bytes = output_tensor.buffer_page_size()  # tiled page (bf8b carries exponents)

    target = list(plan.target)
    nt_h = _prod(target[:-2]) * _div_up(target[-2], tile_h)  # total tile-rows
    wt = _div_up(target[-1], tile_w)  # total tile-columns

    # ========== 2. KNOBS + WORK DISTRIBUTION ==============================
    cb_depth = 2 if (plan.use_double_buffer and LEVERS["double_buffer"]) else 1

    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    if plan.use_multicore and LEVERS["multicore"]:
        full_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    else:
        full_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    num_cores_available = full_grid.num_cores()

    # ---- 2a. placement regime, per side (op_design.md §5.2) ---------------
    # A resident L1 shard IS the per-core block: it pins the cores, the per-core
    # tile-row range and the W extent. Padding materializes the fill INTO the
    # input CB, so a padded call can never alias the input tensor; single-core is
    # refused for a sharded call by validate() (a shard is inherently multi-core)
    # and the A0 off-arm forces one core, so both fall back to the accessor path.
    shard_eligible = plan.use_multicore and bool(LEVERS["multicore"]) and not plan.has_pad_region
    in_shard = shard_side_plan(input_tensor, plan.in_padded, tile_h, tile_w) if shard_eligible else None
    out_shard = shard_side_plan(output_tensor, target, tile_h, tile_w) if shard_eligible else None

    if in_shard is not None and out_shard is not None and _same_placement(in_shard, out_shard):
        in_placement, out_placement, shard = P_LOCAL_SHARD, P_LOCAL_SHARD, in_shard
    elif out_shard is not None:
        # Output-local: pack straight into the destination shard, read the source
        # (interleaved, or another core's L1 shard) through the accessor.
        in_placement, out_placement, shard = P_ACCESSOR, P_LOCAL_SHARD, out_shard
    elif in_shard is not None:
        in_placement, out_placement, shard = P_LOCAL_SHARD, P_ACCESSOR, in_shard
    else:
        in_placement, out_placement, shard = P_ACCESSOR, P_ACCESSOR, None

    # ---- 2b. knobs + work distribution -----------------------------------
    # An aliased CB costs no extra L1 (it IS the tensor), so only the STREAMING
    # sides enter the L1 budget — that is what keeps a wide-W sharded crossover
    # bounded in W.
    def _streaming_bytes(in_p, out_p):
        return (0 if in_p == P_LOCAL_SHARD else in_tile_bytes), (0 if out_p == P_LOCAL_SHARD else out_tile_bytes)

    if shard is not None:
        work_mode = W_REGION
        stream_in, stream_out = _streaming_bytes(in_placement, out_placement)
        if in_placement == P_LOCAL_SHARD:
            # The aliased RM shard's own geometry pins the block width: a block of
            # WT_CHUNK pages must be one tile_h x (WT_CHUNK*32) row-major region,
            # which is the full shard width and nothing else.
            wt_chunk, n_chunks = shard["shard_wt"], 1
        else:
            wt_chunk, n_chunks = derive_shard_blocking(shard["shard_wt"], wt_cap(cb_depth, stream_in, stream_out))
        while cb_depth > 1 and cb_bytes(cb_depth, wt_chunk, stream_in, stream_out) > CB_L1_BUDGET:
            cb_depth -= 1
        if cb_bytes(1, wt_chunk, stream_in, stream_out) > CB_L1_BUDGET:
            # The one shape zero-copy cannot buy: the shard pins a block width
            # whose streaming partner CB will not fit. Take the accessor path on
            # both sides so WT_CHUNK is free to shrink again.
            in_placement, out_placement, shard, work_mode = P_ACCESSOR, P_ACCESSOR, None, W_BLOCKS

    if shard is None:
        work_mode = W_BLOCKS
        if LEVERS["w_split"]:
            wt_chunk, n_chunks, num_blocks_total = derive_blocking(
                nt_h, wt, in_tile_bytes, out_tile_bytes, num_cores_available, cb_depth
            )
        else:
            # w_split OFF — the pure height split (op_design.md §1.3 candidate 2).
            wt_chunk, n_chunks, num_blocks_total = wt, 1, nt_h

        # never OOM: fall back to depth-1 rather than exceed the L1 budget
        # (same cb_bytes() source as derive_blocking's ceiling and the CBs below)
        while cb_depth > 1 and cb_bytes(cb_depth, wt_chunk, in_tile_bytes, out_tile_bytes) > CB_L1_BUDGET:
            cb_depth -= 1

        (
            num_cores,
            all_cores,
            core_group_1,
            core_group_2,
            blocks_per_core_1,
            blocks_per_core_2,
        ) = ttnn.split_work_to_cores(
            full_grid, num_blocks_total, bool(LEVERS["row_wise"])
        )  # row_wise=True (master.md A1)

        cores = ttnn.corerange_to_cores(all_cores, num_cores, bool(LEVERS["row_wise"]))
    else:
        # Cores are the cores that HOLD the shards, in shard order (master.md A2:
        # launch only where the data is). Each owns its own shard's tile region.
        cores = shard["cores"]
        all_cores = ttnn.CoreRangeSet({ttnn.CoreRange(core, core) for core in cores})
        blocks_per_core_shard = shard["shard_ht"] * n_chunks
        num_blocks_total = blocks_per_core_shard * len(cores)

    # ========== 3. CIRCULAR BUFFERS =======================================
    # A streaming CB is CB_DEPTH * NT_BLK * WT_CHUNK pages — a function of the
    # knobs only, never of WT / NT_H / any tensor dimension. cb_pages() is the
    # one place that formula lives. An aliased CB is the shard itself.
    pages_per_cb = cb_pages(cb_depth, wt_chunk)
    tile_descriptor = ttnn.TileDescriptor(tile_h, tile_w)

    def _aliased_cb(cb_index, tensor, page_bytes, dtype):
        """CB placed ON the resident L1 shard (design lamp L1, master.md C14).

        ``cb_descriptor_from_sharded_tensor`` carries the buffer pointer (that is
        what makes it zero-copy); the format is then restated in TILE-page terms
        because tilize counts pages in tiles on both sides, while an RM shard's
        own page is one stick.
        """
        descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_index, tensor, core_ranges=all_cores)
        descriptor.total_size = shard["shard_ht"] * shard["shard_wt"] * page_bytes
        descriptor.format_descriptors = [
            ttnn.CBFormatDescriptor(
                buffer_index=cb_index,
                data_format=dtype,
                page_size=page_bytes,
                tile=tile_descriptor,
            )
        ]
        return descriptor

    if in_placement == P_LOCAL_SHARD:
        cb_input_sticks_descriptor = _aliased_cb(CB_INPUT_STICKS, input_tensor, in_tile_bytes, input_tensor.dtype)
    else:
        cb_input_sticks_descriptor = ttnn.CBDescriptor(
            total_size=pages_per_cb * in_tile_bytes,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_INPUT_STICKS,
                    data_format=input_tensor.dtype,
                    page_size=in_tile_bytes,
                    tile=tile_descriptor,
                )
            ],
        )

    if out_placement == P_LOCAL_SHARD:
        cb_output_tiles_descriptor = _aliased_cb(CB_OUTPUT_TILES, output_tensor, out_tile_bytes, output_tensor.dtype)
    else:
        cb_output_tiles_descriptor = ttnn.CBDescriptor(
            total_size=pages_per_cb * out_tile_bytes,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_OUTPUT_TILES,
                    data_format=output_tensor.dtype,
                    page_size=out_tile_bytes,
                    tile=tile_descriptor,
                )
            ],
        )

    # ========== 4. KERNELS =================================================
    regime = R_PAD if (plan.has_pad_region or not LEVERS["regime_select"]) else R_ALIGNED

    in_shape = list(plan.in_shape)
    h_in = in_shape[-2]
    w_in_bytes = in_shape[-1] * elem_in
    n_img_in = _prod(in_shape[:-2])
    nth_per_img = _div_up(target[-2], tile_h)
    pad_word = _pack_pad_word(plan.pad_value, input_tensor.dtype)

    # -- reader (NCRISC / NOC0) --
    reader_ct_args = [
        regime,
        in_placement,
        work_mode,
        tile_h,
        wt_chunk,
        nt_h,
        n_chunks,  # W chunks per shard row (W_REGION); 1 collapses the inner loop
        nth_per_img,
        h_in,
        n_img_in,
        w_in_bytes,
        elem_in,
        ABLATE["dm"],
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    # -- writer (BRISC / NOC1) --
    writer_ct_args = [
        out_placement,
        work_mode,
        wt_chunk,
        nt_h,
        wt,
        n_chunks,
        out_tile_bytes,
        LEVERS["block_write"],
        ABLATE["dm"],
        LEVERS["page_write"],
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    needs_cast = input_tensor.dtype != output_tensor.dtype
    compute_ct_args = [wt_chunk, 1 if needs_cast else 0, ABLATE["compute"]]

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()

    src_addr = input_tensor.buffer_address()
    dst_addr = output_tensor.buffer_address()

    # Two work assignments, one runtime-arg shape. W_BLOCKS hands each core a
    # contiguous range of the global W-chunk-major block index (tile_row0 /
    # tile_col0 unused); W_REGION hands it the origin of its own shard's tile
    # region and walks it tile-row-major, W chunk innermost.
    start_block = 0
    for shard_index, core in enumerate(cores):
        if work_mode == W_REGION:
            blocks_this_core = blocks_per_core_shard
            tile_row0, tile_col0 = shard["regions"][shard_index]
        else:
            if core_group_1.contains(core):
                blocks_this_core = blocks_per_core_1
            elif core_group_2.contains(core):
                blocks_this_core = blocks_per_core_2
            else:
                blocks_this_core = 0
            tile_row0, tile_col0 = 0, 0
        reader_rt_args[core.x][core.y] = [
            src_addr,
            start_block,
            blocks_this_core,
            pad_word,
            tile_row0,
            tile_col0 * tile_w * elem_in,  # the region's byte offset within a stick
        ]
        writer_rt_args[core.x][core.y] = [dst_addr, start_block, blocks_this_core, tile_row0, tile_col0]
        compute_rt_args[core.x][core.y] = [blocks_this_core]
        if work_mode == W_BLOCKS:
            start_block += blocks_this_core

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tilize_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor() if LEVERS["noc_split"] else ttnn.WriterConfigDescriptor(),
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tilize_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor() if LEVERS["noc_split"] else ttnn.ReaderConfigDescriptor(),
    )

    # fp32 -> fp32 must be BIT-EXACT: keep Dest in fp32 and stop the unpacker
    # downgrading fp32 to tf32 on its way to Dest. Only legal when the fast
    # tilize path is off (it is: fp32 OUTPUT disables it), which is exactly the
    # fp32-in/fp32-out case.
    lossless_fp32 = (
        input_tensor.dtype == ttnn.float32 and output_tensor.dtype == ttnn.float32 and bool(LEVERS["fp32_dest"])
    )
    compute_config = ttnn.ComputeConfigDescriptor()
    compute_config.fp32_dest_acc_en = lossless_fp32
    if lossless_fp32:
        unpack_modes = [ttnn.UnpackToDestMode.Default] * NUM_CIRCULAR_BUFFERS
        unpack_modes[CB_INPUT_STICKS] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute_config.unpack_to_dest_mode = unpack_modes

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tilize_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=compute_config,
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=[cb_input_sticks_descriptor, cb_output_tiles_descriptor],
    )
