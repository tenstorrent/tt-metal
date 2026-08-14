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

Placement (op_design.md §5.2, design lamp L1) is decided **per side** and is
orthogonal to the blocking: a resident L1 shard backs its CB directly
(``P_LOCAL_SHARD``, zero NoC on that side) while interleaved memory — and a
non-local shard — goes through a ``TensorAccessor`` (``P_ACCESSOR``). A shard
does not change the loop nest; it pins the cores, the per-core tile region and
the W extent, which is why the sharded work assignment (``W_REGION``) is a
different index map over the same block.
"""

from __future__ import annotations

import struct
from pathlib import Path

import ttnn

from ttnn.operations._op_contract import SupportRefusal


KERNEL_DIR = Path(__file__).parent / "kernels"

# --- fixed hardware / library facts (not knobs) ----------------------------
DEFAULT_TILE_WIDTH = 32  # a tile is always 32 wide
NUM_CIRCULAR_BUFFERS = 32
NOC_ALIGN_BYTES = 32  # transfer granularity a split gather has to keep

# Dtypes whose DATUM is one byte. Not the same question as `element_size() == 1`:
# bfloat8_b also reports one byte per element but is a block-float format with a
# shared exponent, and the LLK's 8-bit tilize path must not be selected for it.
EIGHT_BIT_DTYPES = (ttnn.uint8,)

# Block-float formats: a group of datums shares one exponent, so an individual
# element word cannot be written into a tile after the pack (which is why the
# output-format pad stamp below never applies to them).
BLOCK_FLOAT_DTYPES = tuple(d for d in (getattr(ttnn, "bfloat8_b", None), getattr(ttnn, "bfloat4_b", None)) if d)

# Faces are 16 wide on every supported arch and 16 tall on a full 32-row tile; a
# TINY tile's face height is the tile height itself (tile.cpp TILE_FACE_HW_CHOICES:
# 8x32 -> 8x16, ... 1x32 -> 1x16). The output-format pad stamp addresses a tiled
# tile through that geometry, so `fill_tile_pad` derives the same rule from tile_h
# — this constant is only the FULL-tile face height, and the assert below is what
# pins that every supported tile height has whole faces.
FACE_HEIGHT = 16

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

# --- placement heuristic: when zero-copy on the DESTINATION actually pays ----
# An aliased shard pins WT_CHUNK to the shard's own width, so a destination shard
# that is narrow in W (or a source shard whose page is narrow) forces the
# STREAMING reader into tiny per-row transfers issued by only the shard's cores.
# Measured on Wormhole B0 (bf16, `_bench_tilize.py` + Refinement-2 probes):
# destination-local wins 1.19x / 2.06x / 1.30x once the read transfer is >= 256 B
# and LOSES 1.75x / 3.19x / 3.45x below it (128 B and 64 B reads), where the
# generic full-grid split is faster even though it moves the bytes over the NoC.
# The writer side never trips this — it always moves whole TILE pages (>= 1 KB) —
# so the gate is read-side only, and a SOURCE-local plan is never gated (measured
# 0.94x-3.85x, i.e. parity at worst).
MIN_STREAM_READ_BYTES = 256

# --- pipeline depth: how many BLOCKS a core should get (Refinement 3) --------
# A core that owns exactly ONE block cannot overlap anything: it reads the whole
# block, tilizes it, then writes it, strictly serialized — which is why Phase 0
# measured the depth-2 CB (master.md C16) as a no-payoff lever on the
# grid-filling square. There was no second block to work on. Splitting W finer
# hands each core several blocks so read(i+1) / compute(i) / write(i-1) run
# concurrently and the tilize LLK disappears into the DM shadow.
#
# The cost is a SMALLER per-row read transfer (master.md B5), so the split is
# capped: never chunk W so finely that a source-row transfer drops below
# MIN_PIPELINE_READ_BYTES. Both are named knobs with one source here; the grid
# fill floor and the L1 ceiling still dominate them (this only ever ADDS chunks
# to a shape that would otherwise land one block per core).
PIPELINE_BLOCKS_PER_CORE = 4
MIN_PIPELINE_READ_BYTES = 1024

# --- CB indices (semantic names; the numeric slot is just a buffer index) ---
CB_INPUT_STICKS = 0  # reader -> compute  (row-major sticks, tile-sized pages)
CB_RETILE_STAGE = 1  # reader -> reader   (R_RETILE: staged SOURCE tile pages)
CB_PAD_SCRATCH = 2  # writer -> writer   (out_fill: ONE pre-stamped whole-pad tile)
CB_INPUT_STICKS_B = 3  # reader#2 -> compute (Perf 2 split reader: the second half)
CB_OUTPUT_TILES = 16  # compute -> writer (tiled pages)

# --- reader regime selector (op_design.md §5.1) -----------------------------
R_ALIGNED = 0
R_PAD = 1
# R_RETILE (Refinement 5): the source is ALREADY tiled, at a different tile
# height. Perf 2: the reader now lands the face permutation DIRECTLY in the
# OUTPUT TILE rather than routing it through a row-major intermediate — see
# `retile_direct` below for the mechanism and the measured numbers.
R_RETILE = 2

# --- Perf 2: the retile-direct reader ---------------------------------------
# DRAM NoC transfer alignment. A DRAM-sourced transfer must start on this
# boundary; the retile-direct reader's source offsets are all multiples of its
# run length, so `run_bytes % NOC_DRAM_ALIGN_BYTES` is the WHOLE predicate.
NOC_DRAM_ALIGN_BYTES = 32  # wormhole_b0 (64 on blackhole — read from the hal)
# The DRAM-direct form's transaction floor. At a 32 B run it is a measured
# REGRESSION on every geometry tried (1->32 bf16: 99,142 vs 60,837 ns for the
# staged form); at 64 B it is already a 1.33x win (32->2 bf16: 42,052 vs 55,839).
# Nothing between was measured, so 64 is the lowest PROVEN floor.
MIN_DIRECT_DRAM_RUN_BYTES = 64

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


# Perf 2: the INPUT CB carries one extra group of SLACK behind the reader's one
# outstanding read. The measured rule is slack, not depth and not window size:
# every arm with `cb_depth == ahead + 1` (zero slack) landed on the baseline or
# below it, and every arm with `cb_depth >= ahead + 2` won — while `ahead >= 2`
# is flat at best and a regression on interleaved W_BLOCKS. So: ONE outstanding
# group, ONE group of slack. The OUTPUT CB is untouched, which is what holds the
# L1 cost to `wt_chunk * in_tile_bytes`.
IN_CB_EXTRA_DEPTH = 1


def cb_pages_in(cb_depth, wt_chunk, in_extra=0):
    """Pages in the INPUT streaming CB — `cb_pages()` plus the issue-ahead slack.

    ``in_extra`` is ``IN_CB_EXTRA_DEPTH`` on a plan that actually takes the
    issue-ahead schedule and 0 everywhere else, so no path pays L1 (and no path
    pays the ``wt_cap()`` consequence) for a window it does not use.
    """
    return cb_pages(cb_depth + in_extra, wt_chunk)


def cb_bytes(cb_depth, wt_chunk, in_tile_bytes, out_tile_bytes, in_extra=0):
    """L1 bytes held by the two streaming CBs together, from ``cb_pages()``."""
    return cb_pages_in(cb_depth, wt_chunk, in_extra) * in_tile_bytes + cb_pages(cb_depth, wt_chunk) * out_tile_bytes


def stage_tile_bytes(tile_h, in_tile_h, tile_w, elem_bytes):
    """L1 the R_RETILE scratch holds per tile-column of chunk width (Refinement 5).

    One output tile-row spans ``ceil(tile_h / in_tile_h)`` SOURCE tile-rows (both
    heights are powers of two <= 32, so one divides the other and the ceiling is
    exact), and the reader stages a whole source tile page for each. 0 off the
    retile path — THE single source for the scratch size, shared by the CB
    descriptor and the L1 ceiling below so they cannot drift.
    """
    if not in_tile_h:
        return 0
    return _div_up(tile_h, in_tile_h) * in_tile_h * tile_w * elem_bytes


def wt_cap(cb_depth, in_tile_bytes, out_tile_bytes, stage_bytes=0, in_extra=0):
    """L1 ceiling on the W block factor — THE single source for that cap.

    ``in_tile_bytes`` / ``out_tile_bytes`` are the *streaming* page sizes: pass 0
    for a side whose CB is aliased on a resident shard, since that side costs no
    extra L1 (it IS the tensor). ``stage_bytes`` is the R_RETILE scratch, which is
    depth-1 (it is reader-private scratch, not a pipelined CB). When nothing
    streams, only the library's fast-tilize width bound remains.
    """
    per_chunk_tile = cb_bytes(cb_depth, 1, in_tile_bytes, out_tile_bytes, in_extra) + stage_bytes
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


def read_bytes_per_stick(wt_chunk, in_tile_bytes, tile_h):
    """Bytes ONE source-row transfer moves for a ``WT_CHUNK``-wide block.

    THE single source for the transfer-size side of every blocking decision
    (master.md B5 / B0): the reader issues one transfer of this many bytes per
    source stick, so it is what a finer W split actually costs.
    """
    return wt_chunk * in_tile_bytes // tile_h


def derive_blocking(
    nt_h, wt, in_tile_bytes, out_tile_bytes, num_cores, cb_depth, tile_h, pipeline=True, stage_bytes=0, in_extra=0
):
    """The three block knobs — single source of truth (op_design.md §1.4).

    Returns ``(wt_chunk, n_chunks, num_blocks)``.

    * ``WT_CHUNK`` is the COARSEST chunk that fits: the whole tile-row width
      unless the L1 ceiling, the grid-fill floor or the pipeline floor forces it
      smaller.
    * ``n_chunks`` divides ``WT`` exactly, so every block has the same width and
      there is exactly one compute kernel (no cliff-width variant).
    * ``NT_H >= NUM_CORES * PIPELINE_BLOCKS_PER_CORE`` implies ``n_chunks == 1``,
      i.e. the wide-shape machinery is inert on tall shapes (byte-identical to a
      pure height split, which already gives every core several blocks).
    * ``pipeline`` is the Refinement-3 blocking rule; False reproduces the Phase-0
      "one block per core is enough" rule exactly (kept as a parameter so the
      blocking pins in test_tilize_levers.py can compare the two rules).
    """
    # Bytes both CBs hold per tile-column of chunk width — read from cb_bytes()
    # via wt_cap() so the ceiling can never drift from the CB sizing below (it
    # carries the NT_BLK factor too, which a hand-written formula here would have
    # dropped) and the sharded path shares the same cap.
    cap = wt_cap(cb_depth, in_tile_bytes, out_tile_bytes, stage_bytes, in_extra)

    n_want = max(1, _div_up(num_cores, nt_h))  # grid-fill floor
    if pipeline:
        # Pipeline floor: enough blocks per core for read/compute/write to
        # overlap — but never at the price of a read transfer below the B5 floor.
        n_pipeline = _div_up(num_cores * PIPELINE_BLOCKS_PER_CORE, nt_h)
        n_transfer_cap = max(1, read_bytes_per_stick(wt, in_tile_bytes, tile_h) // MIN_PIPELINE_READ_BYTES)
        n_want = max(n_want, min(n_pipeline, n_transfer_cap))
    n_want = max(n_want, _div_up(wt, cap))  # L1 ceiling
    n_want = min(n_want, wt)  # can never split W finer than one tile-column

    n_chunks = next(c for c in range(n_want, wt + 1) if wt % c == 0)
    wt_chunk = wt // n_chunks
    return wt_chunk, n_chunks, nt_h * n_chunks


def shard_side_plan(tensor, padded_shape, tile_h, tile_w):
    """The per-core tile region of an L1-sharded tensor — or None when the shard
    cannot back a zero-copy CB (op_design.md §5.2 ``side_regime``).

    Returns ``{"cores", "shard_ht", "shard_wt", "regions"}`` where ``regions[i]``
    is shard *i*'s ``(tile_row0, tile_col0, tile_rows)`` in the folded (tile-row,
    tile-col) grid and ``cores[i]`` is the core that holds it. Both legacy 2-D and
    ND specs go through the SAME derivation: a legacy ShardSpec is exactly an ND
    spec whose shard shape has rank 2 over the folded 2-D view of the tensor,
    which ``memory_config.nd_shard_spec`` already reports.

    The grid may be UNEVEN along the split ROW dim (Refinement 2): the last shard
    then carries fewer tile-rows, which is why a region records its own
    ``tile_rows`` and the per-core block count is derived from it rather than from
    a single uniform shard height. ``shard_ht`` stays the ALLOCATED shard height —
    that is what the aliased CB's ring spans.

    None (→ the accessor path) whenever a shard is not a whole number of tile-rows,
    the W dim does not divide exactly (``WT_CHUNK`` is one compile-time value for
    every core, so shards may not differ in width), the shard does not fold to a
    CONTIGUOUS band of tile-rows, or a core would hold more than one shard.
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

    # Whole tile-rows, a tile-multiple width, and an exact split in W.
    if any(s <= 0 for s in shard):
        return None
    if shard[-1] % tile_w or ref[-1] % shard[-1]:
        return None
    rows_per_shard = _prod(shard[:-1])
    if rows_per_shard % tile_h:
        return None
    # The ROW dims may split UNEVENLY (the last shard is short); W may not.
    chunks = [_div_up(r, s) for r, s in zip(ref, shard)]

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
        rows, extent = 0, 1
        for d in range(len(ref) - 1):  # every ROW dim (0 .. N-2)
            rows += idx[d] * shard[d] * _prod(ref[d + 1 : len(ref) - 1])
            # What this shard actually covers in dim d — short on the last chunk
            # of an unevenly split dim, the whole dim otherwise (the contiguity
            # rule above allows at most one split row dim, so the product is one
            # contiguous band of folded rows).
            extent *= min(shard[d], ref[d] - idx[d] * shard[d])
        if rows % tile_h or extent % tile_h:
            return None
        regions.append((rows // tile_h, (idx[-1] * shard[-1]) // tile_w, extent // tile_h))

    cores = list(ttnn.get_optimal_worker_cores_for_sharded_tensor(tensor))  # shard order
    if len(cores) != n_shards or len({(c.x, c.y) for c in cores}) != n_shards:
        return None  # a core holding two shards needs two region assignments

    return {
        "cores": cores,
        "shard_ht": rows_per_shard // tile_h,  # ALLOCATED height (the CB ring)
        "shard_wt": shard[-1] // tile_w,
        "regions": regions,
    }


def src_page_geometry(tensor, padded_shape, elem_bytes):
    """``(page_bytes, pages_per_row)`` for a ROW_MAJOR source read through a
    ``TensorAccessor`` — THE single source for the reader's page arithmetic
    (op_design.md §5.2, the cross-spec L1 gather).

    A ROW_MAJOR page is one row (stick) when the tensor is interleaved **or** its
    shard spans the whole row: ``pages_per_row == 1`` and the page id IS the
    folded row index (the Phase-0 identity). A shard NARROWER than the row
    (WIDTH / BLOCK sharded, or an ND shard split on the last dim) makes a page one
    SHARD row, so a row is ``ceil(W / shard_W)`` pages and a span of bytes inside
    one row may cross page boundaries — which is what ``read_row_span()`` in the
    reader splits. Derived from the shard SPEC (authoritative) rather than from the
    buffer's aligned page size, which can round a page up and mis-count the row.
    """
    row_elems = int(padded_shape[-1])
    memory_config = tensor.memory_config()
    if not memory_config.is_sharded():
        return row_elems * elem_bytes, 1
    nd_spec = getattr(memory_config, "nd_shard_spec", None)
    if nd_spec is None:
        return row_elems * elem_bytes, 1
    shard_w = int(nd_spec.shard_shape[-1])
    if shard_w <= 0 or shard_w >= row_elems:
        return row_elems * elem_bytes, 1
    return shard_w * elem_bytes, _div_up(row_elems, shard_w)


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


def _quantize_to_dtype(value, dtype):
    """`value` as it survives a round trip through ``dtype``'s element format.

    Built on ``_pack_pad_word`` so there is exactly ONE definition of "the fill in
    format X" — the quantizer cannot drift from the word the kernel actually stores.
    """
    word = _pack_pad_word(value, dtype)
    if dtype == ttnn.float32:
        return struct.unpack("<f", struct.pack("<I", word))[0]
    if dtype == ttnn.bfloat16:
        return struct.unpack("<f", struct.pack("<I", word << 16))[0]
    return word  # an integer element format stores the value verbatim


def needs_output_format_fill(value, in_dtype, out_dtype):
    """Does the pad fill need a SECOND word packed in the output format?

    The reader always fills the input CB in the **input** element format — that is
    a hard contract (the fill travels the data path, so packing it in
    ``output_dtype`` is garbage the moment a cast is requested). The consequence is
    that a WIDENING cast delivers an input-rounded fill: bf16 cannot hold 10.2, so
    an fp32 output would carry 10.1875 where the oracle wants 10.2.

    True exactly when that round-trip loses the value, which is what makes the
    writer's output-format stamp worth its L1 stores. THE single source for the
    decision (host gate + `out_fill` compile-time arg).
    """
    if value is None or in_dtype == out_dtype:
        return False
    if in_dtype in BLOCK_FLOAT_DTYPES or out_dtype in BLOCK_FLOAT_DTYPES:
        # A block-float tile's datums share an exponent, so a raw element word
        # cannot be stamped into one. The input-format fill is quantized by the
        # packer exactly as real data is, which IS the definition of correct here.
        return False
    return _quantize_to_dtype(value, out_dtype) != _quantize_to_dtype(_quantize_to_dtype(value, in_dtype), out_dtype)


def create_program_descriptor(input_tensor, output_tensor, plan) -> ttnn.ProgramDescriptor:
    # ========== 1. TENSOR / TILE GEOMETRY =================================
    tile_h, tile_w = plan.tile_h, plan.tile_w
    elem_in = input_tensor.element_size()

    in_tile_bytes = tile_h * tile_w * elem_in  # row-major CB page (tile-sized)
    out_tile_bytes = output_tensor.buffer_page_size()  # tiled page (bf8b carries exponents)

    target = list(plan.target)
    nt_h = _prod(target[:-2]) * _div_up(target[-2], tile_h)  # total tile-rows
    wt = _div_up(target[-1], tile_w)  # total tile-columns

    # --- Refinement 5: the RETILE path ------------------------------------
    # A TILE-layout input is re-tiled to `tile_h`. `in_tile_h` is the SOURCE tile's
    # height (0 off this path — the sentinel every retile-only derivation keys on),
    # and `stage_per_chunk_tile` is the L1 the reader's page staging costs per
    # tile-column of chunk width; both have ONE source and feed the CB descriptor
    # and the L1 ceiling alike.
    retile = input_tensor.layout == ttnn.TILE_LAYOUT
    in_tile_h = int(input_tensor.tile.tile_shape[0]) if retile else 0
    stage_per_chunk_tile = stage_tile_bytes(tile_h, in_tile_h, tile_w, elem_in)

    # How the reader addresses a ROW_MAJOR source page (single source; §5.2).
    # `src_row_pages > 1` is the cross-spec gather: the source shard is narrower
    # than a tensor row, so a row is several pages and a span has to be split.
    if retile:
        # A TILE source's page IS a tile, so the ROW_MAJOR stick geometry does not
        # describe it (and its "row" split would be meaningless). The retile reader
        # does its own page math off `in_tile_h` / `wt`.
        src_page_bytes, src_row_pages = in_tile_h * tile_w * elem_in, 1
    else:
        src_page_bytes, src_row_pages = src_page_geometry(input_tensor, plan.read_padded, elem_in)

    # ========== 2. KNOBS + WORK DISTRIBUTION ==============================
    cb_depth = 2 if plan.use_double_buffer else 1

    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    if plan.use_multicore:
        full_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    else:
        full_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    num_cores_available = full_grid.num_cores()

    # ---- 2a. placement regime, per side (op_design.md §5.2) ---------------
    # A resident L1 shard IS the per-core block: it pins the cores, the per-core
    # tile-row range and the W extent. Single-core is refused for a sharded call by
    # validate() (a shard is inherently multi-core), so a single-core call falls
    # back to the accessor path.
    #
    # Eligibility is PER SIDE, because the one thing padding rules out is aliasing
    # the INPUT: the fill is materialized into the input CB, and that CB is the
    # source tensor itself on the zero-copy path. The OUTPUT side is untouched by
    # the fill — compute packs whole (already padded) tiles — so a padded call
    # still packs straight into a resident destination shard instead of writing it
    # over the NoC.
    shard_eligible = plan.use_multicore
    # A RETILE source is disqualified from aliasing for a structural reason, not a
    # heuristic one: the input CB holds ROW-MAJOR sticks (the tilize helper's
    # contract) and a tiled shard is not that. Its bytes must be permuted, so there
    # is nothing to consume in place and the accessor read is the implementation.
    in_shard = (
        shard_side_plan(input_tensor, plan.read_padded, tile_h, tile_w)
        if (shard_eligible and not plan.has_pad_region and not retile)
        else None
    )
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
    # Perf 2: will this plan take the reader's ISSUE-AHEAD schedule? Decided here,
    # BEFORE the blocking, because the schedule needs one extra group of input-CB
    # slack and that group is part of the L1 budget every block knob is derived
    # against. A plan that will not take it passes in_extra = 0 and keeps exactly
    # the blocking it had before this round.
    #
    # These are the R_ALIGNED conditions restated on values known this early (the
    # regime itself is derived after the shard decision): no fill region, no
    # retile, one page per stick. A DRAM source is never aliased on a local shard,
    # so `in_placement == P_ACCESSOR` follows from `src_in_dram` rather than
    # needing the placement decision.
    src_in_dram = input_tensor.memory_config().buffer_type == ttnn.BufferType.DRAM
    # Perf 2: tiles per SOURCE shard row on the cross-core gather. Whole by
    # construction — the op already refuses a paged source whose page is not
    # NOC_ALIGN_BYTES-aligned, and a tile-width-aligned shard makes the page a
    # whole number of tile-columns. 0 off the gather.
    src_page_tiles = src_page_bytes // (tile_w * elem_in) if src_row_pages > 1 else 0
    # Rows per SOURCE shard. A block may only be moved as ONE transfer if it stays
    # inside a single shard — pages of one shard column are contiguous in THAT
    # core's L1, but the next shard down lives on a different core — so the shard
    # must hold a whole number of tile-rows. MEASURED, not assumed: with this term
    # missing, the BLOCK-sharded sources with 50- and 64-row shards produced wrong
    # values (tests/.../test_tilize_sharded.py::test_padded_sharded_identity and
    # ::test_reshard_placement_and_source_page_geometry).
    src_shard_rows = (
        int(input_tensor.memory_config().shard_spec.shape[0])
        if (input_tensor.memory_config().is_sharded() and input_tensor.memory_config().shard_spec is not None)
        else 0
    )
    wants_read_ahead = bool(src_in_dram and not retile and not plan.has_pad_region and src_row_pages == 1)
    in_extra = IN_CB_EXTRA_DEPTH if wants_read_ahead else 0

    # Derived here (ahead of the blocking) because the split reader below needs
    # it: a plan whose WRITER still has to stamp the pad region cannot give that
    # RISC away to reading.
    out_fill = int(
        plan.has_pad_region
        # The stamp addresses whole faces: 16 rows on a full tile, tile_h rows on a
        # tiny one (Refinement 5). Every legal tile height satisfies one or the other.
        and (tile_h % FACE_HEIGHT == 0 or FACE_HEIGHT % tile_h == 0)
        and needs_output_format_fill(plan.pad_value, input_tensor.dtype, output_tensor.dtype)
    )

    # Perf 2 — SPLIT READER. On a destination-local plan the writer issues no NoC
    # traffic at all (the output CB is the resident shard), so BRISC sits ~90%
    # idle while NCRISC is the whole wall. Letting both DM RISCs read, each owning
    # every other block into its OWN input CB, buys the second issuer.
    #
    # Measured end-to-end, with the real library tilize in the loop:
    #   crossover [1,1,2048,256] DRAM->H x8   14,875 -> 9,919 ns  1.50x
    #   crossover_tall [1,1,8192,256]         55,182 -> 33,543    1.65x
    #   reshard [1,1,1024,256] W x2 -> H x8   18,109 -> 10,814    1.67x
    #   reshard fp32                          33,919 -> 19,119    1.77x
    # The mechanism is issue capacity, not fabric: the plans that win sit at
    # 50-76 GB/s achieved read bandwidth and the split lifts them to 103-125 GB/s
    # -- the same ceiling the plans that are ALREADY at 125-152 GB/s sit at, which
    # is why those measure flat rather than better.
    #
    # The flavor is picked by the SOURCE BUFFER TYPE, not by the read regime: the
    # two DM RISCs are NOT interchangeable issuers at DRAM volume (the same read
    # work on BRISC/NOC_1 is 2.2x slower, which Metal itself encodes as
    # preferred_noc_for_dram_read() == NOC_0), so a DRAM source puts BOTH readers
    # on NOC_0 under DM_DYNAMIC_NOC and separates their barriers with per-RISC
    # transaction ids; a source in another core's L1 is faster with a dedicated
    # NoC each.
    SPLIT_NONE, SPLIT_DUAL_NOC, SPLIT_SHARED_NOC0 = 0, 1, 2
    wants_split = bool(
        out_placement == P_LOCAL_SHARD  # destination-local: BRISC has no write duty
        and in_placement == P_ACCESSOR  # ... and there IS a read to split
        and not out_fill  # the writer must still stamp the pad region
        and not retile  # untested; the retile reader is L1-permute bound
    )
    split_flavor = (SPLIT_SHARED_NOC0 if src_in_dram else SPLIT_DUAL_NOC) if wants_split else SPLIT_NONE
    # The second input CB is real L1 and must enter the budget with the first, or
    # a wide shard throws at program creation (measured: [1,1,2048,2048] -> H x8,
    # "static circular buffers clash with L1 buffers").
    in_cbs = 2 if wants_split else 1
    # The issue-ahead slack group would be dead L1 on the split path: the split
    # carries its own per-RISC transaction ids and was measured without a window,
    # so the host turns issue-ahead off there and the group is not allocated.
    in_extra = IN_CB_EXTRA_DEPTH if (wants_read_ahead and not wants_split) else 0

    # An aliased CB costs no extra L1 (it IS the tensor), so only the STREAMING
    # sides enter the L1 budget — that is what keeps a wide-W sharded crossover
    # bounded in W.
    def _streaming_bytes(in_p, out_p):
        # Perf 2: the split reader gives the input side `in_cbs` streaming CBs.
        return (
            0 if in_p == P_LOCAL_SHARD else in_cbs * in_tile_bytes,
            0 if out_p == P_LOCAL_SHARD else out_tile_bytes,
        )

    if shard is not None:
        work_mode = W_REGION
        stream_in, stream_out = _streaming_bytes(in_placement, out_placement)
        if in_placement == P_LOCAL_SHARD:
            # The aliased RM shard's own geometry pins the block width: a block of
            # WT_CHUNK pages must be one tile_h x (WT_CHUNK*32) row-major region,
            # which is the full shard width and nothing else.
            wt_chunk, n_chunks = shard["shard_wt"], 1
        else:
            cap = wt_cap(cb_depth, stream_in, stream_out, stage_per_chunk_tile, in_extra)
            # Perf 2 — CROSS-CORE GATHER: a block whose width IS one source shard
            # row is contiguous at BOTH ends (its tile_h rows are tile_h
            # consecutive pages of one shard, and the CB slot is contiguous
            # because the block width is the page), so the reader moves it in ONE
            # transfer instead of tile_h. Measured end-to-end 1.27x on the reshard
            # focus, 1.54x on the 128 B-page plan, 2.91x on a 1-tile page; flat,
            # never slower, once the per-row transfer already saturates source L1
            # egress. Preferring this width costs nothing elsewhere: it only
            # applies where the source is a narrower-than-a-row shard.
            if (
                src_page_tiles
                and src_shard_rows
                and src_shard_rows % tile_h == 0
                and shard["shard_wt"] % src_page_tiles == 0
                and src_page_tiles <= cap
            ):
                wt_chunk, n_chunks = src_page_tiles, shard["shard_wt"] // src_page_tiles
            else:
                wt_chunk, n_chunks = derive_shard_blocking(shard["shard_wt"], cap)
        # Read-transfer gate (MIN_STREAM_READ_BYTES): an aliased DESTINATION pins
        # the reader's per-row transfer to the shard's own width, and below the
        # measured knee the generic full-grid split beats packing in place. Only
        # the read side can trip it — the writer always moves whole tile pages.
        if in_placement == P_ACCESSOR:
            # What ONE source transfer moves. A retile reads whole source TILE
            # pages (that is what makes it addressable at all), so its transfer is
            # the page — the per-STICK size the gate was measured on describes the
            # row-major reader only.
            read_bytes = src_page_bytes if retile else min(wt_chunk * tile_w * elem_in, src_page_bytes)
            # Perf 2: the gate exists because an aliased destination pins the
            # reader to a small PER-ROW transfer. Once the block is ONE contiguous
            # transfer that premise is gone — the gated W x4 -> H x8 plan is
            # 20,828 ns with the gate firing and 13,512 with the block kept local.
            if src_page_tiles and src_shard_rows and src_shard_rows % tile_h == 0 and wt_chunk == src_page_tiles:
                read_bytes = tile_h * src_page_bytes
            if read_bytes < MIN_STREAM_READ_BYTES:
                in_placement, out_placement, shard, work_mode = P_ACCESSOR, P_ACCESSOR, None, W_BLOCKS
    if shard is not None:
        while (
            cb_depth > 1
            and cb_bytes(cb_depth, wt_chunk, stream_in, stream_out, in_extra) + wt_chunk * stage_per_chunk_tile
            > CB_L1_BUDGET
        ):
            cb_depth -= 1
        if cb_bytes(1, wt_chunk, stream_in, stream_out, in_extra) + wt_chunk * stage_per_chunk_tile > CB_L1_BUDGET:
            # The one shape zero-copy cannot buy: the shard pins a block width
            # whose streaming partner CB will not fit. Take the accessor path on
            # both sides so WT_CHUNK is free to shrink again.
            in_placement, out_placement, shard, work_mode = P_ACCESSOR, P_ACCESSOR, None, W_BLOCKS

    if shard is None:
        work_mode = W_BLOCKS
        wt_chunk, n_chunks, num_blocks_total = derive_blocking(
            nt_h,
            wt,
            in_tile_bytes,
            out_tile_bytes,
            num_cores_available,
            cb_depth,
            tile_h,
            stage_bytes=stage_per_chunk_tile,
            in_extra=in_extra,
        )

        # never OOM: fall back to depth-1 rather than exceed the L1 budget
        # (same cb_bytes() source as derive_blocking's ceiling and the CBs below)
        while (
            cb_depth > 1
            and cb_bytes(cb_depth, wt_chunk, in_tile_bytes, out_tile_bytes, in_extra) + wt_chunk * stage_per_chunk_tile
            > CB_L1_BUDGET
        ):
            cb_depth -= 1

        # -- Perf 1: buy read/write OVERLAP on a ONE-BLOCK-PER-CORE shape --------
        # When the block count lands at exactly one block per core the pipeline
        # degenerates: the core reads its whole block, computes, then writes, with
        # nothing to overlap against. Measured on [1,1,32,16384] bf16 by cumulative
        # ablation: read 4,590 + write 8,680 + sync 1,020 = 14,290 against a
        # measured wall of 13,830 — the exclusive costs SUM, i.e. the stages do not
        # overlap at all, against a max(read, write) + floor ideal of ~9,700.
        #
        # Halving the lit core count gives every core TWO blocks and leaves
        # WT_CHUNK — and therefore the 512 B read transfer — completely untouched.
        # That distinction is the whole lever: shrinking the transfer instead
        # (n_chunks x2 on the full grid) was measured at 0.908x, reconfirming the
        # transfer-size floor.
        #
        # The predicate is over the BLOCKING, not over shapes: it fires only where
        # the split really would light one core per block, and only for an EXACT
        # halving that keeps every core's load equal. Everything outside it was
        # measured and regresses, which is why each clause is there:
        #   * already >= 2 blocks/core -> 0.968x on [1,1,2048,2048] (3 sessions)
        #   * too few blocks to halve  -> 0.706x on [1,1,32,64]
        #   * an unbalanced cap (48 cores, 1-or-2 blocks/core) -> 0.949x
        # Measured ON: [1,1,32,16384] bf16 13,578 -> 13,066 ns (15-round in-session
        # A/B, ~4 sigma), [1,1,32,8192] 8,589 -> 8,146, fp32 26,911 -> 26,447.
        grid_x = grid.x if plan.use_multicore else 1
        one_block_per_core = num_blocks_total <= num_cores_available
        # The two cases that CANNOT take the trade, each carved out on a MEASURED
        # regression rather than a suspicion. Both share one mechanism: halving the
        # lit cores only pays when the wall is the DRAM interface (the same
        # aggregate bandwidth arrives through half the cores, so the freed
        # pipelining is profit). Where the wall is PER-CORE work instead, halving
        # the cores just halves the parallelism.
        #   * R_RETILE — the reader's payload is a local L1 face permutation, not a
        #     DRAM read. Measured [1,1,1024,1024] 1->32: 68,125 -> 111,618 ns (0.61x).
        #   * a sub-128 B output page — the write is transaction-RATE bound per
        #     core, so fewer cores means fewer issuing RISCs, not more overlap.
        #     Measured tile_h=1 on [1,1,2048,2048]: 249,507 -> 267,661 ns (0.93x).
        #     tile_h=8 (512 B) is FLAT and therefore stays IN: 95,610 -> 94,765.
        # Written as an exception around what cannot, not as an allow-list around
        # what was benchmarked, so it shrinks as understanding grows.
        core_halving_meaningless = retile or out_tile_bytes < 128
        halve_cores = bool(
            one_block_per_core
            and not core_halving_meaningless
            and num_blocks_total >= 2 * grid_x
            and num_blocks_total % (2 * grid_x) == 0  # an EXACT, balanced halving
        )
        split_grid = (
            ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, num_blocks_total // (2 * grid_x) - 1))]
            )
            if halve_cores
            else full_grid
        )

        (
            num_cores,
            all_cores,
            core_group_1,
            core_group_2,
            blocks_per_core_1,
            blocks_per_core_2,
        ) = ttnn.split_work_to_cores(
            split_grid, num_blocks_total, True
        )  # row_wise=True (master.md A1)

        cores = ttnn.corerange_to_cores(all_cores, num_cores, True)
    else:
        # Cores are the cores that HOLD the shards, in shard order (master.md A2:
        # launch only where the data is). Each owns its own shard's tile region.
        halve_cores = False  # the sharded plan pins cores to shards, never to the split
        cores = shard["cores"]
        all_cores = ttnn.CoreRangeSet({ttnn.CoreRange(core, core) for core in cores})
        # Per shard, not per grid: an unevenly split grid gives its last shard
        # fewer tile-rows, so the block count comes from that region's own extent.
        blocks_per_shard = [ht * n_chunks for (_, _, ht) in shard["regions"]]
        num_blocks_total = sum(blocks_per_shard)

    # Perf 2: the split reader's last two clauses, both of which need the settled
    # blocking. `wants_split` above only established that the PLAN has an idle
    # BRISC and a read worth splitting.
    split_reader = SPLIT_NONE
    if wants_split and shard is not None and in_placement == P_ACCESSOR and out_placement == P_LOCAL_SHARD:
        # A core with one block has nothing to split (measured flat, and it would
        # leave one of the two readers idle).
        if min(blocks_per_shard) >= 2:
            split_reader = split_flavor
        # The DRAM flavor is issue-bound only while the per-stick transfer is
        # small. Measured ladder at 512 B/stick 1.50-1.65x, at 1024 B flat (1.07x),
        # at 2048 B a 0.86x REGRESSION, and at 4096 B the second CB does not fit
        # L1 at all. So the gate is a measured boundary, not a guess.
        if split_reader == SPLIT_SHARED_NOC0 and read_bytes_per_stick(wt_chunk, in_tile_bytes, tile_h) > 1024:
            split_reader = SPLIT_NONE

    # The issue-ahead slack is allocated only where the schedule is actually
    # taken. `halve_cores` is not known until the split above is settled, so the
    # blocking was derived with the slack included (conservative — it can only
    # make WT_CHUNK smaller, never wrong) and the group is dropped here, before
    # the CB descriptors and the kernel's slot count are built from it.
    if halve_cores:
        in_extra = 0

    # ========== 3. CIRCULAR BUFFERS =======================================

    # The rank->=2 promoted view (plan.read_shape): a rank-0 scalar reads as a
    # 1x1 source, so h_in / w_in_bytes / the image count stay well defined.
    in_shape = list(plan.read_shape)
    h_in = in_shape[-2]
    w_in_bytes = in_shape[-1] * elem_in
    n_img_in = _prod(in_shape[:-2])
    nth_per_img = _div_up(target[-2], tile_h)
    pad_word = _pack_pad_word(plan.pad_value, input_tensor.dtype)

    # -- Refinement 4: the OUTPUT-format pad stamp (`out_fill`) --------------
    # The reader's input-format fill is arithmetically exact for every path except
    # a WIDENING cast with a fill the input format cannot hold. There the writer
    # re-stamps the pad region of each finished tile with a second word packed in
    # the OUTPUT format (see kernels/tilize_fill.hpp). Gated on the round-trip
    # actually losing the value, so every other cell is byte-identical to before.
    pad_word_out = _pack_pad_word(plan.pad_value, output_tensor.dtype) if out_fill else 0
    # Perf 1: does the padded TARGET contain at least one WHOLE pad tile — a tile
    # every element of which is pad? Those, and only those, are the ones the writer
    # can produce from a single pre-stamped scratch tile instead of ~tile_h*32
    # element stores. A geometry with only ragged W/H tails has none, and there the
    # whole mechanism (and its L1 page) must not exist at all: stamping the scratch
    # unconditionally was MEASURED at +4.3 us on a 6.7 us [1,1,50,50]->[1,1,64,64].
    # THIS is the single source of that predicate — the writer takes it as a
    # compile-time arg rather than re-deriving it, so the CB allocation below and
    # the kernel branch can never disagree.
    pad_scratch = int(
        out_fill
        and (
            h_in + tile_h <= nth_per_img * tile_h  # a whole pad tile-ROW exists
            or in_shape[-1] + tile_w <= wt * tile_w  # a whole pad tile-COLUMN exists
            or n_img_in * nth_per_img < nt_h  # trailing whole-pad images
        )
    )

    # A streaming CB is CB_DEPTH * NT_BLK * WT_CHUNK pages — a function of the
    # knobs only, never of WT / NT_H / any tensor dimension. cb_pages() is the
    # one place that formula lives. An aliased CB is the shard itself.
    pages_per_cb = cb_pages(cb_depth, wt_chunk)
    # Perf 2: the input side carries one extra group so the reader can keep a read
    # in flight across the block boundary and still have a free slot to issue into.
    pages_per_in_cb = cb_pages_in(cb_depth, wt_chunk, in_extra)
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
            total_size=pages_per_in_cb * in_tile_bytes,
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

    # Perf 2: the split reader's SECOND input CB — same page size, same tile, same
    # dtype, same depth. Two separate CBs, never one CB with two issuers:
    # cb_push_back moves a single shared write pointer, so ordering two producers
    # into block order needs a per-block semaphore handshake, which would
    # re-serialize exactly the issue the split is parallelizing.
    cb_input_sticks_b_descriptor = (
        ttnn.CBDescriptor(
            total_size=pages_per_in_cb * in_tile_bytes,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_INPUT_STICKS_B,
                    data_format=input_tensor.dtype,
                    page_size=in_tile_bytes,
                    tile=tile_descriptor,
                )
            ],
        )
        if split_reader
        else None
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

    # R_RETILE scratch: reader-private L1 for the staged SOURCE tile pages. It is
    # ONE page of `wt_chunk * stage_per_chunk_tile` bytes (depth-1 — there is no
    # producer/consumer handshake on it; the reader both writes and reads it), and
    # like every other buffer here it is a function of the block knobs only, never
    # of WT or NT_H. Off the retile path it is not created at all.
    cbs = [cb_input_sticks_descriptor, cb_output_tiles_descriptor]
    if cb_input_sticks_b_descriptor is not None:
        cbs.append(cb_input_sticks_b_descriptor)
    # Perf 1 — the OUTPUT-format pad stamp's pre-stamped tile. ONE page of one
    # output tile, writer-private (it has no producer/consumer handshake: the
    # writer both fills it and sources from it), created ONLY when the geometry can
    # actually contain a whole pad tile. See `pad_scratch` above for why that gate
    # is not optional.
    if pad_scratch:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=out_tile_bytes,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_PAD_SCRATCH,
                        data_format=output_tensor.dtype,
                        page_size=out_tile_bytes,
                        tile=ttnn.TileDescriptor(tile_h, tile_w),
                    )
                ],
            )
        )
    if retile:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=wt_chunk * stage_per_chunk_tile,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_RETILE_STAGE,
                        data_format=input_tensor.dtype,
                        page_size=wt_chunk * stage_per_chunk_tile,
                        tile=ttnn.TileDescriptor(in_tile_h, tile_w),
                    )
                ],
            )
        )

    # ========== 4. KERNELS =================================================
    # The gather (src_page_geometry, §1) cannot be expressed by the library
    # reader — its contract walks consecutive page ids as consecutive sticks — so
    # a paged source takes the general per-row loop, where an aligned source
    # simply fills nothing (valid_bytes == row_bytes).
    if in_placement == P_LOCAL_SHARD:
        src_page_bytes, src_row_pages = in_tile_bytes, 1  # no accessor read at all
    paged_src = src_row_pages > 1
    if paged_src and src_page_bytes % NOC_ALIGN_BYTES:
        # A row is gathered as one transfer per page slice, so every split has to
        # land on a NoC-alignable boundary: the slice lengths are
        # `page_bytes - (multiple of 32)`, which stays 32B-aligned only when the
        # page itself is. Every TILE-width-aligned shard satisfies this for every
        # supported dtype (32 elements is 32 B at 1 byte/elem and 64 B at 2), so
        # this only fires for a source shard whose width is not a multiple of 32
        # BYTES — a geometry a TILE tensor cannot even hold. Refuse it rather than
        # return a silently shifted gather.
        raise SupportRefusal(
            f"tilize: cross-core gather of a ROW_MAJOR shard needs a "
            f"{NOC_ALIGN_BYTES}-byte-aligned page; this source shard's row is "
            f"{src_page_bytes} B ({src_row_pages} pages per tensor row)"
        )
    if retile:
        # The retile reader is not a fill regime at all — its source is tiled, so
        # neither the library stick reader nor the R_PAD row loop can address it.
        regime = R_RETILE
        # ---- Perf 2: the RETILE-DIRECT reader ------------------------------
        # MECHANISM. The permutation's contiguous run is min(out_face_h,
        # src_face_h) FACE ROWS. Routing it through a ROW-MAJOR intermediate (what
        # Refinement 5 did, and what Perf 1 sped up but kept) throws that away —
        # stepping the destination one row steps the source a whole face, so the
        # run collapses to ONE face row (32 B at bf16). Landing the permutation
        # directly in the OUTPUT TILE keeps the run: 8 face rows = 256 B on a
        # 32->8 bf16 retile, so an output tile is 2 transfers instead of 16, every
        # byte crosses L1 once instead of twice, and the tilize compute has
        # nothing left to do (the reader IS the op unless a cast is requested).
        #
        # Measured [1,1,1024,1024] bf16 32->8 DRAM->DRAM: 41,949 -> 23,982 ns.
        # 32->16 2.00x, 32->4 2.07x, 8->32 1.97x, 16->32 2.11x, uint8 32->8 2.95x,
        # sharded destination 4.37x, L1-interleaved source 5.38x, fp32 1.22-1.26x.
        src_face_h = min(in_tile_h, 16)  # tile.cpp TILE_FACE_HW_CHOICES
        out_face_h = min(tile_h, 16)
        retile_run_bytes = min(src_face_h, out_face_h) * 16 * elem_in
        # THE ONE CARVE-OUT, written as the exception (`if cannot: legacy`):
        # a 1-row OUTPUT tile. out_face_h == 1 makes every run a single face row
        # with no reuse to gain, and the direct form MEASURED 0.79-0.89x there
        # (bf16 32->1: 79,959-89,195 vs 70,788; uint8 32->1: 20,220-24,545 vs
        # 20,078). tile_h == 2 is already a 1.33x WIN, so the boundary is exactly
        # 1 — not "small tiles". Everything else, tested or not, takes the new path.
        retile_direct = tile_h > 1
    else:
        regime = R_PAD if (plan.has_pad_region or paged_src) else R_ALIGNED
        retile_run_bytes, retile_direct = 0, False

    # Inside the direct path, WHERE the run's bytes come from. Both forms produce
    # byte-identical output; this is purely a transfer-efficiency choice.
    #   DRAM-direct: one DRAM transfer per run, no L1 round trip. Needs the run to
    #                be DRAM-alignable and at/above the transaction floor.
    #   staged     : one DRAM transfer per whole source PAGE (always aligned) plus
    #                a local NoC loopback per run. Wins when the write leg is gone,
    #                i.e. a resident L1 output shard makes the DRAM read the wall:
    #                32->8 sharded 14,818 staged vs 20,291 DRAM-direct.
    # The alignment term fires on exactly one measured family — uint8 with a
    # one-row face (16 B run) — and it selects the SOURCE of the same loop, not a
    # different kernel.
    retile_direct_dram = int(
        retile_direct
        and retile_run_bytes % NOC_DRAM_ALIGN_BYTES == 0
        and retile_run_bytes >= MIN_DIRECT_DRAM_RUN_BYTES
        and out_placement != P_LOCAL_SHARD
    )

    # -- Refinement 3, master.md B6: the one-packet read issue ---------------
    # Needs the custom reader loop, so it applies only where that loop lives: the
    # aligned W_BLOCKS accessor read.
    aligned_accessor_read = regime == R_ALIGNED and work_mode == W_BLOCKS and in_placement == P_ACCESSOR
    # B8's WRITE twin needs the output CB EXACTLY two blocks deep, so the writer's
    # two slot addresses are fixed and it can hold one block in flight while the
    # next is issued. cb_pages() is the single source for that geometry.
    trid_ok = cb_pages(cb_depth, wt_chunk) == 2 * NT_BLK * wt_chunk and NT_BLK == 1
    read_one_packet = int(aligned_accessor_read)

    # -- Perf 2: ONE R_ALIGNED reader loop (issue-ahead + stick coalescing) ---
    # `read_trid` (master.md B8's read half) is GONE, subsumed here: it was gated
    # on exactly "one outstanding group with ZERO slack", which a full sweep
    # measured as baseline-or-worse on every cell, while the same loop with one
    # group of SLACK (IN_CB_EXTRA_DEPTH) wins 1.18-1.24x. `ahead == 1` is best or
    # tied everywhere measured and `ahead >= 2` regresses, so the window is 1 and
    # is not a knob.
    #
    # This predicate deliberately drops the `work_mode == W_BLOCKS` clause the
    # Refinement-3 read lever above keeps: the unified loop covers W_REGION too,
    # which is where the crossover's 90%-of-wall read lives. B6 was priced on
    # W_BLOCKS only, so its gate stays narrow rather than being widened by
    # association.
    # Perf 2: the whole-block gather transfer is expressible exactly when the
    # block width IS one source shard row (so both ends are contiguous).
    gather_coalesce = int(
        src_row_pages > 1
        and in_placement == P_ACCESSOR
        and src_shard_rows
        and src_shard_rows % tile_h == 0
        and wt_chunk * tile_w * elem_in == src_page_bytes
    )
    aligned_read = regime == R_ALIGNED and in_placement == P_ACCESSOR
    # ISSUE-AHEAD. The source must be DRAM: on a source in ANOTHER core's L1 the
    # transaction-id machinery is pure added RISC work with no fabric latency to
    # hide behind, and it MEASURED 0.81-0.94x on three such cells (3 repeats
    # each). That is the one carve-out, and the coalescing knob below is what
    # wins on that topology instead. The slack term is mechanical: if the L1
    # fallback dropped cb_depth to 1 there is no slack group, and a zero-slack
    # window is a 0.73x regression at fp32.
    # The one carve-out on the schedule itself, and it is the SAME mechanism Perf 1
    # measured for B8's read half: where `halve_cores` fired, each core owns
    # exactly TWO blocks, so issuing block i+1 before barriering block i publishes
    # the first block a whole issue loop late with no steady state to amortize it
    # against. Re-measured this round on [1,1,32,16384] over 3 paired reps:
    # bf16 13,798 on vs 13,113 off (0.95x), fp32 26,737 vs 25,171 (0.94x). Every
    # other cell swept is a win or flat, including the ones with tiny transfers
    # (tile_h=8 0.99x, uint8 1.00x — both inside the noise band, so they keep the
    # schedule rather than being fenced off it).
    read_ahead = int(
        aligned_read and wants_read_ahead and cb_depth + in_extra >= 3 and not halve_cores and not split_reader
    )
    # TRANSACTION MERGE. `coal` consecutive source sticks become ONE transfer,
    # legal exactly when they are one contiguous address range and the block takes
    # the whole stick: a SHARDED source (an interleaved page walk round-robins the
    # banks, and the merged read measured NOT bit-exact), one page per tensor row,
    # the block spanning the whole row, and a shard height that is a whole number
    # of tile-rows so a block never straddles two shards. Measured 1.12-1.18x on
    # exactly the cells issue-ahead loses on.
    read_coalesce = (
        tile_h
        if (
            aligned_read
            and not src_in_dram
            and src_shard_rows
            and src_row_pages == 1
            and n_chunks == 1
            and wt_chunk * tile_w * elem_in == src_page_bytes
            and src_shard_rows % tile_h == 0
        )
        else 1
    )
    # A coalesced transfer is tile_h times a row and blows past the one-packet
    # NoC burst size, so B6's one-packet form is mutually exclusive with it by
    # construction.
    if read_coalesce > 1:
        read_one_packet = 0
    # B8's WRITE twin. Same two-slot CB precondition, and every write must be a
    # whole page.
    write_trid = int(out_placement == P_ACCESSOR and trid_ok)

    # A real value-preserving conversion between element formats. Derived here
    # (not at the compute kernel) because the retile-direct reader needs it too:
    # with no cast it produces the finished OUTPUT TILE itself; with a cast it
    # produces an output-SHAPED tile in the INPUT dtype for compute to datacopy.
    needs_cast = input_tensor.dtype != output_tensor.dtype

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
        src_page_bytes,
        src_row_pages,
        read_one_packet,
        # Perf 2: the issue-ahead window (0/1) and the sticks-per-transfer merge
        # factor. Together they replace B8's read-side `read_trid`.
        read_ahead,
        read_coalesce,
        # The reader's input-format fill is DEAD WORK when the writer re-stamps
        # every pad position (Refinement 4): identical regions, and the writer's
        # word is the exact one. Measured on the worst-case padded widening shape:
        # the fill is most of the R_PAD reader's cost.
        out_fill,
        # Refinement 5 (R_RETILE only): the SOURCE tile height, and the tile-column
        # count the source page id `tile_row * wt + tile_col` needs.
        in_tile_h,
        wt,
        # Perf 2 (R_RETILE only): land the face permutation directly in the output
        # tile (`retile_direct`), sourcing each run straight from DRAM rather than
        # from a staged page (`retile_direct_dram`), and — when a cast is also
        # requested — leave the finished tile in the INPUT dtype for compute to
        # convert (`retile_cast`) instead of handing the writer raw bytes nobody
        # would have packed.
        int(retile_direct),
        retile_direct_dram,
        int(retile and needs_cast),
        # Perf 2: block slots in the (one-group-deeper) input CB — the reader
        # walks them itself while a read is outstanding, since get_write_ptr only
        # advances on push_back.
        cb_pages_in(cb_depth, wt_chunk, in_extra) // wt_chunk,
        # Perf 2: the cross-core gather moves a WHOLE BLOCK in one transfer when
        # the block width is exactly one source shard row.
        gather_coalesce,
        # Perf 2 SPLIT READER: the flavor (0 none / 1 dedicated dual-NoC / 2
        # shared NOC_0 + per-RISC trid) and this RISC's phase. Phase 0 is NCRISC
        # and publishes CB_INPUT_STICKS; phase 1 is BRISC and publishes
        # CB_INPUT_STICKS_B. Each owns every other block, so the block subset is a
        # compile-time stride and NO runtime arg changes.
        split_reader,
        0,  # phase — reader #2 below is the same kernel with this set to 1
    ]
    phase_arg_index = len(reader_ct_args) - 1
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    # Reader #2 differs from reader #1 in exactly ONE compile-time value, which is
    # what keeps the two halves from drifting: same source file, same args.
    reader_b_ct_args = list(reader_ct_args)
    reader_b_ct_args[phase_arg_index] = 1

    # -- writer (BRISC / NOC1) --
    writer_ct_args = [
        out_placement,
        work_mode,
        wt_chunk,
        nt_h,
        wt,
        n_chunks,
        out_tile_bytes,
        write_trid,
        out_fill,
        # Only queried on the stamp path: `element_size()` is undefined for a
        # block-float output, which `needs_output_format_fill` already excludes.
        output_tensor.element_size() if out_fill else 4,
        tile_h,
        h_in,
        in_shape[-1],  # valid columns inside the padded target
        nth_per_img,
        n_img_in,
        # Perf 1: 1 => CB_PAD_SCRATCH exists and whole-pad tiles are produced from
        # it instead of being stamped element-by-element. Derived once, above.
        pad_scratch,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # Perf 2: on the retile-direct path the READER already produced the output
    # tile layout, so compute either has nothing to do at all (no cast) or owns
    # the conversion ALONE as a datacopy pass (cast) — never a tilize.
    # Perf 2: on the split path compute alternates between the two input CBs and
    # takes over the OUTPUT CB's drain (the writer kernel is not launched, so the
    # aliased CB would otherwise have no consumer).
    compute_ct_args = [wt_chunk, 1 if needs_cast else 0, int(retile_direct), split_reader]

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
            blocks_this_core = blocks_per_shard[shard_index]
            tile_row0, tile_col0, _ = shard["regions"][shard_index]
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
            tile_col0,  # W_REGION origin in TILE columns (R_RETILE pages are tiles)
        ]
        writer_rt_args[core.x][core.y] = [
            dst_addr,
            start_block,
            blocks_this_core,
            tile_row0,
            tile_col0,
            pad_word_out,  # the fill in the OUTPUT element format (0 when unused)
        ]
        compute_rt_args[core.x][core.y] = [blocks_this_core]
        if work_mode == W_BLOCKS:
            start_block += blocks_this_core

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "tilize_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    if split_reader:
        # Perf 2: BRISC is the SECOND READER. It gets the reader source, the reader
        # runtime args and reader #2's compile-time args; tilize_writer.cpp is not
        # launched at all on this path, and compute drains the aliased output CB
        # in its place so that CB still has exactly one consumer.
        #
        # SHARED_NOC0: both RISCs issue on NOC_0 (the two DM RISCs are not
        # interchangeable at DRAM volume — the same read work on BRISC/NOC_1 is
        # 2.2x slower, which Metal encodes as preferred_noc_for_dram_read()) under
        # DM_DYNAMIC_NOC, which is REQUIRED on BOTH: the dynamic read barrier sums
        # the two RISCs' issue counters, so a dedicated-mode partner would not
        # publish into the shared counter. Each reader then barriers on its own
        # transaction id, which is per-id hardware state and RISC-agnostic.
        if split_reader == SPLIT_SHARED_NOC0:
            _t = ttnn._ttnn.types
            reader_kernel = ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "tilize_reader.cpp"),
                core_ranges=all_cores,
                compile_time_args=reader_ct_args,
                runtime_args=reader_rt_args,
                config=ttnn.DataMovementConfigDescriptor(
                    _t.DataMovementProcessor.RISCV_1, _t.NOC.RISCV_0_default, ttnn.NOC_MODE.DM_DYNAMIC_NOC
                ),
            )
            writer_kernel = ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "tilize_reader.cpp"),
                core_ranges=all_cores,
                compile_time_args=reader_b_ct_args,
                runtime_args=reader_rt_args,
                config=ttnn.DataMovementConfigDescriptor(
                    _t.DataMovementProcessor.RISCV_0, _t.NOC.RISCV_0_default, ttnn.NOC_MODE.DM_DYNAMIC_NOC
                ),
            )
        else:
            writer_kernel = ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "tilize_reader.cpp"),
                core_ranges=all_cores,
                compile_time_args=reader_b_ct_args,
                runtime_args=reader_rt_args,
                config=ttnn.WriterConfigDescriptor(),
            )
    else:
        writer_kernel = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "tilize_writer.cpp"),
            core_ranges=all_cores,
            compile_time_args=writer_ct_args,
            runtime_args=writer_rt_args,
            config=ttnn.WriterConfigDescriptor(),
        )

    # fp32 -> fp32 must be BIT-EXACT: keep Dest in fp32 and stop the unpacker
    # downgrading fp32 to tf32 on its way to Dest. Only legal when the fast
    # tilize path is off (it is: fp32 OUTPUT disables it), which is exactly the
    # fp32-in/fp32-out case.
    lossless_fp32 = input_tensor.dtype == ttnn.float32 and output_tensor.dtype == ttnn.float32
    # 8-bit datums (Refinement 4) need fp32 DEST as well — for a different reason.
    # The tilize LLK's 8-bit path (ckernel_defs.h IS_8BIT_FORMAT) is only validated
    # with DEST accumulation on (tt-llk `test_unpack_tilize_int8` runs
    # DestAccumulation.Yes), and the reference C++ tilize set the same `fp32_llk_acc`
    # flag for its own 8-bit dtype. With a 16-bit DEST the 8-bit tile packs as ZEROS
    # (measured: probes/probe_021.py, every element 0). Keyed on the DTYPE, not on
    # element_size(), because bfloat8_b also reports 1 byte/elem and is a block-float
    # format, not an 8-bit datum.
    eight_bit = input_tensor.dtype in EIGHT_BIT_DTYPES or output_tensor.dtype in EIGHT_BIT_DTYPES
    compute_config = ttnn.ComputeConfigDescriptor()
    compute_config.fp32_dest_acc_en = lossless_fp32 or eight_bit
    # master.md F24: the FAST (truncating) block-float packer, never the precise
    # one. The fast packer already clears the bf8b accuracy gate from both bf16 and
    # fp32 inputs on this op (measured PCC 0.99997 / 1.00000 against the pad
    # oracle) and tilize does no arithmetic that could accumulate the truncation,
    # while the precise arm measured inside the noise band both ways (65,253 vs
    # 64,981 ns on the square; 2,930 vs 3,008 on the smallest regime).
    compute_config.bfp8_pack_precise = False
    if lossless_fp32:
        unpack_modes = [ttnn.UnpackToDestMode.Default] * NUM_CIRCULAR_BUFFERS
        unpack_modes[CB_INPUT_STICKS] = ttnn.UnpackToDestMode.UnpackToDestFp32
        if split_reader:
            # The split's second input CB is the SAME operand read by the same
            # unpacker — it must carry the same mode, or the two halves of one
            # tensor unpack differently (measured: the fp32 padded sharded cells
            # hang the device with this line missing).
            unpack_modes[CB_INPUT_STICKS_B] = ttnn.UnpackToDestMode.UnpackToDestFp32
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
        cbs=cbs,
    )
