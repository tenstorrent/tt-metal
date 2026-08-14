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
CB_OUTPUT_TILES = 16  # compute -> writer (tiled pages)

# --- reader regime selector (op_design.md §5.1) -----------------------------
R_ALIGNED = 0
R_PAD = 1
# R_RETILE (Refinement 5): the source is ALREADY tiled, at a different tile
# height. The reader stages whole source tile pages in L1 and moves face rows out
# of them as row-major sticks; compute then tilizes to the requested height, so
# the retile is a reader-only change. Whole-page staging is not an optimization:
# a face row is 16 elements (32 B at bf16, 16 B at uint8) and DRAM read alignment
# is 32 B on Wormhole but **64 B on Blackhole**, so face rows are not directly
# addressable in DRAM on the arch that runs these cells.
R_RETILE = 2

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
#   xfer_gate     0 -> alias the destination shard even when the read transfer it
#                      pins is below MIN_STREAM_READ_BYTES (the pre-Refinement-2
#                      behaviour: prefer local placement unconditionally).
#   pipeline      0 -> Phase-0 blocking: chunk W only as far as the grid-fill
#                      floor, so a grid-filling shape lands ONE block per core
#                      and read/compute/write cannot overlap (Refinement 3;
#                      this is what made master.md C16 read as a no-payoff).
#   write_trid    0 -> one plain write barrier per block (no double-issue), i.e.
#                      the write NoC drains between blocks (master.md B8, WRITE
#                      side — the reader lever's twin; the split-DM ablation put
#                      the write half on the critical path).
#   read_trid     0 -> one plain read barrier per block (no double-issue), i.e.
#                      the NoC drains between blocks (master.md B8 off arm).
#   read_vc       0 -> every reader issues on the default read VC
#                      (master.md B10 off arm).
#   pack_fast     0 -> bfp8_pack_precise=True, i.e. the PRECISE block-float packer
#                      (rounds instead of truncating, one extra pack pass) on a
#                      bfloat8_b output (master.md F24 off arm). 1 (shipped) is
#                      the CHEAP default: the fast packer already clears the
#                      bf8b PCC gate from both bf16 and fp32 inputs.
#   out_fill      0 -> skip the writer's OUTPUT-format pad stamp, leaving the
#                      reader's input-format fill as the only fill (the
#                      pre-Refinement-4 behaviour: exact everywhere EXCEPT a
#                      widening cast whose fill the input format cannot hold,
#                      which is precisely what Phase 0 put in EXCLUSIONS).
#   zero_copy     0 -> never alias a CB on a resident L1 shard: take the
#                     TensorAccessor path on both sides and the generic block
#                     split over the whole grid, i.e. re-read/re-write the local
#                     shard over the NoC (master.md C14 + A2 off arm; this is
#                     precisely the "tolerated, not implemented" sharded path).
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
    "zero_copy": 1,
    "xfer_gate": 1,
    "pipeline": 1,
    "read_trid": 1,
    "write_trid": 1,
    # master.md B10 ships PARKED at its byte-identical default (every reader on
    # the default request VC). Measured over 5-sample medians on Wormhole B0 it
    # is neutral on the grid-filling square (86,136 vs 85,720 ns) and a 2.6% LOSS
    # on the wide/short shape (13,478 with vs 13,131 without): spreading requests
    # over 4 VCs does not help an op whose readers are already spread over 64
    # cores and whose source pages round-robin over every DRAM bank. Kept as a
    # live knob (turning it to 1 is the ON arm the bench measures) rather than
    # deleted — the mechanism is correct, it just has nothing to break up here.
    "read_vc": 0,
    "read_one_packet": 1,
    "out_fill": 1,
    "pack_fast": 1,
}

# Levers deliberately SHIPPED in their 0 arm, with the measurement that put them
# there. A lever lands here (rather than being deleted) when the mechanism is
# correct and still a live knob, but its ON arm measured neutral-or-worse on this
# op's shapes — the ledger's `measured-no-payoff` disposition. Anything OFF that
# is NOT listed here is a bench arm that leaked into production
# (test_production_switches_ship_in_their_optimal_state pins exactly that).
PARKED_LEVERS = {
    "read_vc": "master.md B10: neutral on (a) 86,136 vs 85,720 ns, -2.6% on (b) "
    "13,478 vs 13,131 ns (5-sample medians). 64 readers over 12 round-robin DRAM "
    "banks have no first-come-first-serve route to break up.",
}

# --- B10: how many read-request VCs the readers spread over ------------------
# Unicast VCs are 0-3 (dataflow_api.h read/write `vc` parameter). Core i issues
# its read requests on `i % NUM_READ_VCS`, so cores sharing a NoC route do not
# all queue behind one another on the single default VC.
NUM_READ_VCS = 4

# --- classification ablation (perf-only; op_design.md §9.1) ------------------
# Stub a stage's PAYLOAD while keeping every CB reserve/push/wait/pop, barrier
# and loop trip count, so the duration diff attributes time to that stage.
# Production is always {0, 0}; the bench flips these. Output is wrong by design
# when either is set — never assert PCC on an ablated run.
ABLATE = {
    "compute": 0,  # 1 -> compute does the CB handshake but no tilize_block
    "dm": 0,  # 1 -> reader/writer issue no NoC transfers
    # The two DM halves separately (Refinement 3): reader and writer are one
    # pipeline, so attributing the wall to a half is what says whether a
    # reader-side lever even has a writer twin worth building.
    "dm_read": 0,  # 1 -> reader issues no NoC reads (writer untouched)
    "dm_write": 0,  # 1 -> writer issues no NoC writes (reader untouched)
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


def wt_cap(cb_depth, in_tile_bytes, out_tile_bytes, stage_bytes=0):
    """L1 ceiling on the W block factor — THE single source for that cap.

    ``in_tile_bytes`` / ``out_tile_bytes`` are the *streaming* page sizes: pass 0
    for a side whose CB is aliased on a resident shard, since that side costs no
    extra L1 (it IS the tensor). ``stage_bytes`` is the R_RETILE scratch, which is
    depth-1 (it is reader-private scratch, not a pipelined CB). When nothing
    streams, only the library's fast-tilize width bound remains.
    """
    per_chunk_tile = cb_bytes(cb_depth, 1, in_tile_bytes, out_tile_bytes) + stage_bytes
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


def derive_blocking(nt_h, wt, in_tile_bytes, out_tile_bytes, num_cores, cb_depth, tile_h, pipeline=True, stage_bytes=0):
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
    * ``pipeline`` is the Refinement-3 knob (lever ``pipeline``); False reproduces
      the Phase-0 "one block per core is enough" rule exactly.
    """
    # Bytes both CBs hold per tile-column of chunk width — read from cb_bytes()
    # via wt_cap() so the ceiling can never drift from the CB sizing below (it
    # carries the NT_BLK factor too, which a hand-written formula here would have
    # dropped) and the sharded path shares the same cap.
    cap = wt_cap(cb_depth, in_tile_bytes, out_tile_bytes, stage_bytes)

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
    decision (host gate + `out_fill` compile-time arg + the ledger's off-arm).
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
    # tile-row range and the W extent. Single-core is refused for a sharded call by
    # validate() (a shard is inherently multi-core) and the A0 off-arm forces one
    # core, so both fall back to the accessor path.
    #
    # Eligibility is PER SIDE, because the one thing padding rules out is aliasing
    # the INPUT: the fill is materialized into the input CB, and that CB is the
    # source tensor itself on the zero-copy path. The OUTPUT side is untouched by
    # the fill — compute packs whole (already padded) tiles — so a padded call
    # still packs straight into a resident destination shard instead of writing it
    # over the NoC.
    shard_eligible = plan.use_multicore and bool(LEVERS["multicore"]) and bool(LEVERS["zero_copy"])
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
            wt_chunk, n_chunks = derive_shard_blocking(
                shard["shard_wt"], wt_cap(cb_depth, stream_in, stream_out, stage_per_chunk_tile)
            )
        # Read-transfer gate (MIN_STREAM_READ_BYTES): an aliased DESTINATION pins
        # the reader's per-row transfer to the shard's own width, and below the
        # measured knee the generic full-grid split beats packing in place. Only
        # the read side can trip it — the writer always moves whole tile pages.
        if in_placement == P_ACCESSOR and LEVERS["xfer_gate"]:
            # What ONE source transfer moves. A retile reads whole source TILE
            # pages (that is what makes it addressable at all), so its transfer is
            # the page — the per-STICK size the gate was measured on describes the
            # row-major reader only.
            read_bytes = src_page_bytes if retile else min(wt_chunk * tile_w * elem_in, src_page_bytes)
            if read_bytes < MIN_STREAM_READ_BYTES:
                in_placement, out_placement, shard, work_mode = P_ACCESSOR, P_ACCESSOR, None, W_BLOCKS
    if shard is not None:
        while (
            cb_depth > 1
            and cb_bytes(cb_depth, wt_chunk, stream_in, stream_out) + wt_chunk * stage_per_chunk_tile > CB_L1_BUDGET
        ):
            cb_depth -= 1
        if cb_bytes(1, wt_chunk, stream_in, stream_out) + wt_chunk * stage_per_chunk_tile > CB_L1_BUDGET:
            # The one shape zero-copy cannot buy: the shard pins a block width
            # whose streaming partner CB will not fit. Take the accessor path on
            # both sides so WT_CHUNK is free to shrink again.
            in_placement, out_placement, shard, work_mode = P_ACCESSOR, P_ACCESSOR, None, W_BLOCKS

    if shard is None:
        work_mode = W_BLOCKS
        if LEVERS["w_split"]:
            wt_chunk, n_chunks, num_blocks_total = derive_blocking(
                nt_h,
                wt,
                in_tile_bytes,
                out_tile_bytes,
                num_cores_available,
                cb_depth,
                tile_h,
                pipeline=bool(LEVERS["pipeline"]),
                stage_bytes=stage_per_chunk_tile,
            )
        else:
            # w_split OFF — the pure height split (op_design.md §1.3 candidate 2).
            wt_chunk, n_chunks, num_blocks_total = wt, 1, nt_h

        # never OOM: fall back to depth-1 rather than exceed the L1 budget
        # (same cb_bytes() source as derive_blocking's ceiling and the CBs below)
        while (
            cb_depth > 1
            and cb_bytes(cb_depth, wt_chunk, in_tile_bytes, out_tile_bytes) + wt_chunk * stage_per_chunk_tile
            > CB_L1_BUDGET
        ):
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
        # Per shard, not per grid: an unevenly split grid gives its last shard
        # fewer tile-rows, so the block count comes from that region's own extent.
        blocks_per_shard = [ht * n_chunks for (_, _, ht) in shard["regions"]]
        num_blocks_total = sum(blocks_per_shard)

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

    # R_RETILE scratch: reader-private L1 for the staged SOURCE tile pages. It is
    # ONE page of `wt_chunk * stage_per_chunk_tile` bytes (depth-1 — there is no
    # producer/consumer handshake on it; the reader both writes and reads it), and
    # like every other buffer here it is a function of the block knobs only, never
    # of WT or NT_H. Off the retile path it is not created at all.
    cbs = [cb_input_sticks_descriptor, cb_output_tiles_descriptor]
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
    else:
        regime = R_PAD if (plan.has_pad_region or paged_src or not LEVERS["regime_select"]) else R_ALIGNED

    # The rank->=2 promoted view (plan.read_shape): a rank-0 scalar reads as a
    # 1x1 source, so h_in / w_in_bytes / the image count stay well defined.
    in_shape = list(plan.read_shape)
    h_in = in_shape[-2]
    w_in_bytes = in_shape[-1] * elem_in
    n_img_in = _prod(in_shape[:-2])
    nth_per_img = _div_up(target[-2], tile_h)
    pad_word = _pack_pad_word(plan.pad_value, input_tensor.dtype)

    # -- Refinement 4: the OUTPUT-format pad stamp (lever `out_fill`) ---------
    # The reader's input-format fill is arithmetically exact for every path except
    # a WIDENING cast with a fill the input format cannot hold. There the writer
    # re-stamps the pad region of each finished tile with a second word packed in
    # the OUTPUT format (see kernels/tilize_fill.hpp). Gated on the round-trip
    # actually losing the value, so every other cell is byte-identical to before.
    out_fill = int(
        plan.has_pad_region
        # The stamp addresses whole faces: 16 rows on a full tile, tile_h rows on a
        # tiny one (Refinement 5). Every legal tile height satisfies one or the other.
        and (tile_h % FACE_HEIGHT == 0 or FACE_HEIGHT % tile_h == 0)
        and bool(LEVERS["out_fill"])
        and needs_output_format_fill(plan.pad_value, input_tensor.dtype, output_tensor.dtype)
    )
    pad_word_out = _pack_pad_word(plan.pad_value, output_tensor.dtype) if out_fill else 0

    # -- Refinement-3 read levers (interleaved aligned path only) ------------
    # All three need the custom reader loop; with all three off the reader
    # compiles to the library-helper call verbatim (the Phase-0 hot path).
    # They apply only where that loop lives: the aligned W_BLOCKS accessor read.
    aligned_accessor_read = regime == R_ALIGNED and work_mode == W_BLOCKS and in_placement == P_ACCESSOR
    # B8 needs the input CB to be EXACTLY two blocks deep, so the reader's two
    # slot addresses are fixed and it can hold one block in flight while the
    # next is issued. cb_pages() is the single source for that geometry.
    trid_ok = cb_pages(cb_depth, wt_chunk) == 2 * NT_BLK * wt_chunk and NT_BLK == 1
    read_one_packet = int(aligned_accessor_read and LEVERS["read_one_packet"])
    read_trid = int(aligned_accessor_read and trid_ok and LEVERS["read_trid"])
    read_vc_enable = int(aligned_accessor_read and LEVERS["read_vc"])
    # B8's WRITE twin. Same two-slot CB precondition, and every write must be a
    # whole page (the page_write OFF arm splits them, and an ablated arm issues
    # none), so those two arms fall back to the plain barrier-per-block loop.
    write_trid = int(
        out_placement == P_ACCESSOR
        and trid_ok
        and LEVERS["write_trid"]
        and LEVERS["page_write"]
        and not (ABLATE["dm"] or ABLATE["dm_write"])
    )

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
        ABLATE["dm"] or ABLATE["dm_read"],
        src_page_bytes,
        src_row_pages,
        read_one_packet,
        read_trid,
        read_vc_enable,
        # The reader's input-format fill is DEAD WORK when the writer re-stamps
        # every pad position (Refinement 4): identical regions, and the writer's
        # word is the exact one. Measured on the worst-case padded widening shape:
        # the fill is most of the R_PAD reader's cost.
        out_fill,
        # Refinement 5 (R_RETILE only): the SOURCE tile height, and the tile-column
        # count the source page id `tile_row * wt + tile_col` needs.
        in_tile_h,
        wt,
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
        ABLATE["dm"] or ABLATE["dm_write"],
        LEVERS["page_write"],
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
            shard_index % NUM_READ_VCS,  # B10: this core's read-request VC
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
    # master.md F24: the expensive setting of the block-float packer is NOT the
    # default here — the FAST (truncating) packer already clears the bf8b accuracy
    # gate from both bf16 and fp32 inputs on this op (measured PCC 0.99997 /
    # 1.00000 against the pad oracle), and tilize does no arithmetic that could
    # accumulate the truncation. Only the counterfactual arm turns it on.
    compute_config.bfp8_pack_precise = output_tensor.dtype in BLOCK_FLOAT_DTYPES and not LEVERS["pack_fast"]
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
        cbs=cbs,
    )
