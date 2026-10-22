# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm ProgramDescriptor — regime R1 (row-parallel, row-resident) and regime R3 (W-split + cross-core
Σx² combine), op_design.md -> Regimes.

Regimes (select_plan):
    R1  interleaved, rows too narrow to need a W-split: tile-rows over the full grid, each core holds whole
        rows resident (no cross-Tensix-core traffic).
    R3  every core owns a (tile-row range × W-slice) block. Cores are grouped into ROW-GROUPS of G_w cores that
        share a tile-row range; group = a consecutive run of G_w cores in (y, x) order, root = its first core.
        Per block: each core reduces its slice (square_accumulate + local REDUCE_ROW collapse, column 0
        valid), unicasts the partial to the root's gather CB (+ a semaphore increment), the root sums the G_w
        partials (reduce<AccumulateViaAdd, Skip>) and finalizes (×1/W, +ε, rsqrt), and multicasts inv_rms
        back to the group (ttnn.Mcast + McastArgs, receiver handshake). With owners > 1 the combine is a
        REDUCE-SCATTER instead (catalog tensix_all_reduce): the block's tile-rows are split over `owners` group
        members, each member sends an owner its partials of that owner's rows (one write), every owner sums +
        finalizes its ceil(R_b / owners) rows and unicasts them into every member's cb_inv_rms (all-gather; a
        second semaphore counts the arrivals).
        Placements:
          - WIDTH_SHARDED / BLOCK_SHARDED input (native, zero-copy: cb_x_tiles backed on the input shard,
            cb_output backed on the output shard; the slice/row map is read off the shard spec);
          - INTERLEAVED TILE input when R1 refuses (row too wide for L1) or on occupancy (few tile-rows):
            a logical W-split, each core reads its W-slice from DRAM once.

Blocking model (op_design.md -> Blocking Model). Every knob is defined ONCE here and every
dependent quantity (CB page counts, loop trip counts, grid) derives from it:

    W_CHUNK_TILES  -> cap on w_chunk_tiles. The cap actually used is the largest of
                      W_CHUNK_TILES, W_CHUNK_TILES/2, ... that fits L1 (preferring one that still
                      allows x_depth = 2); then, to minimise pad tiles, over the core's slice S
                      (S = Wt in R1):
                      num_w_chunks       = ceil(S / cap)
                      w_chunk_tiles (C)  = ceil(S / num_w_chunks)
                      width_padded_tiles = num_w_chunks * C      (Wp)
                      x_row_stride (WS)  = Wp (interleaved ring, pad tiles zeroed) or S (shard-backed: the
                                           shard's own row stride; the last chunk is S - (nch-1)*C wide)
    STREAM_DEPTH   -> depth (in C-tile quanta) of cb_x_rm_sticks, cb_gamma_tiles (streaming γ), cb_output
    GAMMA_RESIDENT -> allow the resident-γ fast path: all Wp row-0 γ tiles expanded ONCE per kernel
                      (cb_gamma_tiles = Wp pages) when that fits L1 at the block size already chosen;
                      otherwise (or False) γ chunks stream, expanded per (block, chunk). Both expansions
                      run on the WRITER RISC-V, so γ never sits on the reader's critical path (lamp L4).
    GAMMA_DIRECT_READ -> TILE γ on the resident path: read row 0 of each γ tile from DRAM straight into its
                      cb_gamma_tiles page — no staging row, no L1->L1 expansion. Modes:
                        GAMMA_DIRECT_PREFIX   one read of the row-0 byte prefix (gamma_row0_prefix_bytes:
                                              face 0 row 0 .. face 1 row 0, + the Bfp8_b exponent head)
                        GAMMA_DIRECT_SEGMENTS one read per row-0 segment (fewer bytes, more requests)
                        GAMMA_DIRECT_OFF      staging fill + expansion (2 reads per segment per tile)
    GAMMA_SEG_SLOT -> bytes per row-0 segment slot of the TILE-γ staging row (>= every arch's DRAM read
                      alignment, so a DRAM->L1 segment read keeps matching low address bits)
    SCALE_BLOCK_TILES -> tiles per DEST window of the scaling stages (scale_by_gamma / scale_by_inv_rms);
                      the eltwise chain clamps it to DEST capacity (8 under 16-bit DEST, 4 under fp32
                      DEST). 1 = one dst-sync + CB handshake per tile.
    FINALIZE_COL0   -> perf lamp L7: the finalize post-op (x1/W, +eps, rsqrt) runs as one fused SFPU pass over
                      the column-0 vectors of each REDUCE_ROW result (8 of the tile's 32 vectors, once) instead
                      of three full-tile passes. False = the stock mul_unary / add_unary / rsqrt_tile sequence.
    BANK_STAGGER    -> perf lamp L9: core k rotates its DRAM issue order (x reads, output writes, γ reads)
                      by k mod NUM_DRAM_BANKS inside each batch, so cores in lockstep do not all hit one
                      bank (every tile-row starts on the same bank when Wt % banks == 0). False = in order.
    MIN_BLOCKS_PER_CORE -> block_rows cap so an interleaved core with >= 2 tile-rows walks >= 2 blocks (the
                      reader fetches block k+1 while block k is scaled and written; perf lamp L1). Sharded
                      cores take the coarsest block (the whole shard, clamped by L1): nothing to prefetch.
    R3_ROWS_PER_GROUP, R3_MAX_GROUP_CORES, R3_MIN_SLICE_TILES -> interleaved regime R3 layout (perf lamp L2:
                      fewer slice cores = cheaper combine, more = more occupancy). Aim for row-groups of
                      ~R3_ROWS_PER_GROUP tile-rows: G_w target = clamp(cores // ceil(rows / R3_ROWS_PER_GROUP),
                      1, R3_MAX_GROUP_CORES); S = max(R3_MIN_SLICE_TILES, ceil(Wt / target)); G_w = ceil(Wt / S);
                      G_h = min(rows, cores // G_w).
    R3_SCATTER_ROWS_PER_OWNER -> regime R3 combine topology (perf lamp L2, catalog tensix_all_reduce): the
                      row-group splits the combine of each block over owners = min(G, rows_max //
                      R3_SCATTER_ROWS_PER_OWNER) members (reduce-scatter + unicast all-gather), so the sum of the G
                      partials and the finalize of R rows no longer serialize on one root. owners == 1 (one-row
                      groups such as decode, or the knob at 0) is the flat root reduce + one multicast back.
    R3_OCCUPANCY_GAIN, R3_MIN_GROUP_CORES -> interleaved regime R3 is chosen over R1 (when R1 also fits) only if
                      the W-split puts at least R3_OCCUPANCY_GAIN x as many Tensix cores to work AND cuts each
                      row into >= R3_MIN_GROUP_CORES slices (a 2-way split does not pay for its combine:
                      32x256 4.02 -> 4.23 us). R3_OCCUPANCY_GAIN = 0 disables the occupancy branch.
    block_rows (R), x_depth -> the largest R whose L1 ledger footprint fits (choose_block_rows; footprint is
                               monotone in R)
    grid / core count       -> R1: split_work_to_cores over the full compute grid (row_wise=True);
                               R3: the shard spec (sharded) / R3 row-groups (interleaved)
"""

from __future__ import annotations

import dataclasses
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# ---- perf_experiments/combine_bytes variant switches (set by the bench before each dispatch) ----
# COMBINE_FACES: 0 = baseline full-tile combine writes, 1 = faces 0/2 (two writes per tile), 2 = faces 0/2 via the
# one-packet stateful write. POISON_FACES13: NaN-fill faces 1/3 of cb_gather / cb_inv_rms (correctness probe).
COMBINE_FACES = 0
POISON_FACES13 = 0
END_ATOMIC_BARRIER = 0  # 1: noc_async_atomic_barrier() at the end of the reader (latent end-of-kernel atomics race)

TILE_DIM = 32
FACE_DIM = 16
TILE_HW = TILE_DIM * TILE_DIM
BFP8_EXP_BYTES = 64  # Bfp8_b tile: exponent section (4 faces x 16 face rows, 1 B each) before the mantissas

# ---- Knobs (single source of truth) ----
W_CHUNK_TILES = 32  # streaming chunk width cap (tiles) — perf lamp L3 (RM: C*32*es B NoC segments)
STREAM_DEPTH = 2  # depth of the three streaming CBs, in C-tile quanta
SCALE_BLOCK_TILES = 8  # compute block of the scaling chains (clamped to DEST capacity by the chain)
FINALIZE_COL0 = True  # perf lamp L7: finalize (x1/W, +eps, rsqrt) as ONE SFPU pass over column 0 (8 vectors, not 96)
BANK_STAGGER = False  # perf lamp L9: per-core rotation of the DRAM issue order (kernels reduce it mod #banks)
MIN_BLOCKS_PER_CORE = 2  # perf lamp L1: cross-block read/write overlap needs >= 2 blocks per core
# TILE reader: C-tile chunks per NoC read barrier, as a cap; None = a whole tile-row per barrier.
READ_BATCH_CHUNKS = None
GAMMA_RESIDENT = True  # perf lamp L4: expand γ once per kernel when L1 allows (never shrinks the block)
GAMMA_DIRECT_OFF, GAMMA_DIRECT_PREFIX, GAMMA_DIRECT_SEGMENTS = 0, 1, 2  # mirrored by the writer kernel
GAMMA_DIRECT_READ = GAMMA_DIRECT_PREFIX  # TILE γ resident path: DRAM -> cb_gamma_tiles, no staging
GAMMA_SEG_SLOT = 64  # B; Blackhole DRAM read alignment (Wormhole: 32) — a multiple of both
R3_ROWS_PER_GROUP = 2  # perf lamp L2: target tile-rows per interleaved row-group
R3_MAX_GROUP_CORES = 24  # perf lamp L2: cap on W-slice cores per interleaved row-group (combine fan-in)
R3_MIN_SLICE_TILES = 4  # perf lamp L2: smallest interleaved W-slice per core (tiles)
R3_SCATTER_ROWS_PER_OWNER = 1  # perf lamp L2: min tile-rows per reduce-scatter owner (0 = flat root reduce)
R3_OCCUPANCY_GAIN = 2  # interleaved R3 over R1 only when it puts >= this many times the cores to work
R3_MIN_GROUP_CORES = 4  # ... and splits each row into at least this many W-slices

# Packed reduce scaler (1.0, exact in bfloat16) — calculate_and_prepare_reduce_scaler's format.
SCALER_FORMAT = ttnn.bfloat16

# ---- CB indices (semantic names; the kernels declare the same slots) ----
CB_X_RM_STICKS = 0
CB_X_TILES = 1
CB_REDUCE_SCALER = 2
CB_SUMSQ_PARTIAL = 3
CB_INV_RMS = 4
CB_GAMMA_ROW = 5
CB_GAMMA_TILES = 6
CB_X_GAMMA = 7
CB_Y_TILES = 8
CB_PARTIAL_COL = 9  # R3: this core's collapsed Σx² partial (column 0) — compute -> reader
CB_GATHER = 10  # R3: the root's G_w partials per tile-row — reader (root) -> compute (root)
CB_RMS_ROOT = 11  # R3: the root's finalized inv_rms, before the multicast — compute (root) -> reader (root)
CB_SUMSQ_LAST = 12  # sharded W-pad mask: x² of the partial last tile — compute -> compute
CB_SUMSQ_ACC = 13  # sharded W-pad mask: collapsed Σx² of the full tiles (reduce accumulator) — compute
CB_OUTPUT = 16

# ---- Semaphores ----
SEM_GATHER = 0  # R3: root counts partial arrivals (contributors increment it after their write lands)
SEM_MCAST_BASE = 1  # R3, owners == 1: first id of the inv_rms multicast channel's semaphores (ttnn.Mcast.attach)
SEM_INV = 1  # R3, owners > 1: members count inv_rms arrivals (the multicast channel is absent, its ids are free)
MCAST_PREFIX = "mcast"


def _fp32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


@dataclass(frozen=True)
class Topology:
    """Cross-core structure of the chosen regime — the host constants every footprint term depends on."""

    sharded: bool = False  # TILE: cb_x_tiles / cb_output backed on the input / output shard (zero-copy)
    rm_sharded: bool = False  # ROW_MAJOR shard: sticks re-laid locally into the tilize staging and back
    combine: bool = False  # cross-core Σx² combine (G > 1)
    G: int = 1  # cores per row-group (W-slices sharing a tile-row range)
    owners: int = 1  # combine owners per row-group: 1 = root reduce + multicast, > 1 = reduce-scatter + all-gather
    masked: bool = False  # sharded and W % 32 != 0: the core owning tile Wt-1 masks its pad columns in compute


R1_TOPOLOGY = Topology()


@dataclass(frozen=True)
class Geometry:
    is_rm: bool
    has_gamma: bool
    W: int  # logical last-dim elements
    Wt: int  # ceil(W / 32) — the whole row (DRAM page addressing)
    S: int  # tiles of the W-slice one core owns (nominal; R1: Wt)
    C: int  # w_chunk_tiles
    num_w_chunks: int
    Wp: int  # width_padded_tiles of the slice = num_w_chunks * C
    WS: int  # cb_x_tiles / cb_output row stride (tiles): Wp (interleaved ring) or S (shard-backed)
    tensor_tile_rows: int
    num_sticks: int  # RM only (rows); 0 for TILE
    es: int  # input element bytes
    es_g: int  # gamma element bytes (0 when absent)
    gamma_is_tile: bool  # γ arrives as Layout::TILE (row 0 of each γ tile holds 32 γ values)
    gamma_row_page: int  # bytes of cb_gamma_row per γ tile: RM 32*es_g (compact stick), TILE segs*GAMMA_SEG_SLOT
    T_in: int
    T_out: int
    T_g: int  # gamma tile bytes (0 when absent)
    interm_format: object  # page format of the compute-internal CBs (follows DEST width)
    T_interm: int  # intermediate tile bytes — the ONE source for both CB sizing and the footprint
    topo: Topology
    # TILE γ streaming: read row 0 straight from DRAM per (block, chunk) instead of through the S-page staging
    # row — the L1 fallback choose_blocking takes only when the staging row does not fit.
    gamma_stream_direct: bool = False


def datum_bytes(dtype) -> int:
    """Bytes per datum. Bfp8_b has no per-datum byte size (Tensor.element_size() raises): its 1 B is the
    mantissa only, the shared exponents live in a 64 B section. Bfp8_b is TILE-only and W-aligned here,
    so no consumer of this value (RM stick bytes, W-pad element zeroing) ever runs on a Bfp8_b tensor."""
    return 1 if dtype == ttnn.bfloat8_b else ttnn.tile_size(dtype) // TILE_HW


def intermediate_format(input_tensor, fp32_dest_acc_en: bool):
    """Page format of every compute-internal CB (cb_sumsq_partial, cb_inv_rms, cb_x_gamma, and the R3
    partial / gather / root CBs). It follows the DEST accumulation width (l1_ledger.md audit 2): Float32
    under fp32 DEST, Float16_b under 16-bit DEST — never wider than DEST (a Float32 page would store no
    extra bits), and never Bfp8_b even for a Bfp8_b input (the raw Σx² lanes and x·γ products would lose
    their per-datum exponent)."""
    return ttnn.float32 if fp32_dest_acc_en else ttnn.bfloat16


def gamma_row_segments(gamma) -> tuple:
    """Byte offsets, inside one TILE γ tile, of the segments that carry row 0 of faces 0 and 1 — the
    only rows the BroadcastDim::Row multiply reads. Mirrored by the writer kernel (gamma_segment_offset).
      Float32 / Float16_b: face 0 row 0 at 0, face 1 row 0 at one face (16x16 datums) in.
      Bfp8_b: 64 B exponent section first (one shared exponent per 16-datum face row, face-major:
              face f row r at byte 16f + r), then 1 B mantissas face-major. Row 0 of faces 0/1 needs
              the exponents [0, 32) and the mantissas at 64 and 64 + 256. Every offset is a multiple
              of the DRAM read alignment, so a segment read keeps matching low address bits."""
    if gamma.dtype == ttnn.bfloat8_b:
        return (0, BFP8_EXP_BYTES, BFP8_EXP_BYTES + FACE_DIM * FACE_DIM)
    return (0, FACE_DIM * FACE_DIM * datum_bytes(gamma.dtype))


def gamma_segment_bytes(gamma) -> int:
    """Bytes per row-0 segment read (uniform across a γ tile's segments). Bfp8_b: 32 B covers both the
    face 0 / face 1 exponent bytes (16 each) and a 16 B mantissa row (the extra 16 B is face row 1,
    never consumed)."""
    return 2 * FACE_DIM if gamma.dtype == ttnn.bfloat8_b else FACE_DIM * datum_bytes(gamma.dtype)


def gamma_row0_prefix_bytes(gamma) -> int:
    """Bytes [0, n) of a TILE γ tile that contain every row-0 segment (gamma_row_segments): one contiguous
    read covers them all. bf16 544 B, fp32 1088 B, Bfp8_b 352 B (vs 2048 / 4096 / 1088 B per tile).
    Mirrored by the writer kernel (gamma_prefix_bytes)."""
    return gamma_row_segments(gamma)[-1] + gamma_segment_bytes(gamma)


def tensor_tile_rows_of(input_tensor) -> int:
    shape = list(input_tensor.shape)
    if input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT:
        return math.ceil(_prod(shape[:-1]) / TILE_DIM)
    return _prod(shape[:-2]) * math.ceil(shape[-2] / TILE_DIM)


def compute_geometry(
    input_tensor,
    gamma,
    chunk_cap: int,
    fp32_dest_acc_en: bool,
    slice_tiles=None,
    topo: Topology = R1_TOPOLOGY,
    gamma_stream_direct: bool = False,
) -> Geometry:
    shape = list(input_tensor.shape)
    is_rm = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    has_gamma = gamma is not None
    W = shape[-1]
    Wt = math.ceil(W / TILE_DIM)
    S = Wt if slice_tiles is None else slice_tiles
    num_w_chunks = math.ceil(S / chunk_cap)
    C = math.ceil(S / num_w_chunks)
    if topo.sharded:
        # The shard's row stride is S; chunks walk it with a narrower last chunk (never past the row).
        num_w_chunks = math.ceil(S / C)
    Wp = num_w_chunks * C
    WS = S if topo.sharded else Wp
    num_sticks = _prod(shape[:-1]) if is_rm else 0
    es = datum_bytes(input_tensor.dtype)
    T_in = ttnn.tile_size(input_tensor.dtype)
    if has_gamma:
        es_g = datum_bytes(gamma.dtype)
        T_g = ttnn.tile_size(gamma.dtype)
        gamma_is_tile = gamma.layout == ttnn.TILE_LAYOUT
        # TILE γ: the row-0 segments of each tile (gamma_row_segments), each in its own aligned slot.
        # RM γ: the compact stick (two 16-element halves, contiguous).
        assert gamma_segment_bytes(gamma) <= GAMMA_SEG_SLOT
        gamma_row_page = len(gamma_row_segments(gamma)) * GAMMA_SEG_SLOT if gamma_is_tile else TILE_DIM * es_g
    else:
        es_g = 0
        T_g = 0
        gamma_is_tile = False
        gamma_row_page = 0
    interm_format = intermediate_format(input_tensor, fp32_dest_acc_en)
    return Geometry(
        is_rm=is_rm,
        has_gamma=has_gamma,
        W=W,
        Wt=Wt,
        S=S,
        C=C,
        num_w_chunks=num_w_chunks,
        Wp=Wp,
        WS=WS,
        tensor_tile_rows=tensor_tile_rows_of(input_tensor),
        num_sticks=num_sticks,
        es=es,
        es_g=es_g,
        gamma_is_tile=gamma_is_tile,
        gamma_row_page=gamma_row_page,
        T_in=T_in,
        T_out=T_in,  # output dtype == input dtype
        T_g=T_g,
        interm_format=interm_format,
        T_interm=ttnn.tile_size(interm_format),
        topo=topo,
        gamma_stream_direct=gamma_stream_direct and has_gamma and gamma_is_tile,
    )


def _per_row_bytes(g: Geometry, x_depth: int) -> int:
    """Bytes that scale with block_rows (one tile-row of the block) — l1_ledger.md F(R, d)."""
    t = g.topo
    return (
        int(not t.sharded) * x_depth * g.WS * g.T_in  # cb_x_tiles (shard-backed: zero-copy, no L1 of its own)
        + 2 * g.T_interm  # cb_sumsq_partial + cb_inv_rms
        + int(g.has_gamma) * g.C * g.T_interm  # cb_x_gamma
        + int(g.is_rm) * g.C * g.T_out  # cb_y_tiles
        + int(t.masked) * 2 * g.T_interm  # cb_sumsq_last + cb_sumsq_acc
    )


def scatter_rows(t: Topology, block_rows: int) -> int:
    """Tile-rows each combine owner sums + finalizes per full block (the uniform push quantum, ragged split)."""
    return math.ceil(block_rows / t.owners)


def gather_pages(t: Topology, block_rows: int) -> int:
    """cb_gather: G partials per owned tile-row (uniform quantum on every owner, so the base address every
    contributor targets is where each owner's write pointer sits at a block start)."""
    return scatter_rows(t, block_rows) * t.G


def combine_bytes(g: Geometry, block_rows: int) -> int:
    """cb_partial_col (R) + cb_gather + cb_rms_root (one page per owned tile-row) — l1_ledger.md."""
    t = g.topo
    if not t.combine:
        return 0
    return (block_rows + gather_pages(t, block_rows) + scatter_rows(t, block_rows)) * g.T_interm


def gamma_tiles_pages(g: Geometry, gamma_resident: bool) -> int:
    """cb_gamma_tiles capacity: the whole padded slice (resident) or STREAM_DEPTH C-tile chunks."""
    return g.Wp if gamma_resident else STREAM_DEPTH * g.C


def gamma_direct(g: Geometry, gamma_resident: bool) -> int:
    """Direct-read mode of the γ supply: TILE γ on the resident path (or streaming, under the L1 fallback
    gamma_stream_direct) reads row 0 straight into cb_gamma_tiles (no staging row); every other γ path stays on
    the staging row (GAMMA_DIRECT_OFF)."""
    if g.has_gamma and g.gamma_is_tile and (gamma_resident or g.gamma_stream_direct):
        return GAMMA_DIRECT_READ
    return GAMMA_DIRECT_OFF


def gamma_row_pages(g: Geometry, gamma_resident: bool) -> int:
    """cb_gamma_row (staging row) capacity: S pages (the core's slice), or 0 when the direct prefix read
    replaces it. A ROW_MAJOR-sharded slice starts at any element, so its DRAM read starts at the aligned-down
    address: one more page of slack."""
    if not g.has_gamma or gamma_direct(g, gamma_resident) != GAMMA_DIRECT_OFF:
        return 0
    return g.S + int(g.topo.rm_sharded)


def scaler_pages(g: Geometry) -> int:
    """cb_reduce_scaler: [full 1.0] — or [full, partial] when a sharded core masks its W-pad columns in the
    reduce (ReducePartialScaler::with_partial, l1_ledger.md)."""
    return 2 if g.topo.masked else 1


def _fixed_bytes(g: Geometry, gamma_resident: bool = False) -> int:
    """Bytes independent of block_rows — l1_ledger.md F(R, d)."""
    return (
        scaler_pages(g) * ttnn.tile_size(SCALER_FORMAT)  # cb_reduce_scaler
        + int(not g.topo.sharded) * STREAM_DEPTH * g.C * g.T_out  # cb_output (shard-backed: zero-copy)
        + int(g.is_rm) * STREAM_DEPTH * g.C * g.T_in  # cb_x_rm_sticks
        + gamma_row_pages(g, gamma_resident) * g.gamma_row_page  # cb_gamma_row
        + int(g.has_gamma) * gamma_tiles_pages(g, gamma_resident) * g.T_g  # cb_gamma_tiles
    )


def footprint(g: Geometry, block_rows: int, x_depth: int, gamma_resident: bool = False) -> int:
    return block_rows * _per_row_bytes(g, x_depth) + combine_bytes(g, block_rows) + _fixed_bytes(g, gamma_resident)


def _max_block_rows(g: Geometry, x_depth: int, cap: int, budget: int) -> int:
    """Largest R in [1, cap] whose footprint fits (footprint is monotone in R); 0 when R = 1 does not fit."""
    lo, hi = 0, cap
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if footprint(g, mid, x_depth) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


def choose_gamma_resident(g: Geometry, block_rows: int, x_depth: int, budget: int) -> bool:
    """Resident γ only when it fits on top of the blocking already chosen with streaming γ — it never
    costs block rows, x_depth or chunk width (l1_ledger.md: _WIDE shapes stay on the streaming path)."""
    return GAMMA_RESIDENT and g.has_gamma and footprint(g, block_rows, x_depth, True) <= budget


def l1_budget(device) -> int:
    return ttnn.get_memory_view(device, ttnn.BufferType.L1).largest_contiguous_bytes_free_per_bank


def choose_block_rows(g: Geometry, rows_per_core_max: int, budget: int):
    """Closed form (op_design.md -> Work Distribution). Returns (block_rows, x_depth) or None when a
    single resident tile-row does not fit."""
    if g.topo.sharded:
        # The shard is resident already (zero-copy): the coarsest block that fits, up to the whole shard.
        # x_depth is meaningless here (no ring); 2 marks "the preferred choice" for choose_blocking.
        r = _max_block_rows(g, 1, rows_per_core_max, budget)
        return (r, 2) if r >= 1 else None
    if rows_per_core_max <= 1:
        # Every core walks exactly one block: there is no block k+1 to prefetch, so a second x buffer
        # buys nothing — x_depth = 1 frees R*Wp*T_in bytes (e.g. for resident γ on decode shapes).
        return (1, 1) if footprint(g, 1, 1) <= budget else None
    rows_per_block_cap = max(1, math.ceil(rows_per_core_max / MIN_BLOCKS_PER_CORE))
    r2 = _max_block_rows(g, 2, rows_per_block_cap, budget)
    if r2 >= 1:
        return r2, 2
    if footprint(g, 1, 1) <= budget:
        return 1, 1
    return None


def choose_blocking(
    input_tensor,
    gamma,
    rows_per_core_max: int,
    budget: int,
    fp32_dest_acc_en: bool,
    slice_tiles=None,
    topo: Topology = R1_TOPOLOGY,
    refuse: bool = True,
):
    """Pick (geometry, block_rows, x_depth): the widest chunk cap (W_CHUNK_TILES, halving) that keeps
    x_depth = 2; failing that, the widest that fits a resident row at x_depth = 1; failing that with TILE γ,
    the same search without the γ staging row (gamma_stream_direct); else refuse (or None when refuse is
    False — the caller then tries the next regime)."""
    choice = _search_blocking(input_tensor, gamma, rows_per_core_max, budget, fp32_dest_acc_en, slice_tiles, topo)
    if choice is None and gamma is not None and gamma.layout == ttnn.TILE_LAYOUT:
        choice = _search_blocking(
            input_tensor, gamma, rows_per_core_max, budget, fp32_dest_acc_en, slice_tiles, topo, True
        )
    if choice is not None or not refuse:
        return choice
    g = compute_geometry(input_tensor, gamma, 1, fp32_dest_acc_en, slice_tiles, topo)
    raise NotImplementedError(
        f"rms_norm: a W-slice of {g.S} tiles (W={g.W}) does not fit a resident tile-row in L1 (needs "
        f"{footprint(g, 1, 1)} B, budget {budget} B)"
    )


def _search_blocking(
    input_tensor, gamma, rows_per_core_max, budget, fp32_dest_acc_en, slice_tiles, topo, gamma_stream_direct=False
):
    caps = []
    cap = W_CHUNK_TILES
    while cap >= 1:
        caps.append(cap)
        cap //= 2
    fallback = None
    for cap in caps:
        g = compute_geometry(input_tensor, gamma, cap, fp32_dest_acc_en, slice_tiles, topo, gamma_stream_direct)
        choice = choose_block_rows(g, rows_per_core_max, budget)
        if choice is None:
            continue
        if choice[1] == 2:
            return g, choice[0], choice[1]
        if fallback is None:
            fallback = (g, choice[0], choice[1])
    return fallback


# ---------------------------------------------------------------------------
# Work plans (regime selection + per-core assignment)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoreWork:
    core: object  # ttnn.CoreCoord (logical)
    start_tile_row: int
    rows: int  # tile-rows this core (and its whole row-group) processes
    w0: int  # first tile of this core's W-slice (TILE; ROW_MAJOR shard slices need not be tile-aligned)
    V: int  # valid tiles of the slice (<= S; the ragged last slice is shorter)
    index: int  # position in the row-group (0 = root)
    root: object  # ttnn.CoreCoord (logical) of the row-group root
    e0: int  # first element of the W-slice
    ve: int  # valid elements of the W-slice
    sticks: int = 0  # ROW_MAJOR shard: valid sticks (rows) this core holds; 0 otherwise


@dataclass(frozen=True)
class Plan:
    regime: str  # "R1" or "R3"
    topo: Topology
    slice_tiles: int  # S
    rows_max: int  # max tile-rows per core
    work: tuple  # CoreWork per active core, in (y, x) order
    geometry: Geometry
    block_rows: int
    x_depth: int

    @property
    def core_ranges(self):
        return _core_range_set([w.core for w in self.work])


def _core_range_set(cores):
    """CoreRangeSet of the given cores as maximal x-runs per grid row."""
    ranges = []
    run = None
    for c in _yx_sorted(cores):
        if run is not None and c.y == run[1].y and c.x == run[1].x + 1:
            run = (run[0], c)
        else:
            if run is not None:
                ranges.append(ttnn.CoreRange(run[0], run[1]))
            run = (c, c)
    if run is not None:
        ranges.append(ttnn.CoreRange(run[0], run[1]))
    return ttnn.CoreRangeSet(ranges)


def _cores_of(core_range_set):
    cores = []
    for rng in core_range_set.ranges():
        for y in range(rng.start.y, rng.end.y + 1):
            for x in range(rng.start.x, rng.end.x + 1):
                cores.append(ttnn.CoreCoord(x, y))
    return cores


def _yx_sorted(cores):
    return sorted(cores, key=lambda c: (c.y, c.x))


def _r1_work(input_tensor, Wt, tensor_tile_rows):
    W = input_tensor.shape[-1]
    grid = input_tensor.device().compute_with_storage_grid_size()
    (_, _, core_group_1, core_group_2, rows_g1, rows_g2) = ttnn.split_work_to_cores(
        grid, tensor_tile_rows, row_wise=True
    )
    work = []
    start = 0
    for group, rows in ((core_group_1, rows_g1), (core_group_2, rows_g2)):
        if rows == 0:
            continue
        for core in _cores_of(group):
            work.append(CoreWork(core, start, rows, 0, Wt, 0, core, 0, W))
            start += rows
    assert start == tensor_tile_rows
    return work, rows_g1


def _group_work(cores, group_size, slice_elems, W, row_ranges):
    """Assign (row range, W-slice of slice_elems elements) to cores laid out as consecutive (y, x)-ordered
    row-groups of group_size — exactly how ttnn.Mcast partitions its receiver set. row_ranges holds
    (start_tile_row, tile_rows, sticks) per row-group."""
    work = []
    for gi, (start, rows, sticks) in enumerate(row_ranges):
        group = cores[gi * group_size : (gi + 1) * group_size]
        for j, core in enumerate(group):
            e0 = j * slice_elems
            ve = min(slice_elems, W - e0)
            if ve <= 0 or rows <= 0:
                raise NotImplementedError("rms_norm: a shard with no valid data is not supported")
            work.append(
                CoreWork(core, start, rows, e0 // TILE_DIM, math.ceil(ve / TILE_DIM), j, group[0], e0, ve, sticks)
            )
    return work


def _sharded_work(input_tensor, tensor_tile_rows):
    """A sharded input (HEIGHT: regime R1-HS, G = 1; WIDTH / BLOCK: regime R3): the shard spec IS the work split. Returns
    (work, S, rows_max, shard_w): S = slice tiles (nominal), rows_max = tile-rows per core (max).
      TILE:      shard = (tile-rows x S tiles); the core's tile-rows and tiles are the tensor's.
      ROW_MAJOR: shard = (sticks x shard_w elements) — neither need be tile-aligned; a core's sticks are
                 grouped into ceil(sticks / 32) local tile-rows, its slice padded to S = ceil(shard_w / 32)."""
    mc = input_tensor.memory_config()
    spec = mc.shard_spec
    if spec is None or mc.buffer_type != ttnn.BufferType.L1:
        raise NotImplementedError("rms_norm: sharded inputs must carry a 2-D L1 shard spec")
    height = mc.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    if spec.orientation != ttnn.ShardOrientation.ROW_MAJOR and not height:
        # HEIGHT cores are independent (no row-group), so any shard -> core order works there.
        raise NotImplementedError("rms_norm: only ROW_MAJOR shard orientation is supported")
    shape = list(input_tensor.shape)
    W = shape[-1]
    is_rm = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    shard_h, shard_w = int(spec.shape[0]), int(spec.shape[1])
    if is_rm:
        unit_h, total_h = 1, _prod(shape[:-1])  # sticks
    else:
        assert shard_h % TILE_DIM == 0 and shard_w % TILE_DIM == 0, "TILE shards are tile-multiples"
        unit_h, total_h = TILE_DIM, tensor_tile_rows  # tile-rows
    Hsh = shard_h // unit_h
    S = math.ceil(shard_w / TILE_DIM)

    def row_range(start, count):
        # (start_tile_row, tile_rows, sticks): ROW_MAJOR rows are local to the shard.
        if is_rm:
            return (0, math.ceil(count / TILE_DIM), count)
        return (start, count, 0)

    if height:
        # Regime R1-HS: every core holds full-W rows (one W-slice, G = 1 -> no cross-core combine); shard i
        # sits on the i-th core of the grid in the buffer's own shard order (corerange_to_cores, orientation).
        # A ragged last shard holds fewer valid rows; trailing grid cores past the last shard hold none.
        if shard_w < W:
            raise NotImplementedError("rms_norm: height shard spec does not cover the row width")
        n = math.ceil(total_h / Hsh)
        cores = ttnn.corerange_to_cores(spec.grid, n, row_wise=spec.orientation == ttnn.ShardOrientation.ROW_MAJOR)
        if len(cores) < n:
            raise NotImplementedError("rms_norm: height shard spec does not cover the tensor")
        row_ranges = [row_range(i * Hsh, min(Hsh, total_h - i * Hsh)) for i in range(n)]
        return _group_work(list(cores), 1, shard_w, W, row_ranges), S, max(r[1] for r in row_ranges), shard_w
    cores = _yx_sorted(_cores_of(spec.grid))
    if mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        # One row-group: every core holds the full height and one W-slice, in (y, x) shard order (the
        # slices may span several grid rows — eval/SHARDING.md §2).
        n = math.ceil(W / shard_w)
        if Hsh < total_h or n > len(cores):
            raise NotImplementedError("rms_norm: width shard spec does not cover the tensor")
        rr = row_range(0, total_h)
        return _group_work(cores[:n], n, shard_w, W, [rr]), S, rr[1], shard_w
    # BLOCK_SHARDED, ROW_MAJOR orientation: shard (r, c) on core (x0 + c, y0 + r); row-group = grid row.
    bbox = spec.grid.bounding_box()
    nx = bbox.end.x - bbox.start.x + 1
    ny = bbox.end.y - bbox.start.y + 1
    if len(cores) != nx * ny:
        raise NotImplementedError("rms_norm: BLOCK_SHARDED needs a rectangular shard grid")
    gx = math.ceil(W / shard_w)
    gy = math.ceil(total_h / Hsh)
    if gx > nx or gy > ny:
        raise NotImplementedError("rms_norm: block shard spec does not cover the tensor")
    used = [c for c in cores if c.x - bbox.start.x < gx and c.y - bbox.start.y < gy]
    row_ranges = [row_range(r * Hsh, min(Hsh, total_h - r * Hsh)) for r in range(gy)]
    return _group_work(used, gx, shard_w, W, row_ranges), S, max(r[1] for r in row_ranges), shard_w


def _r3_interleaved_layout(Wt, tensor_tile_rows, num_cores):
    """(S, G_w, G_h): W-slices of S tiles (G_w = ceil(Wt / S) cores per row-group), G_h row-groups."""
    groups_target = math.ceil(tensor_tile_rows / R3_ROWS_PER_GROUP)
    G_w_target = max(1, min(R3_MAX_GROUP_CORES, num_cores // groups_target))
    S = max(R3_MIN_SLICE_TILES, math.ceil(Wt / G_w_target))
    S = min(S, Wt)
    G_w = math.ceil(Wt / S)
    G_h = max(1, min(tensor_tile_rows, num_cores // G_w))
    return S, G_w, G_h


def _r3_interleaved_work(input_tensor, Wt, tensor_tile_rows, S, G_w, G_h):
    grid = input_tensor.device().compute_with_storage_grid_size()
    cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)][: G_w * G_h]
    base, extra = divmod(tensor_tile_rows, G_h)
    row_ranges = []
    start = 0
    for gi in range(G_h):
        rows = base + (1 if gi < extra else 0)
        row_ranges.append((start, rows, 0))
        start += rows
    W = input_tensor.shape[-1]
    return _group_work(cores, G_w, S * TILE_DIM, W, row_ranges), base + (1 if extra else 0)


def select_plan(input_tensor, gamma, fp32_acc: bool) -> Plan:
    """Regime-selection function (op_design.md -> Regimes):
    HEIGHT_SHARDED -> R1-HS (the sharded path with G = 1: local reduction, no combine).
    WIDTH/BLOCK_SHARDED -> R3 sharded. INTERLEAVED -> R1 unless R1 refuses (row too wide) or the W-split puts
    >= R3_OCCUPANCY_GAIN x the cores to work on a few-tile-row shape (TILE only; RM stays R1)."""
    device = input_tensor.device()
    budget = l1_budget(device)
    W = input_tensor.shape[-1]
    Wt = math.ceil(W / TILE_DIM)
    tensor_tile_rows = tensor_tile_rows_of(input_tensor)
    memory_layout = input_tensor.memory_config().memory_layout
    is_rm = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    def r3_plan(work, S, rows_max, sharded):
        G = max(w.index for w in work) + 1
        tile_shard = sharded and not is_rm
        masked = tile_shard and (W % TILE_DIM != 0)
        owners = 1
        if G > 1 and R3_SCATTER_ROWS_PER_OWNER > 0:
            owners = max(1, min(G, rows_max // R3_SCATTER_ROWS_PER_OWNER))
        topo = Topology(
            sharded=tile_shard, rm_sharded=sharded and is_rm, combine=G > 1, G=G, owners=owners, masked=masked
        )
        choice = choose_blocking(input_tensor, gamma, rows_max, budget, fp32_acc, S, topo, refuse=sharded)
        if choice is None:
            return None
        g, block_rows, x_depth = choice
        # Owners split each BLOCK's rows: when L1 (or the interleaved prefetch cap) cut the rows into blocks,
        # owners beyond block_rows // R3_SCATTER_ROWS_PER_OWNER would never own a row. Fewer owners make the
        # gather quantum larger, so keep the trim only if the footprint still fits.
        if topo.owners > 1:
            owners = max(1, min(topo.owners, block_rows // R3_SCATTER_ROWS_PER_OWNER))
            if owners != topo.owners:
                trimmed = dataclasses.replace(g, topo=dataclasses.replace(topo, owners=owners))
                if footprint(trimmed, block_rows, x_depth) <= budget:
                    g, topo = trimmed, trimmed.topo
        regime = "R1-HS" if memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED else "R3"
        return Plan(regime, topo, S, rows_max, tuple(work), g, block_rows, x_depth)

    if memory_layout in (
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ):
        work, S, rows_max, _ = _sharded_work(input_tensor, tensor_tile_rows)
        if is_rm and gamma is not None and gamma.layout == ttnn.TILE_LAYOUT and any(w.e0 % TILE_DIM for w in work):
            # A ROW_MAJOR shard slice that starts mid-tile would need γ re-tiled across tile boundaries.
            raise NotImplementedError("rms_norm: TILE γ with a ROW_MAJOR shard that is not tile-aligned in W")
        return r3_plan(work, S, rows_max, sharded=True)

    if memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise NotImplementedError(f"rms_norm: memory_layout {memory_layout} is not built")

    r1_work, rows_g1 = _r1_work(input_tensor, Wt, tensor_tile_rows)
    r1 = choose_blocking(input_tensor, gamma, rows_g1, budget, fp32_acc, refuse=False)

    r3 = None
    if not is_rm:
        grid = device.compute_with_storage_grid_size()
        num_cores = grid.x * grid.y
        S, G_w, G_h = _r3_interleaved_layout(Wt, tensor_tile_rows, num_cores)
        r1_cores = len(r1_work)
        wants_r3 = r1 is None or (
            R3_OCCUPANCY_GAIN > 0 and G_w >= R3_MIN_GROUP_CORES and G_w * G_h >= R3_OCCUPANCY_GAIN * r1_cores
        )
        if wants_r3 and G_w > 1:
            work, rows_max = _r3_interleaved_work(input_tensor, Wt, tensor_tile_rows, S, G_w, G_h)
            r3 = r3_plan(work, S, rows_max, sharded=False)
    if r3 is not None:
        return r3
    if r1 is None:
        g = compute_geometry(input_tensor, gamma, 1, fp32_acc)
        raise NotImplementedError(
            f"rms_norm: W={g.W} does not fit a resident tile-row in L1 (needs {footprint(g, 1, 1)} B, budget "
            f"{budget} B) and the W-split regime R3 is not built for this layout"
        )
    g, block_rows, x_depth = r1
    return Plan("R1", R1_TOPOLOGY, Wt, rows_g1, tuple(r1_work), g, block_rows, x_depth)


# ---------------------------------------------------------------------------
# Program descriptor
# ---------------------------------------------------------------------------


# Reader runtime args (per core) ahead of the optional packed row-group member coordinates — the kernel's
# get_arg_val indices.
READER_RT_BASE_ARGS = (
    "input_addr",
    "start_tile_row",
    "rows",
    "bank_stagger",
    "slice_w0",
    "slice_valid",
    "is_root",
    "group_index",
    "root_x",
    "root_y",
    "slice_e0",
    "slice_elems",
    "stick_count",
)


# Measurement-only kernel defines (perf investigations; never set in production — unset env => no defines, so the
# production JIT hash and binaries are unchanged):
#   RMS_NORM_PERF_ZONES=1      -> KERNEL_PERF_ZONES: turns on the permanent MaybeDeviceZoneScope stage zones
#                                 (kernel_lib/perf_instrumentation.hpp; they also need the device profiler).
#   RMS_NORM_ABLATE=<a,b,...>  -> payload ablation switches that keep every CB / semaphore / barrier handshake:
#     reads   reader: no x NoC reads (interleaved TILE / RM)          -> RMS_ABLATE_READS
#     writes  writer: no output NoC writes                            -> RMS_ABLATE_WRITES
#     gamma   writer: no γ DRAM reads                                 -> RMS_ABLATE_GAMMA
#     combine reader: no R3 partial / inv_rms data writes (semaphores kept) -> RMS_ABLATE_COMBINE
#     compute compute: every eltwise_chain emits only its CB sync     -> CKL_ELTWISE_CHAIN_SKIP_COMPUTE
_ABLATE_DEFINES = {
    "reads": "RMS_ABLATE_READS",
    "writes": "RMS_ABLATE_WRITES",
    "gamma": "RMS_ABLATE_GAMMA",
    "combine": "RMS_ABLATE_COMBINE",
    "scatter": "RMS_ABLATE_SCATTER",
    "allgather": "RMS_ABLATE_ALLGATHER",
}


def _measurement_defines(compute: bool = False) -> list:
    defines = []
    if os.environ.get("RMS_NORM_PERF_ZONES", "0") not in ("", "0"):
        defines.append(("KERNEL_PERF_ZONES", "1"))
    ablate = {a.strip() for a in os.environ.get("RMS_NORM_ABLATE", "").split(",") if a.strip()}
    unknown = ablate - set(_ABLATE_DEFINES) - {"compute"}
    if unknown:
        raise ValueError(f"RMS_NORM_ABLATE: unknown switches {sorted(unknown)}")
    if compute:
        if "compute" in ablate:
            defines.append(("CKL_ELTWISE_CHAIN_SKIP_COMPUTE", "1"))
    else:
        defines += [(d, "1") for a, d in _ABLATE_DEFINES.items() if a in ablate]
    return defines


def _cb(index, data_format, page_size, num_pages, core_ranges):
    return ttnn.CBDescriptor(
        total_size=page_size * num_pages,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    gamma=None,
    epsilon: float,
    compute_kernel_config: ttnn.ComputeConfigDescriptor,
) -> ttnn.ProgramDescriptor:
    device = input_tensor.device()
    fp32_acc = bool(compute_kernel_config.fp32_dest_acc_en)
    plan = select_plan(input_tensor, gamma, fp32_acc)
    g, block_rows, x_depth, topo = plan.geometry, plan.block_rows, plan.x_depth, plan.topo
    all_cores = plan.core_ranges
    budget = l1_budget(device)
    gamma_resident = choose_gamma_resident(g, block_rows, x_depth, budget)

    # ---------- Circular buffers ----------
    C, R = g.C, block_rows
    cbs = [
        _cb(CB_REDUCE_SCALER, SCALER_FORMAT, ttnn.tile_size(SCALER_FORMAT), scaler_pages(g), all_cores),
        _cb(CB_SUMSQ_PARTIAL, g.interm_format, g.T_interm, R, all_cores),
        _cb(CB_INV_RMS, g.interm_format, g.T_interm, R, all_cores),
    ]
    if topo.sharded:
        # Zero-copy: compute reads x straight out of the input shard and packs into the output shard.
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_X_TILES, input_tensor))
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT, output_tensor))
    else:
        cbs.append(_cb(CB_X_TILES, input_tensor.dtype, g.T_in, x_depth * R * g.WS, all_cores))
        cbs.append(_cb(CB_OUTPUT, output_tensor.dtype, g.T_out, STREAM_DEPTH * C, all_cores))
    if g.is_rm:
        cbs.append(_cb(CB_X_RM_STICKS, input_tensor.dtype, g.T_in, STREAM_DEPTH * C, all_cores))
        cbs.append(_cb(CB_Y_TILES, output_tensor.dtype, g.T_out, R * C, all_cores))
    if g.has_gamma and gamma_row_pages(g, gamma_resident) > 0:
        cbs.append(_cb(CB_GAMMA_ROW, gamma.dtype, g.gamma_row_page, gamma_row_pages(g, gamma_resident), all_cores))
    if g.has_gamma:
        cbs.append(_cb(CB_GAMMA_TILES, gamma.dtype, g.T_g, gamma_tiles_pages(g, gamma_resident), all_cores))
        cbs.append(_cb(CB_X_GAMMA, g.interm_format, g.T_interm, R * C, all_cores))
    if topo.combine:
        cbs.append(_cb(CB_PARTIAL_COL, g.interm_format, g.T_interm, R, all_cores))
        # Allocated identically on every core: a contributor addresses the root's gather slots through its
        # own (never advanced) write pointer.
        cbs.append(_cb(CB_GATHER, g.interm_format, g.T_interm, gather_pages(topo, R), all_cores))
        cbs.append(_cb(CB_RMS_ROOT, g.interm_format, g.T_interm, scatter_rows(topo, R), all_cores))
    if topo.masked:
        cbs.append(_cb(CB_SUMSQ_LAST, g.interm_format, g.T_interm, R, all_cores))
        cbs.append(_cb(CB_SUMSQ_ACC, g.interm_format, g.T_interm, R, all_cores))

    read_batch_chunks = g.num_w_chunks if READ_BATCH_CHUNKS is None else min(READ_BATCH_CHUNKS, g.num_w_chunks)
    # ROW_MAJOR shard: bytes per shard stick (the input and output shards share the spec).
    shard_row_bytes = int(input_tensor.memory_config().shard_spec.shape[1]) * g.es if topo.rm_sharded else 0

    # ---------- Compile-time args ----------
    # Shared geometry prefix (same order in all three kernels; SHARED_CT_ARGS entries).
    common_ct = [
        int(g.is_rm),
        int(g.has_gamma),
        block_rows,
        C,
        g.num_w_chunks,
        g.Wt,
        g.W,
        g.num_sticks,
        x_depth,
        read_batch_chunks,
        int(g.gamma_is_tile),
        int(gamma_resident),
        g.gamma_row_page,
        gamma_direct(g, gamma_resident),
        int(topo.sharded),
        int(topo.combine),
        g.WS,
        topo.G,
        int(topo.masked),
        SEM_GATHER,
        int(topo.rm_sharded),
        shard_row_bytes,
    ]
    reader_ct = list(common_ct)
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    # Reader-only args after the accessor: combine owners, the all-gather semaphore, and where the packed
    # row-group member coordinates start in the runtime args (owners > 1 only).
    reader_ct.extend([topo.owners, SEM_INV, len(READER_RT_BASE_ARGS)])
    # γ is supplied by the writer (its RISC-V is idle until the first output chunk) — lamp L4.
    writer_ct = list(common_ct)
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_ct.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if gamma is not None
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
    # Compute-only args follow the shared prefix.
    compute_ct = list(common_ct) + [SCALE_BLOCK_TILES, int(FINALIZE_COL0), topo.owners]

    # ---------- Runtime args ----------
    inv_w_bits = _fp32_bits(1.0 / g.W)
    eps_bits = _fp32_bits(epsilon)
    in_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    gamma_addr = gamma.buffer_address() if gamma is not None else 0

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    # Reduce-scatter (owners > 1): every member addresses every other member of its row-group (partials to the
    # owners, finalized rows to all). Virtual coordinates packed (x << 16) | y, in group-index order.
    group_members = {}
    if topo.owners > 1:
        for w in plan.work:
            group_members.setdefault((w.root.x, w.root.y), {})[w.index] = device.worker_core_from_logical_core(w.core)
    for core_ordinal, w in enumerate(plan.work):
        core = w.core
        # Kernels reduce the stagger mod NUM_DRAM_BANKS (the arch's bank count is a kernel define).
        stagger = core_ordinal if BANK_STAGGER else 0
        root_virtual = device.worker_core_from_logical_core(w.root)
        # The core owning tile Wt-1 of a non-aligned row masks that tile's pad columns (sharded only: the
        # interleaved reader zeroes them in its own CB instead).
        masked_core = int(topo.masked and w.w0 + w.V == g.Wt)
        reader_rt[core.x][core.y] = [
            in_addr,
            w.start_tile_row,
            w.rows,
            stagger,
            w.w0,
            w.V,
            int(w.index == 0),
            w.index,
            root_virtual.x,
            root_virtual.y,
            w.e0,
            w.ve,
            w.sticks,
        ]
        assert len(reader_rt[core.x][core.y]) == len(READER_RT_BASE_ARGS)
        if topo.owners > 1:
            members = group_members[(w.root.x, w.root.y)]
            reader_rt[core.x][core.y] = list(reader_rt[core.x][core.y]) + [
                (members[i].x << 16) | members[i].y for i in range(topo.G)
            ]
            if COMBINE_FACES == 6:
                # all-gather multicast rectangles: runs of consecutive virtual x on one virtual y
                me = device.worker_core_from_logical_core(core)
                pts = sorted(((members[i].y, members[i].x) for i in range(topo.G)))
                rects = []
                for y, x in pts:
                    if rects and rects[-1][0] == y and rects[-1][2] == x - 1:
                        rects[-1][2] = x
                    else:
                        rects.append([y, x, x])
                words = [len(rects)]
                for y, x0, x1 in rects:
                    has_self = int(me.y == y and x0 <= me.x <= x1)
                    words += [(x0 << 16) | y, (x1 << 16) | y, (x1 - x0 + 1) | (has_self << 16)]
                reader_rt[core.x][core.y] = list(reader_rt[core.x][core.y]) + words
        writer_rt[core.x][core.y] = [
            out_addr,
            gamma_addr,
            w.start_tile_row,
            w.rows,
            stagger,
            w.w0,
            w.V,
            w.e0,
            w.ve,
            w.sticks,
        ]
        compute_rt[core.x][core.y] = [w.rows, inv_w_bits, eps_bits, w.V, w.index, masked_core]

    dm_defines = [("RMS_BANK_STAGGER", "1" if BANK_STAGGER else "0")] + _measurement_defines()
    dm_defines += [
        ("RMS_COMBINE_FACES", str(int(COMBINE_FACES))),
        ("RMS_POISON_FACES13", str(int(POISON_FACES13))),
        ("RMS_END_ATOMIC_BARRIER", str(int(END_ATOMIC_BARRIER))),
    ]
    compute_defines = _measurement_defines(compute=True)
    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct,
        defines=dm_defines,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        defines=dm_defines,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct,
        defines=compute_defines,
        runtime_args=compute_rt,
        config=compute_kernel_config,
    )

    descriptor = ttnn.ProgramDescriptor()
    descriptor.cbs = cbs
    # Row-groups are consecutive (y, x)-ordered runs of G cores, root = index 0 = the fixed sender — the same
    # partition ttnn.Mcast builds from (receivers, group size, RowMajor order).
    if topo.combine and topo.owners > 1:
        # Reduce-scatter: the all-gather of finalized rows is unicast + SEM_INV (no multicast channel; concurrent
        # multicasts from every owner into one row-group measured slower, see the reader's scatter_block).
        descriptor.semaphores = [
            ttnn.SemaphoreDescriptor(id=SEM_GATHER, core_ranges=all_cores, initial_value=0),
            ttnn.SemaphoreDescriptor(id=SEM_INV, core_ranges=all_cores, initial_value=0),
        ]
        ttnn.attach_absent_mcast(reader, MCAST_PREFIX)
    elif topo.combine:
        descriptor.semaphores = [ttnn.SemaphoreDescriptor(id=SEM_GATHER, core_ranges=all_cores, initial_value=0)]
        mcast = ttnn.Mcast(
            device,
            ttnn.McastConfig(),
            all_cores,
            topo.G,
            ttnn.McastFixedSenderConfig(sender_index=0),
            ttnn.McastCoreOrder.RowMajor,
        )
        mcast.attach(descriptor, MCAST_PREFIX, [reader], SEM_MCAST_BASE)
    else:
        ttnn.attach_absent_mcast(reader, MCAST_PREFIX)
    descriptor.kernels = [reader, writer, compute]
    return descriptor
