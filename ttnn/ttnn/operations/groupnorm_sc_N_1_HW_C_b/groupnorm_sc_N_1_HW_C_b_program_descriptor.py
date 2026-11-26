# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Program descriptor for groupnorm_sc_N_1_HW_C_b (op_design.md, l1_ledger.md).

Host-side planning:
  * _choose_gx(): column-slice count per image (span-aligned, G_core <= 32, L1-feasible),
    minimizing the max per-core tile count.
  * work split: units u = n*gx + xs; U < P -> cpi = min(P // U, Ht) cores per unit with a
    balanced row split and a gather/mcast combine on slot 0; U >= P -> cpi = 1 and whole-unit
    blocks round-robined contiguously over the P cores.
  * residency: K_alloc = min(B_max, K_max) resident chunks (or the TUNING_OVERRIDES value,
    clamped to [0, K_max]); the input ring adds the streaming depth iff B_max > K_alloc.

L1 BLOCK_SHARDED input (Refinement 1, even shards): a core's shard is its block. The unit is one image's
grid column (grid columns split C, grid rows split N*HW, cpi = gy / N shards per image); the gather goes up
the column to its root (shard row 0 of the image) and the stats mcast comes back down in ColumnMajor groups.
TILE: the input ring is the shard (two views, K = B), the output tiles are the output shard. RM: k = lcm(sw,
32) / sw consecutive sticks form one virtual row of W = lcm(sw, 32) / 32 tiles (lane v holds channel v mod sw),
compute tilizes chunks from the shard into the resident ring and untilizes the result into the output shard.

in_place (Refinement 2): the output shard is the input shard - the output CBs are borrowed from the same buffer, so
no second shard is allocated and the L1 it would take returns to the static CBs. Write ordering keeps it safe:
TILE pass 2 packs tile i over tile i in the chunk it just read (unpack precedes pack per tile); RM pass 2 untilizes a
chunk only after it was tilized (resident tiles first; a non-resident head chunk is re-tilized from the shard right
before its rows are overwritten); the members' pilot reads of the root shard complete before their gather, so before
the root's pass 2.

Every block knob (gx, chunk_ht, K, buffer depths, max_cores) is defined once here and flows into
CB sizes and kernel args.

Formats (Refinement 4): every page / element size comes from one Formats record, derived from the tensors' dtypes
(ttnn.tile_size): the input ring, RM sticks and output CBs follow the input dtype (bf16 or Float32; output dtype =
input dtype), the reader-built gamma / beta maps follow the affine dtype, gamma / beta may be ROW_MAJOR or TILE. The
statistics, partials and channel vectors stay Float32 (fp32 DEST); the group map, pilot rows and -1 tile stay bf16
(exact 0 / 1 / bf16 values).
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# ---------------------------------------------------------------------------
# Tunable constants (single source of truth; TUNING_OVERRIDES may override some)
# ---------------------------------------------------------------------------
CHUNK_TILES_TARGET = 32  # streaming quantum target (tiles per chunk)
D_IN = 2  # TILE input stream depth (chunks)
D_RM = 2  # RM stick CB depth (chunks)
# DEST window of the compute's two-op per-chunk chains (fp32 DEST, half sync: the device DEST_AUTO_LIMIT).
# Interleaved output: the pass-2 output is split across both DM RISCs (perf_experiments/dram_dynamic_balance). The
# writer's share CB holds 1 chunk (2 when L1 has a chunk to spare after the resident ring) and the reader's 1 chunk:
# D_OUT = 2 chunks at minimum, the op's former output ring. Compute packs each chunk into whichever share has a free
# slot (the writer's first) and names it on that RISC's sequence CB (OUT_SEQ_PAGES uint32 pages).
D_OUT = 2  # output chunks at minimum (writer's share 1 + reader's share 1)
OUT_READER_CHUNKS = 1
OUT_SEQ_PAGES = 8  # per sequence CB; at most 2 + the writer's depth outstanding
OUT_SEQ_PAGE_BYTES = 16
MAX_GROUP_LANES = 32  # group lanes in one tile row
# Per-chunk device zones (perf_instrumentation.hpp MaybeDeviceZoneScopeIf) split each chunk's wait from its work.
# They are opt-in (TUNING_OVERRIDES["chunk_zones"]): in a profiled build they cost about 0.9 us on a 25 us sharded
# case on top of the always-on block-level zones (about 1.3 us), so routine device-time runs leave them off. Even
# when requested they are compiled in only when a core's chunk steps (pass 1 + pass-2 re-reads) keep every RISC
# inside the ~125 zone-execution marker budget: at most 4 per-chunk zones per step, plus about a dozen block zones.
# 16 leaves margin: a 22-step DRAM block (vae_dram_262144x256) crashed the Tracy capture with chunk zones on.
CHUNK_ZONE_MAX_STEPS = 16

BF16_TILE_BYTES = 2048  # the bf16 CBs whatever the tensors' dtypes: group map, pilot rows, -1 tile
F32_TILE_BYTES = 4096  # the Float32 statistic CBs
FACE_ROW0_READ_BYTES = 64  # a TILE pilot / gamma read: row 0 (+ row 1 for bf16) of one face, one 64 B DRAM read
GATHER_SEMAPHORE_ID = 0
MCAST_FIRST_SEMAPHORE_ID = 1

# ---------------------------------------------------------------------------
# CB indices (semantic names; see op_design.md "Circular Buffers")
# ---------------------------------------------------------------------------
CB_INPUT_RM = 0
CB_OUTPUT_RM = 1  # sharded RM: untilized output, backed on the output shard
CB_INPUT_TILES = 2  # input ring, pass-1 view (alias group A)
CB_INPUT_KEEP = 3  # input ring, pass-2 view (alias group A)
CB_OUTPUT_TILES = 5
CB_NEG_ONE = 6  # constant -1 (bf16, compute-generated once): in0 of the root finalize's -k and sender-sum matmuls
CB_SUM_ACC = 7
CB_SUMSQ_ACC = 8
CB_CHAN_SCALE = 9
CB_CHAN_WORK = 10
CB_GROUP_MAP = 12
CB_GAMMA_ROWS = 13  # expand in1: gamma map (W tiles, gamma's dtype), + beta map (W) when beta; reader-built
CB_LOCAL_PARTIALS = 15
CB_GATHERED_PARTIALS = 16
CB_STATS_OUT = 19
CB_GROUP_STATS = 20
CB_AFFINE_STAGING = 21  # reader-only staging: pilot row (step 0) / affine slices (step B), disjoint in time
CB_PILOT_ROWS = 22  # -k per channel (row 0, bf16), one per block, held for pass 1
CB_GROUP_SHIFT = 24  # unit root: k per group lane in row 0 (Float32, rows 1..31 zero), read by the finalize
CB_RM_STAGE_IN = 25  # sharded RM (R3): staged partial tile rows (k-filled), reader -> compute tilize (alias group F)
CB_RM_STAGE_OUT = 26  # sharded RM (R3): their normalized rows, compute untilize -> writer copy-back (alias group F)
CB_OUTPUT_B = 27  # interleaved: the reader's share of the pass-2 output (written on the reader's NoC)
CB_OUT_SEQ_A = 28  # interleaved: pass-2 index of each chunk packed into CB_OUTPUT_TILES (-> writer)
CB_OUT_SEQ_B = 29  # interleaved: pass-2 index of each chunk packed into CB_OUTPUT_B (-> reader)

NUM_CB_SLOTS = 64  # unpack_to_dest_mode vector length (covers WH 32 / BH 64)

# Sharded op words (Refinement 3): one per piece of a core's shard, in shard row order. Bits 0..15 = tile rows.
OP_STAGED = 1 << 16  # RM: one tile row staged by the reader (partial / image-mixed), tilized from CB_RM_STAGE_IN
OP_SEG_END = 1 << 17  # the last ring piece of an image segment (pass 1 collapses, pass 2 drops its a / b')
OP_SKIP = 1 << 18  # RM: the shard FIFOs step over a staged tile-row position (W pages, no data)
PILOT_SLOT_BYTES = 16  # sharded TILE pilot: one 16 B NoC read per group lane (the lane's k sample)


def _ceil_div(a, b):
    return -(-a // b)


def _align_up(a, b):
    return _ceil_div(a, b) * b


def _lcm(a, b):
    return a * b // math.gcd(a, b)


def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def _f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


_ELEM_BYTES = {ttnn.bfloat16: 2, ttnn.float32: 4}


@dataclass(frozen=True)
class Formats:
    """Page formats of the op's tensors: the single source of every dtype-dependent size (Refinement 4)."""

    in_dtype: object = ttnn.bfloat16  # input = output dtype: input ring, RM sticks, output CBs
    aff_dtype: object = ttnn.bfloat16  # gamma / beta: the reader-built affine maps (bf16 when there is no affine)
    aff_tile: bool = False  # gamma / beta in TILE layout (row 0 of each tile carries the values)

    @property
    def in_tile(self):
        return ttnn.tile_size(self.in_dtype)

    @property
    def in_elem(self):
        return _ELEM_BYTES[self.in_dtype]

    @property
    def amap_tile(self):
        return ttnn.tile_size(self.aff_dtype)

    @property
    def aff_elem(self):
        return _ELEM_BYTES[self.aff_dtype]


def formats_of(input_tensor, gamma=None, beta=None):
    ref = gamma if gamma is not None else beta
    return Formats(
        in_dtype=input_tensor.dtype,
        aff_dtype=ttnn.bfloat16 if ref is None else ref.dtype,
        aff_tile=ref is not None and ref.layout == ttnn.TILE_LAYOUT,
    )


def l1_cb_budget(device):
    """Bytes available to this program's statically allocated CBs on every worker core.

    CBs are placed from the L1 allocator base upward, so the region is [allocator base, lowest live L1
    buffer). ``ttnn.get_max_worker_l1_unreserved_size()`` is NOT this: it is measured from the
    kernel-config base, so it also counts the kernel-config ring (70688 B on BH) that sits below the
    allocator base, and the CBs it admits overflow L1 by exactly that much. The memory view's per-bank
    size is measured from the allocator base, and its block table (offsets relative to that base)
    gives the lowest live L1 buffer, so caller-owned L1 tensors (e.g. an R1 shard) are excluded too.
    """
    view = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    budget = int(view.total_bytes_per_bank)
    for blk in view.block_table:
        if blk.get("allocated") == "yes":
            budget = min(budget, int(blk["address"]))
    return budget


def _tuning():
    from .groupnorm_sc_N_1_HW_C_b import TUNING_OVERRIDES

    return dict(TUNING_OVERRIDES)


def input_ring_pages(*, T, K_alloc, stream, is_rm):
    """Input ring capacity: the resident tail plus the streaming depth (TILE: D_IN, RM: 1 tilized chunk)."""
    return K_alloc * T + int(stream) * (1 if is_rm else D_IN) * T


def output_ring_bytes(T, fmt):
    """The minimum interleaved output CBs: the writer's and the reader's shares (D_OUT chunks, output dtype) + the
    two sequence CBs. The writer's optional second chunk is added by plan_or_raise when it fits."""
    return D_OUT * T * fmt.in_tile + 2 * OUT_SEQ_PAGES * OUT_SEQ_PAGE_BYTES


def affine_stage_bytes(*, W, map_period, fmt):
    """Reader staging of one of gamma / beta: RM, the slice's P channels from a 64 B-aligned start; TILE, one 64 B
    row-0 read of faces 0 / 1 per channel tile (the kernel's AFFINE_STAGE_BYTES)."""
    if fmt.aff_tile:
        return W * 2 * FACE_ROW0_READ_BYTES
    return _align_up(map_period * fmt.aff_elem, 64) + 64


def staging_bytes(*, W, map_period, is_rm, n_affine, fmt, sharded=False, pilot_cols=1):
    """Reader staging: the pilot (interleaved TILE: a 64 B row-0 read of faces 0/1 per tile; interleaved RM: the
    slice of stick 0; sharded TILE: one 16 B slot per group lane; sharded RM: one 16 B-aligned span of stick 0 per
    source column (pilot_cols of the unit), then the k stick built over it) or the n_affine gamma/beta slices,
    whichever is larger. Disjoint in time."""
    stick = map_period * fmt.in_elem
    if sharded and is_rm:
        pilot = max(stick, pilot_cols * (_align_up(stick, 16) + 16))
    elif sharded:
        pilot = MAX_GROUP_LANES * PILOT_SLOT_BYTES
    else:
        pilot = stick if is_rm else W * 2 * FACE_ROW0_READ_BYTES
    affine = n_affine * affine_stage_bytes(W=W, map_period=map_period, fmt=fmt)
    return _align_up(max(pilot, affine), 64)


def footprint_bytes(
    *,
    W,
    T,
    R,
    K_alloc,
    stream,
    is_rm,
    has_gamma,
    has_beta,
    fmt,
    map_period=None,
    sharded=False,
    ring_pages=None,
    stage_pages=0,
    NU=1,
    NSEG=1,
    pilot_cols=1,
):
    """Per-core static L1 footprint of every CB (l1_ledger.md "Per-core footprint"). Sharded: the input
    sticks (RM) / the input ring (TILE) and the output (RM sticks / TILE tiles) are the shards, not static CBs.
    ring_pages overrides the RM ring (sharded fully resident: every tile row of the shard's pieces); stage_pages
    are the RM staged tile rows (+ the k-fill template); NU images per statistic unit, NSEG segments per core.
    Page sizes follow fmt: input-dtype pages (RM sticks, ring, staged rows, output), affine-dtype maps, bf16 map /
    pilot / -1 tile, Float32 statistics."""
    hg, hb = int(has_gamma), int(has_beta)
    map_period = W * 32 if map_period is None else map_period
    tile_shard = sharded and not is_rm
    if tile_shard:
        ring = 0
    elif ring_pages is not None:
        ring = ring_pages
    else:
        ring = input_ring_pages(T=T, K_alloc=K_alloc, stream=stream, is_rm=is_rm)
    in_pages = (D_RM * T if is_rm and not sharded else 0) + ring + stage_pages
    bf16_pages = 1 + 2 * W  # the constant -1 tile (root finalize in0), map, pilot rows
    amap_pages = W * (hg + hb)  # gamma map, beta map
    # Sharded: TILE packs into the output shard, RM pack-untilizes into it (no output tile ring).
    out_ring = 0 if sharded else output_ring_bytes(T, fmt)
    # 2W: alias groups E (sum acc / chan scale) and C (sumsq acc / chan work).
    # Per image of the unit: gathered partials (2R), stats out (2), group stats (2) + e (1). Per segment of the
    # core: local partials (2). Group shift (1). NU = NSEG = 1: 2W + 2R + 8.
    f32_bytes = out_ring + F32_TILE_BYTES * (2 * W + 2 * R * NU + 2 * NSEG + 4 * NU + 2)
    staging = staging_bytes(
        W=W, map_period=map_period, is_rm=is_rm, n_affine=hg + hb, fmt=fmt, sharded=sharded, pilot_cols=pilot_cols
    )
    return fmt.in_tile * in_pages + BF16_TILE_BYTES * bf16_pages + fmt.amap_tile * amap_pages + f32_bytes + staging


@dataclass
class CoreWork:
    core: object  # ttnn.CoreCoord
    unit_start: int
    num_blocks: int
    row_start: int
    rows_core: int
    is_root: int
    slot: int
    root: object  # ttnn.CoreCoord (logical)
    # Sharded slice geometry (Refinement 3). Interleaved: derived per block in the kernels (these stay unused).
    xs: int = 0  # first channel of the core's slice
    vc: int = 0  # valid channels of the slice (< the period on a padded last core column)
    l0: int = 0  # group lane of the slice's first channel (lanes count the unit's groups)
    off0: int = 0  # offset of the slice's first channel inside its group
    g_unit: int = 0  # group lanes of the unit
    n_img: int = 1  # images of the unit (statistic sets the root finalizes)
    n_seg: int = 1  # image segments of the core's rows
    seg_img0: int = 0  # unit image index of the core's first segment
    ops: list = field(default_factory=list)  # op words (OP_*), shard row order
    staged: list = field(default_factory=list)  # RM: (shard stick, position in its tile row, sticks) per staged row
    rm_credit_pages: int = 0  # RM: shard FIFO pages stepped through (full tile rows + skipped positions)
    pilot_cols: list = field(default_factory=list)  # logical x of the unit's columns (pilot k sources)
    w_valid: int = 0  # TILE: channel tiles holding data (the rest of the slice is padding)


@dataclass
class GroupNormPlan:
    N: int
    HW: int
    C: int
    G: int
    Ht: int
    Wt: int
    Cg: int
    span_tiles: int
    gps: int
    S: int
    gx: int
    W: int
    G_core: int
    U: int
    P: int
    cpi: int
    h: int
    T: int
    R: int
    rows_core_max: int
    B_max: int
    K_max: int
    K_alloc: int
    stream: bool
    is_rm: bool
    has_gamma: bool
    has_beta: bool
    grid_x: int
    grid_y: int
    l1_budget: int
    footprint: int
    work: list = field(default_factory=list)
    out_writer_chunks: int = 1  # interleaved: chunk depth of the writer's output share (CB_OUTPUT_TILES)
    sharded: bool = False
    in_place: bool = False  # sharded only: the output is the input shard
    map_period: int = 0  # channels per slice (lane v holds channel v mod map_period)
    shard_grid: object = None  # sharded: the shard grid (ttnn.CoreRangeSet)
    # Refinement 3 (sharded): statistic units = column cluster x image band.
    NU: int = 1  # images per unit (max)
    NSEG: int = 1  # image segments per core (max)
    ops_path: bool = False  # sharded, every core fully resident: pieces walked from RT op words
    stage_pages: int = 0  # RM staged tile rows + k-fill template (alias group F)
    col_mask: bool = False  # RM padded last column: SFPU lane mask after the column collapse
    mcast_row_major: bool = False  # unit = whole grid width x band rows (groups straddle core columns)
    ring_pages: int = 0  # RM ops path: the tilized ring holds every piece's tile rows
    pilot_cols: int = 1  # sharded: core columns of a statistic unit (the RM pilot's span reads, one per column)
    fmt: Formats = field(default_factory=Formats)  # page formats (Refinement 4)


def _evaluate_gx(gx, *, N, Ht, S, span_tiles, gps, P, chunk_target, is_rm, has_gamma, has_beta, fmt, l1_budget):
    W = (S // gx) * span_tiles
    G_core = (S // gx) * gps
    if G_core > MAX_GROUP_LANES:
        return None
    U = N * gx
    cpi = min(P // U, Ht) if U < P else 1
    rows_core_max = _ceil_div(Ht, cpi)
    h = max(1, min(chunk_target // W, rows_core_max))
    # Balance the chunks (no small ragged tail) and keep >= 2 per block so pass 1 overlaps its own read.
    n_chunks = max(_ceil_div(rows_core_max, h), min(rows_core_max, 2))
    h = _ceil_div(rows_core_max, n_chunks)
    T = h * W
    R = _ceil_div(cpi, 32)
    f_fixed = footprint_bytes(
        W=W, T=T, R=R, K_alloc=0, stream=True, is_rm=is_rm, has_gamma=has_gamma, has_beta=has_beta, fmt=fmt
    )
    if f_fixed > l1_budget:
        return None
    if U < P:
        max_core_tiles = rows_core_max * W
        cores_used = U * cpi
    else:
        max_core_tiles = _ceil_div(U, P) * Ht * W
        cores_used = P
    return dict(
        gx=gx,
        W=W,
        G_core=G_core,
        U=U,
        cpi=cpi,
        rows_core_max=rows_core_max,
        h=h,
        T=T,
        R=R,
        f_fixed=f_fixed,
        max_core_tiles=max_core_tiles,
        cores_used=cores_used,
    )


def _choose_gx(*, N, Ht, S, span_tiles, gps, P, chunk_target, is_rm, has_gamma, has_beta, fmt, l1_budget):
    best = None
    best_key = None
    for gx in _divisors(S):
        cand = _evaluate_gx(
            gx,
            N=N,
            Ht=Ht,
            S=S,
            span_tiles=span_tiles,
            gps=gps,
            P=P,
            chunk_target=chunk_target,
            is_rm=is_rm,
            has_gamma=has_gamma,
            has_beta=has_beta,
            fmt=fmt,
            l1_budget=l1_budget,
        )
        if cand is None:
            continue
        # Minimize max per-core tiles; ties: more cores, then larger gx (TILE) / smaller gx (RM).
        key = (cand["max_core_tiles"], -cand["cores_used"], gx if is_rm else -gx)
        if best_key is None or key < best_key:
            best, best_key = cand, key
    return best


def _distribute(plan_fields, grid_x):
    """Per-core work list (row-major core order, so Mcast RowMajor groups == units)."""
    U, P, cpi, Ht = plan_fields["U"], plan_fields["P"], plan_fields["cpi"], plan_fields["Ht"]

    def coord(c):
        return ttnn.CoreCoord(c % grid_x, c // grid_x)

    work = []
    if U < P:
        q, rem = divmod(Ht, cpi)
        for c in range(U * cpi):
            unit, slot = divmod(c, cpi)
            row_start = slot * q + min(slot, rem)
            rows = q + (1 if slot < rem else 0)
            work.append(
                CoreWork(
                    core=coord(c),
                    unit_start=unit,
                    num_blocks=1,
                    row_start=row_start,
                    rows_core=rows,
                    is_root=int(slot == 0),
                    slot=slot,
                    root=coord(unit * cpi),
                )
            )
    else:
        base, extra = divmod(U, P)
        for c in range(P):
            n_units = base + (1 if c < extra else 0)
            start = c * base + min(c, extra)
            work.append(
                CoreWork(
                    core=coord(c),
                    unit_start=start,
                    num_blocks=n_units,
                    row_start=0,
                    rows_core=Ht,
                    is_root=1,
                    slot=0,
                    root=coord(c),
                )
            )
    return work


def plan_or_raise(input_tensor, num_groups, *, has_gamma, has_beta, in_place=False, fmt=None):
    """Build the blocking plan; NotImplementedError when no feasible column-slice count exists. fmt: the page
    formats (formats_of(input, gamma, beta)); None = the input's dtype with bf16 ROW_MAJOR gamma / beta."""
    tuning = _tuning()
    if fmt is None:
        fmt = Formats(in_dtype=input_tensor.dtype)
    if input_tensor.is_sharded():
        return _plan_sharded(
            input_tensor, num_groups, has_gamma=has_gamma, has_beta=has_beta, in_place=in_place, tuning=tuning, fmt=fmt
        )
    N, _, HW, C = tuple(input_tensor.shape)
    G = num_groups
    Ht, Wt = HW // 32, C // 32
    Cg = C // G
    L = _lcm(Cg, 32)
    if C % L != 0:
        raise NotImplementedError(
            f"groupnorm_sc_N_1_HW_C_b: lcm(Cg={Cg}, 32)={L} does not divide C={C} "
            "(cross_column_group_combine regime, not built)"
        )
    span_tiles, gps = L // 32, L // Cg
    S = Wt // span_tiles
    is_rm = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    P = grid.x * grid.y
    if tuning.get("max_cores") is not None:
        P = max(1, min(P, int(tuning["max_cores"])))
    chunk_target = CHUNK_TILES_TARGET if tuning.get("chunk_tiles_target") is None else int(tuning["chunk_tiles_target"])
    chunk_target = max(1, chunk_target)
    l1_budget = l1_cb_budget(device)

    best = _choose_gx(
        N=N,
        Ht=Ht,
        S=S,
        span_tiles=span_tiles,
        gps=gps,
        P=P,
        chunk_target=chunk_target,
        is_rm=is_rm,
        has_gamma=has_gamma,
        has_beta=has_beta,
        fmt=fmt,
        l1_budget=l1_budget,
    )
    if best is None:
        raise NotImplementedError(
            f"groupnorm_sc_N_1_HW_C_b: no feasible column-slice count for C={C}, G={G} "
            "(G_core <= 32 and L1 fit); cross_column_group_combine regime not built"
        )

    W, T, R, h = best["W"], best["T"], best["R"], best["h"]
    B_max = _ceil_div(best["rows_core_max"], h)
    chunk_bytes = fmt.in_tile * T  # one chunk of the input ring (and of an output share: output dtype = input)
    K_max = (l1_budget - best["f_fixed"]) // chunk_bytes
    # f_fixed carries the streaming depth; a fully resident ring (K = B) streams nothing, so solve K = B against
    # the stream-free footprint too (flux1 1048576x64: K 17 -> 19 = B, no pass-2 re-reads).
    f_resident = footprint_bytes(
        W=W, T=T, R=R, K_alloc=0, stream=False, is_rm=is_rm, has_gamma=has_gamma, has_beta=has_beta, fmt=fmt
    )
    if (l1_budget - f_resident) // chunk_bytes >= B_max:
        K_max = max(K_max, B_max)
    override = tuning.get("max_resident_chunks")
    if override is None:
        K_alloc = min(B_max, K_max)
    else:
        K_alloc = max(0, min(int(override), K_max, B_max))
    stream = B_max > K_alloc
    footprint = footprint_bytes(
        W=W, T=T, R=R, K_alloc=K_alloc, stream=stream, is_rm=is_rm, has_gamma=has_gamma, has_beta=has_beta, fmt=fmt
    )
    # The writer's output share keeps a double buffer when L1 has a chunk to spare after the resident ring; an
    # L1-bound plan (the ring took every spare chunk) runs 1 + 1, the former 2-chunk ring's bytes.
    out_writer_chunks = 2 if l1_budget - footprint >= chunk_bytes else 1
    footprint += (out_writer_chunks - 1) * chunk_bytes

    fields = dict(U=best["U"], P=P, cpi=best["cpi"], Ht=Ht)
    plan = GroupNormPlan(
        N=N,
        HW=HW,
        C=C,
        G=G,
        Ht=Ht,
        Wt=Wt,
        Cg=Cg,
        span_tiles=span_tiles,
        gps=gps,
        S=S,
        gx=best["gx"],
        W=W,
        G_core=best["G_core"],
        U=best["U"],
        P=P,
        cpi=best["cpi"],
        h=h,
        T=T,
        R=R,
        rows_core_max=best["rows_core_max"],
        B_max=B_max,
        K_max=K_max,
        K_alloc=K_alloc,
        stream=stream,
        is_rm=is_rm,
        has_gamma=has_gamma,
        has_beta=has_beta,
        grid_x=grid.x,
        grid_y=grid.y,
        l1_budget=l1_budget,
        footprint=footprint,
        map_period=W * 32,
        out_writer_chunks=out_writer_chunks,
        fmt=fmt,
    )
    plan.work = _distribute(fields, grid.x)
    return plan


def _rm_core_ops(segs, tr, h):
    """RM op words + staged rows of one core: segs = [(image, first stick, end stick)] of its valid rows.

    A tile row (tr = 32k sticks) wholly inside one segment is read from the shard (runs of up to h rows, OP full);
    any other tile row - the shard's partial tail, or one holding an image boundary - is staged once per segment
    it touches (OP_STAGED: the reader copies the segment's sticks and k-fills the rest). After the last staged
    piece of a position that full rows follow, OP_SKIP steps the shard FIFOs over it."""
    items = []  # (segment index, position, staged triple or None)
    for si, (_, a, b) in enumerate(segs):
        t = a // tr
        while t * tr < b:
            lo, hi = t * tr, t * tr + tr
            if a <= lo and hi <= b:
                items.append((si, t, None))
            else:
                st = max(a, lo)
                items.append((si, t, (st, st - lo, min(b, hi) - st)))
            t += 1
    max_full = max((t for _, t, trip in items if trip is None), default=-1)
    ops, staged = [], []
    run = 0
    for i, (si, t, trip) in enumerate(items):
        nxt = items[i + 1] if i + 1 < len(items) else None
        seg_last = nxt is None or nxt[0] != si
        if trip is None:
            run += 1
            if run == h or seg_last or nxt[2] is not None:
                ops.append(run | (OP_SEG_END if seg_last else 0))
                run = 0
            continue
        ops.append(1 | OP_STAGED | (OP_SEG_END if seg_last else 0))
        staged.append(trip)
        if (nxt is None or nxt[1] != t) and t < max_full:
            ops.append(OP_SKIP)
    return ops, staged


def _plan_sharded(input_tensor, num_groups, *, has_gamma, has_beta, in_place, tuning, fmt):
    """Blocking plan of an L1 BLOCK_SHARDED input: one block per core, the core's own shard (in_place: the output
    shard is that same buffer).

    Refinement 3 (ragged shards): a statistic unit is a column cluster x an image band. Columns cluster when a
    group straddles a core-column boundary (sw % Cg != 0: the whole grid width, group lanes = the unit's groups);
    bands split the grid rows only where a core-row boundary is an image boundary, so a core's rows may hold
    several image segments (one statistic set each; the unit root finalizes one per image of the band). Padded
    edges: only valid rows / channels contribute (TILE: whole tiles; RM: a partial or image-mixed tile row is
    staged k-filled, a padded last column is lane-masked after the column collapse)."""
    mem = input_tensor.memory_config()
    spec = mem.shard_spec
    N, _, HW, C = tuple(input_tensor.shape)
    G = num_groups
    Cg = C // G
    sh, sw = int(spec.shape[0]), int(spec.shape[1])
    bbox = spec.grid.bounding_box()
    x0, y0 = bbox.start.x, bbox.start.y
    gx, gy = bbox.end.x - x0 + 1, bbox.end.y - y0 + 1
    is_rm = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    def refuse(why):
        raise NotImplementedError(f"groupnorm_sc_N_1_HW_C_b: block-sharded input: {why}")

    if mem.buffer_type != ttnn.BufferType.L1:
        refuse("only L1 shards are supported")
    if spec.orientation != ttnn.ShardOrientation.ROW_MAJOR or spec.grid.num_cores() != gx * gy:
        refuse("the shard grid must be one ROW_MAJOR rectangle")
    NHW = N * HW
    vrows = [min(sh, NHW - y * sh) for y in range(gy)]
    vcols = [min(sw, C - x * sw) for x in range(gx)]
    if min(vrows) <= 0 or min(vcols) <= 0:
        refuse(f"shard {sh}x{sw} on a {gx}x{gy} grid leaves a core without data")
    if is_rm:
        L = _lcm(sw, 32)
        k, W = L // sw, L // 32
    else:
        if HW % 32 != 0:
            refuse(f"TILE images of {HW} rows are not whole tile rows")
        k, W = 1, sw // 32
    tr = 32 * k  # shard rows (sticks) per tile row

    # Column clusters: single columns when every core-column boundary is a group boundary, else the whole width.
    split_cols = sw % Cg == 0
    cl_cols = 1 if split_cols else gx
    lanes_max = sw // Cg if split_cols else G
    if lanes_max > MAX_GROUP_LANES:
        refuse(f"{lanes_max} groups per statistic unit exceed one tile row of group lanes")
    # Image bands: core rows split where a core-row boundary is an image boundary; equal heights (mcast groups).
    band_starts = [0] + [y for y in range(1, gy) if (y * sh) % HW == 0]
    bh = band_starts[1] if len(band_starts) > 1 else gy
    if any(b - a != bh for a, b in zip(band_starts, band_starts[1:] + [gy])):
        refuse(f"image bands of unequal height (shard height {sh}, HW={HW})")
    unit_cores = bh * cl_cols
    R = _ceil_div(unit_cores, 32)

    def band_images(y):
        yb = y - y % bh
        first = (yb * sh) // HW
        last = (min((yb + bh) * sh, NHW) - 1) // HW
        return first, last - first + 1

    def segments(y):
        r0, r1 = y * sh, y * sh + vrows[y]
        segs, n = [], r0 // HW
        while n * HW < r1:
            segs.append((n, max(n * HW, r0) - r0, min((n + 1) * HW, r1) - r0))
            n += 1
        return segs

    NU = max(band_images(y)[1] for y in range(gy))
    seg_rows = [segments(y) for y in range(gy)]
    NSEG = max(len(sg) for sg in seg_rows)
    simple = NSEG == 1 and all(v % tr == 0 for v in vrows)  # every piece is a whole shard tile row

    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    # Out of place, the output shard is allocated after this plan, below the live L1 buffers the budget is measured
    # against. In place there is no second shard: the output CBs borrow the input shard already excluded.
    out_shard_bytes = 0 if in_place else _align_up(sh * sw * fmt.in_elem, 64)
    l1_budget = l1_cb_budget(device) - out_shard_bytes

    rows_nominal = _ceil_div(sh, tr)  # tile-row positions of a shard
    rows_full = [v // tr for v in vrows]  # whole tile rows of each core row (TILE: its valid tile rows)
    cap = rows_nominal if tuning.get("chunk_tiles_target") is None else max(1, int(tuning["chunk_tiles_target"]) // W)
    fp_common = dict(
        W=W,
        R=R,
        is_rm=is_rm,
        has_gamma=has_gamma,
        has_beta=has_beta,
        map_period=sw,
        sharded=True,
        NU=NU,
        NSEG=NSEG,
        pilot_cols=cl_cols,
        fmt=fmt,
    )
    col_mask = is_rm and vcols[-1] < sw

    # RM pieces per core row (op words + staged rows) for a fully resident ring of every piece's tile rows.
    rm_pieces = None
    if is_rm:
        h_ops = min(cap, rows_nominal)
        rm_pieces = [_rm_core_ops(seg_rows[y], tr, h_ops) for y in range(gy)]
        ring_rows = [sum(op & 0xFFFF for op in ops if not op & OP_SKIP) for ops, _ in rm_pieces]
        n_staged = max(len(st) for _, st in rm_pieces)
        stage_pages = (n_staged + 1) * W if n_staged else 0  # + the k-fill template row
        fp_ops = footprint_bytes(
            T=h_ops * W, K_alloc=0, stream=False, ring_pages=max(ring_rows) * W, stage_pages=stage_pages, **fp_common
        )

    if not is_rm:
        # TILE: the input ring is the shard (K = B); the chunk only paces the reader's credit pushes.
        rows_core = rows_nominal
        divisors = [d for d in range(min(cap, rows_core), 0, -1) if rows_core % d == 0]
        h = divisors[0]
        T = h * W
        B = rows_core // h
        K_max = K_alloc = B
        stage_pages = 0
        footprint = footprint_bytes(T=T, K_alloc=K_alloc, stream=False, **fp_common)
        ops_path = True
    elif simple and fp_ops > l1_budget:
        # Even RM shards whose tilized copy does not fit: partial residency (K < B) on whole-tile-row chunks.
        rows_core = rows_nominal
        stage_pages = 0
        h = _largest_divisor_at_most(rows_core, max(1, CHUNK_TILES_TARGET // W))
        T = h * W
        B = rows_core // h
        fp_kw = dict(T=T, **fp_common)
        chunk_bytes = fmt.in_tile * T
        f_fixed = footprint_bytes(K_alloc=0, stream=True, **fp_kw)
        K_part = (l1_budget - f_fixed) // chunk_bytes
        if K_part < 0:
            refuse(f"the static CBs ({f_fixed} B) exceed the L1 left beside the shards ({l1_budget} B)")
        K_max = K_part
        override = tuning.get("max_resident_chunks")
        K_alloc = min(K_part if override is None else max(0, int(override)), B)
        footprint = footprint_bytes(K_alloc=K_alloc, stream=B > K_alloc, **fp_kw)
        ops_path = False
    else:
        if fp_ops > l1_budget:
            refuse(
                f"the fully resident ragged RM pieces ({fp_ops} B) exceed the L1 left beside the shards ({l1_budget} B)"
            )
        rows_core = rows_nominal
        h = h_ops
        T = h * W
        B = _ceil_div(rows_core, h)
        K_max = K_alloc = B
        footprint = fp_ops
        ops_path = True
        override = tuning.get("max_resident_chunks")
        h_l = _largest_divisor_at_most(rows_core, min(cap, rows_core))  # whole-tile-row chunks: the FIFOs wrap
        if override is not None and simple and int(override) < rows_core // h_l:
            # The partial-residency path on request (tuning / tests): chunks of the largest divisor <= the cap, K < B.
            h = h_l
            T = h * W
            B = rows_core // h
            fp_kw = dict(T=T, **fp_common)
            K_part = (l1_budget - footprint_bytes(K_alloc=0, stream=True, **fp_kw)) // (fmt.in_tile * T)
            K_max = K_part
            K_alloc = max(0, min(int(override), K_part))
            stage_pages = 0
            footprint = footprint_bytes(K_alloc=K_alloc, stream=True, **fp_kw)
            ops_path = False
    if footprint > l1_budget:
        refuse(f"the static CBs ({footprint} B) exceed the L1 left beside the shards ({l1_budget} B)")

    plan = GroupNormPlan(
        N=N,
        HW=HW,
        C=C,
        G=G,
        Ht=HW // 32,
        Wt=C // 32,
        Cg=Cg,
        span_tiles=0,
        gps=0,
        S=gx,
        gx=gx,
        W=W,
        G_core=lanes_max,
        U=N * gx,
        P=gx * gy,
        cpi=unit_cores,
        h=h,
        T=T,
        R=R,
        rows_core_max=rows_core,
        B_max=B,
        K_max=K_max,
        K_alloc=K_alloc,
        stream=B > K_alloc,
        is_rm=is_rm,
        has_gamma=has_gamma,
        has_beta=has_beta,
        grid_x=grid.x,
        grid_y=grid.y,
        l1_budget=l1_budget,
        footprint=footprint,
        sharded=True,
        in_place=in_place,
        map_period=sw,
        shard_grid=spec.grid,
        NU=NU,
        NSEG=NSEG,
        ops_path=ops_path,
        stage_pages=stage_pages if ops_path else 0,
        col_mask=col_mask,
        mcast_row_major=not split_cols,
        ring_pages=max(ring_rows) * W if (is_rm and ops_path) else 0,
        pilot_cols=cl_cols,
        fmt=fmt,
    )
    for y in range(gy):
        yb = y - y % bh
        n_first, n_img = band_images(y)
        segs = seg_rows[y]
        if is_rm:
            ops, staged = rm_pieces[y]
            full_rows = sum(op & 0xFFFF for op in ops if not op & (OP_STAGED | OP_SKIP))
            rm_credit_pages = (full_rows + sum(1 for op in ops if op & OP_SKIP)) * W
        else:
            ops = [((b - a) // 32) | OP_SEG_END for _, a, b in segs]
            staged, rm_credit_pages = [], 0
        for x in range(gx):
            cx0 = x if split_cols else 0
            xs = x * sw
            g_lo = (cx0 * sw) // Cg
            last_ch = cx0 * sw + sum(vcols[cx0 : cx0 + cl_cols])  # end of the cluster's valid channels
            slot = (y - yb) if split_cols else (y - yb) * gx + x
            plan.work.append(
                CoreWork(
                    core=ttnn.CoreCoord(x0 + x, y0 + y),
                    unit_start=segs[0][0] * gx + x,
                    num_blocks=1,
                    row_start=0,
                    rows_core=rows_full[y],  # whole tile rows (TILE: the valid ones; RM K < B: every one)
                    is_root=int(slot == 0),
                    slot=slot,
                    root=ttnn.CoreCoord(x0 + cx0, y0 + yb),
                    xs=xs,
                    vc=vcols[x],
                    l0=xs // Cg - g_lo,
                    off0=xs % Cg,
                    g_unit=_ceil_div(last_ch, Cg) - g_lo,
                    n_img=n_img,
                    n_seg=len(segs),
                    seg_img0=segs[0][0] - n_first,
                    ops=ops,
                    staged=staged,
                    rm_credit_pages=rm_credit_pages,
                    pilot_cols=[x0 + c for c in range(cx0, cx0 + cl_cols)],
                    w_valid=W if is_rm else vcols[x] // 32,
                )
            )
    return plan


def _used_core_range_set(num_cores, grid_x):
    """The first num_cores cores of the grid in row-major order."""
    ranges = []
    full_rows, rem = divmod(num_cores, grid_x)
    if full_rows > 0:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, full_rows - 1)))
    if rem > 0:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rem - 1, full_rows)))
    return ttnn.CoreRangeSet(ranges)


def _cb(total_size, cores, formats):
    return ttnn.CBDescriptor(
        total_size=total_size,
        core_ranges=cores,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page) for idx, fmt, page in formats
        ],
    )


def _largest_divisor_at_most(n, cap):
    for d in range(min(n, cap), 0, -1):
        if n % d == 0:
            return d
    return 1


def create_program_descriptor(plan, input_tensor, output_tensor, *, gamma=None, beta=None, eps=1e-5):
    device = input_tensor.device()
    W, T, R, h = plan.W, plan.T, plan.R, plan.h
    is_rm = plan.is_rm
    hg, hb = int(plan.has_gamma), int(plan.has_beta)
    bf16 = ttnn.bfloat16
    f32 = ttnn.float32
    # Page formats (Refinement 4): input-side and output-side CBs follow the input dtype (output dtype = input), the
    # affine maps follow gamma / beta's dtype; one source (plan.fmt, ttnn.tile_size) for every page size below.
    fmt = plan.fmt
    assert fmt.in_dtype == input_tensor.dtype and output_tensor.dtype == input_tensor.dtype
    in_dt, in_tile = fmt.in_dtype, fmt.in_tile

    sharded = plan.sharded
    cores = plan.shard_grid if sharded else _used_core_range_set(len(plan.work), plan.grid_x)
    map_period = plan.map_period

    # In place the output tensor is the input tensor, so the output CBs below borrow the input shard's bytes.
    assert output_tensor is input_tensor or not plan.in_place

    def shard_cb(tensor, formats):
        # Zero-copy: the CB is the core's own shard (no NoC traffic); the views are tile-sized pages over it. A ragged
        # RM shard need not be whole tile rows: the CB spans its whole tile rows (the FIFO walks only those; partial
        # rows are staged).
        spec = tensor.memory_config().shard_spec
        shard_bytes = int(spec.shape[0]) * int(spec.shape[1]) * fmt.in_elem
        total = (shard_bytes // in_tile) * in_tile
        d = ttnn.cb_descriptor_from_sharded_tensor(formats[0][0], tensor, total_size=total)
        d.format_descriptors = [
            ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page) for idx, fmt, page in formats
        ]
        return d

    # ---------------- Circular buffers ----------------
    cbs = []
    if is_rm:
        if sharded:
            cbs.append(shard_cb(input_tensor, [(CB_INPUT_RM, in_dt, in_tile)]))
            cbs.append(shard_cb(output_tensor, [(CB_OUTPUT_RM, in_dt, in_tile)]))
        else:
            cbs.append(_cb(D_RM * T * in_tile, cores, [(CB_INPUT_RM, in_dt, in_tile)]))
    # Alias group A: the input ring, K_alloc resident chunks + the streaming depth, as two FIFO views pushed in
    # lockstep. Pass 1 pops every chunk from CB_INPUT_TILES; CB_INPUT_KEEP drops the streamed head in pass 1 and
    # holds the resident tail (then the re-read head) for pass 2. Sharded TILE: the ring is the shard (K = B).
    ring_views = [(CB_INPUT_TILES, in_dt, in_tile), (CB_INPUT_KEEP, in_dt, in_tile)]
    if sharded and not is_rm:
        cbs.append(shard_cb(input_tensor, ring_views))
        cbs.append(shard_cb(output_tensor, [(CB_OUTPUT_TILES, in_dt, in_tile)]))
    else:
        ring_pages = plan.ring_pages or input_ring_pages(T=T, K_alloc=plan.K_alloc, stream=plan.stream, is_rm=is_rm)
        cbs.append(_cb(ring_pages * in_tile, cores, ring_views))
        if not sharded:
            # Sharded RM pack-untilizes the normalized tiles straight into the output shard: no tile ring.
            # The pass-2 output split across both DM RISCs, each writing its share on its own NoC.
            chunk_bytes = T * in_tile
            seq_bytes = OUT_SEQ_PAGES * OUT_SEQ_PAGE_BYTES
            u32 = ttnn.uint32
            cbs.append(_cb(plan.out_writer_chunks * chunk_bytes, cores, [(CB_OUTPUT_TILES, in_dt, in_tile)]))
            cbs.append(_cb(OUT_READER_CHUNKS * chunk_bytes, cores, [(CB_OUTPUT_B, in_dt, in_tile)]))
            cbs.append(_cb(seq_bytes, cores, [(CB_OUT_SEQ_A, u32, OUT_SEQ_PAGE_BYTES)]))
            cbs.append(_cb(seq_bytes, cores, [(CB_OUT_SEQ_B, u32, OUT_SEQ_PAGE_BYTES)]))
    # Alias group E: the S1' spill accumulator (UnpackToDestFp32, multi-chunk pass 1) / the FPU-readable chan_scale.
    # The last chunk reloads accumulator tile i and packs the collapsed chan tile i over it (in-place lockstep; every
    # view returns to its base each block), so the accumulator costs no L1 of its own.
    cbs.append(
        _cb(
            W * F32_TILE_BYTES,
            cores,
            [(CB_SUM_ACC, f32, F32_TILE_BYTES), (CB_CHAN_SCALE, f32, F32_TILE_BYTES)],
        )
    )
    # Alias group C: the T2 accumulator (pass 1) / chan_work (b', expand -> pass 2), disjoint lifetimes.
    cbs.append(
        _cb(
            W * F32_TILE_BYTES,
            cores,
            [(CB_SUMSQ_ACC, f32, F32_TILE_BYTES), (CB_CHAN_WORK, f32, F32_TILE_BYTES)],
        )
    )
    cbs.append(_cb(W * BF16_TILE_BYTES, cores, [(CB_GROUP_MAP, bf16, BF16_TILE_BYTES)]))
    cbs.append(_cb(W * BF16_TILE_BYTES, cores, [(CB_PILOT_ROWS, bf16, BF16_TILE_BYTES)]))
    cbs.append(_cb(BF16_TILE_BYTES, cores, [(CB_NEG_ONE, bf16, BF16_TILE_BYTES)]))
    cbs.append(_cb(F32_TILE_BYTES, cores, [(CB_GROUP_SHIFT, f32, F32_TILE_BYTES)]))
    if hg:
        # The expand matmul's in1: the gamma map, + the beta map when beta (reader-built in gamma / beta's dtype;
        # beta-only is refused).
        amap_tile = fmt.amap_tile
        cbs.append(_cb((hg + hb) * W * amap_tile, cores, [(CB_GAMMA_ROWS, fmt.aff_dtype, amap_tile)]))
    NU, NSEG = plan.NU, plan.NSEG
    # Two partial tiles per image segment of the core.
    cbs.append(_cb(2 * NSEG * F32_TILE_BYTES, cores, [(CB_LOCAL_PARTIALS, f32, F32_TILE_BYTES)]))
    # Remote-written landing zones (allocated on every used core so the base address is identical): R tile pairs per
    # image of the unit.
    cbs.append(_cb(2 * R * NU * F32_TILE_BYTES, cores, [(CB_GATHERED_PARTIALS, f32, F32_TILE_BYTES)]))
    cbs.append(_cb(2 * NU * F32_TILE_BYTES, cores, [(CB_STATS_OUT, f32, F32_TILE_BYTES)]))
    # [rstd, mean*rstd] per image of the unit (multicast from the unit root) + e = 1.0 (page 2*NU, written once by the
    # writer): the expand's in0.
    cbs.append(_cb((2 * NU + 1) * F32_TILE_BYTES, cores, [(CB_GROUP_STATS, f32, F32_TILE_BYTES)]))
    stage = staging_bytes(
        W=W, map_period=map_period, is_rm=is_rm, n_affine=hg + hb, fmt=fmt, sharded=sharded, pilot_cols=plan.pilot_cols
    )
    cbs.append(_cb(stage, cores, [(CB_AFFINE_STAGING, bf16, stage)]))
    has_stage = plan.stage_pages > 0
    if has_stage:
        # Alias group F: staged partial tile rows (pass 1: reader -> tilize; + the k-fill template row) and their
        # normalized rows (pass 2: untilize -> writer copy-back), disjoint lifetimes.
        cbs.append(
            _cb(
                plan.stage_pages * in_tile,
                cores,
                [(CB_RM_STAGE_IN, in_dt, in_tile), (CB_RM_STAGE_OUT, in_dt, in_tile)],
            )
        )

    # ---------------- Kernel args ----------------
    in_page_bytes = input_tensor.buffer_page_size()
    out_page_bytes = output_tensor.buffer_page_size()
    gamma_addr = gamma.buffer_address() if gamma is not None else 0
    beta_addr = beta.buffer_address() if beta is not None else 0

    reader_ct = [W, h, int(is_rm), hg, hb, R, plan.Cg, plan.Wt, plan.Ht, plan.gx, in_page_bytes, map_period]
    reader_ct.append(int(sharded))
    # Per-chunk zones (CT): every core's chunk steps fit the marker budget.
    chunk_zones = int(
        bool(_tuning().get("chunk_zones"))
        and plan.B_max + (plan.B_max - min(plan.K_alloc, plan.B_max)) <= CHUNK_ZONE_MAX_STEPS
    )
    reader_ct.append(chunk_zones)
    reader_ct.extend([int(has_stage), int(plan.ops_path)])
    # Formats (Refinement 4), CT 16-19: the input tile bytes (bf16 2048 / Float32 4096), gamma / beta's element bytes,
    # gamma / beta TILE layout, gamma / beta's page bytes (RM: the C-channel stick; TILE: one tile).
    aff_ref = gamma if gamma is not None else beta
    reader_ct.extend(
        [in_tile, fmt.aff_elem, int(fmt.aff_tile), aff_ref.buffer_page_size() if aff_ref is not None else 0]
    )
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct.extend(
        (ttnn.TensorAccessorArgs(gamma) if gamma is not None else ttnn.TensorAccessorArgs()).get_compile_time_args()
    )
    reader_ct.extend(
        (ttnn.TensorAccessorArgs(beta) if beta is not None else ttnn.TensorAccessorArgs()).get_compile_time_args()
    )
    # The output, for the reader's pass-2 share (interleaved only; sharded builds carry it unused).
    reader_ct.append(out_page_bytes)
    reader_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # Compute is built once per role too (IS_ROOT is compile-time, CT 7): a runtime role flag lets O3 unswitch the
    # block loop on it, emitting the whole per-block body twice; the root binary must fit the kernel-config ring
    # (also under the watcher / assert build).
    def compute_ct(is_root):
        return [
            W,
            h,
            int(is_rm),
            hg,
            hb,
            R,
            int(sharded),
            int(is_root),
            chunk_zones,
            NU,
            map_period,
            int(plan.col_mask),
            int(has_stage),
            int(plan.ops_path),
        ]

    # The writer is built once per role (IS_ROOT is compile-time): each binary carries only its own mcast pipe
    # (sender on unit roots, receiver elsewhere) and only the root's gather-wait / zero-fill code.
    def writer_ct(is_root):
        ct = [W, h, R, plan.Wt, plan.Ht, plan.gx, out_page_bytes, int(is_root), int(is_rm), int(sharded), chunk_zones]
        ct.extend([NU, int(has_stage), int(plan.ops_path), map_period])
        ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
        return ct

    inv_n_bits = _f32_bits(1.0 / (plan.HW * plan.Cg))
    eps_bits = _f32_bits(eps)

    # NoC pairing per core. Every core reads on one NoC and writes on the other; DRAM interleaved, the odd grid
    # columns take the swapped pairing (read NOC1 / write NOC0), so both NoCs carry half of every phase's DRAM
    # stream (pass-1 reads, pass-2 reads + writes) instead of all reads on NOC0 and all writes on NOC1. Unit roots
    # keep (NOC0, NOC1): the mcast sender's NoC must be the McastConfig NoC. Sharded inputs stream nothing.
    def swapped(wk):
        return int(not sharded and not wk.is_root and wk.core.x % 2 == 1)

    reader_rt = {0: ttnn.RuntimeArgs(), 1: ttnn.RuntimeArgs()}
    reader_cores = {0: [], 1: []}
    compute_rt = {1: ttnn.RuntimeArgs(), 0: ttnn.RuntimeArgs()}
    writer_rt = {(r, sw): ttnn.RuntimeArgs() for r in (1, 0) for sw in (0, 1)}
    writer_cores = {1: [], 0: []}
    writer_cores_sw = {(r, sw): [] for r in (1, 0) for sw in (0, 1)}
    for wk in plan.work:
        x, y = wk.core.x, wk.core.y
        sw = swapped(wk)
        B_core = _ceil_div(wk.rows_core, h)
        K_core = min(plan.K_alloc, B_core)
        root_virt = device.worker_core_from_logical_core(wk.root)
        reader_cores[sw].append(wk.core)
        reader_rt[sw][x][y] = [
            input_tensor.buffer_address(),
            gamma_addr,
            beta_addr,
            wk.unit_start,
            wk.num_blocks,
            wk.row_start,
            wk.rows_core,
            K_core,
            wk.is_root,
            root_virt.x,
            root_virt.y,
            output_tensor.buffer_address(),
            # Sharded slice geometry (Refinement 3); interleaved derives it per block.
            wk.xs,
            wk.vc,
            wk.l0,
            wk.off0,
            wk.g_unit,
            wk.rm_credit_pages,
            len(wk.pilot_cols),
            *[device.worker_core_from_logical_core(ttnn.CoreCoord(cx, wk.root.y)).x for cx in wk.pilot_cols],
            len(wk.staged),
            *[v for trip in wk.staged for v in trip],
        ]
        compute_rt[wk.is_root][x][y] = [
            wk.num_blocks,
            wk.rows_core,
            K_core,
            inv_n_bits,
            eps_bits,
            wk.n_img,
            wk.seg_img0,
            wk.w_valid,
            wk.vc,
            len(wk.ops),
            *wk.ops,
        ]
        writer_cores[wk.is_root].append(wk.core)
        writer_cores_sw[(wk.is_root, sw)].append(wk.core)
        writer_rt[(wk.is_root, sw)][x][y] = [
            output_tensor.buffer_address(),
            wk.unit_start,
            wk.num_blocks,
            wk.row_start,
            wk.rows_core,
            K_core,
            wk.slot,
            root_virt.x,
            root_virt.y,
            plan.cpi,  # gather arrivals: the cores of the unit
            wk.n_img,
            wk.n_seg,
            wk.seg_img0,
            wk.rm_credit_pages,
            len(wk.staged),
            *[v for trip in wk.staged for v in trip],
        ]

    reader_kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "groupnorm_sc_N_1_HW_C_b_reader.cpp"),
            core_ranges=(
                cores if not reader_cores[1] else ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in reader_cores[sw]])
            ),
            compile_time_args=reader_ct,
            runtime_args=reader_rt[sw],
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1 if sw else ttnn.NOC.NOC_0
            ),
        )
        for sw in (0, 1)
        if reader_cores[sw]
    ]
    writer_kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "groupnorm_sc_N_1_HW_C_b_writer.cpp"),
            core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in writer_cores_sw[(role, sw)]]),
            compile_time_args=writer_ct(role),
            runtime_args=writer_rt[(role, sw)],
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0 if sw else ttnn.NOC.NOC_1
            ),
        )
        for role in (1, 0)
        for sw in (0, 1)
        if writer_cores_sw[(role, sw)]
    ]
    compute_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi3,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        dst_full_sync_en=False,
    )
    unpack_modes = [ttnn.UnpackToDestMode.Default] * NUM_CB_SLOTS
    for idx in (CB_SUM_ACC, CB_SUMSQ_ACC):  # cb_group_shift is a matmul operand (srcA) now, not a DEST copy
        unpack_modes[idx] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute_config.unpack_to_dest_mode = unpack_modes
    compute_kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "groupnorm_sc_N_1_HW_C_b_compute.cpp"),
            core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in writer_cores[role]]),
            compile_time_args=compute_ct(role),
            runtime_args=compute_rt[role],
            config=compute_config,
        )
        for role in (1, 0)
        if writer_cores[role]
    ]

    gather_sem = ttnn.SemaphoreDescriptor(id=GATHER_SEMAPHORE_ID, core_ranges=cores, initial_value=0)
    descriptor = ttnn.ProgramDescriptor(kernels=[], semaphores=[gather_sem], cbs=cbs)

    # Stats multicast: one channel over all used cores, groups of cpi consecutive cores, sender = slot 0 (the
    # unit root). DRAM: units are row-major core runs; sharded: a unit is a grid-column segment of one image band
    # (ColumnMajor), or the band's whole grid rows when groups straddle core columns (RowMajor). cpi == 1
    # degenerates to the sender's local copy.
    mcast = ttnn.Mcast(
        device,
        ttnn.McastConfig(noc=ttnn.NOC.NOC_1, handshake=True),
        cores,
        plan.cpi,
        ttnn.McastFixedSenderConfig(sender_index=0),
        ttnn.McastCoreOrder.ColumnMajor if sharded and not plan.mcast_row_major else ttnn.McastCoreOrder.RowMajor,
    )
    mcast.attach(descriptor, "mcast", writer_kernels, MCAST_FIRST_SEMAPHORE_ID)
    descriptor.kernels = [*reader_kernels, *writer_kernels, *compute_kernels]
    return descriptor
