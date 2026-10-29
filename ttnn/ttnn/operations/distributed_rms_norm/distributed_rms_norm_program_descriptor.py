# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""distributed_rms_norm MeshProgramDescriptor (op_design.md -> Blocking Model, Work Distribution, Dataflow Strategy).

One program per chip, one ttnn.generic_op per call. Per chip:

  port cores (2 * L, one per (direction, link))   the fabric_ring_exchange port kernel (relay all-gather of the
                                                  compact fp32 Σx² records, ready fence, per-entry notify)
  worker cores (grid - ports)                     reader / compute / writer — the base op's local work, re-split
                                                  for the worker grid and cut into row WAVES:
      Phase A  load x, square + collapse per wave, on-chip combine of the G_w slice partials at the wave's owner,
               owner publishes the wave's compact records into its link's ports;  γ pass over the whole block
      Phase B  owner collects the G records of each owned row (wave order), folds them in group-position order
               (SFPU, fp32), finalizes rsqrt(ss/W + eps), all-gathers inv_rms to the row-group;
               every member scales its wave rows and stores them.

Regimes (select_plan): D1 (interleaved, whole rows per core), D3 (interleaved, W split over G_w slice cores),
D1-HS (HEIGHT_SHARDED, the shard is the block), D3-WS (WIDTH_SHARDED, one row-group of all shard cores).

Knobs (single source of truth; every CB size / trip count derives from them):
    W_CHUNK_TILES      cap of the streaming chunk C (scale_by_inv_rms -> writer quantum)
    STREAM_DEPTH       cb_output depth in C-tile quanta
    SCALE_BLOCK_TILES  tiles per DEST window of the scaling chains (clamped to DEST by the chain)
    MIN_SLICE_TILES    smallest interleaved W-slice S (fused_split)
    MAX_GROUP_CORES    cap on G_w (on-chip combine fan-in)
    WAVES_MAX          cap on row waves per core (exchange pipelining depth)
    READ_LOOKAHEAD_WAVES  x-read waves in flight per core beyond the oldest unlanded one (0: wave-ordered DRAM reads)
    COMBINE_LAG        waves between a wave's landing / square and its on-chip combine + publish (0 since Refinement 2)
    FOLD_AHEAD         waves an owner's cross-device fold may run ahead of its own scaling (Refinement 3)
    NOC_SWAP_COLUMNS   westmost logical worker columns whose x reads run on NoC1 and output writes on NoC0
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, field, replace
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_DIM = 32
FACE_DIM = 16
TILE_HW = TILE_DIM * TILE_DIM

# ---- Knobs (single source of truth) ----
W_CHUNK_TILES = 32
STREAM_DEPTH = 2
SCALE_BLOCK_TILES = 8
MIN_SLICE_TILES = 4
MAX_GROUP_CORES = 24
WAVES_MAX = 8
FABRIC_PORT_SEMAPHORES = 3  # ids setup_fabric_connection appends on a port core (teardown, buffer index, flow control)
WAVES_MAX_PASSES = 12  # wave cap of the pass regime (smaller waves shrink every per-wave CB); port own_stats ids
#                        0..num_waves-1 plus the fabric's ids must fit NUM_SEMAPHORES
NUM_SEMAPHORES = 16
COMBINE_DEPTH = 2  # cb_partial_col depth in waves: compute collapses wave w+1 while the reader still sends wave w
LANDING_MIN_WORDS = 1024  # landing scratch floor (4 KiB): most plans then share one allocation
# Refinement 2 (LOCAL perf, 1x4 ring, traced slowest chip, same-session A/B against the Phase 0 values 1 / 0 / 1):
# 640x1792 op 26.3 -> 21.6 us (exchange stubbed 19.7 -> 17.9), 2048x1280 44.8 -> 33.2 (31.9 -> 28.7); D1 / D1-HS /
# D3-WS / decode within noise (decode stub +0.12 us from the swap, its op time no worse).
READ_LOOKAHEAD_WAVES = 0  # x-read waves in flight beyond the oldest unlanded one: 0 = each wave's reads go out once the
#                           previous wave landed, so DRAM serves the grid wave by wave and wave 0 publishes early
#                           (1: op +2.5 us at 640x1792; 2: +2.0)
NOC_SWAP_COLUMNS = 3  # westmost logical worker columns (next to DRAM column 0) whose x reads go on NoC1 and output
#                       writes on NoC0 (base op Perf 2 dram_noc_balance). 640x1792 stub 19.96 -> 17.7 us; 1 / 2 / 4 / 5
#                       columns: 19.1 / 18.3 / 17.8 / 18.7
COMBINE_LAG = 0  # waves between a wave's landing / square and its on-chip combine + publish (1: Phase 0 order — the
#                  contribution of wave w waited for wave w+1's reads)
FOLD_AHEAD_ALL = 1 << 16  # "every owned fold first" (any value >= num_waves)
FOLD_AHEAD = 1  # waves an owner's fold (cross-device finalize) may run ahead of its own scaling pass, combine
#                  regimes (G_w > 1): >= num_waves = every owned fold before any scaling (Phase 0 order); 0 = fold of
#                  wave w right before the scaling of wave w. 1 lets an owner store its earlier waves before its last
#                  records land (2048x1280 33.4 / 32.5 / 32.2 -> 30.8 / 30.8 / 30.7 us; 0: last-launched chip +2.6 us)
FOLD_AHEAD_SOLO = FOLD_AHEAD_ALL  # G_w == 1 (D1 / D1-HS): the core owns every wave and its inv comes back through
#                  its own reader, so interleaving folds with scales only adds a round trip per wave (D1-HS 1 vs all:
#                  +0.35 us Ring, +0.6 us Linear)

# ---- Exchange constants ----
RECORD_BYTES = 32 * 4  # one fp32 per row of a tile-row
SEG_ALIGN = 64  # landing slot alignment (Blackhole DRAM/L1 NoC alignment)
MAX_HEADERS = 8  # packet headers per port (one per destination chip)
PORT_PLACEMENT = "row0"  # Refinement 4 (TRANSPORT): "row0" = 2·L cores nearest the middle of logical row 0, the same on
#                          every chip; "eth" = per chip, each port on the free core with the fewest NoC1 hops to its
#                          link's Ethernet core (fabric_all_gather plan()); the worker grid excludes the union
STATS_LINKS_MAX = (
    1 << 16
)  # Refinement 4 (TRANSPORT, lamp L3): links that carry the stats records, min'ed with num_links;
#                            only those links get (and reserve) port cores, the rest of the grid works
FWD, BWD = 0, 1
_OPP = {FWD: BWD, BWD: FWD}

SCALER_FORMAT = ttnn.bfloat16
STATS_FORMAT = ttnn.float32  # cb_stats / cb_record_scratch: the exchanged records are exact fp32 (wire format)

# ---- CB indices (the kernels declare the same slots) ----
CB_X_TILES = 1
CB_REDUCE_SCALER = 2
CB_SUMSQ_PARTIAL = 3
CB_INV_RMS = 4
CB_GAMMA_TILES = 6
CB_X_GAMMA = 7
CB_X_REREAD = 8
CB_PARTIAL_COL = 9
CB_GATHER = 10
CB_RMS_ROOT = 11
CB_SUMSQ_LAST = 12
CB_SUMSQ_ACC = 13
CB_CHIP_PARTIAL = 14
CB_STATS = 15
CB_OUTPUT = 16
CB_RECORD_SCRATCH = 17
NUM_CBS = 64


def _fp32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


def _align_up(v, a):
    return (v + a - 1) // a * a


def _crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def intermediate_format(fp32_dest_acc_en: bool):
    """Compute-internal CB format follows the DEST width (base op intermediate_format)."""
    return ttnn.float32 if fp32_dest_acc_en else ttnn.bfloat16


def tensor_tile_rows_of(shape) -> int:
    return _prod(shape[:-2]) * math.ceil(shape[-2] / TILE_DIM)


def l1_budget(device) -> int:
    """Bytes a program's CBs may take per worker: CBs grow up from the bank base, L1 buffers are placed top-down, so
    the budget is the gap below the lowest allocated L1 block (the largest free block may be a hole higher up)."""
    try:
        view = ttnn.get_memory_view(device, ttnn.BufferType.L1)
    except Exception:  # noqa: BLE001 — mesh devices without a memory view: Blackhole worker L1 minus reserves
        return 1300 * 1024
    try:
        blocks = [
            (int(b["address"]), int(b["size"]), str(b["allocated"]).lower() in ("yes", "true", "1"))
            for b in view.block_table
        ]
    except Exception:  # noqa: BLE001
        blocks = []
    if not blocks:
        return int(view.largest_contiguous_bytes_free_per_bank)
    base = min(a for a, _, _ in blocks)
    allocated = [a for a, _, used in blocks if used]
    return (min(allocated) - base) if allocated else int(view.total_bytes_per_bank)


# ---------------------------------------------------------------------------------------------------------------
# Local plan (the base op's blocking, re-split for the worker grid left after the ports)
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Member:
    core: object  # logical CoreCoord
    group: int  # row-group index
    index: int  # member index j within the row-group
    start_row: int  # first tile-row of the group
    rows: int  # tile-rows of the group
    w0: int  # first tile of the slice
    V: int  # valid tiles of the slice


# γ placement modes of the scaling pass (fallback order: block -> wave -> chunk; Refinement 1)
GAMMA_BLOCK = 0  # x·γ of the whole pass block into cb_x_gamma before Phase B (the exchange's stall shadow)
GAMMA_WAVE = 1  # x·γ per wave, right before the wave's inv scaling (cb_x_gamma = q_w rows)
GAMMA_CHUNK = 2  # x·γ per C-chunk of a wave, fused ahead of its inv scaling (cb_x_gamma = q_w · C tiles)
_GAMMA_MODE_NAMES = {GAMMA_BLOCK: "block", GAMMA_WAVE: "wave", GAMMA_CHUNK: "chunk"}


@dataclass
class LocalPlan:
    regime: str
    sharded: bool
    S: int  # slice tiles (nominal)
    G_w: int
    G_h: int
    rows_max: int
    num_waves: int
    q_w: int
    owners: int
    q: int
    masked: bool
    members: list = field(default_factory=list)  # Member per worker core, in group order
    # ---- L1 blocking (Refinement 1; Phase 0 = PW num_waves, cap W_CHUNK_TILES, block γ, x_depth 1) ----
    PW: int = 0  # waves per pass (0 -> num_waves: one pass). Waves stay global across passes (exchange unchanged)
    cap: int = W_CHUNK_TILES  # chunk cap (halved by the L1 fallback)
    gamma_mode: int = GAMMA_BLOCK
    gamma_stream: bool = False  # cb_gamma_tiles streams STREAM_DEPTH C-tile chunks instead of the resident slice
    x_depth: int = 1  # interleaved x CB depth in pass blocks (2: the next pass's reads overlap this pass)
    out_resident: bool = False  # output CB backed on the output shard (input's own placement)
    combine_depth: int = COMBINE_DEPTH  # cb_partial_col depth in waves (1: the lean rung of the L1 ladder)
    sq_rows: int = 0  # tile-rows per square -> collapse round (0: the whole wave; 1: the lean rung)
    # x re-read (Refinement 1a, interleaved only, the last rung): x is never resident — Phase A streams it per
    # square -> collapse round through cb_x_tiles (x_depth rounds deep), Phase B's writer re-reads it per
    # (wave, chunk, tile-row) into cb_x_reread. x is read twice from DRAM, only on this rung.
    reread: bool = False

    def __post_init__(self):
        if self.PW <= 0:
            self.PW = self.num_waves

    @property
    def chunks(self):
        return chunking(self.S, self.sharded, self.cap)

    @property
    def C(self):
        return self.chunks[0]

    @property
    def nch(self):
        return self.chunks[1]

    @property
    def WS(self):
        return self.chunks[2]

    @property
    def Wp(self):
        return self.nch * self.C

    @property
    def passes(self):
        return math.ceil(self.num_waves / self.PW)

    @property
    def R_pass(self):
        """Tile-rows of one pass block (the whole assignment when there is one pass)."""
        return min(self.rows_max, self.PW * self.q_w)

    @property
    def regions(self):
        """cb_gather regions (one per owned wave) live within one pass."""
        return max(1, max_owned_waves(self.PW, self.owners, self.G_w))

    @property
    def SQ(self):
        """Raw Σx² CB capacity in tile-rows (cb_sumsq_partial, and cb_sumsq_last / cb_sumsq_acc when masked)."""
        return self.q_w if self.sq_rows <= 0 else min(self.sq_rows, self.q_w)

    @property
    def gamma_per_wave(self):
        return self.gamma_mode == GAMMA_WAVE


def gather_field_bits(G_w: int) -> int:
    """Bits per owned-wave field of SEM_GATHER (counts up to G_w contributions)."""
    return max(1, G_w.bit_length())


def max_owned_waves(num_waves: int, owners: int, G_w: int) -> int:
    """Most waves one member owns among num_waves consecutive waves (owner k of wave w = member (w * owners + k) mod
    G_w: the owner slots of consecutive waves are consecutive integers, so each residue appears at most this often)."""
    return math.ceil(num_waves * owners / G_w) if G_w > 1 else 0


def wave_params(rows_max: int, G_w: int, waves_max: int = WAVES_MAX):
    """(num_waves, q_w, owners, q). Waves shrink until every member's owned waves fit SEM_GATHER's 32 bits (for the
    pass regime's larger wave cap the bound is per pass: first_fit checks it per option)."""
    waves = max(1, min(rows_max, waves_max))
    while True:
        q_w = math.ceil(rows_max / waves)
        num_waves = math.ceil(rows_max / q_w)
        owners = max(1, min(G_w, q_w))
        q = math.ceil(q_w / owners)
        if (
            waves == 1
            or waves_max > WAVES_MAX
            or max_owned_waves(num_waves, owners, G_w) * gather_field_bits(G_w) <= 32
        ):
            return num_waves, q_w, owners, q
        waves -= 1


def chunking(S: int, sharded: bool, cap: int = W_CHUNK_TILES):
    nch = math.ceil(S / cap)
    C = math.ceil(S / nch)
    if sharded:
        nch = math.ceil(S / C)
    Wp = nch * C
    WS = S if sharded else Wp
    return C, nch, WS


@dataclass(frozen=True)
class Sizes:
    T_in: int
    T_out: int
    T_g: int  # 0 when absent
    T_i: int
    has_gamma: bool
    G: int


def cb_table(plan: LocalPlan, sz: Sizes):
    """(cb index, page size, pages) of every CB this plan ALLOCATES per worker (zero-copy shard-backed CBs excluded).
    The single source of the L1 footprint and of the CB descriptors (l1_ledger.md)."""
    p = plan
    T_s = ttnn.tile_size(STATS_FORMAT)
    t = [
        (CB_REDUCE_SCALER, ttnn.tile_size(SCALER_FORMAT), 2 if p.masked else 1),
        (CB_SUMSQ_PARTIAL, sz.T_i, p.SQ),
        (CB_INV_RMS, sz.T_i, p.PW * p.q_w),
        (CB_CHIP_PARTIAL, sz.T_i, 2 * p.q),
        (CB_RMS_ROOT, sz.T_i, p.q),
        (CB_STATS, T_s, sz.G * p.q),
        (CB_RECORD_SCRATCH, RECORD_BYTES, p.q),  # publish staging (collect reads straight into cb_stats)
    ]
    if p.reread:
        t.append((CB_X_TILES, sz.T_in, p.x_depth * p.SQ * p.WS))
        t.append((CB_X_REREAD, sz.T_in, STREAM_DEPTH * p.C))
    elif not p.sharded:
        t.append((CB_X_TILES, sz.T_in, p.x_depth * p.R_pass * p.WS))
    if not p.out_resident:
        t.append((CB_OUTPUT, sz.T_out, STREAM_DEPTH * p.C))
    if p.G_w > 1:  # the on-chip combine CBs exist only on D3 / D3-WS
        t.append((CB_PARTIAL_COL, sz.T_i, p.combine_depth * p.q_w))
        t.append((CB_GATHER, sz.T_i, p.regions * p.G_w * p.q))
    if sz.has_gamma:
        t.append((CB_GAMMA_TILES, sz.T_g, STREAM_DEPTH * p.C if p.gamma_stream else p.Wp))
        xg = {GAMMA_BLOCK: p.R_pass * p.WS, GAMMA_WAVE: p.q_w * p.WS, GAMMA_CHUNK: p.q_w * p.C}[p.gamma_mode]
        if p.reread:
            xg = p.C  # x·γ of one tile-row chunk
        t.append((CB_X_GAMMA, sz.T_i, xg))
    if p.masked:
        t.append((CB_SUMSQ_LAST, sz.T_i, p.SQ))
        t.append((CB_SUMSQ_ACC, sz.T_i, p.SQ))
    return t


def footprint(sz: Sizes, plan: LocalPlan) -> int:
    """Per-worker L1 bytes of every allocated CB (l1_ledger.md -> Total per-core footprint)."""
    return sum(ps * n for _, ps, n in cb_table(plan, sz))


def _caps():
    c, out = W_CHUNK_TILES, []
    while c >= 1:
        out.append(c)
        c //= 2
    return out


def blocking_options(num_waves: int, has_gamma: bool, sharded: bool, combine: bool = False):
    """The L1 fallback ladder, predicate-guarded first fit (Refinement 1). Phase 0's plan is the first option (one
    pass, full chunk cap, block γ) and its per-wave γ fallback the second, so every shape that fit keeps its plan:
      1. per-wave x·γ;  2. per-chunk x·γ (resident γ, then streamed γ);  3. chunk-cap halving;
      4. passes (multi-block rows: waves per pass PW < num_waves; interleaved x_depth 2, then 1);
      5. lean: the whole ladder again with cb_partial_col single-buffered (on-chip combine) and one tile-row per
         square -> collapse round (raw Σx² CBs of one tile-row)."""
    pws = []
    for P in range(1, num_waves + 1):
        pw = math.ceil(num_waves / P)
        if pw not in pws:
            pws.append(pw)
    modes = (
        [(GAMMA_BLOCK, False), (GAMMA_WAVE, False), (GAMMA_CHUNK, False), (GAMMA_CHUNK, True)]
        if has_gamma
        else [(GAMMA_BLOCK, False)]
    )
    for lean in (False, True):
        depth = 1 if (lean and combine) else COMBINE_DEPTH
        sq = 1 if lean else 0
        for pw in pws:
            for cap in _caps():
                for gm, gs in modes:
                    for xd in [1] if (sharded or pw >= num_waves) else [2, 1]:
                        yield dict(
                            PW=pw, cap=cap, gamma_mode=gm, gamma_stream=gs, x_depth=xd, combine_depth=depth, sq_rows=sq
                        )


def reread_options(num_waves: int, has_gamma: bool):
    """The x re-read rung (Refinement 1a, interleaved only), tried after every other rung: x streamed one tile-row per
    square -> collapse round (x_depth rows deep, double-buffered first) and re-read per tile-row chunk for the scaling
    pass, so the x footprint no longer scales with the wave. cb_partial_col keeps COMBINE_DEPTH (the streamed reader
    pushes wave w+1 before it services wave w: a single-wave cb_partial_col would deadlock)."""
    pws = []
    for P in range(1, num_waves + 1):
        pw = math.ceil(num_waves / P)
        if pw not in pws:
            pws.append(pw)
    modes = [(GAMMA_CHUNK, False), (GAMMA_CHUNK, True)] if has_gamma else [(GAMMA_BLOCK, False)]
    for pw in pws:
        for cap in _caps():
            for gm, gs in modes:
                for xd in (2, 1):
                    yield dict(
                        PW=pw,
                        cap=cap,
                        gamma_mode=gm,
                        gamma_stream=gs,
                        x_depth=xd,
                        combine_depth=COMBINE_DEPTH,
                        sq_rows=1,
                        reread=True,
                    )


def rewave(base: LocalPlan, waves_max: int) -> LocalPlan:
    nw, q_w, owners, q = wave_params(base.rows_max, base.G_w, waves_max)
    return replace(base, num_waves=nw, q_w=q_w, owners=owners, q=q, PW=nw)


def first_fit(base: LocalPlan, sz: Sizes, budget: int):
    """(option rank, plan) of the first ladder option whose footprint fits, or None. The whole ladder at the Phase 0
    wave count first; then, as the last resort, the pass rungs at up to WAVES_MAX_PASSES smaller waves."""
    rank = 0
    bases = [base]
    more = rewave(base, WAVES_MAX_PASSES)
    if more.num_waves > base.num_waves:
        bases.append(more)
    ladders = [blocking_options]
    if not base.sharded:
        ladders.append(lambda nw, g, sh, cmb: reread_options(nw, g))  # last resort: x re-read (Refinement 1a)
    for ladder in ladders:
        for b in bases:
            for opt in ladder(b.num_waves, sz.has_gamma, b.sharded, b.G_w > 1):
                rank += 1
                p = replace(b, **opt)
                if b is not base and p.PW >= p.num_waves:
                    continue  # beyond the Phase 0 wave count only the pass regime is used
                if p.regions * gather_field_bits(p.G_w) > 32 and p.G_w > 1:
                    continue  # SEM_GATHER holds one field per owned wave of a pass
                if footprint(sz, p) <= budget:
                    return rank, p
    return None


def _wave_plan(regime, sharded, S, G_w, G_h, rows_max, masked, out_resident, members=None):
    nw, q_w, owners, q = wave_params(rows_max, G_w)
    return LocalPlan(
        regime=regime,
        sharded=sharded,
        S=S,
        G_w=G_w,
        G_h=G_h,
        rows_max=rows_max,
        num_waves=nw,
        q_w=q_w,
        owners=owners,
        q=q,
        masked=masked,
        members=members or [],
        out_resident=out_resident,
    )


def fused_split(T: int, Wt: int, N: int, sz: Sizes, budget: int):
    """(S, G_w, G_h) for an interleaved input on N worker cores (op_design.md -> Work Distribution). Phase 0's rule
    (the whole block resident, per-wave γ allowed) first; when no slice fits it, the slice whose first-fit L1
    fallback ranks best (Refinement 1), ties broken by the Phase 0 key."""
    cands = []
    forced = _measure_overrides().get("slice")
    for S in range(min(Wt, MIN_SLICE_TILES), Wt + 1):
        if forced and S != forced:
            continue
        G_w = math.ceil(Wt / S)
        if G_w > MAX_GROUP_CORES or G_w > N:
            continue
        if S > 1 and math.ceil(Wt / (S - 1)) == G_w and S - 1 >= min(Wt, MIN_SLICE_TILES):
            continue  # a narrower slice gives the same G_w: never better
        G_h = min(T, N // G_w)
        rows_max = math.ceil(T / G_h)
        key = (rows_max * S, -min(rows_max, WAVES_MAX), G_w)
        cands.append((key, S, G_w, G_h, rows_max))
    best = None
    for key, S, G_w, G_h, rows_max in cands:
        base = _wave_plan("D", False, S, G_w, G_h, rows_max, False, False)
        # feasibility with the per-wave γ fallback (the smaller cb_x_gamma); local_plan prefers whole-block γ
        if footprint(sz, replace(base, gamma_mode=GAMMA_WAVE if sz.has_gamma else GAMMA_BLOCK)) > budget:
            continue
        if best is None or key < best[0]:
            best = (key, S, G_w, G_h)
    if best is None:
        for key, S, G_w, G_h, rows_max in cands:
            fit = first_fit(_wave_plan("D", False, S, G_w, G_h, rows_max, False, False), sz, budget)
            if fit is not None and (best is None or (fit[0], key) < best[0]):
                best = ((fit[0], key), S, G_w, G_h)
    if best is None:
        raise NotImplementedError(
            f"distributed_rms_norm: no W-split of Wt={Wt} fits L1 even one wave per pass at a 1-tile chunk "
            f"(budget {budget} B)"
        )
    return best[1], best[2], best[3]


def _shard_cores(spec, n):
    return list(ttnn.corerange_to_cores(spec.grid, n, row_wise=spec.orientation == ttnn.ShardOrientation.ROW_MAJOR))


def _measure_overrides() -> dict:
    """Measurement-only plan overrides (unset in production): DRMS_WAVES (wave cap), DRMS_GAMMA_MODE (0/1/2),
    DRMS_SLICE (the interleaved slice width S)."""
    import os

    ov = {}
    if os.environ.get("DRMS_WAVES"):
        ov["waves"] = int(os.environ["DRMS_WAVES"])
    if os.environ.get("DRMS_SLICE"):
        ov["slice"] = int(os.environ["DRMS_SLICE"])
    if os.environ.get("DRMS_GAMMA_MODE"):
        ov["gamma_mode"] = int(os.environ["DRMS_GAMMA_MODE"])
    return ov


def local_plan(input_tensor, workers, sz: Sizes, budget: int, out_resident=None) -> LocalPlan:
    shape = list(input_tensor.shape)
    W = shape[-1]
    Wt = math.ceil(W / TILE_DIM)
    T = tensor_tile_rows_of(shape)
    mc = input_tensor.memory_config()
    ml = mc.memory_layout

    if ml == ttnn.TensorMemoryLayout.INTERLEAVED:
        S, G_w, G_h = fused_split(T, Wt, len(workers), sz, budget)
        base, extra = divmod(T, G_h)
        members = []
        start = 0
        for g in range(G_h):
            rows = base + (1 if g < extra else 0)
            for j in range(G_w):
                w0 = j * S
                members.append(Member(workers[g * G_w + j], g, j, start, rows, w0, min(S, Wt - w0)))
            start += rows
        rows_max = base + (1 if extra else 0)
        regime = "D3" if G_w > 1 else "D1"
        sharded = False
        masked = False
    else:
        spec = mc.shard_spec
        if spec is None or mc.buffer_type != ttnn.BufferType.L1:
            raise NotImplementedError("distributed_rms_norm: sharded inputs must carry an L1 shard spec")
        shard_h, shard_w = int(spec.shape[0]), int(spec.shape[1])
        if shard_h % TILE_DIM or shard_w % TILE_DIM:
            raise NotImplementedError("distributed_rms_norm: TILE shards must be tile multiples")
        Hsh = shard_h // TILE_DIM
        S = shard_w // TILE_DIM
        sharded = True
        masked = W % TILE_DIM != 0
        members = []
        if ml == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
            if shard_w < W:
                raise NotImplementedError("distributed_rms_norm: height shard does not cover the row")
            n = math.ceil(T / Hsh)
            cores = _shard_cores(spec, n)
            for i in range(n):
                rows = min(Hsh, T - i * Hsh)
                members.append(Member(cores[i], i, 0, i * Hsh, rows, 0, Wt))
            G_w = 1
            rows_max = max(m.rows for m in members)
            regime = "D1-HS"
        elif ml == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
            if spec.orientation != ttnn.ShardOrientation.ROW_MAJOR:
                raise NotImplementedError("distributed_rms_norm: WIDTH_SHARDED needs ROW_MAJOR orientation")
            n = math.ceil(Wt / S)
            if Hsh < T:
                raise NotImplementedError("distributed_rms_norm: width shard does not cover the height")
            cores = _shard_cores(spec, n)
            for j in range(n):
                members.append(Member(cores[j], 0, j, 0, T, j * S, min(S, Wt - j * S)))
            G_w, rows_max = n, T
            regime = "D3-WS"
        else:
            raise NotImplementedError(f"distributed_rms_norm: memory layout {ml} is not built")
        worker_set = {(c.x, c.y) for c in workers}
        if any((m.core.x, m.core.y) not in worker_set for m in members):
            raise NotImplementedError("distributed_rms_norm: a shard sits on a port core (PORT-ON-SHARD, deferred)")
        if G_w > 1 and any(m.V <= 0 for m in members):
            raise NotImplementedError("distributed_rms_norm: a width shard with no valid tile")

    G_h = max(m.group for m in members) + 1
    base_plan = _wave_plan(regime, sharded, S, G_w, G_h, rows_max, masked, bool(out_resident), members)
    ov = _measure_overrides()
    if "waves" in ov:
        base_plan = rewave(base_plan, ov["waves"])
    fit = first_fit(base_plan, sz, budget)
    if fit is not None and "gamma_mode" in ov and sz.has_gamma:
        alt = replace(fit[1], gamma_mode=ov["gamma_mode"])
        if footprint(sz, alt) <= budget:
            fit = (fit[0], alt)
    if fit is None:
        smallest = replace(
            rewave(base_plan, WAVES_MAX_PASSES),
            PW=1,
            cap=1,
            combine_depth=1,
            sq_rows=1,
            gamma_mode=GAMMA_CHUNK if sz.has_gamma else GAMMA_BLOCK,
            gamma_stream=sz.has_gamma,
        )
        raise NotImplementedError(
            f"distributed_rms_norm: per-core footprint {footprint(sz, smallest)} B (one wave per pass, 1-tile chunk) "
            f"exceeds L1 budget {budget} B"
        )
    return fit[1]


def member_of(plan: LocalPlan, w: int, k: int) -> int:
    """Member index of owner k of wave w (rotated so consecutive waves land on different members)."""
    return (w * plan.owners + k) % plan.G_w


def n_wave_rows(plan: LocalPlan, rows: int, w: int) -> int:
    r0 = w * plan.q_w
    return 0 if r0 >= rows else min(plan.q_w, rows - r0)


# ---------------------------------------------------------------------------------------------------------------
# Exchange plan (TRANSPORT: fabric_ring_exchange relay, generalized to Linear)
# ---------------------------------------------------------------------------------------------------------------


def exchange_plan(G: int, ring: bool):
    """Per position s: sends[d] = [(slot q, dest pos, dest entry, hops, wait)], entries[d] = [slots landing on port d],
    ready_out[d] = [(target pos, target direction, hops)]. Relay: item 0 = own slot, item i >= 1 relays entry i-1."""
    info = [dict(sends={FWD: [], BWD: []}, entries={FWD: [], BWD: []}, ready_out={FWD: [], BWD: []}) for _ in range(G)]
    for s in range(G):
        if ring:
            kf = G // 2
            kb = G - 1 - kf
            fwd = [(s - i) % G for i in range(kf)]
            bwd = [(s + i) % G for i in range(kb)]
            nxt, prv = (s + 1) % G, (s - 1) % G
        else:
            fwd = [s - i for i in range(s + 1)] if s < G - 1 else []
            bwd = [s + i for i in range(G - s)] if s > 0 else []
            nxt = s + 1 if s < G - 1 else None
            prv = s - 1 if s > 0 else None
        for d, lst, peer in ((FWD, fwd, nxt), (BWD, bwd, prv)):
            if not lst:
                continue
            info[s]["sends"][d] = [(q, peer, i, 1, 0 if i == 0 else i) for i, q in enumerate(lst)]
            info[peer]["entries"][d] = list(lst)
    for s in range(G):
        for d in (FWD, BWD):
            for t in sorted({it[1] for it in info[s]["sends"][d]}):
                info[t]["ready_out"][_OPP[d]].append((s, d, 1))
    return info


def port_cores(grid, n_ports, taken=frozenset()):
    """n_ports free cores nearest the middle of logical row 0 (next to the Ethernet row), row by row."""
    cx = (grid.x - 1) / 2.0
    cand = [(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in taken]
    cand.sort(key=lambda c: (c[1], abs(c[0] - cx), c[0]))
    if len(cand) < n_ports:
        raise NotImplementedError("distributed_rms_norm: too few free cores for the ports (PORT-ON-SHARD)")
    sel = sorted(cand[:n_ports], key=lambda c: (c[1], c[0]))
    return [ttnn.CoreCoord(x, y) for x, y in sel]


def port_placement() -> str:
    """PORT_PLACEMENT, overridable for measurement only (DRMS_PORT_PLACEMENT; unset in production)."""
    import os

    return os.environ.get("DRMS_PORT_PLACEMENT", "") or PORT_PLACEMENT


# Nearest-Ethernet placement helpers, copied from fabric_all_gather (program_descriptor_with_inline_kernels.py
# _eth_physical_lists / _chip_maps / _noc_grid / _worker_nearest_noc1). No probe program (a second dispatch): the
# Ethernet channel is the first word setup_fabric_connection returns (WorkerToFabricEdmSender::build_from_args reads
# it as the connection-table index), channel i = the i-th non-harvested entry of the arch Ethernet list.
_ETH_CACHE = {}
_PLACEMENT_CACHE = {}


def _eth_tables(mesh):
    """(uid -> (non-harvested eth physical (x, y) list, translated column map), (noc x size, noc y size))."""
    if id(mesh) in _ETH_CACHE:
        return _ETH_CACHE[id(mesh)][1]
    import os

    import yaml

    desc = yaml.safe_load(open(ttnn.cluster.serialize_cluster_descriptor()))
    root = os.environ.get("TT_METAL_HOME", os.getcwd())
    arch_yaml = {"BLACKHOLE": "blackhole_140_arch.yaml", "WORMHOLE_B0": "wormhole_b0_80_arch.yaml"}[
        str(mesh.arch()).split(".")[-1]
    ]
    soc = yaml.safe_load(open(os.path.join(root, "tt_metal", "soc_descriptors", arch_yaml)))
    arch_eth = [tuple(int(v) for v in e.split("-")) for e in soc["eth"]]
    tensix_cols = sorted({int(w.split("-")[0]) for w in soc["functional_workers"]})
    lists = {}
    for chip, uid in desc.get("chip_unique_ids", {}).items():
        h = desc.get("harvesting", {}).get(chip, {})
        mask = int(h.get("eth_harvesting_mask", 0))
        tmask = int(h.get("harvest_mask", 0))
        live = [x for i, x in enumerate(tensix_cols) if not (tmask >> i) & 1]
        t2p = {tensix_cols[k]: px for k, px in enumerate(live)} if h.get("noc_translation", False) else {}
        lists[int(uid)] = ([e for i, e in enumerate(arch_eth) if not (mask >> i) & 1], t2p)
    out = (lists, (int(soc["grid"]["x_size"]), int(soc["grid"]["y_size"])))
    _ETH_CACHE[id(mesh)] = (mesh, out)
    return out


def _noc1_hops(mesh, t2p, noc_grid, eth_xy, xy):
    """NoC1 hops (-Y then -X, wrapping) from logical worker xy to the Ethernet core at physical eth_xy, with the
    fabric_all_gather _worker_nearest_noc1 tie-breaks (lower row, then lower column)."""
    gx, gy = noc_grid
    v = mesh.worker_core_from_logical_core(ttnn.CoreCoord(*xy))
    px = t2p.get(v.x, v.x)
    return ((v.y - eth_xy[1]) % gy + (px - eth_xy[0]) % gx, v.y, px)


def _eth_targets(mesh, cluster_axis, ring, n_links):
    """{chip coord: ([eth physical (x, y) or None per port index 2·l + d], translated column map)}."""
    lists, _ = _eth_tables(mesh)
    out = {}
    for grp in build_groups(tuple(mesh.shape), cluster_axis):
        G = len(grp)
        for p, coord in enumerate(grp):
            me = _node(mesh, coord)
            uid = int(ttnn.cluster.get_chip_unique_id_from_fabric_node_id(int(me.mesh_id), int(me.chip_id)))
            eth_list, t2p = lists.get(uid, (None, {}))
            tg = [None] * (2 * n_links)
            for d, q in ((FWD, p + 1), (BWD, p - 1)):
                if not (ring or 0 <= q < G) or eth_list is None:
                    continue
                pn = _node(mesh, grp[q % G])
                link_ids = ttnn.get_forwarding_link_indices(me, pn)
                for l in range(min(n_links, len(link_ids))):
                    conn = ttnn.setup_fabric_connection(
                        me, pn, link_ids[l], ttnn.ProgramDescriptor(), ttnn.CoreCoord(0, 0)
                    )
                    ch = int(conn[0])
                    tg[2 * l + d] = eth_list[ch] if ch < len(eth_list) else None
            out[coord] = (tg, t2p)
    return out


def chip_port_maps(mesh, grid, cluster_axis, ring, n_links, taken=frozenset(), mode=None):
    """{chip coord: [port core of (direction d, link l) at index 2·l + d]} for every chip of every group.

    "row0": port_cores() on every chip. "eth": per chip, the free core with the fewest NoC1 hops to the port's link's
    Ethernet core. "eth_uniform": one core per port index for every chip, the one with the fewest NoC1 hops summed over
    the chips. A port with no peer (a line end's missing direction) takes the next free row-0 core."""
    mode = mode or port_placement()
    coords = [c for grp in build_groups(tuple(mesh.shape), cluster_axis) for c in grp]
    if mode not in ("eth", "eth_uniform"):
        same = port_cores(grid, 2 * n_links, taken)
        return {c: same for c in coords}
    key = (id(mesh), str(ttnn.get_fabric_config()), cluster_axis, ring, n_links, tuple(sorted(taken)), mode)
    if key in _PLACEMENT_CACHE:
        return _PLACEMENT_CACHE[key][1]
    _, noc_grid = _eth_tables(mesh)
    targets = _eth_targets(mesh, cluster_axis, ring, n_links)
    free = [(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in taken]

    def fill(chosen):
        used = {c for c in chosen if c is not None}
        rest = [(c.x, c.y) for c in port_cores(grid, 2 * n_links, taken | used) if (c.x, c.y) not in used]
        return [ttnn.CoreCoord(*(c if c is not None else rest.pop(0))) for c in chosen]

    out = {}
    if mode == "eth":
        for coord in coords:
            tg, t2p = targets[coord]
            chosen = [None] * (2 * n_links)
            for i, exy in enumerate(tg):
                if exy is not None:
                    cand = [c for c in free if c not in chosen]
                    chosen[i] = min(cand, key=lambda c: _noc1_hops(mesh, t2p, noc_grid, exy, c)) if cand else None
            out[coord] = fill(chosen)
    else:
        chosen = [None] * (2 * n_links)
        for i in range(2 * n_links):
            legs = [(targets[c][0][i], targets[c][1]) for c in coords if targets[c][0][i] is not None]
            cand = [c for c in free if c not in chosen]
            if legs and cand:
                chosen[i] = min(
                    cand, key=lambda c: sum(_noc1_hops(mesh, t2p, noc_grid, exy, c)[0] for exy, t2p in legs)
                )
        same = fill(chosen)
        out = {c: same for c in coords}
    _PLACEMENT_CACHE[key] = (mesh, out)
    return out


def port_union(maps, n_used_links=None) -> list:
    """Sorted union over chips of the port cores (of the first n_used_links links, default all)."""
    s = set()
    for lst in maps.values():
        for i, c in enumerate(lst):
            if n_used_links is None or i < 2 * n_used_links:
                s.add((c.x, c.y))
    return [ttnn.CoreCoord(x, y) for x, y in sorted(s, key=lambda c: (c[1], c[0]))]


def port_exclusion(mesh, grid, cluster_axis, ring, num_links, taken, maps) -> set:
    """Logical cores kept off the worker grid: the union over chips of the port cores (identical worker count N on
    every chip). DRMS_PORT_EXCLUDE_ALL (measurement only) excludes every placement's ports, so placements A/B on one
    worker grid."""
    import os

    pset = {(c.x, c.y) for c in port_union(maps)}
    if os.environ.get("DRMS_PORT_EXCLUDE_ALL", "") not in ("", "0"):
        for mode in ("row0", "eth", "eth_uniform"):
            other = chip_port_maps(mesh, grid, cluster_axis, ring, reserved_links(num_links), taken, mode)
            pset |= {(c.x, c.y) for c in port_union(other)}
    return pset


# ---------------------------------------------------------------------------------------------------------------
# Persistent exchange resources (cold call only)
# ---------------------------------------------------------------------------------------------------------------

_RESOURCES = {}


class Resources:
    def __init__(self, mesh, ports, G, words):
        import torch  # host-side only

        self.ports = ports
        self.words = words
        crs = _crs(ports)
        self.sems = [ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(G)]  # ready + G-1 arrivals
        n_dev = mesh.get_num_devices()
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(crs, (1, words), ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.landing = ttnn.from_torch(
            torch.zeros((n_dev * len(ports), words), dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=mc,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        ttnn.synchronize_device(mesh)  # every chip's counters are zero before any chip's first program

    def release(self):
        """Free the L1 now. A dropped Resources can stay referenced (e.g. by a failed call's traceback frames) and
        would keep its landing scratch and semaphores pinned low in L1, capping every later call's CB budget."""
        if self.landing is not None:
            self.landing.deallocate(True)
            self.landing = None
        self.sems = []

    @property
    def ready_addr(self):
        return int(ttnn.get_global_semaphore_address(self.sems[0]))

    @property
    def arrival_addrs(self):
        return [int(ttnn.get_global_semaphore_address(s)) for s in self.sems[1:]]

    @property
    def landing_addr(self):
        return int(self.landing.buffer_address())


def get_resources(mesh, ports, G, num_waves, seg_stride):
    """One landing scratch + semaphore set per (mesh, port cores, G), grown to the largest request seen. Keying on the
    plan's waves / segment stride too left one L1 allocation per distinct plan alive for the process: over a long
    run those pile up and shrink the CB budget of every later call (Refinement 1)."""
    words = max(1, num_waves * G * seg_stride // 4)
    key = (id(mesh), tuple((c.x, c.y) for c in ports), G)
    res = _RESOURCES.get(key)
    # grow on demand; shrink back once a much smaller plan runs (a prefill-sized scratch must not tax every later
    # call's CB budget)
    want = max(words, LANDING_MIN_WORDS)
    if res is None or res[1].words < words or res[1].words > 4 * want:
        if res is not None:
            _RESOURCES.pop(key)
            res[1].release()  # free the old scratch before allocating its replacement (in-order queue: safe)
        res = (mesh, Resources(mesh, ports, G, want))
        _RESOURCES[key] = res
    res[1].key = key
    return res[1]


def evict_resources(mesh) -> bool:
    """Drop every cached exchange resource of this mesh (freed now, re-created on demand). True if any existed."""
    keys = [k for k, v in _RESOURCES.items() if v[0] is mesh]
    for k in keys:
        _RESOURCES.pop(k)[1].release()
    return bool(keys)


def forget_resources(res):
    """Drop cached resources whose device-side initialization never ran (see dispatch_collected)."""
    entry = _RESOURCES.pop(getattr(res, "key", None), None)
    if entry is not None:
        entry[1].release()


def collected_count() -> int:
    """Ops stashed by the up-front precompile collector (tests/plugins/up_front_collect.py). That collector runs a test
    body with dispatch blocked: a cold call made there creates its global semaphores and landing scratch without the
    device ever seeing their initial values, so they must not be cached for the real run."""
    fn = getattr(getattr(ttnn, "graph", None), "up_front_num_collected", None)
    try:
        return int(fn()) if fn is not None else 0
    except Exception:  # noqa: BLE001
        return 0


# ---------------------------------------------------------------------------------------------------------------
# Groups / links
# ---------------------------------------------------------------------------------------------------------------


def build_groups(mesh_shape, cluster_axis):
    rows, cols = mesh_shape
    if cluster_axis == 0:
        return [[(r, c) for r in range(rows)] for c in range(cols)]
    return [[(r, c) for c in range(cols)] for r in range(rows)]


def _node(mesh, coord):
    return mesh.get_fabric_node_id(ttnn.MeshCoordinate(*coord))


def group_hops(G, ring):
    hops = [(i, i + 1) for i in range(G - 1)]
    if ring and G >= 3:
        hops.append((G - 1, 0))
    return hops


def discovered_links(mesh, cluster_axis, ring):
    """min over every hop (both directions) of every group of the forwarding links; raises on a missing hop."""
    best = None
    for grp in build_groups(tuple(mesh.shape), cluster_axis):
        for a, b in group_hops(len(grp), ring):
            for s, d in ((grp[a], grp[b]), (grp[b], grp[a])):
                n = len(ttnn.get_forwarding_link_indices(_node(mesh, s), _node(mesh, d)))
                if n == 0:
                    raise ValueError(
                        f"distributed_rms_norm: no direct forwarding link {s} -> {d} "
                        f"(topology={'Ring' if ring else 'Linear'}, fabric={ttnn.get_fabric_config()})"
                    )
                best = n if best is None else min(best, n)
    return best or 0


# ---------------------------------------------------------------------------------------------------------------
# Program descriptor
# ---------------------------------------------------------------------------------------------------------------


_ABLATE_DEFINES = {
    "exchange": "DRMS_ABLATE_EXCHANGE",
    "reads": "DRMS_ABLATE_READS",  # x NoC reads skipped (CB pushes kept)
    "writes": "DRMS_ABLATE_WRITES",  # output NoC writes skipped (CB pops kept)
    "gamma": "DRMS_ABLATE_GAMMA",  # γ NoC reads skipped (CB pushes kept)
    "chains": "CKL_ELTWISE_CHAIN_SKIP_COMPUTE",  # every eltwise_chain (square, γ, inv scaling) without its math
}


def noc_swap_columns() -> int:
    """NOC_SWAP_COLUMNS, overridable for measurement only (DRMS_NOC_SWAP; unset in production)."""
    import os

    v = os.environ.get("DRMS_NOC_SWAP", "")
    return int(v) if v else NOC_SWAP_COLUMNS


def noc_swap_cores(plan) -> list:
    """Worker cores whose x reads / output writes swap NoCs (base op Perf 2): the noc_swap_columns() westmost logical
    columns (next to DRAM column 0) of an interleaved input. Sharded inputs move no x DRAM traffic: nothing swaps."""
    if plan.sharded:
        return []
    n = noc_swap_columns()
    return [m.core for m in plan.members if m.core.x < n]


def _split_noc_swapped(kernels, cores, swapped):
    """kernels with the reader / writer (kernels[0] / kernels[1]) restricted to the unswapped cores, plus same-source
    descriptors on the swapped cores with the reader on NoC1 and the writer on NoC0 (base _split_noc_swapped)."""
    if not swapped:
        return kernels
    sw = {(c.x, c.y) for c in swapped}
    normal = [c for c in cores if (c.x, c.y) not in sw]

    def sub(kd, cs, cfg):
        rt = ttnn.RuntimeArgs()
        for c in cs:
            rt[c.x][c.y] = list(kd.runtime_args[c.x][c.y])
        return ttnn.KernelDescriptor(
            kernel_source=kd.kernel_source,
            core_ranges=_crs(cs),
            compile_time_args=list(kd.compile_time_args),
            defines=list(kd.defines),
            runtime_args=rt,
            config=cfg,
        )

    reader, writer = kernels[0], kernels[1]
    out = [
        sub(
            reader,
            swapped,
            ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1),
        ),
        sub(
            writer,
            swapped,
            ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
        ),
    ]
    if normal:  # an empty-range descriptor carries no CB formats (base op: fails to compile)
        out += [sub(reader, normal, ttnn.ReaderConfigDescriptor()), sub(writer, normal, ttnn.WriterConfigDescriptor())]
    return out + kernels[2:]


def combine_lag() -> int:
    """COMBINE_LAG, overridable for measurement only (DRMS_COMBINE_LAG; unset in production)."""
    import os

    v = os.environ.get("DRMS_COMBINE_LAG", "")
    return int(v) if v else COMBINE_LAG


def fold_ahead(plan) -> int:
    """FOLD_AHEAD (combine regimes) / FOLD_AHEAD_SOLO (G_w == 1), clamped to 16 bits, overridable for measurement
    only (DRMS_FOLD_AHEAD; unset in production)."""
    import os

    v = os.environ.get("DRMS_FOLD_AHEAD", "")
    return min(int(v) if v else (FOLD_AHEAD if plan.G_w > 1 else FOLD_AHEAD_SOLO), FOLD_AHEAD_ALL)


def stats_links(num_links: int) -> int:
    """Links carrying the records (and reserving port cores): min(num_links, STATS_LINKS_MAX), overridable for
    measurement only (DRMS_STATS_LINKS; unset in production)."""
    import os

    v = os.environ.get("DRMS_STATS_LINKS", "")
    return max(1, min(num_links, int(v) if v else STATS_LINKS_MAX))


def reserved_links(num_links: int) -> int:
    """Links whose port cores are kept off the worker grid: stats_links(num_links). DRMS_RESERVE_LINKS (measurement
    only) reserves that many regardless, to separate the transport effect from the worker-count effect."""
    import os

    v = os.environ.get("DRMS_RESERVE_LINKS", "")
    return max(stats_links(num_links), int(v)) if v else stats_links(num_links)


def read_lookahead_waves() -> int:
    """READ_LOOKAHEAD_WAVES, overridable for measurement only (DRMS_LOOKAHEAD; unset in production)."""
    import os

    v = os.environ.get("DRMS_LOOKAHEAD", "")
    return int(v) if v else READ_LOOKAHEAD_WAVES


def _measurement_defines() -> list:
    """Measurement-only kernel defines (never set in production: unset env => no defines, binaries unchanged).
    DRMS_ABLATE=exchange: ports return at once, workers skip the counted gate and the landed waits."""
    import os

    ablate = {a.strip() for a in os.environ.get("DRMS_ABLATE", "").split(",") if a.strip()}
    unknown = ablate - set(_ABLATE_DEFINES)
    if unknown:
        raise ValueError(f"DRMS_ABLATE: unknown switches {sorted(unknown)}")
    defines = [(d, "1") for a, d in _ABLATE_DEFINES.items() if a in ablate]
    if os.environ.get("DRMS_PERF_ZONES", "0") not in ("", "0"):
        defines.append(("DRMS_PERF_ZONES", "1"))
    return defines


SCRATCH_ALIGN = 64  # byte alignment of each CB carved out of the scratch buffer
SCRATCH_SLACK = 4096  # headroom left in the hole for the allocator's own alignment


def l1_largest_hole(device) -> int:
    try:
        return int(ttnn.get_memory_view(device, ttnn.BufferType.L1).largest_contiguous_bytes_free_per_bank)
    except Exception:  # noqa: BLE001
        return 0


def _scratch_backed_cbs(mesh, worker_crs, n_workers, table, fmt, interm):
    """Every CB of `table` as a slice of one per-worker scratch L1 buffer (dynamic CB addresses). The scratch tensor
    must stay alive through the dispatch (the caller passes it to generic_op with the other io tensors)."""
    offsets, off = [], 0
    for _, ps, n in table:
        offsets.append(off)
        off = _align_up(off + ps * n, SCRATCH_ALIGN)
    words = max(SCRATCH_ALIGN // 4, off // 4)
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(worker_crs, (1, words), ttnn.ShardOrientation.ROW_MAJOR),
    )
    try:
        scratch = ttnn.allocate_tensor_on_device(
            ttnn.Shape([n_workers, words]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh, mc
        )
    except RuntimeError as e:
        raise NotImplementedError(f"distributed_rms_norm: no L1 hole for the {off} B CB scratch ({e})") from None
    cbs = []
    for (idx, ps, n), o in zip(table, offsets):
        d = ttnn.cb_descriptor_from_sharded_tensor(
            idx, scratch, address_offset=o, total_size=ps * n, core_ranges=worker_crs
        )
        d.format_descriptors = [
            ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt.get(idx, interm), page_size=ps)
        ]
        cbs.append(d)
    return scratch, cbs


def _cb(index, data_format, page_size, num_pages, core_ranges):
    return ttnn.CBDescriptor(
        total_size=page_size * num_pages,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


def _dtype_kind(dtype):
    return {ttnn.float32: 0, ttnn.bfloat16: 1, ttnn.bfloat8_b: 2}[dtype]


def _gamma_es(gamma):
    return ttnn.tile_size(gamma.dtype) // TILE_HW


def _out_resident(input_tensor, output_tensor) -> bool:
    """The output CB is backed on the output shard (compute packs in place) only when the output carries the input's
    own shard placement; any other output (interleaved, or a different shard spec: resharding) streams through
    cb_output to the writer's output TensorAccessor (Refinement 1)."""
    in_mc = input_tensor.memory_config()
    out_mc = output_tensor.memory_config() if output_tensor is not None else in_mc
    if in_mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED:
        return False
    if out_mc.memory_layout != in_mc.memory_layout or out_mc.buffer_type != in_mc.buffer_type:
        return False
    a, b = out_mc.shard_spec, in_mc.shard_spec
    return (
        a is not None
        and b is not None
        and list(a.shape) == list(b.shape)
        and a.grid == b.grid
        and a.orientation == b.orientation
    )


def plan_summary(
    input_tensor, *, cluster_axis, num_links, gamma=None, fp32_dest_acc_en=True, out_resident=None, ring=False
):
    """Host-only: the selected regime and its split (for tests). out_resident defaults to the input's own placement
    for a sharded input (the golden suite's sharded-output request)."""
    mesh = input_tensor.device()
    grid = mesh.compute_with_storage_grid_size()
    taken = set()
    mc = input_tensor.memory_config()
    if mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        taken = {(c.x, c.y) for r in mc.shard_spec.grid.ranges() for c in _range_cores(r)}
    G = mesh.shape[cluster_axis]
    ring = ring and G >= 3
    pset = port_exclusion(
        mesh,
        grid,
        cluster_axis,
        ring,
        num_links,
        taken,
        chip_port_maps(mesh, grid, cluster_axis, ring, reserved_links(num_links), taken),
    )
    workers = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in pset]
    sz = _sizes(input_tensor, gamma, fp32_dest_acc_en, mesh.shape[cluster_axis])
    if out_resident is None:
        out_resident = mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
    p = local_plan(input_tensor, workers, sz, l1_budget(mesh), out_resident)
    out = dict(
        regime=p.regime,
        gamma_per_wave=p.gamma_per_wave,
        S=p.S,
        G_w=p.G_w,
        G_h=p.G_h,
        rows_max=p.rows_max,
        num_waves=p.num_waves,
        workers=len(workers),
    )
    # Refinement 1 L1 fallbacks: reported only when they leave the Phase 0 plan (Phase 0 summaries are unchanged)
    if (
        p.passes > 1
        or p.cap != W_CHUNK_TILES
        or p.gamma_mode == GAMMA_CHUNK
        or p.sq_rows
        or p.combine_depth != COMBINE_DEPTH
        or p.reread
    ):
        out.update(
            passes=p.passes,
            waves_per_pass=p.PW,
            chunk=p.C,
            gamma_mode=_GAMMA_MODE_NAMES[p.gamma_mode],
            gamma_stream=p.gamma_stream,
            x_depth=p.x_depth,
            combine_depth=p.combine_depth,
            sq_rows=p.SQ,
        )
        if p.reread:
            out.update(reread=True)
    return out


def _range_cores(r):
    return [ttnn.CoreCoord(x, y) for y in range(r.start.y, r.end.y + 1) for x in range(r.start.x, r.end.x + 1)]


def _sizes(input_tensor, gamma, fp32_dest_acc_en, G):
    interm = intermediate_format(fp32_dest_acc_en)
    return Sizes(
        T_in=ttnn.tile_size(input_tensor.dtype),
        T_out=ttnn.tile_size(input_tensor.dtype),
        T_g=ttnn.tile_size(gamma.dtype) if gamma is not None else 0,
        T_i=ttnn.tile_size(interm),
        has_gamma=gamma is not None,
        G=G,
    )


def create_mesh_program_descriptor(
    input_tensor,
    output_tensor,
    *,
    gamma,
    epsilon: float,
    cluster_axis: int,
    ring: bool,
    num_links: int,
    compute_kernel_config,
):
    mesh = input_tensor.device()
    mesh_shape = tuple(mesh.shape)
    grid = mesh.compute_with_storage_grid_size()
    G = mesh_shape[cluster_axis]
    ring = ring and G >= 3  # a 2-chip ring is the line it physically is
    fp32_acc = bool(compute_kernel_config.fp32_dest_acc_en)
    interm = intermediate_format(fp32_acc)
    sz = _sizes(input_tensor, gamma, fp32_acc, G)
    shape = list(input_tensor.shape)
    W_local = shape[-1]
    Wt = math.ceil(W_local / TILE_DIM)
    w_valid_last = W_local - TILE_DIM * (Wt - 1)

    # ---- ports + worker grid ----
    mc = input_tensor.memory_config()
    taken = set()
    if mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        taken = {(c.x, c.y) for r in mc.shard_spec.grid.ranges() for c in _range_cores(r)}
    port_maps = chip_port_maps(mesh, grid, cluster_axis, ring, reserved_links(num_links), taken)
    pset = port_exclusion(mesh, grid, cluster_axis, ring, num_links, taken, port_maps)
    workers = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in pset]
    out_resident = _out_resident(input_tensor, output_tensor)
    # The plan sizes the landing scratch, and a grown scratch is an L1 buffer that lowers the CB budget: re-plan
    # against the budget left once the plan's resources exist (converges at once: the scratch only grows).
    budget = l1_budget(mesh)
    evicted = False
    # Last resort (never taken by a plan that fits the static CB region): when the gap below the lowest L1 buffer is
    # too small for any rung of the ladder — the caller's tensors sit at the bottom of L1 — the CBs are carved out of
    # a scratch L1 buffer allocated wherever the largest free hole is (dynamic CB addresses).
    scratch_cbs = False
    for _ in range(6):
        try:
            plan = local_plan(input_tensor, workers, sz, budget, out_resident)
        except NotImplementedError:
            # Cached exchange scratch allocated while larger tensors were alive can sit low in L1 and cap the CB
            # region; free it (it is re-created, top-down, on demand) and re-measure once before refusing.
            if not evicted and evict_resources(mesh):
                evicted = True
                budget = l1_budget(mesh)
                continue
            hole = l1_largest_hole(mesh) - SCRATCH_SLACK
            if scratch_cbs or hole <= budget:
                raise
            scratch_cbs = True
            budget = hole
            continue

        # ---- publishers -> links ----
        P = plan.G_h * plan.owners  # publishers per wave
        L = min(stats_links(num_links), P)
        n_pub = [len([k for k in range(P) if k % L == l]) for l in range(L)]
        seg_bytes = [n_pub[l] * plan.q * RECORD_BYTES for l in range(L)]
        seg_stride = _align_up(max(seg_bytes), SEG_ALIGN)
        ports_of = {
            coord: {(d, l): lst[2 * l + d] for l in range(L) for d in (FWD, BWD)} for coord, lst in port_maps.items()
        }
        res = get_resources(mesh, port_union(port_maps, L), G, plan.num_waves, seg_stride)
        budget_now = (l1_largest_hole(mesh) - SCRATCH_SLACK) if scratch_cbs else l1_budget(mesh)
        if footprint(sz, plan) <= budget_now:
            break
        budget = budget_now
    virt = lambda c: mesh.worker_core_from_logical_core(c)
    pack = lambda v: (v.x << 16) | v.y

    # roots of link l: every core that publishes on link l in some wave
    groups = {}
    for m in plan.members:
        groups.setdefault(m.group, {})[m.index] = m
    roots = [[] for _ in range(L)]
    for w in range(plan.num_waves):
        for g in range(plan.G_h):
            for k in range(plan.owners):
                l = (g * plan.owners + k) % L
                c = groups[g][member_of(plan, w, k)].core
                pc = pack(virt(c))
                if pc not in roots[l]:
                    roots[l].append(pc)

    xinfo = exchange_plan(G, ring)

    # ---- semaphore ids ----
    SEM_GATHER = 0
    SEM_INV = 1
    SEM_PASS = 2  # workers: members done with a pass (row-group barrier before the next pass reuses gather / inv slots)
    SEM_LANDED_BASE = 3  # + 2*l + d (workers: landed notifies per link and direction)
    n_worker_sems = SEM_LANDED_BASE + 2 * L
    # Port cores carry only the per-wave own_stats semaphores; worker and port cores are disjoint, so the ids start
    # at 0 there (semaphore ids are per core) and the wave count is bounded by NUM_SEMAPHORES alone.
    SEM_OWN_STATS_BASE = 0  # + w (ports: publishers of wave w announced)
    n_port_sems = SEM_OWN_STATS_BASE + plan.num_waves
    assert max(n_worker_sems, n_port_sems + FABRIC_PORT_SEMAPHORES) <= NUM_SEMAPHORES, "NUM_SEMAPHORES"

    worker_crs = _crs([m.core for m in plan.members])

    # ---- CBs (worker cores): cb_table() is the single source of the allocated CBs and of footprint() ----
    fmt = {
        CB_X_TILES: input_tensor.dtype,
        CB_X_REREAD: input_tensor.dtype,
        CB_REDUCE_SCALER: SCALER_FORMAT,
        CB_STATS: STATS_FORMAT,
        CB_RECORD_SCRATCH: STATS_FORMAT,
        CB_OUTPUT: output_tensor.dtype,
        CB_GAMMA_TILES: gamma.dtype if gamma is not None else None,
    }
    scratch = None
    if scratch_cbs:
        scratch, cbs = _scratch_backed_cbs(mesh, worker_crs, len(plan.members), cb_table(plan, sz), fmt, interm)
    else:
        cbs = [_cb(idx, fmt.get(idx, interm), ps, n, worker_crs) for idx, ps, n in cb_table(plan, sz)]
    if plan.sharded:
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_X_TILES, input_tensor))
    if plan.out_resident:
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_OUTPUT, output_tensor))

    # ---- compile-time args ----
    reader_ct = [
        int(plan.sharded),  # 0
        plan.G_w,  # 1
        plan.WS,  # 2
        plan.num_waves,  # 3
        plan.q_w,  # 4
        plan.owners,  # 5
        plan.q,  # 6
        G,  # 7
        L,  # 8
        sz.T_in,  # 9
        sz.T_i,  # 10
        int(interm == ttnn.float32),  # 11
        int(plan.masked),  # 12
        w_valid_last,  # 13
        _dtype_kind(input_tensor.dtype),  # 14
        SEM_GATHER,  # 15
        SEM_INV,  # 16
        SEM_LANDED_BASE,  # 17
        SEM_OWN_STATS_BASE,  # 18
        read_lookahead_waves(),  # 19
        seg_stride,  # 20
        Wt,  # 21
        gather_field_bits(plan.G_w),  # 22
        plan.PW,  # 23 waves per pass
        plan.regions,  # 24 cb_gather regions (owned waves per pass)
        SEM_PASS,  # 25
        int(plan.reread),  # 26 x re-read (streamed Phase A)
        combine_lag(),  # 27 service / chip-sum lag in waves
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())  # 28..
    compute_ct = [
        int(gamma is not None),  # 0
        int(plan.sharded),  # 1
        int(plan.G_w > 1),  # 2
        int(plan.masked),  # 3
        plan.WS,  # 4
        plan.C,  # 5
        plan.nch,  # 6
        plan.num_waves,  # 7
        plan.q_w,  # 8
        plan.owners,  # 9
        plan.q,  # 10
        G,  # 11
        plan.G_w,  # 12
        SCALE_BLOCK_TILES,  # 13
        int(plan.out_resident),  # 14
        plan.gamma_mode,  # 15
        plan.PW,  # 16
        int(plan.gamma_stream),  # 17
        plan.SQ,  # 18 tile-rows per square -> collapse round
        int(plan.reread),  # 19 x re-read
        combine_lag(),  # 20 service / chip-sum lag in waves (= the reader's)
        fold_ahead(plan),  # 21 waves an owner's fold runs ahead of its scaling
    ]
    gamma_is_tile = gamma is not None and gamma.layout == ttnn.TILE_LAYOUT
    writer_ct = [
        int(gamma is not None),  # 0
        int(plan.out_resident),  # 1
        plan.WS,  # 2
        plan.C,  # 3
        plan.nch,  # 4
        plan.num_waves,  # 5
        plan.q_w,  # 6
        Wt,  # 7
        sz.T_out,  # 8
        int(gamma_is_tile),  # 9
        _gamma_es(gamma) if gamma is not None else 0,  # 10
        sz.T_g,  # 11
        plan.Wp,  # 12
        int(plan.gamma_stream),  # 13
        int(plan.reread),  # 14 x re-read (the writer supplies cb_x_reread)
    ]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())  # 15..
    writer_ct.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if gamma is not None
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
    writer_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())  # x re-read
    port_ct = [G, plan.num_waves, int(ttnn.get_tt_fabric_max_payload_size_bytes()), MAX_HEADERS, SEM_OWN_STATS_BASE]

    compute_cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=compute_kernel_config.math_fidelity,
        fp32_dest_acc_en=compute_kernel_config.fp32_dest_acc_en,
        math_approx_mode=compute_kernel_config.math_approx_mode,
        dst_full_sync_en=compute_kernel_config.dst_full_sync_en,
    )
    if fp32_acc:
        modes = [ttnn.UnpackToDestMode.Default] * NUM_CBS
        modes[CB_STATS] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute_cfg.unpack_to_dest_mode = modes

    inv_w_bits = _fp32_bits(1.0 / (G * W_local))
    eps_bits = _fp32_bits(epsilon)
    in_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    gamma_addr = gamma.buffer_address() if gamma is not None else 0

    groups_list = [groups[g] for g in range(plan.G_h)]
    group_coords = [[pack(virt(grp[j].core)) for j in range(plan.G_w)] for grp in groups_list]

    dm_defines = _measurement_defines()
    mesh_desc = ttnn.MeshProgramDescriptor()
    for grp_coords in build_groups(mesh_shape, cluster_axis):
        for p, coord in enumerate(grp_coords):
            info = xinfo[p]
            sends_own = {d: int(len(info["sends"][d]) > 0) for d in (FWD, BWD)}
            n_active = sum(sends_own.values())
            src_dir = [2 if pos == p else (FWD if pos in info["entries"][FWD] else BWD) for pos in range(G)]
            n_in = {d: len(info["entries"][d]) for d in (FWD, BWD)}

            ports = ports_of[coord]  # this chip's port cores
            port_crs = _crs(list(ports.values()))
            port_xy = [pack(virt(ports[(d, l)])) for l in range(L) for d in (FWD, BWD)]
            program = ttnn.ProgramDescriptor()
            # Our semaphores first: setup_fabric_connection (below) appends the port's fabric teardown / buffer-index
            # semaphores at the next free ids, which must not alias the per-wave own_stats ids.
            program.semaphores = [
                ttnn.SemaphoreDescriptor(id=i, core_ranges=worker_crs, initial_value=0) for i in range(n_worker_sems)
            ] + [ttnn.SemaphoreDescriptor(id=i, core_ranges=port_crs, initial_value=0) for i in range(n_port_sems)]
            reader_rt = ttnn.RuntimeArgs()
            writer_rt = ttnn.RuntimeArgs()
            compute_rt = ttnn.RuntimeArgs()
            for m in plan.members:
                c = m.core
                masked_core = int(plan.masked and m.w0 + m.V == Wt)
                rt = [
                    in_addr,
                    m.start_row,
                    m.rows,
                    m.w0,
                    m.V,
                    m.index,
                    p,
                    res.landing_addr,
                    m.group,
                    n_in[FWD],
                    n_in[BWD],
                    int(m.w0 + m.V == Wt),  # holds the W-pad tile (interleaved zeroing)
                    masked_core,
                    n_active,
                ]
                rt += group_coords[m.group]
                rt += port_xy
                rt += [sends_own[FWD], sends_own[BWD]]
                rt += src_dir
                reader_rt[c.x][c.y] = rt
                compute_rt[c.x][c.y] = [m.rows, m.V, m.index, masked_core, inv_w_bits, eps_bits]
                writer_rt[c.x][c.y] = [out_addr, gamma_addr, m.start_row, m.rows, m.w0, m.V, in_addr]

            # ---- port args ----
            port_rt = ttnn.RuntimeArgs()
            peer_of = {
                FWD: grp_coords[(p + 1) % G] if (ring or p < G - 1) else None,
                BWD: grp_coords[(p - 1) % G] if (ring or p > 0) else None,
            }
            for l in range(L):
                for d in (FWD, BWD):
                    core = ports[(d, l)]
                    sends, ready_out, entries = info["sends"][d], info["ready_out"][d], info["entries"][d]
                    has_conn = len(sends) > 0 or len(ready_out) > 0
                    pc = virt(core)
                    dests = sorted({(dst, h) for _, dst, _, h, _ in sends} | {(t, h) for t, _, h in ready_out})
                    assert len(dests) <= MAX_HEADERS
                    root_list = roots[l]
                    n_notify = len(root_list) if len(entries) > 0 else 0
                    args = [
                        res.landing_addr,
                        seg_stride,
                        res.ready_addr,
                        len(dests),
                        len(sends),
                        len(entries),
                        n_pub[l] if len(sends) > 0 else 0,  # n_roots: publishers into my own slot per wave
                        len(ready_out),
                        len({it[1] for it in sends}),
                        pc.x,
                        pc.y,
                        n_notify,
                        int(has_conn),
                        1,  # notify quantum: per landed entry
                        SEM_LANDED_BASE + 2 * l + d,
                        len(root_list),
                    ]
                    args += res.arrival_addrs[: len(entries)]
                    for dst, h in dests:
                        dn = _node(mesh, grp_coords[dst])
                        dc = virt(ports_of[grp_coords[dst]][(d, l)])  # the destination chip's port of (d, l)
                        args += [int(dn.chip_id), int(dn.mesh_id), h, dc.x, dc.y]
                    for q_slot, dst, idx, h, wait in sends:
                        args += [q_slot, dests.index((dst, h)), res.arrival_addrs[idx], wait]
                    for tgt, td, h in ready_out:
                        rc = virt(ports_of[grp_coords[tgt]][(td, l)])
                        args += [dests.index((tgt, h)), rc.x, rc.y]
                    args += [seg_bytes[l]] * plan.num_waves
                    args += root_list
                    if has_conn:
                        peer = peer_of[d]
                        me_node, pn = _node(mesh, coord), _node(mesh, peer)
                        link_ids = ttnn.get_forwarding_link_indices(me_node, pn)
                        args += list(ttnn.setup_fabric_connection(me_node, pn, link_ids[l], program, core))
                    port_rt[core.x][core.y] = args

            kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "distributed_rms_norm_reader.cpp"),
                    core_ranges=worker_crs,
                    compile_time_args=reader_ct,
                    defines=dm_defines,
                    runtime_args=reader_rt,
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "distributed_rms_norm_writer.cpp"),
                    core_ranges=worker_crs,
                    compile_time_args=writer_ct,
                    defines=dm_defines,
                    runtime_args=writer_rt,
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "distributed_rms_norm_compute.cpp"),
                    core_ranges=worker_crs,
                    compile_time_args=compute_ct,
                    defines=dm_defines,
                    runtime_args=compute_rt,
                    config=compute_cfg,
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "distributed_rms_norm_port.cpp"),
                    core_ranges=port_crs,
                    compile_time_args=port_ct,
                    defines=dm_defines,
                    runtime_args=port_rt,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1
                    ),
                ),
            ]
            # Perf (base Perf 2 dram_noc_balance, re-measured on the fused grid): the westmost logical columns run
            # their DRAM traffic on the other NoCs (reader NoC1, writer NoC0)
            kernels = _split_noc_swapped(kernels, [m.core for m in plan.members], noc_swap_cores(plan))
            program.kernels = kernels
            program.cbs = cbs
            key = "|".join(
                str(v)
                for v in (
                    "distributed_rms_norm",
                    tuple(dm_defines),
                    coord,
                    p,
                    G,
                    ring,
                    L,
                    cluster_axis,
                    plan.regime,
                    plan.S,
                    plan.G_w,
                    plan.G_h,
                    plan.rows_max,
                    plan.num_waves,
                    plan.q_w,
                    plan.owners,
                    tuple(shape),
                    input_tensor.dtype,
                    None if gamma is None else (gamma.dtype, gamma.layout),
                    fp32_acc,
                    compute_kernel_config.math_fidelity,
                    compute_kernel_config.math_approx_mode,
                    mc.memory_layout,
                    None if mc.shard_spec is None else (tuple(mc.shard_spec.shape), str(mc.shard_spec.grid)),
                    output_tensor.memory_config().buffer_type,
                    tuple(sorted(pset)),
                    tuple((c.x, c.y) for c in port_maps[coord]),
                    plan.out_resident,
                    (
                        None
                        if output_tensor.memory_config().shard_spec is None
                        else (
                            tuple(output_tensor.memory_config().shard_spec.shape),
                            str(output_tensor.memory_config().shard_spec.grid),
                        )
                    ),
                    plan.gamma_mode,
                    plan.gamma_stream,
                    plan.PW,
                    plan.cap,
                    plan.x_depth,
                    plan.combine_depth,
                    plan.SQ,
                    plan.reread,
                    scratch_cbs,
                    res.landing_addr,
                    read_lookahead_waves(),
                    combine_lag(),
                    noc_swap_columns(),
                    fold_ahead(plan),
                )
            )
            program.custom_program_hash = int.from_bytes(hashlib.sha1(key.encode()).digest()[:8], "little") >> 1
            mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(*coord), ttnn.MeshCoordinate(*coord))] = program
    return mesh_desc, res, ([scratch] if scratch is not None else [])
