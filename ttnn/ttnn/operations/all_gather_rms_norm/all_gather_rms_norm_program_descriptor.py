# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""all_gather_rms_norm MeshProgramDescriptor (op_design.md -> Blocking Model, regime R1 mirror_resident_gather).

Per chip: one ProgramDescriptor. Roles:
  * port cores (ports_union, one per (direction, link) per chip, placed next to the link's Ethernet core):
      port_reader (NCRISC / NoC0) reads outgoing runs from the compute cores' cb_x (own slice after the own members
      loaded it, relays after the upstream entry landed) into cb_port_stage and notifies the members of every
      landed incoming entry; port_sender (BRISC / NoC1) runs the ready fence and sends every run to the SAME core
      and cb_x offset on the neighbour (mirror layout), one fused arrival increment per entry.
  * compute cores (worker grid - ports_union, row-major; row-group g = cores [g*k, g*k + k)): the base op's
      reader / compute / writer, extended: own members read their slice of x from the input, remote members wait
      for the landing; the k members of a row-group combine their per-row moments over the NoC (symmetric
      all-gather for k <= 8; per-row owner with row-form partials beyond, base op R3), normalize their block and store
      it.

Knobs (single source here, handed to the kernels as CT / RT args): w_split_cores k (select_split), core_row_tiles
(the whole per-group assignment, one block), wave_row_groups (plan_waves), port_batch_chunks, chunk_pages (fabric
payload), relay_lag_waves, out_depth / out_unit_tiles.
"""

from __future__ import annotations

import hashlib
import math
import os
import struct
from dataclasses import dataclass, field, replace
from pathlib import Path

import ttnn

from .all_gather_rms_norm_fabric_plan import DIRS, OPP, plan_fabric, schedule

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE = 32

# ---- CB indices (must match kernels/all_gather_rms_norm_common.hpp) ----
CB_X = 0  # FIRST descriptor: landing slot + resident member block (mirror address on every core of every chip)
CB_STAGE = 1  # port cores: outgoing packet staging ring
CB_GAMMA = 2
CB_BETA = 3
CB_ONES = 4  # 1.0 reduce scaler (bf16)
CB_MASK = 5  # LN, W_local % 32 != 0: 0/1 row-0 mask of a slice-last tile (bf16)
CB_MOMENT_A = 6  # rms: per tile-row DEST-accumulated sum x^2; ln: sum x (pass A)
CB_MOMENT_B = 7  # ln: per tile-row (sum c', sum c'^2) (pass B)
CB_KM = 8  # ln: member pivot K_m (bf16)
CB_PARTIAL = 9  # this member's per-row moments (rms: sumsq; ln: mean_m, M2_m)
CB_GATHERED = 10  # k > 1: every member's moments per row
CB_ROWSTAT = 11  # rms: inv_rms; ln: inv_std
CB_OUT = 12
CB_MSTAT = 13  # ln: column-0 (D1, D2) per row
CB_SHIFT = 14  # ln: -(mu - K_g) * inv_std
CB_KG = 15  # ln: K_g = bf16(mu)
CB_MU = 16  # ln: mu (fp32 scratch between the two combine windows)
CB_AFFINE_STAGE = 17  # writer scratch: gamma / beta row-0 staging
# dense assembly (W_local % 32 != 0): output tiles straddle slices -> 0/1 shift-matmul into the dense layout
CB_NORM = 18  # normalized member block (slice-tile space), output dtype
CB_SHIFT_MATS = 19  # 0/1 shift matrices (output dtype), built by the reader
CB_FOREIGN = 20  # contributions of peer members to this member's home output tiles (lockstep slot address)
CB_CONTRIB = 21  # this member's contributions to peers' home output tiles
CB_CENT = 22  # ln FPU path: centered block c' = x - K_m (accumulator format)
# owner combine (k > K_SYM_MAX, base op R3 row-partial topology): row r of a block is owned by member r % k
CB_OWN_RET = 23  # owner -> writer: combined row statistics of the owned rows (rms: inv_rms; ln: inv_std, shift)
CB_OWN_KG = 24  # ln owner -> writer: K_g = bf16(mu) of the owned rows (bf16)
CB_DEV = 25  # ln owner scratch: per gather tile (n_m d_m, M2_m + n_m d_m^2) in row form
CB_WCOL = 26  # ln owner: member weights n_m, column 0 of row m % 32 of gather tile m // 32 (accumulator format)
CB_XG = 27  # rms with gamma, XG_FIRST: x * gamma of the member block (accumulator format)
CB_OWN = 28  # ln, OWN_PIVOT: the reader's contiguous copy of the member's own slice (input format)
CB_PORT_HDR = (
    29  # port cores: the sender's packet-header ring (HDR_RING slots of HDR_STRIDE bytes; raw L1, sender only)
)
NUM_CBS = 32

# ---- semaphores (program-local; declared BEFORE setup_fabric_connection takes its ids) ----
SEM_LANDED = 0  # compute cores: landed entries covering this member (port reader -> member reader)
SEM_PARTIALS = 1  # compute cores: partial exchange arrivals (base op)
SEM_OWN_LOADED = 2  # port cores: own members that loaded their block (member reader -> port reader)
SEM_FOREIGN = 3  # compute cores: contributors that delivered their foreign tiles (dense assembly)
SEM_INV = 4  # compute cores, owner combine: returned row statistics (one increment per returned row)

# ---- knobs (op_design.md: catalog / example measurements) ----
INC_EVERY = int(
    os.environ.get("AGRMS_INC_EVERY", "8")
)  # packets per fabric entry (fabric_all_gather: increment every 8 chunks)
PORT_BATCH_CHUNKS = int(os.environ.get("AGRMS_BATCH", "2"))  # cap: packets per sender flush + pop (batch-mode reader:
# per read barrier); cb_port_stage holds STAGE_BATCHES batches (halved under L1 pressure: port_batch_candidates).
# Refinement 3: 2 measured best on both perf cases (4: +0.25 / +0.56 us, 1: +1.9, 3: flat, 6: +0.9)
RELAY_LAG_WAVES = int(os.environ.get("AGRMS_RELAY_LAG", "1"))  # own entries lead relay entries by this many waves
# (Refinement 4: 0 -> +5.0 us, 2 -> +1.2 us on the focus case)
# last wave's row-groups: member count raised this many candidate steps (tail_split; 0 = uniform split)
TAIL_K_STEPS = int(os.environ.get("AGRMS_TAIL_K_STEPS", "0"))
# writer: gamma / beta pages completed BEFORE the partial exchange (off the post-landing chain; 0 = after it, base op)
AFFINE_FIRST = int(os.environ.get("AGRMS_AFFINE_FIRST", "1"))
# rms with gamma: x * gamma computed while the partials are exchanged, one multiply by inv_rms after the combine
# (only when the chosen split still fits L1 with the block-sized cb_xg: the split itself never changes for it)
XG_FIRST = int(os.environ.get("AGRMS_XG_FIRST", "0"))  # measured +0.26 us on the focus case (Refinement 4)
# port reader: an own entry waits only for the own members it reads (one bit per member in sem_own_loaded, <= 31
# members per port) instead of every own member of the port (0 = wait-all)
OWN_LOADED_MASK = int(os.environ.get("AGRMS_OWN_MASK", "1"))
OWN_MASK_BITS = 31
OWN_MASK_FLAG = 1 << 31  # port reader RT: own entry wait word = flag | member mask
# relay lag of the last wave (build_entries send_slot; RELAY_LAG_WAVES = the uniform schedule)
TAIL_RELAY_LAG = int(os.environ.get("AGRMS_TAIL_RELAY_LAG", str(RELAY_LAG_WAVES)))
STAGE_BATCHES = int(os.environ.get("AGRMS_STAGE_BATCHES", "2"))  # cb_port_stage ring depth in sender batches
PORT_READ_STREAM = int(os.environ.get("AGRMS_READ_STREAM", "1"))  # port reader: per-slot transaction ids, every
# landed slot pushed at once, the whole stage ring in flight (TRANSPORT, Refinement 3); 0 = batch mode below
# ---- short-run transport (TRANSPORT, Refinement 6) ----
# When the member runs of the outgoing entries are short (narrow slices: the indexer case's 1-tile runs, one per
# (row, slice) and compute core), one packet per run makes the port's per-packet work — the sender's header build +
# issue (~0.2 us per packet measured) and the reader's per-slot turnaround — the transport's rate, and each relay
# entry waits behind it. The regime is chosen on the host from the actual entries: it is on when scatter packing
# (<= 4 runs per packet) at least halves the packet count (SHORT_RUN_RATIO). It turns on together (each knob below:
# -1 = auto = on in the short-run regime only, else 0 / the env value everywhere):
#   PORT_PACK      packets span the runs of an entry (fabric scatter write, <= 4 pieces, full chunk_pages packets)
#   SHORT_RUN_BATCH  sender flush + stage-slot release per packet (batch 1) instead of per PORT_BATCH_CHUNKS
#   HDR_RING       every packet's header built ahead of its send (fence / stage / credit waits) into a ring slot;
#                  0 = a header built right before each send (the original sender)
#   PORT_STATEFUL  stateful router sends (command buffers programmed once after the fence) + cached router credits
# Case1 (640x32 LN, 1-tile runs, eval ccl_perf fastest / slowest chip): 12.02 / 12.73 -> 11.42 / 12.23 us with all
# four; pack alone 12.09 (flat), ring without pack +1.1, batch 1 without pack +0.9. Long runs (the focus case's
# 12-tile runs) keep the original path: there pack measured +0.4 and batch 1 +1.9 (Refinement 3).
SHORT_RUN_RATIO = 2
PORT_PACK = int(os.environ.get("AGRMS_PACK", "-1"))
SHORT_RUN_BATCH = int(os.environ.get("AGRMS_SHORT_BATCH", "1"))
HDR_RING = int(os.environ.get("AGRMS_HDR_RING", "-1"))
HDR_RING_AUTO = 16  # ring slots in the short-run regime (8 measured equal; >= the batch)
HDR_STRIDE = 128  # bytes per header slot (>= sizeof(PACKET_HEADER_TYPE), checked by the kernel)
# slot templates: the RISC copies the routed header into a slot at its first build, inside the waits (1), or a DMA
# fills every slot after the connection opens (0: -0.2 us worse on case1, the barrier sits before the fence wait)
HDR_COPY = int(os.environ.get("AGRMS_HDR_COPY", "1"))
PORT_STATEFUL = int(os.environ.get("AGRMS_STATEFUL", "-1"))
PORT_PUSH_PACKETS = int(os.environ.get("AGRMS_PUSH", "0")) or None  # batch mode: packets per read barrier + push
# (None = the batch; 1 measured +3.6 us on the focus case: one 14 KB L1 read in flight is latency-bound)
# ln FPU path (fp32 DEST, k <= 8): the member pivot is the bf16 mean of the member's OWN slice, reduced from a reader
# copy (cb_own) while the remote slices are in flight, instead of the whole block after the last landing (OVERLAP,
# Refinement 5). Enabled only when the split chosen without it still fits L1 with cb_own (the split never changes for
# it); members without an own slice keep the post-landing block pivot. 0 = block pivot everywhere.
OWN_PIVOT = int(os.environ.get("AGRMS_OWN_PIVOT", "1"))
# ln FPU normalize: cap on the output tiles per DEST block (one init per chain step per block; Refinement 5:
# case1 12.42 -> 12.07 us at 4); 0 = the writer unit. The block actually used is norm_block_of() (wrap-safe).
NORM_BLOCK = int(os.environ.get("AGRMS_NORM_BLOCK", "0"))
OUT_DEPTH = 2
OUT_UNIT_CAP = int(
    os.environ.get("AGRMS_OUT_UNIT", "4")
)  # writer batch: small enough that the stores overlap the normalize of the same tile-row
COMBINE_TILE_COST = 3  # base op: per tile-row collapse + exchange + combine cost (x-tile equivalents)
OCC_TILES_PER_MEMBER = 4  # base op occupancy floor
K_SYM_MAX = 8  # symmetric partial all-gather cap (base op K_SYM_MAX_INTERLEAVED); owner combine beyond
# owner combine (base op R3 / Perf 2 row_partial): a member's per-row partial travels in ROW form (two 16-datum face
# rows) into row m % 32 of gather tile m // 32 of the row's owner -> ceil(k / 32) gather tiles per owned row, not k.
OWNER_MEMBER_COST = 0.4  # base op: per gathered partial on the owner (combine + return), x-tile equivalents
ROW_GATHER = 32  # members per gather tile (one 32-datum row each)
# k cap: the writer carries one (x, y) pair per member as runtime args (max_runtime_args = 341 per kernel)
K_CAP_MAX = 128
# perf probe only: tile-rows per row-group at least (Refinement 6: case1 k = 1 at 4 rows per group +8.6 us, k = 4 at
# 1 / 2 / 4 rows +8.2 / +17 / +37 us — the narrow LN chain is per-row bound, so contiguous multi-tile runs cannot be
# bought with fewer, taller row-groups)
GROUP_ROWS_MIN = int(os.environ.get("AGRMS_GROUP_ROWS", "1"))
FORCE_K = (
    int(os.environ["AGRMS_FORCE_K"]) if os.environ.get("AGRMS_FORCE_K") else None
)  # tests / perf probes only: pin the member count (must be a fitting slice-aligned candidate)
LAST_INFO = None  # tests / probes: the split of the last built descriptor (k, owner, batches, ...)
AFFINE_SLOT_BYTES = 128  # one staged gamma / beta "stick" (32 datums, fp32 at most)


# Compact cb_gamma / cb_beta pages (base op Perf 2 "Gamma residency", ported): compute reads gamma / beta only
# ROW-broadcast (row 0 of faces 0 / 1; the unpacker addresses face 1 at page + face_bytes, a format constant), so the
# page STRIDE can be the smallest multiple of 64 B (DRAM reads keep their source offset mod 64 B) whose per-page
# write footprint [0, 64) u [F, F + 64) is disjoint across pages: 192 B for bf16 / fp32 (vs 2048 / 4096 B). The last
# page's unpacker read runs 2F - S bytes past its stride: `tail` extra pages keep it inside the CB.
GAMMA_READ_ALIGN = 64


def gamma_page_layout(g_dtype):
    """(cb_gamma page stride bytes, tail pages) for a gamma / beta of `g_dtype` (fp32 / bf16)."""
    face = (TILE * TILE * _datum_bytes(g_dtype)) // 4
    full = 2 * face
    foot = [(0, GAMMA_READ_ALIGN), (face, GAMMA_READ_ALIGN)]  # bytes one page's loader may write

    def disjoint(s):
        for j in range(1, full // s + 2):
            for o1, l1 in foot:
                for o2, l2 in foot:
                    a, b = o2 + j * s, o2 + j * s + l2
                    if a < o1 + l1 and o1 < b:
                        return False
        return True

    for s in range(GAMMA_READ_ALIGN, full, GAMMA_READ_ALIGN):
        if disjoint(s):
            return s, _div_up(full - s, s)
    return full, 0


def _f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def _div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


def acc_dtype(fp32_dest_acc_en: bool):
    return ttnn.float32 if fp32_dest_acc_en else ttnn.bfloat16


# ---------------------------------------------------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class Geometry:
    G: int
    W_local: int
    W: int
    H: int
    num_images: int
    image_row_tiles: int
    trt: int  # tensor tile-rows (leading dims folded)
    sw: int  # slice width in tiles
    v: int  # valid columns of a slice-last tile
    out_w_tiles: int


def derive_geometry(shape, G) -> Geometry:
    shape = [int(d) for d in shape]
    W_local, H = shape[-1], shape[-2]
    num_images = math.prod(shape[:-2]) if len(shape) > 2 else 1
    irt = _div_up(H, TILE)
    sw = _div_up(W_local, TILE)
    W = G * W_local
    return Geometry(
        G=G,
        W_local=W_local,
        W=W,
        H=H,
        num_images=num_images,
        image_row_tiles=irt,
        trt=num_images * irt,
        sw=sw,
        v=W_local - TILE * (sw - 1),
        out_w_tiles=_div_up(W, TILE),
    )


# ---------------------------------------------------------------------------------------------------------------------
# Member mapping (source slice -> members): k = G * k_s (k_s members per slice) or k = G / d (d slices per member)
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class MemberSpec:
    parts: list  # [(q, c0, n, off)]: slice q, slice-local tiles [c0, c0 + n) at row offset `off` of cb_x
    core_w: int  # valid tiles per row
    valid_cols: int  # valid datums per row (n_m)


@dataclass
class MemberMap:
    k: int
    pitch: int  # cb_x tiles per tile-row (uniform)
    members: list


def member_map(k: int, geo: Geometry) -> MemberMap:
    G, sw, Wl = geo.G, geo.sw, geo.W_local
    members = []
    if k >= G:
        ks = k // G
        mwk = _div_up(sw, ks)
        for q in range(G):
            for i in range(ks):
                c0 = i * mwk
                n = min(mwk, sw - c0)
                assert n >= 1
                cols = min(TILE * (c0 + n), Wl) - TILE * c0
                members.append(MemberSpec([(q, c0, n, 0)], n, cols))
        pitch = mwk
    else:
        d = G // k
        for m in range(k):
            parts = [(m * d + j, 0, sw, j * sw) for j in range(d)]
            members.append(MemberSpec(parts, d * sw, d * Wl))
        pitch = d * sw
    assert len(members) == k
    return MemberMap(k, pitch, members)


def _k_candidates(geo: Geometry, N: int):
    G, sw = geo.G, geo.sw
    cands = set()
    for ks in range(1, sw + 1):
        mwk = _div_up(sw, ks)
        cands.add(G * _div_up(sw, mwk))  # normalized: every member owns >= 1 tile
    for d in range(1, G + 1):
        if G % d == 0:
            cands.add(G // d)
    return sorted(c for c in cands if c <= min(K_CAP_MAX, N))


def is_owner(k: int) -> bool:
    """Combine topology: symmetric all-gather up to K_SYM_MAX members, per-row owner beyond (base op R3)."""
    return k > K_SYM_MAX


def gather_tiles(k: int) -> int:
    """Owner combine: row-form gather tiles per owned row (32 members per tile)."""
    return _div_up(k, ROW_GATHER)


# ---------------------------------------------------------------------------------------------------------------------
# CB plan (single source of truth for footprint() and the CB declarations; l1_ledger.md rows)
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Sizes:
    T_in: int
    T_out: int
    T_g: int
    T_acc: int
    T_bf16: int
    is_ln: bool
    has_gamma: bool
    has_beta: bool
    mask: bool
    ln_fpu: bool = False
    m16: bool = False  # LN with 16-bit DEST: fp32-LREG moments + combine (no DEST-accumulated moment scratch)
    g_tail: int = 0  # compact gamma: extra pages covering the last page's unpacker read extent


def make_sizes(x_dtype, g_dtype, fp32_dest, is_ln, has_gamma, has_beta, mask) -> Sizes:
    g_stride, g_tail = gamma_page_layout(g_dtype) if has_gamma else (0, 0)
    return Sizes(
        T_in=ttnn.tile_size(x_dtype),
        T_out=ttnn.tile_size(x_dtype),
        T_g=g_stride,  # compact page stride (not a whole tile)
        g_tail=g_tail,
        T_acc=ttnn.tile_size(acc_dtype(fp32_dest)),
        T_bf16=ttnn.tile_size(ttnn.bfloat16),
        is_ln=is_ln,
        has_gamma=has_gamma,
        has_beta=has_beta,
        mask=mask,
        m16=is_ln and not fp32_dest,
    )


def out_unit(pitch: int) -> int:
    return min(pitch, OUT_UNIT_CAP)


def port_batch_candidates():
    """Port packets per batch from the cap down (cb_port_stage = 2 batches): the largest that fits L1 is used."""
    pb = PORT_BATCH_CHUNKS
    while True:
        yield pb
        if pb == 1:
            return
        pb = max(1, pb // 2)


def hdr_ring_bytes() -> int:
    """Port-core L1 of the sender's header ring at its largest (the regime is chosen after the split)."""
    return max(HDR_RING, HDR_RING_AUTO) * HDR_STRIDE


def entry_packets(runs, chunk_pages: int, pack: bool) -> int:
    """Packets of one outgoing entry (runs = tile counts in order): mirror of agrms_pack::EntryPacker."""
    max_pieces = 4 if pack else 1  # NOC_SCATTER_WRITE_MAX_CHUNKS
    packets = r = t = 0
    while r < len(runs):
        tiles = pieces = 0
        while r < len(runs) and tiles < chunk_pages and pieces < max_pieces:
            take = min(runs[r] - t, chunk_pages - tiles)
            tiles += take
            t += take
            pieces += 1
            if t == runs[r]:
                r, t = r + 1, 0
        packets += 1
    return packets


def short_run_transport(out_entries: dict, chunk_pages: int) -> bool:
    """The short-run regime: scatter packing at least halves the packets of the busiest port."""
    worst = (0, 0)
    for entries in out_entries.values():
        runs = [[r[2] for r in e["runs"]] for e in entries]
        plain = sum(entry_packets(rr, chunk_pages, False) for rr in runs)
        packed = sum(entry_packets(rr, chunk_pages, True) for rr in runs)
        worst = max(worst, (plain, packed))
    return worst[0] > 0 and worst[0] >= SHORT_RUN_RATIO * worst[1]


def _auto(knob: int, on: bool, value: int = 1) -> int:
    return knob if knob >= 0 else (value if on else 0)


def stage_pages(batch: int, chunk_pages: int) -> int:
    return STAGE_BATCHES * batch * chunk_pages


def out_unit_candidates(pitch: int):
    """Writer batch (cb_out unit) from the cap down: the largest that fits L1 is used."""
    u = out_unit(pitch)
    while True:
        yield u
        if u == 1:
            return
        u = max(1, u // 2)


def cb_plan(sz: Sizes, k: int, pitch: int, rows: int, n_affine_slots: int, dense=None, unit=None) -> dict:
    """{cb: (pages, page_bytes)} of the compute cores (cb_x included)."""
    n_stat = 2 if sz.is_ln else 1
    out_pages = OUT_DEPTH * (unit or out_unit(pitch))
    if dense is None:
        # the aligned stream never holds more than the member block (per-tile reserve in compute)
        out_pages = min(out_pages, rows * pitch)
    plan = {
        CB_X: (rows * pitch, sz.T_in),
        CB_ONES: (1, sz.T_bf16),
        CB_PARTIAL: (n_stat * rows, sz.T_acc),
        CB_ROWSTAT: (rows, sz.T_acc),
        CB_OUT: (out_pages, sz.T_out),
    }
    if (
        not sz.ln_fpu and not sz.m16
    ):  # rms: sum x^2 tiles; ln SFPU path: sum x tiles (the LN FPU path reduces cb_x straight to cb_km)
        plan[CB_MOMENT_A] = (rows, sz.T_acc)
    if k > 1 and not is_owner(k):
        plan[CB_GATHERED] = (n_stat * k * rows, sz.T_acc)
    if is_owner(k):
        # uniform over members (lockstep slot base); owned rows of member m: r = m, m + k, ...
        owned = _div_up(rows, k)
        gp = gather_tiles(k)
        plan[CB_GATHERED] = (owned * n_stat * gp, sz.T_acc)
        plan[CB_OWN_RET] = (owned * n_stat, sz.T_acc)  # rms: inv_rms; ln: (inv_std, shift) per owned row
        if sz.is_ln:
            plan[CB_OWN_KG] = (owned, sz.T_bf16)
            plan[CB_WCOL] = (gp, sz.T_acc)
            if not sz.m16:
                plan[CB_DEV] = (2 * gp, sz.T_acc)
    if sz.has_gamma:
        plan[CB_GAMMA] = (pitch + sz.g_tail, sz.T_g)  # compact row-0 pages (gamma_page_layout)
        plan[CB_AFFINE_STAGE] = (n_affine_slots, AFFINE_SLOT_BYTES)
    if sz.has_beta:
        plan[CB_BETA] = (pitch + sz.g_tail, sz.T_g)
    if sz.is_ln:
        plan[CB_SHIFT] = (rows, sz.T_acc)
        plan[CB_KG] = (rows, sz.T_bf16)
        if sz.mask:
            plan[CB_MASK] = (1, sz.T_bf16)
    if sz.m16:  # row-form statistics -> column form (unpack transpose round trip): K_g, inv_std, shift of one row
        plan[CB_MU] = (3, sz.T_acc)
    if sz.is_ln and not sz.m16:  # DEST-accumulated moments scratch (fp32 DEST only)
        plan[CB_MOMENT_B] = ((1 if sz.ln_fpu else 2) * rows, sz.T_acc)  # FPU: D2 tiles; SFPU: (D1, D2) pairs
        # owner: cb_km also carries the owner's pivot K and K_g = bf16(mu) after the member pivots are popped
        plan[CB_KM] = (max(rows, 2) if is_owner(k) else rows, sz.T_bf16)
        plan[CB_MSTAT] = (2 * rows, sz.T_acc)  # (D1, D2); own pivot at k = 1: + K_m staged ahead (own_pivot)
        plan[CB_MU] = (2 if is_owner(k) else 1, sz.T_acc)  # owner: (mu, inv_std) between its two windows
        if sz.ln_fpu:
            plan[CB_CENT] = (rows * pitch, sz.T_acc)
    if dense is not None:
        plan[CB_NORM] = (rows * pitch, sz.T_out)
        plan[CB_SHIFT_MATS] = (max(1, dense.n_mats), sz.T_out)
        if dense.max_fin:
            plan[CB_FOREIGN] = (rows * dense.max_fin, sz.T_out)
        if dense.max_fout:
            plan[CB_CONTRIB] = (rows * dense.max_fout, sz.T_out)
    return plan


def cb_alias_groups(sz: Sizes) -> list:
    """CBs with disjoint lifetimes sharing one L1 region (one CBDescriptor, several buffer indices). All are
    compute -> compute (never written over the NoC, so no peer can land in them early):
      * ln moments scratch (popped by the end of member_moments) with the owner combine's cb_dev: SFPU path
        cb_moment_a (pass A, popped by its row reduce before pass B packs) + cb_moment_b; FPU path cb_cent (cb_moment_b
        is live together with cb_cent there);
      * ln cb_mstat (popped by the member finalize) with cb_mu (combine)."""
    if not sz.is_ln:
        return []
    moments = (CB_CENT, CB_DEV) if sz.ln_fpu else (CB_MOMENT_A, CB_MOMENT_B, CB_DEV)
    return [moments, (CB_MSTAT, CB_MU)]


def alias_regions(plan: dict, sz: Sizes) -> list:
    """[(indices present, region bytes)] for every alias group with >= 1 member in `plan`. The region is rounded to a
    whole number of cb_dev / cb_mu blocks (they cycle per owned row: a block must never straddle the region end)."""
    regions = []
    for grp in cb_alias_groups(sz):
        idx = [c for c in grp if c in plan]
        if not idx:
            continue
        nbytes = max(plan[c][0] * plan[c][1] for c in idx)
        for c in (CB_DEV, CB_MU):
            if c in idx:
                q = plan[c][0] * plan[c][1]
                nbytes = _div_up(nbytes, q) * q
        regions.append((idx, nbytes))
    return regions


def plan_bytes(plan: dict, sz: Sizes = None) -> int:
    regions = alias_regions(plan, sz) if sz is not None else []
    aliased = {c for idx, _ in regions for c in idx}
    return sum(p * b for c, (p, b) in plan.items() if c not in aliased) + sum(n for _, n in regions)


def n_affine_slots(mm: MemberMap, geo: Geometry) -> int:
    # sticks touched by one member: per part ceil + 1 (non-aligned shift spans one extra stick)
    return max(sum(n + 1 for (_, _, n, _) in m.parts) for m in mm.members) + 1  # + 1: 64 B alignment slack


def cb_l1_budget(device) -> int:
    """Free L1 between the allocator base and the lowest allocated L1 buffer (base op)."""
    try:
        view = ttnn.get_memory_view(device, ttnn.BufferType.L1)
        allocated = [int(b["address"]) for b in view.block_table if b["allocated"] == "yes"]
        # block addresses are relative to the allocator base, where the CB region starts
        return min(allocated) if allocated else int(view.total_bytes_per_bank)
    except Exception:  # noqa: BLE001
        return int(ttnn.get_max_worker_l1_unreserved_size()) - 80 * 1024


# ---------------------------------------------------------------------------------------------------------------------
# Split selection (base op rule re-run on the fused grid)
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class Split:
    k: int
    mm: MemberMap
    num_groups: int
    group_rows: list
    group_row_start: list
    rows_max: int
    # tail class (OVERLAP, Refinement 4): row-groups [tail_g0, num_groups) — the last wave's — use tail_k members
    # (narrower member blocks: the moments + normalize + store chain after the LAST landing shrinks with them).
    # tail_k == 0: uniform split.
    tail_k: int = 0
    tail_mm: MemberMap = None
    tail_g0: int = 0

    def is_tail(self, g: int) -> bool:
        return self.tail_k > 0 and g >= self.tail_g0

    def k_of(self, g: int) -> int:
        return self.tail_k if self.is_tail(g) else self.k

    def mm_of(self, g: int) -> MemberMap:
        return self.tail_mm if self.is_tail(g) else self.mm

    def core0(self, g: int) -> int:
        """Index of group g's first member in the compute-core list (groups are contiguous runs of k_of(g))."""
        if not self.is_tail(g):
            return g * self.k
        return self.tail_g0 * self.k + (g - self.tail_g0) * self.tail_k

    def cores_used(self) -> int:
        return self.core0(self.num_groups - 1) + self.k_of(self.num_groups - 1)

    def classes(self):
        """[(k, mm, groups)] member classes (one kernel set each)."""
        if self.tail_k == 0:
            return [(self.k, self.mm, list(range(self.num_groups)))]
        out = []
        if self.tail_g0 > 0:
            out.append((self.k, self.mm, list(range(self.tail_g0))))
        out.append((self.tail_k, self.tail_mm, list(range(self.tail_g0, self.num_groups))))
        return out


def select_split(geo: Geometry, N: int, fits_fn) -> Split:
    """k = argmin over slice-aligned candidates of rows(k) * (member_w(k) + COMBINE_TILE_COST [k > 1]),
    members keeping >= OCC_TILES_PER_MEMBER tiles (the smallest fitting candidate always admitted); narrow rows
    (G * sw < 2 * OCC) take the smallest fitting candidate. Ties -> smaller k."""
    trt = geo.trt
    cands = _k_candidates(geo, N)

    def groups(k):
        return min(trt, N // k, max(1, trt // GROUP_ROWS_MIN))

    def rows(k):
        return _div_up(trt, groups(k))

    fitting = [k for k in cands if groups(k) >= 1 and fits_fn(k, rows(k))]
    if not fitting:
        raise RuntimeError(
            f"all_gather_rms_norm: OOM: no member split fits L1 (the credited streaming regime is deferred): "
            f"candidates "
            f"{cands}, {getattr(fits_fn, 'report', lambda: '')()}"
        )
    if FORCE_K is not None:
        assert FORCE_K in fitting, f"FORCE_K={FORCE_K} not in fitting candidates {fitting}"
        k = FORCE_K
    elif geo.G * geo.sw < 2 * OCC_TILES_PER_MEMBER:
        k = min(fitting)
    else:
        admitted = {min(fitting)} | {k for k in fitting if member_map(k, geo).pitch >= OCC_TILES_PER_MEMBER}

        def cost(k):
            c = rows(k) * (member_map(k, geo).pitch + (COMBINE_TILE_COST if k > 1 else 0))
            if is_owner(k):  # the busiest owner gathers k partials for each of its ceil(rows / k) rows
                c += _div_up(rows(k), k) * k * OWNER_MEMBER_COST
            return c

        k = min(sorted(admitted), key=lambda kk: (cost(kk), kk))
    ng = groups(k)
    base, rem = divmod(trt, ng)
    g_rows = [base + (1 if g < rem else 0) for g in range(ng)]
    starts = [sum(g_rows[:g]) for g in range(ng)]
    return Split(k, member_map(k, geo), ng, g_rows, starts, max(g_rows))


# ---------------------------------------------------------------------------------------------------------------------
# Transport entries (waves x shards per port)
# ---------------------------------------------------------------------------------------------------------------------


def member_runs(spec: MemberSpec, q: int, rows: int, pitch: int):
    """cb_x tile runs of member `spec` holding slice q: [(tile_offset, ntiles)]."""
    for qq, c0, n, off in spec.parts:
        if qq != q:
            continue
        if n == pitch and off == 0:
            return [(0, rows * pitch)]
        return [(r * pitch + off, n) for r in range(rows)]
    return []


def tail_split(split: Split, geo: Geometry, N: int, wave_rg: int, fits_fn, dense: bool) -> Split:
    """Last-wave narrowing (OVERLAP lamp "tail chain block size"): the row-groups of the last wave land last, and the
    whole chain after their landing (member moments, partial exchange, combine, normalize, store) scales with the
    member block. Give those groups the TAIL_K_STEPS-th wider member count (k_s x 2 on the focus case: 12 -> 6 tiles
    per member) when the cores, the symmetric combine and L1 allow it; every earlier wave keeps the cost-rule k (its
    tail is hidden behind the later waves' transport). Uniform split otherwise."""
    if TAIL_K_STEPS <= 0 or dense or FORCE_K is not None:
        return split
    k = split.k
    n_waves = _div_up(split.num_groups, wave_rg)
    if n_waves < 2 or k < geo.G or is_owner(k):
        return split
    cands = [c for c in _k_candidates(geo, N) if c > k and not is_owner(c)]
    if len(cands) < TAIL_K_STEPS:
        return split
    kt = cands[TAIL_K_STEPS - 1]
    g0 = (n_waves - 1) * wave_rg
    if g0 * k + (split.num_groups - g0) * kt > N:
        return split
    if not fits_fn(kt, split.rows_max):
        return split
    return replace(split, tail_k=kt, tail_mm=member_map(kt, geo), tail_g0=g0)


def wave_row_groups(split: Split, geo: Geometry, L: int, chunk_pages: int) -> int:
    mp = 1
    for spec in split.mm.members:
        for q, _, _, _ in spec.parts:
            runs = member_runs(spec, q, split.rows_max, split.mm.pitch)
            mp = max(mp, sum(_div_up(n, chunk_pages) for _, n in runs))
    return max(1, min(split.num_groups, 2 * L * _div_up(INC_EVERY, 2 * mp)))


def build_entries(chips, ring, L, split: Split, geo: Geometry, compute_cores, vxy, wave_rg, chunk_pages):
    """Per (coord, d, l): out entries [dict(q, w, wait, runs=[(xy, off, n)])] and in entries; expected landings and
    own-member signalling. Entries with no runs are dropped."""
    n_waves = _div_up(split.num_groups, wave_rg)
    last = n_waves - 1
    # send slots: own entries of wave w at slot w, relays RELAY_LAG_WAVES later; the LAST wave's own entries move
    # behind every earlier relay and its relays follow them TAIL_RELAY_LAG slots later, so only the last wave's
    # (tail_split-narrowed) groups land at the very end (ties: own before relays, schedule order)
    own_last = last + RELAY_LAG_WAVES - min(TAIL_RELAY_LAG, RELAY_LAG_WAVES) if n_waves > 1 else 0

    def send_slot(w, idx):
        if w == last:
            return own_last if idx == 0 else own_last + min(TAIL_RELAY_LAG, RELAY_LAG_WAVES)
        return w if idx == 0 else w + RELAY_LAG_WAVES

    def runs_for(q, w, l, part):
        runs = []
        for g in range(w * wave_rg, min((w + 1) * wave_rg, split.num_groups)):
            if g % L != l:
                continue
            if part == 1 and (g // L) % 2 != 0:
                continue
            if part == 2 and (g // L) % 2 != 1:
                continue
            gmm = split.mm_of(g)
            for m, spec in enumerate(gmm.members):
                core = compute_cores[split.core0(g) + m]
                for off, n in member_runs(spec, q, split.group_rows[g], gmm.pitch):
                    runs.append((vxy(core), off, n, (core.x, core.y)))
        return runs

    out = {}
    for coord, ch in chips.items():
        p, G = ch["p"], ch["G"]
        fwd, bwd = schedule(p, G, ring, balance=True)
        sched = {"fwd": fwd if ch["next"] is not None else [], "bwd": bwd if ch["prev"] is not None else []}
        for d in DIRS:
            for l in range(L):
                if (d, l) not in ch["ports"]:
                    continue
                keyed = []
                for w in range(n_waves):
                    for idx, (q, part) in enumerate(sched[d]):
                        runs = runs_for(q, w, l, part)
                        if runs:
                            keyed.append(((send_slot(w, idx), idx), dict(q=q, w=w, own=(idx == 0), runs=runs)))
                out[(coord, d, l)] = [e for _, e in sorted(keyed, key=lambda t: t[0])]
    # incoming lists and relay wait thresholds
    inc = {}
    for coord, ch in chips.items():
        for d in DIRS:
            up = ch["prev"] if d == "fwd" else ch["next"]
            for l in range(L):
                if (d, l) not in ch["ports"]:
                    continue
                inc[(coord, d, l)] = out.get((up, d, l), []) if up is not None else []
    for (coord, d, l), entries in out.items():
        ins = inc[(coord, d, l)]
        for e in entries:
            if e["own"]:
                e["wait"] = 0
                continue
            idx = next(i for i, ie in enumerate(ins) if ie["q"] == e["q"] and ie["w"] == e["w"])
            have = {(r[1], r[3]) for r in ins[idx]["runs"]}
            assert all((r[1], r[3]) in have for r in e["runs"]), "relay entry not covered by its incoming entry"
            e["wait"] = idx + 1
    return out, inc, n_waves


# ---------------------------------------------------------------------------------------------------------------------
# Dense assembly plan (w_non_aligned): output tile j of every tile-row = sum over source runs of (member tile) x S,
# S a 0/1 shift matrix (S[e][e + delta] = 1 for e in [e0, e1)); home member of j = owner of dense column 32 j;
# other members' runs reach the home as one partial tile each (foreign contributions, added with S = identity).
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class DensePlan:
    n_mats: int
    max_fin: int
    max_fout: int
    members: list  # per member: dict(mats, j0, home=[(srcs, foreign_slots)], fout=[(home_m, slot, srcs)], n_contrib, homes)


def dense_plan(geo: Geometry, mm: MemberMap) -> DensePlan:
    G, Wl, W, k = geo.G, geo.W_local, geo.W, mm.k

    def owner(c):
        q, cl = divmod(c, Wl)
        i = cl // TILE
        for m, spec in enumerate(mm.members):
            for qq, c0, n, off in spec.parts:
                if qq == q and c0 <= i < c0 + n:
                    # local tile index inside the member's row (cb_norm walks core_w tiles per row in part order)
                    lj = sum(pn for (_, _, pn, _) in spec.parts[: spec.parts.index((qq, c0, n, off))]) + (i - c0)
                    return m, lj, cl % TILE
        raise AssertionError(c)

    per_tile = []  # j -> (home, [(m, lj, (e0, e1, delta))])
    for j in range(geo.out_w_tiles):
        runs = []
        c = TILE * j
        end = min(TILE * (j + 1), W)
        while c < end:
            m, lj, e = owner(c)
            n = min(TILE - e, end - c)
            # stop the run at the slice end
            q = c // Wl
            n = min(n, (q + 1) * Wl - c)
            runs.append((m, lj, (e, e + n, (c - TILE * j) - e)))
            c += n
        per_tile.append((owner(TILE * j)[0], runs))
    members = [dict(mats=[], home=[], fout=[], contrib_from=set(), homes_to=[]) for _ in range(k)]

    def mat_idx(m, mat):
        if mat not in members[m]["mats"]:
            members[m]["mats"].append(mat)
        return members[m]["mats"].index(mat)

    slots = [0] * k
    for j, (h, runs) in enumerate(per_tile):
        own = [(lj, mat_idx(h, mat)) for (m, lj, mat) in runs if m == h]
        foreign = []
        for m in sorted({m for (m, _, _) in runs if m != h}):
            srcs = [(lj, mat_idx(m, mat)) for (mm_, lj, mat) in runs if mm_ == m]
            members[m]["fout"].append((h, slots[h], srcs))
            members[h]["contrib_from"].add(m)
            foreign.append(slots[h])
            slots[h] += 1
        members[h]["home"].append((j, own, foreign))
    for m in range(k):
        if members[m]["contrib_from"]:
            mat_idx(m, (0, TILE, 0))  # identity for the foreign tiles
        members[m]["homes_to"] = sorted({h for (h, _, _) in members[m]["fout"]})
    return DensePlan(
        n_mats=max(len(x["mats"]) for x in members),
        max_fin=max(slots),
        max_fout=max(len(x["fout"]) for x in members),
        members=members,
    )


def q0_out(spec: MemberSpec, geo: Geometry) -> int:
    """First output tile column of an aligned member's block (slices are tile-aligned: contiguous)."""
    q0, c0 = spec.parts[0][0], spec.parts[0][1]
    return q0 * geo.sw + c0


# ---------------------------------------------------------------------------------------------------------------------
# Descriptor
# ---------------------------------------------------------------------------------------------------------------------


def _cb(index, pages, page_size, dtype, core_ranges):
    return ttnn.CBDescriptor(
        total_size=pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
    )


def _crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores]).merge(ttnn.CoreRangeSet([]))


def _xy(v):
    return int(v.x) | (int(v.y) << 16)


def _datum_bytes(dtype) -> int:
    return {ttnn.float32: 4, ttnn.bfloat16: 2, ttnn.bfloat8_b: 1}[dtype]


@dataclass
class OpConfig:
    cluster_axis: int
    ring: bool
    num_links: int
    is_ln: bool
    epsilon: float
    compute_kernel_config: object


def create_mesh_program_descriptor(input_tensor, gamma, beta, output_tensor, cfg: OpConfig, sems, fabric):
    mesh = input_tensor.device()
    mesh_shape = tuple(mesh.shape)
    G = mesh_shape[cfg.cluster_axis]
    L = cfg.num_links
    geo = derive_geometry(input_tensor.shape, G)
    has_gamma, has_beta = gamma is not None, beta is not None
    x_dtype = input_tensor.dtype
    g_dtype = gamma.dtype if has_gamma else ttnn.bfloat16
    fp32_dest = bool(cfg.compute_kernel_config.fp32_dest_acc_en)
    mask = cfg.is_ln and geo.v < TILE
    dense = geo.v < TILE  # slices land at non-tile-aligned output offsets: dense assembly regime
    # LN FPU fast path: x and the bf16 pivots are exact FPU operands (bf16 / bfloat8_b) and no slice tail is masked.
    # It holds the centered block (cb_cent, accumulator format) beside cb_x: when no split fits with it and DEST is
    # fp32, the SFPU path (no block-sized scratch; precision-equivalent at fp32 DEST) is solved instead. 16-bit DEST
    # keeps the FPU path (the SFPU path was measured worse there).
    # 16-bit DEST (m16): the moments never use the FPU path's scratch, and the FPU normalize (one rounding per step)
    # replaces the SFPU chain (a truncating SFPU store per step: -0.6 % per-row scale measured on w_non_aligned),
    # so the FPU path holds for any bf16 / bfloat8_b input, slice tail included.
    ln_fpu_ok = cfg.is_ln and x_dtype != ttnn.float32 and (geo.v == TILE or not fp32_dest)
    sz = replace(make_sizes(x_dtype, g_dtype, fp32_dest, cfg.is_ln, has_gamma, has_beta, mask), ln_fpu=ln_fpu_ok)

    chips, ring, ports_union = fabric
    grid = mesh.compute_with_storage_grid_size()
    pu = set(ports_union)
    compute_all = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in pu]
    N = len(compute_all)

    tile_bytes = sz.T_in
    max_payload = int(ttnn.get_tt_fabric_max_payload_size_bytes())
    chunk_pages = max(1, max_payload // tile_bytes)
    budget = cb_l1_budget(mesh)

    fit_log = []
    fit_unit = {}
    fit_batch = {}

    def fits(k, rows):
        mm = member_map(k, geo)
        dpk = dense_plan(geo, mm) if dense else None
        # port cores: cb_x (mirror of the member block) + cb_port_stage; the batch halves under L1 pressure
        port_bytes = None
        for pb in port_batch_candidates():
            port_bytes = rows * mm.pitch * sz.T_in + stage_pages(pb, chunk_pages) * tile_bytes + hdr_ring_bytes()
            if port_bytes <= budget:
                break
        if port_bytes > budget:
            fit_log.append((k, rows, None, port_bytes))
            return False
        for u in out_unit_candidates(mm.pitch):
            cbytes = plan_bytes(cb_plan(sz, k, mm.pitch, rows, n_affine_slots(mm, geo), dpk, u), sz)
            if cbytes <= budget:
                fit_unit[k] = u
                fit_batch[k] = pb
                return True
        fit_log.append((k, rows, cbytes, port_bytes))
        return False

    def _l1_allocated():
        try:
            view = ttnn.get_memory_view(mesh, ttnn.BufferType.L1)
            return [(int(b["address"]), int(b["size"])) for b in view.block_table if b["allocated"] == "yes"]
        except Exception:  # noqa: BLE001
            return "?"

    fits.report = lambda: (
        f"budget {budget} B, (k, rows, compute B, port B): {fit_log}; L1 allocated (rel addr, size): {_l1_allocated()}"
    )

    try:
        split = select_split(geo, N, fits)
    except RuntimeError:
        if not (sz.ln_fpu and fp32_dest):
            raise
        sz = replace(sz, ln_fpu=False)
        split = select_split(geo, N, fits)
    ln_fpu = sz.ln_fpu
    k, mm, pitch, rows_max = split.k, split.mm, split.mm.pitch, split.rows_max
    vxy_cache = {}

    def vxy(c):
        key = (c.x, c.y)
        if key not in vxy_cache:
            vxy_cache[key] = mesh.worker_core_from_logical_core(c)
        return vxy_cache[key]

    wave_rg = wave_row_groups(split, geo, L, chunk_pages)
    dp = dense_plan(geo, mm) if dense else None
    port_batch = fit_batch[k]
    owner = is_owner(k)
    gp = gather_tiles(k) if owner else 1

    def merged_plan(sp: Split) -> dict:
        """CB plan over every member class: one declaration per CB on all compute cores (cb_x's mirror address and
        the lockstep slot addresses need identical layouts), sized to the largest class."""
        merged = {}
        for kk, mmk, _ in sp.classes():
            for idx, (pg, pb) in cb_plan(
                sz, kk, mmk.pitch, rows_max, n_affine_slots(mmk, geo), dp, fit_unit[kk]
            ).items():
                if idx in merged:
                    assert merged[idx][1] == pb
                    pg = max(pg, merged[idx][0])
                merged[idx] = (pg, pb)
        return merged

    split = tail_split(split, geo, N, wave_rg, fits, dense)
    cplan = merged_plan(split)
    if split.tail_k and plan_bytes(cplan, sz) > budget:
        split = replace(split, tail_k=0, tail_mm=None, tail_g0=0)
        cplan = merged_plan(split)
    xg_first = False
    if XG_FIRST and has_gamma and not cfg.is_ln:
        xg_pages = max(rows_max * mmk.pitch for _, mmk, _ in split.classes())
        trial = dict(cplan)
        trial[CB_XG] = (xg_pages, sz.T_acc)
        if plan_bytes(trial, sz) <= budget:
            cplan, xg_first = trial, True
    own_pivot = False
    if OWN_PIVOT and ln_fpu and fp32_dest and not dense and not any(is_owner(kk) for kk, _, _ in split.classes()):
        own_pages = rows_max * max(n for _, mmk, _ in split.classes() for s in mmk.members for (_, _, n, _) in s.parts)
        trial = dict(cplan)
        trial[CB_OWN] = (own_pages, sz.T_in)
        if any(
            kk == 1 for kk, _, _ in split.classes()
        ):  # the compute stages K_m in cb_mstat ahead of (D1, D2) (stage_pivot_block)
            trial[CB_MSTAT] = (3 * rows_max, sz.T_acc)
        if plan_bytes(trial, sz) <= budget:
            cplan, own_pivot = trial, True
    compute_used = compute_all[: split.cores_used()]
    out_e, in_e, n_waves = build_entries(chips, ring, L, split, geo, compute_used, vxy, wave_rg, chunk_pages)
    short_runs = short_run_transport(out_e, chunk_pages)
    port_pack = _auto(PORT_PACK, short_runs)
    if short_runs and os.environ.get("AGRMS_BATCH") is None:
        port_batch = min(port_batch, SHORT_RUN_BATCH)
    port_stateful = _auto(PORT_STATEFUL, short_runs)
    n_stat = 2 if cfg.is_ln else 1
    t_acc = acc_dtype(fp32_dest)
    cb_dtype = {
        CB_X: x_dtype,
        CB_GAMMA: g_dtype,
        CB_BETA: g_dtype,
        CB_ONES: ttnn.bfloat16,
        CB_MASK: ttnn.bfloat16,
        CB_MOMENT_A: t_acc,
        CB_MOMENT_B: t_acc,
        CB_KM: ttnn.bfloat16,
        CB_PARTIAL: t_acc,
        CB_GATHERED: t_acc,
        CB_ROWSTAT: t_acc,
        CB_OUT: x_dtype,
        CB_MSTAT: t_acc,
        CB_SHIFT: t_acc,
        CB_KG: ttnn.bfloat16,
        CB_MU: t_acc,
        CB_AFFINE_STAGE: ttnn.bfloat16,
        CB_NORM: x_dtype,
        CB_SHIFT_MATS: x_dtype,
        CB_FOREIGN: x_dtype,
        CB_CONTRIB: x_dtype,
        CB_CENT: t_acc,
        CB_OWN_RET: t_acc,
        CB_OWN_KG: ttnn.bfloat16,
        CB_DEV: t_acc,
        CB_WCOL: t_acc,
        CB_XG: t_acc,
        CB_OWN: x_dtype,
    }
    port_cores = [ttnn.CoreCoord(x, y) for (x, y) in ports_union]
    crs_compute = _crs(compute_used)
    crs_ports = _crs(port_cores)
    crs_x = _crs(compute_used + port_cores)

    # ---- compute config (LN: copy-only CBs unpack straight to DEST in fp32) ----
    ckc = cfg.compute_kernel_config
    ccfg = ttnn.ComputeConfigDescriptor()
    ccfg.math_fidelity = ckc.math_fidelity
    ccfg.fp32_dest_acc_en = ckc.fp32_dest_acc_en
    ccfg.math_approx_mode = ckc.math_approx_mode
    try:
        ccfg.dst_full_sync_en = ckc.dst_full_sync_en
    except Exception:  # noqa: BLE001
        pass
    if cfg.is_ln and fp32_dest:
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for cb in (CB_GATHERED if k > 1 else CB_PARTIAL, CB_MU, CB_MSTAT):
            modes[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
        if x_dtype == ttnn.float32:
            modes[CB_X] = ttnn.UnpackToDestMode.UnpackToDestFp32
        ccfg.unpack_to_dest_mode = modes

    defines = []
    if cfg.is_ln:
        defines.append(("AGRMS_LN", "1"))
    if has_gamma:
        defines.append(("AGRMS_HAS_GAMMA", "1"))
    if has_beta:
        defines.append(("AGRMS_HAS_BETA", "1"))
    zone_defines = [("AGRMS_ZONES", "1")] if os.environ.get("AGRMS_ZONES") else []  # perf probes only
    if os.environ.get("AGRMS_ABL"):  # perf probes only: port-kernel ablation bits (wrong output)
        zone_defines = zone_defines + [("AGRMS_ABL", os.environ["AGRMS_ABL"])]
    defines += zone_defines
    if dense:
        defines.append(("AGRMS_DENSE", "1"))
    if ln_fpu:
        defines.append(("AGRMS_LN_FPU", "1"))
    if mask:
        defines.append(("AGRMS_MASK", "1"))
    if owner:
        defines.append(("AGRMS_OWNER", "1"))
    if AFFINE_FIRST:
        defines.append(("AGRMS_AFFINE_FIRST", "1"))
    if xg_first:
        defines.append(("AGRMS_XG_FIRST", "1"))
    if own_pivot:
        defines.append(("AGRMS_LN_OWN_PIVOT", "1"))

    x_acc = ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
    o_acc = ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()
    g_acc = (ttnn.TensorAccessorArgs(gamma) if has_gamma else ttnn.TensorAccessorArgs()).get_compile_time_args()
    b_acc = (ttnn.TensorAccessorArgs(beta) if has_beta else ttnn.TensorAccessorArgs()).get_compile_time_args()
    x_elem = _datum_bytes(x_dtype)
    g_elem = _datum_bytes(g_dtype)
    g_is_tile = 1 if (has_gamma and gamma.layout == ttnn.TILE_LAYOUT) else 0
    g_page = gamma.buffer_aligned_page_size() if has_gamma else 0

    def norm_block_of(kk: int, mmk: MemberMap) -> int:
        """ln FPU normalize: output tiles per DEST block (compute CT [6]). A block is packed at contiguous cb_out
        offsets, so it must never straddle the cb_out ring end: every block is full (it divides each member's core_w,
        no ragged row tail) and the ring holds a whole number of blocks; <= the writer's unit (the writer drains whole
        units). Cap: NORM_BLOCK (0 = the writer unit)."""
        return math.gcd(NORM_BLOCK or fit_unit[kk], cplan[CB_OUT][0], *(s.core_w for s in mmk.members))

    def class_cts(kk: int, mmk: MemberMap):
        """(reader, compute, writer) CT args of one member class."""
        reader_ct = [tile_bytes, mmk.pitch, geo.sw, geo.v, x_elem, SEM_LANDED, SEM_OWN_LOADED] + list(x_acc)
        compute_ct = [
            kk,
            mmk.pitch,
            n_stat,
            dp.max_fin if dense else 0,
            dp.max_fout if dense else 0,
            gp,
            norm_block_of(kk, mmk),
        ]
        writer_ct = writer_ct_of(kk, mmk.pitch, fit_unit[kk])
        return reader_ct, compute_ct, writer_ct

    def writer_ct_of(kk, pitch, unit):
        return (
            [
                sz.T_out,
                pitch,
                geo.out_w_tiles,
                kk,
                unit,
                cplan[CB_OUT][0],  # cb_out ring size: single source = cb_plan()
                n_stat,
                SEM_PARTIALS,
                sz.T_acc,
                g_elem,
                g_is_tile,
                g_page,
                sz.T_g,
                AFFINE_SLOT_BYTES,
                SEM_FOREIGN,
                dp.max_fin if dense else 0,
                dp.max_fout if dense else 0,
                0 if dense else 1,  # affine_aligned
                gp,  # owner combine: gather tiles per owned row
                SEM_INV,
            ]
            + list(o_acc)
            + list(g_acc)
            + list(b_acc)
        )

    class_ct = [class_cts(kk, mmk) for kk, mmk, _ in split.classes()]
    port_reader_ct = [
        tile_bytes,
        chunk_pages,
        port_batch,
        SEM_OWN_LOADED,
        SEM_LANDED,
        min(PORT_PUSH_PACKETS or port_batch, port_batch),
        PORT_READ_STREAM,
        port_pack,
        STAGE_BATCHES,
    ]
    hdr_ring = _auto(HDR_RING, short_runs, HDR_RING_AUTO)
    hdr_ring = max(hdr_ring, port_batch) if hdr_ring > 0 else 0  # a slot is reused only after its batch's flush
    port_sender_ct = [tile_bytes, chunk_pages, port_batch, port_pack, hdr_ring, port_stateful, HDR_STRIDE, HDR_COPY]

    inv_w_bits = _f32_bits(1.0 / geo.W)
    eps_bits = _f32_bits(cfg.epsilon)
    arrival_addr, ready_addr = (int(ttnn.get_global_semaphore_address(s)) for s in sems)
    x_addr = input_tensor.buffer_address()
    o_addr = output_tensor.buffer_address()
    g_addr = gamma.buffer_address() if has_gamma else 0
    b_addr = beta.buffer_address() if has_beta else 0
    node = lambda coord: mesh.get_fabric_node_id(ttnn.MeshCoordinate(*coord))

    hash_key = "|".join(
        str(v)
        for v in (
            "all_gather_rms_norm",
            tuple(input_tensor.shape),
            x_dtype,
            input_tensor.memory_config(),
            output_tensor.memory_config(),
            has_gamma and (gamma.dtype, gamma.layout),
            has_beta,
            cfg,
            mesh_shape,
            tuple(ports_union),
            k,
            split.tail_k,
            split.tail_g0,
            wave_rg,
            fit_unit[k],
            port_batch,
            tuple(port_sender_ct),
            ln_fpu,
            xg_first,
            own_pivot,
            ckc.math_fidelity,
            fp32_dest,
            ckc.math_approx_mode,
        )
    )

    mesh_desc = ttnn.MeshProgramDescriptor()
    for coord, ch in chips.items():
        p = ch["p"]
        program = ttnn.ProgramDescriptor()
        program.custom_program_hash = (
            int.from_bytes(hashlib.sha1(f"{hash_key}|{coord}".encode()).digest()[:8], "little") >> 1
        )
        # semaphores first: setup_fabric_connection then takes the lowest free ids on the port cores
        program.semaphores = [
            ttnn.SemaphoreDescriptor(id=SEM_LANDED, core_ranges=crs_x, initial_value=0),
            ttnn.SemaphoreDescriptor(id=SEM_PARTIALS, core_ranges=crs_x, initial_value=0),
            ttnn.SemaphoreDescriptor(id=SEM_OWN_LOADED, core_ranges=crs_x, initial_value=0),
            ttnn.SemaphoreDescriptor(id=SEM_FOREIGN, core_ranges=crs_x, initial_value=0),
            ttnn.SemaphoreDescriptor(id=SEM_INV, core_ranges=crs_x, initial_value=0),
        ]
        # CBs: cb_x FIRST over compute_used + ports_union (identical on every chip -> one landing address)
        cbs = [_cb(CB_X, cplan[CB_X][0], cplan[CB_X][1], x_dtype, crs_x)]
        cbs.append(_cb(CB_STAGE, stage_pages(port_batch, chunk_pages), tile_bytes, x_dtype, crs_ports))
        if hdr_ring > 0:
            cbs.append(_cb(CB_PORT_HDR, hdr_ring, HDR_STRIDE, x_dtype, crs_ports))
        regions = alias_regions(cplan, sz)
        aliased = {c for idx, _ in regions for c in idx}
        for idx in sorted(cplan):
            if idx == CB_X or idx in aliased:
                continue
            pages, pb = cplan[idx]
            cbs.append(_cb(idx, pages, pb, cb_dtype[idx], crs_compute))
        for idx, nbytes in regions:  # one L1 region, several buffer indices (disjoint lifetimes)
            cbs.append(
                ttnn.CBDescriptor(
                    total_size=nbytes,
                    core_ranges=crs_compute,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(buffer_index=c, data_format=cb_dtype[c], page_size=cplan[c][1])
                        for c in idx
                    ],
                )
            )
        program.cbs = cbs

        # ---- expected landings / own-loaded signalling on this chip ----
        landings = {}
        for d in DIRS:
            for l in range(L):
                if (d, l) not in ch["ports"]:
                    continue
                for e in in_e[(coord, d, l)]:
                    for core_xy in sorted({r[3] for r in e["runs"]}):
                        landings[core_xy] = landings.get(core_xy, 0) + 1
        own_sig = {}  # member core -> [(port vxy, increment)]
        own_expected = {}
        own_bit = {}  # (d, l) -> {member core: bit} when the port waits per entry (OWN_LOADED_MASK)
        for d in DIRS:
            for l in range(L):
                if (d, l) not in ch["ports"]:
                    continue
                pc = ch["ports"][(d, l)]
                own_cores = sorted({r[3] for e in out_e[(coord, d, l)] if e["own"] for r in e["runs"]})
                own_expected[(d, l)] = len(own_cores)
                per_entry = OWN_LOADED_MASK and len(own_cores) <= OWN_MASK_BITS
                own_bit[(d, l)] = {cxy: 1 << i for i, cxy in enumerate(own_cores)} if per_entry else {}
                for i, cxy in enumerate(own_cores):
                    own_sig.setdefault(cxy, []).append((_xy(vxy(pc)), (1 << i) if per_entry else 1))

        class_rt = [(ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()) for _ in split.classes()]
        for g in range(split.num_groups):
            reader_rt, compute_rt, writer_rt = class_rt[1 if (split.is_tail(g) and split.tail_g0 > 0) else 0]
            rows = split.group_rows[g]
            row_start = split.group_row_start[g]
            gk, gmm = split.k_of(g), split.mm_of(g)
            members = compute_used[split.core0(g) : split.core0(g) + gk]
            peer_xy = []
            for c in members:
                v = vxy(c)
                peer_xy += [v.x, v.y]
            for m, spec in enumerate(gmm.members):
                c = members[m]
                own = [pt for pt in spec.parts if pt[0] == p]
                own_n, own_c0, own_off = (own[0][2], own[0][1], own[0][3]) if own else (0, 0, 0)
                tails = [off + n - 1 for (_, c0, n, off) in spec.parts if c0 + n == geo.sw] if geo.v < TILE else []
                sig = own_sig.get((c.x, c.y), [])
                rd_args = (
                    [x_addr, row_start, rows, own_n, own_c0, own_off, landings.get((c.x, c.y), 0), len(sig)]
                    + [xy for xy, _ in sig]
                    + [w for _, w in sig]
                    + [len(tails)]
                    + tails
                )
                segs = []
                for _, c0, n, _ in spec.parts:
                    segs += [n, geo.v if (mask and c0 + n == geo.sw) else 0]  # slice-last tile: valid columns
                n_m = [_f32_bits(s.valid_cols) for s in gmm.members]
                dense_c, dense_w, dense_r = [], [], []
                store_col0, store_w = q0_out(spec, geo), spec.core_w
                if dense:
                    dm = dp.members[m]
                    for e0, e1, delta in dm["mats"]:
                        dense_r.append(e0 | (e1 << 8) | ((delta + 64) << 16))
                    dense_c += [len(dm["mats"]), dm["mats"].index((0, TILE, 0)) if dm["contrib_from"] else 0]
                    dense_c.append(len(dm["home"]))
                    for _, srcs, slots in dm["home"]:
                        dense_c.append(len(srcs) | (len(slots) << 8))
                        dense_c += [lj | (mi << 16) for lj, mi in srcs]
                        dense_c += slots
                    dense_c.append(len(dm["fout"]))
                    for _, _, srcs in dm["fout"]:
                        dense_c.append(len(srcs))
                        dense_c += [lj | (mi << 16) for lj, mi in srcs]
                    store_col0 = dm["home"][0][0] if dm["home"] else 0
                    store_w = len(dm["home"])
                    dense_w = [len(dm["fout"])]
                    for h, slot, _ in dm["fout"]:
                        dense_w += [h, slot]
                    dense_w += [len(dm["contrib_from"]), len(dm["homes_to"])] + dm["homes_to"]
                    rd_args = rd_args + [len(dense_r)] + dense_r
                if owner and cfg.is_ln:
                    rd_args = rd_args + [gk] + n_m  # cb_wcol: member weights n_m (fp32 bits)
                reader_rt[c.x][c.y] = rd_args
                compute_rt[c.x][c.y] = (
                    [rows, spec.core_w, m, inv_w_bits, eps_bits, _f32_bits(1.0 / spec.valid_cols), len(spec.parts)]
                    + segs
                    + n_m
                    + dense_c
                    + ([own_n, _f32_bits(1.0 / (own_n * TILE)) if own_n else 0] if own_pivot else [])
                )
                gparts = []
                for q, c0, n, off in spec.parts:
                    gparts += [q * geo.W_local + TILE * c0, geo.W_local - TILE * c0, n, off]
                wr = [o_addr, row_start, rows, store_col0, store_w, m, g_addr, b_addr, len(spec.parts)] + gparts
                wr += [len(wr) + 1]  # index of the first peer (x, y) pair
                writer_rt[c.x][c.y] = wr + peer_xy + dense_w

        # ---- port kernels on every ports_union core (idle ones get zero work) ----
        pr_rt, ps_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        port_of = {(pc.x, pc.y): key for key, pc in ch["ports"].items()}
        for pc in port_cores:
            key = port_of.get((pc.x, pc.y))
            if key is None:
                pr_rt[pc.x][pc.y] = [arrival_addr, 0, 0, 0]
                ps_rt[pc.x][pc.y] = [arrival_addr, ready_addr, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                continue
            d, l = key
            outs = out_e[(coord, d, l)]
            ins = in_e[(coord, d, l)]
            rd = [arrival_addr, own_expected[(d, l)], len(outs), len(ins)]
            sd_entries = []
            bits = own_bit[(d, l)]
            for e in outs:
                runs = []
                for v, off, n, _ in e["runs"]:
                    runs += [_xy(v), off | (n << 16)]
                wait = e["wait"]
                if e["own"] and bits:  # own entry: wait for exactly its own members (bit mask, flag bit 31)
                    wait = OWN_MASK_FLAG | sum({bits[r[3]] for r in e["runs"]})
                rd += [wait, len(e["runs"])] + runs
                sd_entries += [len(e["runs"])] + runs
            for e in ins:
                cores = []
                for v, _, _, cxy in e["runs"]:
                    if not cores or cores[-1] != _xy(v):
                        cores.append(_xy(v))
                cores = list(dict.fromkeys(cores))
                rd += [len(cores)] + cores
            pr_rt[pc.x][pc.y] = rd
            peer = ch["next"] if d == "fwd" else ch["prev"]
            # ready: the peer's opposite-direction port sends into this chip; tell it this program started
            peer_opp_sends = peer is not None and bool(out_e.get((peer, OPP[d], l)))
            send_ready = 1 if peer_opp_sends else 0
            rc = vxy(chips[peer]["ports"][(OPP[d], l)]) if send_ready else None
            sd = [arrival_addr, ready_addr, send_ready, rc.x if rc else 0, rc.y if rc else 0]
            has_conn = peer is not None and (bool(outs) or send_ready)
            if has_conn:
                pcv = vxy(chips[peer]["ports"][(d, l)]) if outs else ttnn.CoreCoord(0, 0)
                pn = node(peer)
                sd += [pcv.x, pcv.y, int(pn.mesh_id), int(pn.chip_id), 1, len(outs)]
                sd += sd_entries
                sd += list(ttnn.setup_fabric_connection(node(coord), pn, ch["links"][d][l], program, pc))
            else:
                sd += [0, 0, 0, 0, 0, 0]
            ps_rt[pc.x][pc.y] = sd

        kernels = []
        for (kk, mmk, groups), (reader_ct, compute_ct, writer_ct), (reader_rt, compute_rt, writer_rt) in zip(
            split.classes(), class_ct, class_rt
        ):
            crs_cls = _crs(compute_used[split.core0(groups[0]) : split.core0(groups[-1]) + kk])
            kernels += [
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "all_gather_rms_norm_reader.cpp"),
                    core_ranges=crs_cls,
                    compile_time_args=reader_ct,
                    defines=defines,
                    runtime_args=reader_rt,
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "all_gather_rms_norm_writer.cpp"),
                    core_ranges=crs_cls,
                    compile_time_args=writer_ct,
                    defines=defines,
                    runtime_args=writer_rt,
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "all_gather_rms_norm_compute.cpp"),
                    core_ranges=crs_cls,
                    compile_time_args=compute_ct,
                    defines=defines,
                    runtime_args=compute_rt,
                    config=ccfg,
                ),
            ]
        program.kernels = kernels + [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "all_gather_rms_norm_port_reader.cpp"),
                core_ranges=crs_ports,
                compile_time_args=port_reader_ct,
                defines=zone_defines,
                runtime_args=pr_rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_0_default  # NoC0
                ),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "all_gather_rms_norm_port_sender.cpp"),
                core_ranges=crs_ports,
                compile_time_args=port_sender_ct,
                defines=zone_defines,
                runtime_args=ps_rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_1_default  # NoC1
                ),
            ),
        ]
        r, c = coord
        mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = program
    info = dict(
        k=k,
        tail_k=split.tail_k,
        tail_g0=split.tail_g0,
        xg_first=xg_first,
        own_pivot=own_pivot,
        group_cores=[
            [(vxy(c).x, vxy(c).y) for c in compute_used[split.core0(g) : split.core0(g) + split.k_of(g)]]
            for g in range(split.num_groups)
        ],
        owner=owner,
        port_batch=port_batch,
        short_runs=short_runs,
        port_pack=port_pack,
        hdr_ring=hdr_ring,
        out_unit=fit_unit[k],
        num_groups=split.num_groups,
        rows=rows_max,
        pitch=pitch,
        wave_rg=wave_rg,
        n_waves=n_waves,
        chunk_pages=chunk_pages,
        ports_union=ports_union,
        N=N,
        budget=budget,
    )
    global LAST_INFO
    LAST_INFO = info
    return mesh_desc, info
