# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""matmul_reduce_scatter host planner + MeshProgramDescriptor (see op_design.md).

Per chip, one program with two core roles:

  matmul cores (grid_m x grid_n, the compute grid minus a transport strip):
    reader  (NCRISC, NoC0): per entry, Kt/kb K-blocks of A (Mcast1D family "a") and W (family "b"); the cross-block
                            operand is resident after entry 0 (re-credited, no NoC). Lands arrivals: waits the global
                            arrival counter, credits the zone slot the upstream port wrote over the fabric.
    compute (TRISCs):       matmul_block into a caller-owned fp32 slot of cb_partial_hold (packer L1 accumulation),
                            then finish: copy (chain start) / hold + arrival (relay, fin) / hold + fwd + bwd (fin).
    writer  (BRISC, NoC1):  start / relay: signal the port's mailbox and pop once the port acks; fin: write the output.
  port cores (2 L, in the strip; one per (direction, link)):
    reader  (NCRISC, NoC0): per unit (entry, core halves), per core: wait the mailbox, NoC-read its finished sub-block
                            into cb_port_chunks, ack the core.
    sender  (BRISC, NoC1):  ready fence, then every chunk as one fabric packet into the downstream core's zone slot; a
                            fused increment of that core's arrival counter on its last chunk.

Every knob is defined once below (or derived in choose_grid / choose_k_block) and flows into CT/RT args.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import ttnn
from ttnn.operations.ccl import Topology

KERNEL_DIR = Path(__file__).parent / "kernels"

# ---- circular buffers ------------------------------------------------------------------------------------------
CB_ACT_K_WINDOW = 0  # A K-block windows (resident: the whole core_m x Kt slice)
CB_WEIGHT_K_WINDOW = 1  # W K-block windows (resident: the whole Kt x core_n slice)
CB_PARTIAL_HOLD = 2  # fp32 matmul partial slots (caller-owned L1-accumulation target)
CB_FINISHED = 3  # finished sub-block (bf16), staging offset 0; read remotely by the ports
CB_ARRIVALS_FWD = 4  # forward arrival zone slots (written by the upstream port over the fabric)
CB_ARRIVALS_BWD = 5  # backward arrival zone slots
CB_PORT_CHUNKS = 6  # port cores: gathered chunks waiting for the fabric sender
CB_ZERO = 7  # one bf16 zero tile (the writer fills it once): the fused finish's DEST += hold + 0 FPU add

# ---- knobs (single source of truth) ------------------------------------------------------------------------------
DEPTH_IN0 = 3  # stream regime: A K-block windows (one computing, one landing, PREFETCH_STEPS read ahead by a sender)
DEPTH_IN1 = 3  # stream regime: W K-block windows
# resident regime: depths the streamed (non-resident) operand's window may take, preferred first. choose_k_block tries
# every kb at the first depth before shrinking the window, so a shape that fits at depth 3 keeps it (focus case) and a
# shape whose resident operand misses the budget at depth 3 (MiMo: W 816 KiB per core) becomes resident at depth 2
# (one window computing, one landing / prefetched) instead of falling back to re-reading the resident operand G times
RESIDENT_STREAM_DEPTHS = tuple(int(d) for d in os.environ.get("MRS_RESIDENT_DEPTHS", "3,2").split(","))  # env: perf
# DEST-resident K (Perf 1): each output subblock accumulates over a super-block of SUPER_BLOCK K-windows in fp32 DEST
# and is packed once per super-block (half the full-sync fp32 L1-acc packs of a per-window pack). The super-block stays
# fronted until its last subblock, so the streamed window CBs need SUPER_BLOCK + SUPER_BLOCK_EXTRA windows (the extra
# ones are the loader's look-ahead; with none, focus 133 us vs 123). Falls back to 1 (pack per window, old depths)
# when Kt / kb is odd or the deeper windows do not fit L1.
SUPER_BLOCK = 2
SUPER_BLOCK_EXTRA = 2
DEPTH_HOLD = 2  # fp32 partial slots when some entry lags (lag 1 + current); a walk with no lagged entry needs 1
# (each finish pops its slot before the next entry reserves one): hold_depth() derives it from the schedule
DEPTH_HANDOFF = 2  # finished slots (a port gathers entry e while compute finishes e+1)
PREFER_PORT_ROW_STRIP = True  # ports in logical row 0, right under the Ethernet row (focus case: 150-160 us with
# the column strip -> 137-145 us with the row strip, despite 14 vs 12 tiles per core: the transport tail shrinks)
PORT_PLACEMENT = "eth"  # "eth": each port nearest its link's Ethernet core | "spread": evenly along the strip
PORT_GROUP = int(os.environ.get("MRS_PORT_GROUP", 4))  # chunks per port batch (cb_port_chunks holds 2 batches); <= 8
# (header pool). env: perf experiments only
MIRROR_HALF_ORDERS = 1  # ring_balanced + a resident cross-block operand: halves split along the streamed operand's
# multicast lines and walk the equal-distance blocks in mirrored orders (build_schedule), so each entry sends half a
# block each way. 0: one shared order (each of the middle entries sends a whole block one way)
# waves per scatter block (Refinement 4, overlap). Each per-core block is split into this many waves along the streamed
# operand's own axis (resident A, scatter -1: n; resident W, scatter -2: m, only when the waves divide core_m), and
# every wave is one entry: a full-K matmul of the wave, then its finish and send, so the first send waits for one
# wave instead of a whole block. 1: one entry per block. The stream regime always runs 1 (a wave would re-stream the
# cross-block operand once per wave). env: perf experiments only
ENTRIES_PER_BLOCK = int(os.environ.get("MRS_WAVES", 1))
# with waves: 1 = one increment per (core, wave), the default: the per-wave arrival lets a relay / fin consume wave 0
# right after its own wave-0 matmul. 0 = every wave's data is sent as soon as it is finished, but only a core's LAST wave of a block carries
# the fused arrival increment (its flush covers the earlier waves' writes to the same core, exactly like the earlier
# chunks of one sub-block), and a receiver consumes any wave of a block once the block's increment landed. Each fused
# increment stalls the receiving router (~0.3 us measured), so per-wave increments make the transport, not the
# matmul, the binding stage. Measured (focus, 2 waves): the send units did not get faster (the ports are per-packet
# bound, not increment bound) while wave-0 relays / fins waited for their block's second wave: +5 us. env: perf only
ARRIVAL_INC_PER_WAVE = int(os.environ.get("MRS_INC_PER_WAVE", 1))
# Output chunks (shapes whose per-core footprint misses L1). The per-chip block (block_m x block_n tiles) is cut into
# n_m x n_n chunks of at most chunk_m x chunk_n tiles, and the whole G-entry walk runs once per chunk, in one program:
# the grid, the slot size P and the zones are sized for one chunk. The chunk walk keeps the resident operand's axis
# outer (scatter -1: m outer, so resident A is reloaded only when the m chunk changes; scatter -2: n outer). Zone slots
# are reused across chunks: ZONE_SETS chunks' arrivals fit the zones at once, and a port sends chunk c only after the
# downstream chip consumed chunk c - ZONE_SETS (a per-chunk credit the downstream's opposite port forwards over the
# fabric). 1 chunk (the default whenever the shape fits) is the unchunked op, byte-identical. env MRS_CHUNKS="n_m,n_n":
# perf experiments only (the test hook _FORCE_CHUNKS does the same)
ZONE_SETS = int(os.environ.get("MRS_ZONE_SETS", 2))
MAX_AUTO_CHUNKS = 256  # auto chunk search: the most chunks a candidate may have (else NotImplementedError)
# Auto split (only when 1 chunk misses L1): "model" plans every distinct candidate up to MAX_AUTO_CHUNKS and keeps the
# one chunk_cost_us() rates fastest; "fewest" keeps the first candidate that fits (fewest chunks, inner axis first).
# Measured (QB2 1x4 ring, traced, generated/mrs_sweep): "fewest" picks Wan2.2 FF2 720p 3x1 (2.44 ms; best 4x1, 2.11
# ms) and 8192x4096x8192 1x4 (11.5 ms; best 8x4, 3.17 ms); "model" picks 4x1 and 8x4. env: perf experiments only
AUTO_CHUNK_POLICY = os.environ.get("MRS_AUTO_CHUNKS", "model")
# K-block cap of a chunked plan: each chunk re-walks Kt / kb K-steps per entry, and a K-step has a fixed cost (~1.4 us
# resident, ~2.1 us stream, fitted below), so a chunked plan prefers the longer K-block (measured, kb cap 32 vs 16:
# Kimi o_proj 1x4 chunks 160 vs 182 us, 2x2 210 vs 231; Wan 8x4 2.18 vs 2.52 ms; 8192x4096x8192 8x4 3.17 vs 3.35 ms;
# 1 chunk keeps KB_MAX: Kimi 1x1 127 vs 122 us at 32). env: perf experiments only
KB_MAX_CHUNKED = int(os.environ.get("MRS_KB_MAX_CHUNKED", 32))
# chunk_cost_us() constants, least-squares fit (mean |log error| 0.13, 222 traced configs of the three sweep shapes):
# compute = padded per-core tiles x Kt x 32 cycles (HiFi2) at COST_FPU_EFF of the clock, plus COST_STEP_US per K-step
# (COST_STEP_STREAM_US under the stream regime, which multicasts both operands every step); DRAM = the operand traffic
# at COST_DRAM_GBPS; transport = COST_HANDOFF_US per (core, sending entry) handoff of one port; the call takes the
# largest of the three plus COST_FLOOR_US
COST_FPU_EFF = 0.93
COST_STEP_US = 1.4
COST_STEP_STREAM_US = 2.1
COST_HANDOFF_US = 0.29
COST_DRAM_GBPS = 200
COST_FLOOR_US = 8.0
CLOCK_MHZ = 1350
# The per-mesh staging tensor (cb_finished + zones) is sized "exact": each call plans as if no staging were allocated,
# and the cached tensor is reallocated whenever its size differs. "grow" keeps the old grow-only cache: the allocated
# size is a floor in every later plan's footprint, so after one large call (Wan 720p, P = 44: 880 KiB of staging) the
# smaller shapes lose their resident regime (measured: Wan 8x4 chunks resident -> stream). env: perf experiments only
STAGING_POLICY = os.environ.get("MRS_STAGING", "exact")  # "exact" | "grow" | "ephemeral" (allocated per call)
KB_MIN_RESIDENT = 8  # smallest K-block the resident regime accepts (min(Kt, this))
# largest K-block: the streamed operand needs several K-blocks per entry to overlap read / mcast / math (focus case,
# diagonal senders: kb 64 -> 212 us, 32 -> 160 us, 16 -> 159 us, 8 -> 164 us on the fastest chip; Refinement 2,
# traced: 32 -> 134-139 us, 16 -> 127 us, 8 -> 131 us). env: perf experiments only
KB_MAX = int(os.environ.get("MRS_KB_MAX", 16))
L1_HEADROOM = 64 * 1024  # bytes kept free per core beyond the op's own footprint
LAG_POLICY = os.environ.get("MRS_LAG", "auto")  # "auto": lag a relay finish only when its arrival is produced late
# upstream; "always": the design's original rule (every add waits behind the next matmul); "fin": auto, plus every
# fin wave but the last waits behind the next wave's matmul. env: perf experiments only
PREFETCH_STEPS = 1  # K-block windows a sender keeps reserved with DRAM reads in flight ahead of its multicast
# Fill levers, per regime ("resident" | "stream"). The resident regime moves both operands only in entry 0, so its
# first entry is DRAM-bound (the fill); the stream regime streams an operand every entry (Refinement 3 tunes it).
# Each value below was measured traced on the focus (resident) and MiMo (stream) cases; the stream values are the
# pre-Refinement-2 behaviour wherever the new one did not win there.
LOOKAHEAD_AFTER_BARRIER = {"resident": 1, "stream": 0}  # 1: a sender issues its look-ahead reads after the current
# step's read barrier, so the current step's reads do not share the DRAM with the next step's (focus: -1.5 us);
# 0: before it (MiMo: better, the streamed reads need the extra overlap)
READ_NOCS = {
    "resident": (0, 1),
    "stream": (0, 0),
}  # (A, W) NoC of the matmul reader's DRAM reads. The reader owns every
# read on both NoCs of its core (the writer only writes), so W on NoC1 adds DRAM-response links (focus: -2 us; both
# operands on NoC1: +8 us; MiMo stream: neutral within noise, so left at NoC0)
SENDERS_PER_LINE = {"resident": 2, "stream": 1}  # each K-block window is split into this many contiguous parts (A by
# m rows, W by k rows), each read + multicast by a different core of the line, so no single RISC serializes a whole
# window's reads and mcast (focus with W on NoC1: -1.3 us; MiMo stream: +3-5 us, noise-level)
MCAST_SENDER_PLACEMENT = {"resident": "diagonal_disjoint", "stream": "diagonal"}  # operand senders per line:
# "uniform" (first core: a column of DRAM readers) | "diagonal" (A and W both start at index 0: the cores (i, i) send
# both families) | "diagonal_disjoint" (diagonal, each W family's start shifted so no core sends two families; focus:
# -0.5-1 us; MiMo: +10 us, so the stream regime keeps "diagonal")
DEST_TILES_FP32_FULL_SYNC = 8  # DEST capacity with fp32 accumulation + full sync (subblock area cap)

TILE = 32
HOLD_TILE_BYTES = 4096  # Float32 tile
STAGE_TILE_BYTES = 2048  # bf16 tile (finished / arrivals / output)
MAILBOX_STRIDE = 16  # bytes per matmul core in a port's mailbox

NOC0 = ttnn.NOC.RISCV_0_default
NOC1 = ttnn.NOC.RISCV_1_default

# compute finish roles
ROLE_COPY, ROLE_ADD_FWD, ROLE_ADD_BWD, ROLE_ADD_BOTH = 0, 1, 2, 3
# writer handoff kinds
KIND_FWD, KIND_BWD, KIND_FIN = 0, 1, 2
_DIRS = ("fwd", "bwd")


# ======================================================================================================================
# Schedule (host, single source: build_schedule)
# ======================================================================================================================


def _role(p, b, h, G, kind):
    """(direction, is_start, distance-to-owner) of block b's half h at chip p; ("fin", False, 0) for b == p."""
    if b == p:
        return ("fin", False, 0)
    if kind == "line":
        if b > p:
            return ("fwd", p == 0, b - p)
        return ("bwd", p == G - 1, p - b)
    df = (b - p) % G
    db = G - df
    if kind == "ring_unbalanced":
        nf = G // 2
        nb = G - 1 - nf
        if df <= nf:
            return ("fwd", df == nf, df)
        return ("bwd", db == nb, db)
    # ring_balanced (even G >= 4): the opposite block is split between the two halves
    H = G // 2
    if h == 0:
        if df <= H:
            return ("fwd", df == H, df)
        return ("bwd", db == H - 1, db)
    if df <= H - 1:
        return ("fwd", df == H - 1, df)
    return ("bwd", db == H, db)


def build_schedule(p, G, kind, mirror=False):
    """roles[(b, h)] and this chip's entry order per half, entries[h]: the other blocks by descending distance-to-owner
    (ties: fwd first, then lower b), then its own block last. Both halves share one order, except with `mirror`
    (ring_balanced only): each half sorts by its own role distance and half 1 breaks ties bwd first. Then the
    equal-distance blocks of the two directions (G = 4: the fwd and bwd neighbours' blocks) run in opposite orders in
    the two halves, so every entry sends half a block each way instead of a whole block one way (the last one-way
    block is the transport tail). Deadlock freedom is unchanged: within a half, a relay at distance d waits for an
    arrival of distance d + 1, which every chip's same half produces at an earlier entry index."""
    roles = {(b, h): _role(p, b, h, G, kind) for b in range(G) for h in (0, 1)}

    def sort_key(b, h):
        if not mirror:
            key = max(roles[(b, hh)][2] for hh in (0, 1))
            return (-key, 0 if roles[(b, 0)][0] == "fwd" else 1, b)
        first = "fwd" if h == 0 else "bwd"
        return (-roles[(b, h)][2], 0 if roles[(b, h)][0] == first else 1, b)

    others = [b for b in range(G) if b != p]
    entries = {h: sorted(others, key=lambda b: sort_key(b, h)) + [p] for h in (0, 1)}
    return roles, entries


def expand_waves(roles, entries, waves):
    """The wave-split schedule: every block b becomes `waves` consecutive virtual blocks v = b * waves + w with b's
    roles, in b's place of each half's order. Deadlock freedom carries over: a relay (b, w) at position
    i * waves + w waits for the upstream's (b, w) at position j * waves + w, and j < i (the per-block argument), so the
    wave's arrival is still produced at a strictly earlier entry index."""
    if waves == 1:
        return roles, entries
    roles_v = {(b * waves + w, h): r for (b, h), r in roles.items() for w in range(waves)}
    entries_v = {h: [b * waves + w for b in order for w in range(waves)] for h, order in entries.items()}
    return roles_v, entries_v


def inc_unit(v, waves):
    """The virtual block whose unit carries the fused arrival increment covering virtual block v (itself, or its
    block's last wave when ARRIVAL_INC_PER_WAVE is 0)."""
    return v if ARRIVAL_INC_PER_WAVE else v - v % waves + waves - 1


def plan_transport(G, kind, mirror=False, waves=1):
    """Per chip position: roles, entries[h], per (half, direction) arrival list (this chip's half-h entry order), and
    the arrival thresholds (position in the upstream's half-h send order). slots[d] = zone slots per direction (max over
    chips and halves, so every chip shares one staging layout). Block ids are virtual (b * waves + w)."""
    ring = kind != "line"
    mirror = mirror and kind == "ring_balanced"
    sch = {p: expand_waves(*build_schedule(p, G, kind, mirror), waves) for p in range(G)}

    def up(p, d):
        q = p - 1 if d == "fwd" else p + 1
        if ring:
            return q % G
        return q if 0 <= q < G else None

    chips = {}
    for p in range(G):
        roles, entries = sch[p]
        arrivals, thresholds, produced_at = {}, {}, {}
        for h in (0, 1):
            for d in _DIRS:
                u = up(p, d)
                lst = []
                for b in entries[h]:
                    r = roles[(b, h)]
                    if r[0] == d and not r[1]:
                        assert u is not None and sch[u][0][(b, h)][0] == d, (p, b, h, d)
                        lst.append(b)
                    elif r[0] == "fin" and u is not None and sch[u][0][(b, h)][0] == d:
                        lst.append(b)
                seq = [b for b in sch[u][1][h] if sch[u][0][(b, h)][0] == d] if u is not None else []
                assert sorted(seq) == sorted(lst), (p, h, d, seq, lst)
                arrivals[(h, d)] = lst
                incs = [b for b in seq if inc_unit(b, waves) == b]  # the upstream's units carrying an increment
                thresholds[(h, d)] = {b: incs.index(inc_unit(b, waves)) + 1 for b in seq}
                # the upstream's entry index that produces each arrival's increment (drives the per-entry lag
                # decision)
                produced_at[(h, d)] = {b: sch[u][1][h].index(inc_unit(b, waves)) for b in seq}
                # deadlock freedom (mirrored orders): a relay waits only for an arrival produced at an earlier
                # upstream entry index (the shared descending-key order instead lags such a finish, see op_design)
                for b in lst if mirror else []:
                    if roles[(b, h)][0] != "fin":
                        assert produced_at[(h, d)][b] < entries[h].index(b), (p, b, h, d)
        chips[p] = dict(roles=roles, entries=entries, arrivals=arrivals, thresholds=thresholds, produced_at=produced_at)
    slots = {d: max(1, max(len(chips[p]["arrivals"][(h, d)]) for p in range(G) for h in (0, 1))) for d in _DIRS}
    return chips, slots


def entry_finish_plan(sc, h):
    """[(arrival mask, lag)] per entry of half h on a chip with schedule sc. mask: bit 0 fwd / bit 1 bwd arrival
    added in (relay: its direction; fin: every direction with an upstream). lag: finish behind the next entry's matmul,
    only when an arrival is produced at the same or a later position of the upstream's walk (an arrival produced
    earlier has, most likely, landed), never on the last entry."""
    roles, entries, arrivals = sc["roles"], sc["entries"][h], sc["arrivals"]
    out = []
    for e_idx, b in enumerate(entries):
        r = roles[(b, h)]
        mask = 0
        if r[0] in _DIRS and not r[1]:
            mask = 1 if r[0] == "fwd" else 2
        elif r[0] == "fin":
            mask = (1 if b in arrivals[(h, "fwd")] else 0) | (2 if b in arrivals[(h, "bwd")] else 0)
        lag = LAG_POLICY == "always" and mask != 0
        lag = lag or (LAG_POLICY == "fin" and r[0] == "fin" and mask != 0)  # a fin wave behind the next wave's matmul
        lag = lag or any(sc["produced_at"][(h, d)][b] >= e_idx for d, bit in (("fwd", 1), ("bwd", 2)) if mask & bit)
        out.append((mask, int(lag and e_idx + 1 < len(entries))))
    return out


def hold_depth(chips_sched):
    """fp32 hold slots: DEPTH_HOLD if any chip's walk lags an entry (its slot stays live across the next matmul),
    else 1."""
    lags = any(lag for sc in chips_sched.values() for h in (0, 1) for _, lag in entry_finish_plan(sc, h))
    return DEPTH_HOLD if lags else 1


# ======================================================================================================================
# Work distribution / regime selection
# ======================================================================================================================


def _subblock(core_m, core_n, cap=DEST_TILES_FP32_FULL_SYNC, prefer_tall=False):
    best = (1, 1)
    for h in range(1, core_m + 1):
        if core_m % h:
            continue
        for w in range(1, core_n + 1):
            if core_n % w or h * w > cap:
                continue
            tie_better = (h > best[0]) if prefer_tall else (w > best[1])
            if h * w > best[0] * best[1] or (h * w == best[0] * best[1] and tie_better):
                best = (h, w)
    return best


def plan_mcast_senders(placement, grid_m, grid_n, a_two_parts, w_two_parts, a_shape_1d, w_shape_1d):
    """[(family, Mcast1D shape, starting_sender_index | None (absent), Mcast1DSenderPlacement)] for "a", "a2", "b",
    "b2". A lines are indexed by gm (length grid_n, sender at gn = (start + gm) % grid_n when diagonal), W lines by gn
    (length grid_m, sender at gm = (start + gn) % grid_m). Part-1 senders start half a line further along. With
    "diagonal_disjoint" each W family's start is the smallest one whose sender cores overlap the senders already
    placed the least, so one NCRISC never serializes the DRAM reads + multicasts of both operands."""
    diag = placement in ("diagonal", "diagonal_disjoint")
    mode = ttnn.Mcast1DSenderPlacement.Diagonal if diag else ttnn.Mcast1DSenderPlacement.Uniform

    def cores(fam, start):
        if fam == "a":
            return {(gm, (start + gm) % grid_n if diag else start) for gm in range(grid_m)}
        return {((start + gn) % grid_m if diag else start, gn) for gn in range(grid_n)}

    taken, plan = set(), []
    for prefix, shape_1d, active, line_len in (
        ("a", a_shape_1d, True, grid_n),
        ("a2", a_shape_1d, a_two_parts, grid_n),
        ("b", w_shape_1d, True, grid_m),
        ("b2", w_shape_1d, w_two_parts, grid_m),
    ):
        if not active:
            plan.append((prefix, shape_1d, None, mode))
            continue
        start = 0 if len(prefix) == 1 else line_len // 2
        if placement == "diagonal_disjoint" and prefix[0] == "b":
            start = min(range(line_len), key=lambda s0: (len(cores("b", s0) & taken), (s0 - start) % line_len))
        taken |= cores(prefix[0], start)
        plan.append((prefix, shape_1d, start, mode))
    return plan


def choose_grid(gx, gy, block_m, block_n, L):
    """(strip, orientation, grid_m, grid_n, core_m, core_n, (sb_h, sb_w)) minimizing the per-core block."""
    best = None
    for strip in [t for t in ("row", "col") if os.environ.get("MRS_STRIP", t) == t]:  # env: perf experiments only
        strip_len = gx if strip == "row" else gy
        if 2 * L > strip_len:
            continue
        rect = (gx, gy - 1) if strip == "row" else (gx - 1, gy)
        for orient in [o for o in ("m_on_x", "m_on_y") if os.environ.get("MRS_ORIENT", o) == o]:
            ext_m, ext_n = (rect[0], rect[1]) if orient == "m_on_x" else (rect[1], rect[0])
            core_m = -(-block_m // min(ext_m, block_m))
            grid_m = -(-block_m // core_m)
            core_n = -(-block_n // min(ext_n, block_n))
            grid_n = -(-block_n // core_n)
            sb = _subblock(core_m, core_n)
            ragged = grid_m * core_m * grid_n * core_n - block_m * block_n
            key = (
                PREFER_PORT_ROW_STRIP and strip != "row",
                core_m * core_n,
                -sb[0] * sb[1],
                ragged,
                grid_m * grid_n,
                strip != "row",
            )
            cand = dict(strip=strip, orient=orient, grid_m=grid_m, grid_n=grid_n, core_m=core_m, core_n=core_n, sb=sb)
            if best is None or key < best[0]:
                best = (key, cand)
    if best is None:
        raise ValueError(f"matmul_reduce_scatter: no transport strip of the {gx}x{gy} grid fits {2 * L} ports")
    return best[1]


def window_depths(resident, depth):
    """(A, W) window depths in K-blocks: the stream regime's DEPTH_IN0 / DEPTH_IN1, or under the resident regime
    `depth` for the streamed operand (the resident operand's CB holds the whole Kt slice; its depth is unused)."""
    return (depth, depth) if resident else (DEPTH_IN0, DEPTH_IN1)


def wave_plan(resident, scatter_dim, core_m, core_n, waves=None):
    """The per-core block's waves: dict(n=waves, win_m, win_n, P (= slot tiles, the largest wave), descs). descs[w] =
    (row_off, col_off, rows, cols) inside the per-core block; a wave's tiles are dense (rows x cols, TileRowMajor) at
    the front of its nominal P-tile slot. Waves run along the streamed operand's axis (resident A: n, ragged last wave
    allowed; resident W: m, only when they divide core_m, since matmul_block pops in0 by its shape), so the resident
    operand is shared by every wave unchanged and the streamed one still crosses DRAM once."""
    waves = ENTRIES_PER_BLOCK if waves is None else waves
    W, win_m, win_n = 1, core_m, core_n
    if resident and waves > 1:
        if scatter_dim == -1 and core_n >= 2:
            win_n = -(-core_n // min(waves, core_n))
            W = -(-core_n // win_n)
        elif scatter_dim == -2 and core_m % waves == 0 and core_m >= waves:
            W, win_m = waves, core_m // waves
    descs = []
    for w in range(W):  # only one axis is split, so the other offset is 0
        r0, c0 = (w * win_m if win_m < core_m else 0), (w * win_n if win_n < core_n else 0)
        descs.append((r0, c0, min(win_m, core_m - r0), min(win_n, core_n - c0)))
    return dict(n=W, win_m=win_m, win_n=win_n, P=win_m * win_n, descs=descs)


def footprint(
    kb,
    resident,
    *,
    depth=None,
    Kt,
    core_m,
    core_n,
    tA,
    tW,
    scatter_dim,
    slots,
    hold_depths,
    waves_of,
    staging_floor_tiles=0,
    depths=None,
    zone_sets=1,
):
    """Per matmul core L1 bytes (l1_ledger.md): operand windows + fp32 hold + the staging slice. The staging tensor is
    cached grow-only per mesh, so it is counted at max(what this call needs, what is already allocated).
    Per regime (keyed by `resident`): hold_depths (hold_depth() of the schedule that regime runs), slots (its zone
    slots per direction) and waves_of (its wave_plan: the streamed window and the slot size are a wave's).
    zone_sets: chunks whose arrivals the zones hold at once (1 unchunked)."""
    wv = waves_of[resident]
    P = wv["P"]
    rA = resident and scatter_dim == -1
    rW = resident and scatter_dim == -2
    dA, dW = depths or window_depths(resident, depth if depth is not None else RESIDENT_STREAM_DEPTHS[0])
    a = tA * (core_m * Kt if rA else dA * wv["win_m"] * kb)
    w = tW * (Kt * core_n if rW else dW * kb * wv["win_n"])
    hold = HOLD_TILE_BYTES * hold_depths[resident] * P
    sl = slots[resident]
    staging = STAGE_TILE_BYTES * max(staging_floor_tiles, (DEPTH_HANDOFF + zone_sets * (sl["fwd"] + sl["bwd"])) * P)
    return a + w + hold + staging + STAGE_TILE_BYTES  # + the zero tile (CB_ZERO)


def choose_k_block(Kt, budget, force_resident, kb_max=None, **geo):
    """(resident, kb, depth): resident with the preferred kb | Kt (largest <= KB_MAX, then smaller, then larger), kb >=
    min(Kt, KB_MIN_RESIDENT), at the first RESIDENT_STREAM_DEPTHS depth where any such kb fits; else stream with the
    preferred kb | Kt that fits (depth: None, the stream regime's DEPTH_IN0 / DEPTH_IN1). force_resident (test hook)
    replaces the residency decision. The resident regime's cap is KB_MAX x its wave count, so a wave's streamed window
    keeps the bytes (and the multicast rounds per block) of an unsplit one (focus / GLM, 2 waves: kb 16 -> 128.5 /
    214 us, kb 32 -> 126.6 / 197 us)."""

    def divisors(cap):
        # preference order: the largest divisor <= cap first, then smaller ones, then (no fit below the cap) larger
        out = [d for d in range(min(Kt, cap), 0, -1) if Kt % d == 0]
        return out + [d for d in range(cap + 1, Kt + 1) if Kt % d == 0]

    kb_max = KB_MAX if kb_max is None else kb_max
    if force_resident is not False:
        for depth in RESIDENT_STREAM_DEPTHS:
            for kb in divisors(kb_max * geo["waves_of"][True]["n"]):
                if kb < min(Kt, KB_MIN_RESIDENT):
                    continue
                if footprint(kb, True, depth=depth, Kt=Kt, **geo) <= budget:
                    return True, kb, depth
        if force_resident is True:
            raise ValueError("matmul_reduce_scatter: the resident cross-block operand does not fit L1 for this shape")
    for kb in divisors(kb_max):
        if footprint(kb, False, Kt=Kt, **geo) <= budget:
            return False, kb, None
    raise NotImplementedError(
        "matmul_reduce_scatter: even kb=1 exceeds the L1 budget (regime DRAM-landing is deferred)"
    )


def super_block_plan(Kt, kb, resident, depth, budget, geo):
    """(S, (dA, dW)): the compute's DEST-resident super-block (K-windows per pack) and the operand window depths. S =
    SUPER_BLOCK when it divides Kt / kb and the streamed windows deepened to S + SUPER_BLOCK_EXTRA still fit L1; else
    1 at the regime's own depths. The resident operand's depth is unused (its CB holds the whole Kt slice)."""
    base = window_depths(resident, depth)
    S = SUPER_BLOCK
    if S > 1 and (Kt // kb) % S == 0:
        d = S + SUPER_BLOCK_EXTRA
        deep = (max(d, base[0]), max(d, base[1]))
        if footprint(kb, resident, depth=depth, Kt=Kt, depths=deep, **geo) <= budget:
            return S, deep
    return 1, base


def chunk_candidates(block_m, block_n, scatter_dim):
    """(n_m, n_n) in the auto search order: fewest chunks first; at equal count, more splits along the axis the chunk
    walk keeps inner (scatter -1: n, the resident A rows stay put; scatter -2: m). Only distinct chunk grids (a split
    whose chunks would come out the same size as a smaller split is skipped)."""
    inner_n = scatter_dim == -1
    seen = set()
    for total in range(1, min(MAX_AUTO_CHUNKS, block_m * block_n) + 1):  # a generator: the first fit ends the search
        pairs = [(total // d, d) for d in range(1, total + 1) if total % d == 0]  # (n_m, n_n)
        pairs.sort(key=lambda p: -(p[1] if inner_n else p[0]))
        for n_m, n_n in pairs:
            if n_m > block_m or n_n > block_n:
                continue
            key = (-(-block_m // n_m), -(-block_n // n_n))
            if key in seen:
                continue
            seen.add(key)
            yield n_m, n_n


def plan_chunk_geometry(
    n_m,
    n_n,
    *,
    G,
    kind,
    L,
    Kt,
    block_m,
    block_n,
    gx,
    gy,
    tA,
    tW,
    scatter_dim,
    staging_floor,
    budget,
    force_resident,
):
    """Grid, regime and schedule for one chunk of chunk_m x chunk_n tiles (the whole block when n_m = n_n = 1). Raises
    NotImplementedError when the chunk's footprint misses the L1 budget."""
    chunk_m, chunk_n = -(-block_m // n_m), -(-block_n // n_n)
    n_m, n_n = -(-block_m // chunk_m), -(-block_n // chunk_n)  # no empty trailing chunk
    zone_sets = 1 if n_m * n_n == 1 else max(1, min(ZONE_SETS, n_m * n_n))
    gd = choose_grid(gx, gy, chunk_m, chunk_n, L)
    core_m, core_n = gd["core_m"], gd["core_n"]
    if kind == "ring_balanced" and gd["grid_m"] * gd["grid_n"] < 2:
        raise ValueError("matmul_reduce_scatter: a balanced ring needs >= 2 matmul cores (two halves)")
    # halves: with a resident cross-block operand, split along the streamed operand's multicast lines (W lines = fixed
    # gn for scatter -1, A lines = fixed gm for scatter -2), so the two halves may compute different blocks and walk
    # mirrored orders; else the row-major split of the core index. Each regime's schedule fixes its hold depth, which
    # the footprint (and so the regime choice) needs.
    mirror_split = "k"
    if int(os.environ.get("MRS_MIRROR", MIRROR_HALF_ORDERS)) and kind == "ring_balanced":  # env: perf experiments only
        if scatter_dim == -1 and gd["grid_n"] >= 2:
            mirror_split = "n"
        elif scatter_dim == -2 and gd["grid_m"] >= 2:
            mirror_split = "m"
    waves_of = {r: wave_plan(r, scatter_dim, core_m, core_n) for r in (False, True)}
    sched_of = {
        r: plan_transport(G, kind, mirror=r and mirror_split != "k", waves=waves_of[r]["n"]) for r in (False, True)
    }
    hold_depths = {r: hold_depth(sched_of[r][0]) for r in (False, True)}
    geo = dict(
        core_m=core_m,
        core_n=core_n,
        tA=tA,
        tW=tW,
        scatter_dim=scatter_dim,
        slots={r: sched_of[r][1] for r in (False, True)},
        hold_depths=hold_depths,
        waves_of=waves_of,
        staging_floor_tiles=staging_floor,
        zone_sets=zone_sets,
    )
    resident, kb, depth = choose_k_block(
        Kt, budget, force_resident, kb_max=KB_MAX if n_m * n_n == 1 else KB_MAX_CHUNKED, **geo
    )
    if os.environ.get("MRS_KB"):  # perf experiments only: cap the K-block (largest divisor of Kt <= MRS_KB)
        kb = max(d for d in range(1, kb + 1) if Kt % d == 0 and d <= int(os.environ["MRS_KB"]))
    super_block, depths = super_block_plan(Kt, kb, resident, depth, budget, geo)
    chips_sched, slots = sched_of[resident]
    waves = waves_of[resident]
    return dict(
        n_m=n_m,
        n_n=n_n,
        chunk_m=chunk_m,
        chunk_n=chunk_n,
        zone_sets=zone_sets,
        gd=gd,
        resident=resident,
        kb=kb,
        depths=depths,
        super_block=super_block,
        depth_hold=hold_depths[resident],
        half_split=mirror_split if resident else "k",
        sched=chips_sched,
        slots=slots,
        waves=waves,
        P=waves["P"],  # slot tiles: every hold / finished / zone slot holds one wave
        footprint=footprint(kb, resident, depth=depth, Kt=Kt, depths=depths, **geo),
    )


def chunk_cost_us(pl, *, G, L, Mt, Kt, Nt, tA, tW, scatter_dim):
    """Estimated traced latency (us) of plan `pl` (plan_chunk_geometry): the largest of compute, DRAM and transport
    (constants above) plus the program floor. Only ranks the auto split's candidates."""
    n_m, n_n = pl["n_m"], pl["n_n"]
    C = n_m * n_n
    cores = pl["gd"]["grid_m"] * pl["gd"]["grid_n"]
    steps = C * G * pl["waves"]["n"] * (Kt // pl["kb"])
    step_us = COST_STEP_US if pl["resident"] else COST_STEP_STREAM_US
    compute = C * G * pl["P"] * pl["waves"]["n"] * Kt * 32 / CLOCK_MHZ / COST_FPU_EFF + step_us * steps
    a_bytes, w_bytes = Mt * Kt * tA, Kt * Nt * tW
    if scatter_dim == -1:  # A shared by every block (resident: once), W columns per block (re-read per m chunk)
        traffic = (a_bytes if pl["resident"] else G * n_n * a_bytes) + n_m * w_bytes
    else:  # W shared by every block, A rows per block (re-read per n chunk)
        traffic = (w_bytes if pl["resident"] else G * n_m * w_bytes) + n_n * a_bytes
    dram = traffic / (COST_DRAM_GBPS * 1e3)
    transport = COST_HANDOFF_US * C * (G - 1) * cores / L
    return COST_FLOOR_US + max(compute, dram, transport)


_PLAN_CACHE = {}  # auto split per (shape, dtypes, group, mesh grid, budget, knobs): the search plans many candidates


def _place_ports(strip, gx, gy, L):
    """{(d, l): logical core} in the transport strip, nearest the Ethernet row."""
    cores = {}
    n = 2 * L
    for i in range(n):
        d, l = _DIRS[i % 2], i // 2
        if strip == "col":
            cores[(d, l)] = ttnn.CoreCoord(gx - 1, i)
        else:
            cores[(d, l)] = ttnn.CoreCoord(min(gx - 1, (2 * i + 1) * gx // (2 * n)), 0)
    return cores


def _place_ports_near_eth(mesh, coord, strip, gx, gy, L, peers, node, links):
    """{(d, l): logical core} in the strip, each linked port at the strip cell with the fewest NoC1 hops to its link's
    Ethernet core (fabric rule: a sender right below its router). No probe dispatch: the router's Ethernet channel is
    the first runtime arg setup_fabric_connection emits (asked on a scratch descriptor); it indexes the chip's
    non-harvested Ethernet list. Falls back to _place_ports when the Ethernet map is unavailable."""
    from ttnn.operations.examples.fabric_all_gather.program_descriptor_with_inline_kernels import (
        _chip_maps,
        _worker_nearest_noc1,
    )

    allowed = {(x, 0) for x in range(gx)} if strip == "row" else {(gx - 1, y) for y in range(gy)}
    eth_list = _chip_maps(mesh, coord)[0]
    if eth_list is None:
        return _place_ports(strip, gx, gy, L)
    cores, taken = {}, set()
    for l in range(L):
        for d in _DIRS:
            peer = peers[d]
            if peer is None:
                continue
            scratch = ttnn.ProgramDescriptor()
            link = links[(coord, peer)][l]
            chan = list(ttnn.setup_fabric_connection(node(coord), node(peer), link, scratch, ttnn.CoreCoord(0, 0)))[0]
            if chan >= len(eth_list):
                return _place_ports(strip, gx, gy, L)
            core = _worker_nearest_noc1(mesh, coord, eth_list[chan], taken, allowed)
            if core is None:
                return _place_ports(strip, gx, gy, L)
            cores[(d, l)] = core
            taken.add((core.x, core.y))
    for key in _place_ports(strip, gx, gy, L):  # receive-only ends: any free strip cell
        if key not in cores:
            free = sorted(allowed - taken)
            cores[key] = ttnn.CoreCoord(*free[0])
            taken.add(free[0])
    return cores


# ======================================================================================================================
# Per-mesh persistent resources (created once; global semaphores synchronized once at creation)
# ======================================================================================================================

_MESH_CACHE = {}


def _full_grid(mesh):
    g = mesh.compute_with_storage_grid_size()
    return int(g.x), int(g.y)


def _crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def _mesh_resources(mesh, staging_tiles):
    key = id(mesh)
    gx, gy = _full_grid(mesh)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
    res = _MESH_CACHE.get(key)
    if res is None or res["mesh"] is not mesh:
        import torch

        sems = [ttnn.create_global_semaphore(mesh, grid, 0) for _ in range(4)]  # arr_fwd, arr_bwd, ready, credit
        n = gx * gy
        mbox_cfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, (1, n * MAILBOX_STRIDE // 4), ttnn.ShardOrientation.ROW_MAJOR),
        )
        mailbox = ttnn.from_torch(
            torch.zeros((n, n * MAILBOX_STRIDE // 4), dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=mbox_cfg,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        ttnn.synchronize_device(mesh)  # every counter / mailbox zero before any chip's first call
        res = dict(mesh=mesh, sems=sems, mailbox=mailbox, staging=None, staging_tiles=0)
        _MESH_CACHE[key] = res
    if res["staging_tiles"] < staging_tiles or (STAGING_POLICY == "exact" and res["staging_tiles"] != staging_tiles):
        if res["staging"] is not None:
            ttnn.deallocate(res["staging"])
            res["staging"] = None
        n = gx * gy
        cfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, (staging_tiles * TILE, TILE), ttnn.ShardOrientation.ROW_MAJOR),
        )
        res["staging"] = ttnn.allocate_tensor_on_device(
            ttnn.Shape([n * staging_tiles * TILE, TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, cfg
        )
        res["staging_tiles"] = staging_tiles
    return res


def _l1_budget(mesh):
    """L1_bank_size - allocator base - headroom: the allocatable L1 region of a worker core minus L1_HEADROOM."""
    l1_top = int(ttnn._ttnn.reports.get_device_info(mesh).l1_bank_size)
    # Live L1 buffers grow down from the top of the bank; the CB region must end below the lowest of them.
    lowest = ttnn._ttnn.device.GetLowestOccupiedComputeL1Address(mesh)
    if lowest is not None:
        l1_top = min(l1_top, int(lowest) - int(ttnn._ttnn.device.GetAllocatorBaseAddress(mesh, ttnn.BufferType.L1)))
    return l1_top - L1_HEADROOM


# ======================================================================================================================
# Caller checks
# ======================================================================================================================


def _check_dram_interleaved_tile(t, what):
    mc = t.memory_config()
    if mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:  # the readers address it through TensorAccessor
        raise ValueError(f"matmul_reduce_scatter: {what} must be interleaved (DRAM or L1)")
    if t.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"matmul_reduce_scatter: {what} must be TILE layout")


def _config_wraps(fabric_config, cluster_axis):
    fc = ttnn.FabricConfig
    if fabric_config in (fc.FABRIC_2D_TORUS_XY, fc.FABRIC_1D_RING):
        return True
    if fabric_config == fc.FABRIC_2D_TORUS_X:
        return cluster_axis == 1
    if fabric_config == fc.FABRIC_2D_TORUS_Y:
        return cluster_axis == 0
    return False


def build_groups(mesh_shape, cluster_axis):
    rows, cols = mesh_shape
    if cluster_axis == 0:
        return [[(r, c) for r in range(rows)] for c in range(cols)]
    return [[(r, c) for c in range(cols)] for r in range(rows)]


# ======================================================================================================================
# Entry
# ======================================================================================================================


def run(a, w, *, cluster_axis, scatter_dim, topology, num_links, compute_kernel_config, memory_config, hooks):
    mesh = a.device()
    a_shape, w_shape = list(a.shape), list(w.shape)
    if not 2 <= len(a_shape) <= 4 or any(s != 1 for s in a_shape[:-2]):
        raise ValueError(f"matmul_reduce_scatter: A must be rank 2-4 with leading dims 1, got {a_shape}")
    if not 2 <= len(w_shape) <= 4 or any(s != 1 for s in w_shape[:-2]):
        raise ValueError(f"matmul_reduce_scatter: W must be (K, N) with leading dims 1, got {w_shape}")
    w_shape = w_shape[-2:]
    M, K = a_shape[-2:]
    K2, N = w_shape
    if K != K2:
        raise ValueError(f"matmul_reduce_scatter: K mismatch, A {a_shape} vs W {w_shape}")
    _check_dram_interleaved_tile(a, "input_tensor")
    _check_dram_interleaved_tile(w, "weight")
    if memory_config is not None and memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise NotImplementedError("matmul_reduce_scatter: sharded output memory_config (no placement axis in TARGET)")
    out_mem = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    mesh_shape = tuple(mesh.shape)
    G = mesh_shape[cluster_axis]
    if G < 2:
        raise ValueError(f"matmul_reduce_scatter: cluster_axis {cluster_axis} has {G} device(s); need >= 2")
    extent = M if scatter_dim == -2 else N
    if extent % (TILE * G) != 0:
        raise ValueError(f"matmul_reduce_scatter: scattered extent {extent} does not split into {G} tile blocks")

    ring = topology == Topology.Ring and G >= 3  # a 2-device ring is the line
    kind = "line" if not ring else ("ring_balanced" if G % 2 == 0 else "ring_unbalanced")
    groups = build_groups(mesh_shape, cluster_axis)
    node = lambda coord: mesh.get_fabric_node_id(ttnn.MeshCoordinate(*coord))
    if ring:
        if not _config_wraps(ttnn.get_fabric_config(), cluster_axis):
            raise ValueError(
                f"matmul_reduce_scatter: Ring along axis {cluster_axis} needs a fabric config that wraps it "
                f"(active: {ttnn.get_fabric_config()})"
            )
    hops = []
    for grp in groups:
        pairs = list(zip(grp[:-1], grp[1:])) + ([(grp[-1], grp[0])] if ring else [])
        for s, d in pairs:
            hops += [(s, d), (d, s)]
    links = {(s, d): list(ttnn.get_forwarding_link_indices(node(s), node(d))) for s, d in hops}
    avail = min(len(v) for v in links.values())
    if avail == 0:
        raise ValueError("matmul_reduce_scatter: a hop of the group has no usable fabric link (no wrap link?)")
    if num_links is None:
        L = avail
    else:
        if num_links <= 0 or num_links > avail:
            raise ValueError(f"matmul_reduce_scatter: num_links={num_links}, but a hop has only {avail} link(s)")
        L = num_links

    # ---- geometry ----
    Mt, Kt, Nt = M // TILE, K // TILE, N // TILE
    block_m, block_n = (Mt // G, Nt) if scatter_dim == -2 else (Mt, Nt // G)
    gx, gy = _full_grid(mesh)
    tA = int(a.buffer_aligned_page_size())
    tW = int(w.buffer_aligned_page_size())
    cached = _MESH_CACHE.get(id(mesh))
    staging_floor = cached["staging_tiles"] if cached is not None and cached["mesh"] is mesh else 0
    if STAGING_POLICY == "exact":
        staging_floor = 0  # reallocated to this call's size below
    budget = _l1_budget(mesh)
    if cached is not None and cached["mesh"] is mesh and cached["staging"] is not None:
        budget += cached["staging_tiles"] * STAGE_TILE_BYTES  # our own staging is below the other L1 buffers
    force_resident = getattr(hooks, "_FORCE_XBLOCK_RESIDENCY", None)
    forced = getattr(hooks, "_FORCE_CHUNKS", None) or (
        tuple(int(v) for v in os.environ["MRS_CHUNKS"].split(",")) if os.environ.get("MRS_CHUNKS") else None
    )
    plan_args = dict(
        G=G,
        kind=kind,
        L=L,
        Kt=Kt,
        block_m=block_m,
        block_n=block_n,
        gx=gx,
        gy=gy,
        tA=tA,
        tW=tW,
        scatter_dim=scatter_dim,
        staging_floor=staging_floor,
        budget=budget,
        force_resident=force_resident,
    )
    if forced is not None:
        pl = plan_chunk_geometry(*forced, **plan_args)
    else:
        pl = None
        try:
            pl = plan_chunk_geometry(1, 1, **plan_args)  # whatever fits unchunked keeps its plan
        except NotImplementedError:
            key = (
                tuple(sorted((k, v) for k, v in plan_args.items() if k != "force_resident")),
                force_resident,
                AUTO_CHUNK_POLICY,
                ZONE_SETS,
                KB_MAX_CHUNKED,
            )
            if key in _PLAN_CACHE:
                pl = plan_chunk_geometry(*_PLAN_CACHE[key], **plan_args)
            else:
                cost_args = dict(G=G, L=L, Mt=Mt, Kt=Kt, Nt=Nt, tA=tA, tW=tW, scatter_dim=scatter_dim)
                best = None
                for n_m, n_n in chunk_candidates(block_m, block_n, scatter_dim):
                    if n_m * n_n == 1:
                        continue
                    try:
                        cand = plan_chunk_geometry(n_m, n_n, **plan_args)
                    except (NotImplementedError, ValueError):
                        continue
                    if AUTO_CHUNK_POLICY == "fewest":
                        best = (0.0, cand)
                        break
                    cost = chunk_cost_us(cand, **cost_args)
                    if best is None or cost < best[0]:
                        best = (cost, cand)
                if best is None:
                    raise NotImplementedError(
                        f"matmul_reduce_scatter: no split of the {block_m}x{block_n}-tile block into <= "
                        f"{MAX_AUTO_CHUNKS} chunks fits L1"
                    )
                pl = best[1]
                _PLAN_CACHE[key] = (pl["n_m"], pl["n_n"])
    gd, resident, kb, P = pl["gd"], pl["resident"], pl["kb"], pl["P"]
    chips_sched, slots, waves = pl["sched"], pl["slots"], pl["waves"]
    hooks._LAST_PLAN = {
        "xblock": "resident" if resident else "stream",
        "schedule": kind,
        "kb": kb,
        "depths": pl["depths"],
        "super_block": pl["super_block"],
        "hold_depth": pl["depth_hold"],
        "grid": (gd["grid_m"], gd["grid_n"]),
        "strip": gd["strip"],
        "half_split": pl["half_split"],
        "waves": waves["n"],
        "chunks": (pl["n_m"], pl["n_n"]),
        "chunk_tiles": (pl["chunk_m"], pl["chunk_n"]),
        "zone_sets": pl["zone_sets"],
        "P": P,
        "footprint_kib": pl["footprint"] // 1024,
    }

    staging_tiles = (DEPTH_HANDOFF + pl["zone_sets"] * (slots["fwd"] + slots["bwd"])) * P
    res = _mesh_resources(mesh, staging_tiles)

    out_shape = list(a_shape)
    out_shape[-1] = N
    if scatter_dim == -2:
        out_shape[-2] = M // G
    else:
        out_shape[-1] = N // G
    output = ttnn.allocate_tensor_on_device(ttnn.Shape(out_shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, out_mem)

    ctx = dict(
        mesh=mesh,
        a=a,
        w=w,
        output=output,
        res=res,
        G=G,
        kind=kind,
        L=L,
        links=links,
        groups=groups,
        node=node,
        scatter_dim=scatter_dim,
        Mt=Mt,
        Kt=Kt,
        Nt=Nt,
        block_m=block_m,
        block_n=block_n,
        gd=gd,
        P=P,
        kb=kb,
        depths=pl["depths"],
        super_block=pl["super_block"],
        depth_hold=pl["depth_hold"],
        resident=resident,
        slots=slots,
        sched=chips_sched,
        half_split=pl["half_split"],
        waves=waves,
        n_m=pl["n_m"],
        n_n=pl["n_n"],
        chunk_m=pl["chunk_m"],
        chunk_n=pl["chunk_n"],
        zone_sets=pl["zone_sets"],
        compute_kernel_config=compute_kernel_config,
        gx=gx,
        gy=gy,
    )
    desc = create_mesh_program_descriptor(ctx)
    result = ttnn.generic_op([a, w, output], desc)
    if STAGING_POLICY == "ephemeral":
        # Free the L1 staging once the program is enqueued: later ops reuse it only after this one finishes, and
        # peers write into it only while this program runs. Keeps it from crowding other ops' static CBs.
        ttnn.deallocate(res["staging"])
        res["staging"], res["staging_tiles"] = None, 0
    return result


# ======================================================================================================================
# Program descriptors
# ======================================================================================================================


def create_mesh_program_descriptor(ctx):
    mesh = ctx["mesh"]
    G, L, kind, P = ctx["G"], ctx["L"], ctx["kind"], ctx["P"]
    gd = ctx["gd"]
    gx, gy = ctx["gx"], ctx["gy"]
    core_m, core_n, kb, Kt, Nt = gd["core_m"], gd["core_n"], ctx["kb"], ctx["Kt"], ctx["Nt"]
    waves = ctx["waves"]
    W = waves["n"]
    win_m, win_n = waves["win_m"], waves["win_n"]  # streamed windows: A win_m x kb, W kb x win_n (one wave)
    num_entries = G * W
    # per wave: reader / writer (offsets + dense extents), compute (extents + subblock), ports (tiles)
    wave_words = [r0 | (c0 << 8) | (rows << 16) | (cols << 24) for r0, c0, rows, cols in waves["descs"]]
    tall = bool(os.environ.get("MRS_SB_TALL"))  # perf experiments only: prefer taller subblocks on ties
    wave_subblocks = [_subblock(rows, cols, prefer_tall=tall) for _, _, rows, cols in waves["descs"]]
    wave_mm_words = [
        rows | (cols << 8) | (h << 16) | (w_ << 24)
        for (_, _, rows, cols), (h, w_) in zip(waves["descs"], wave_subblocks)
    ]
    wave_tiles = [rows * cols for _, _, rows, cols in waves["descs"]]
    grid_m, grid_n = gd["grid_m"], gd["grid_n"]
    block_m, block_n = ctx["block_m"], ctx["block_n"]
    scatter_dim, resident, slots = ctx["scatter_dim"], ctx["resident"], ctx["slots"]
    a, w, output, res = ctx["a"], ctx["w"], ctx["output"], ctx["res"]
    ring = kind != "line"
    virt = lambda c: mesh.worker_core_from_logical_core(c)
    node = ctx["node"]

    # ---- matmul grid ----
    x0, y0 = (0, 1) if gd["strip"] == "row" else (0, 0)

    def mm_core(gm, gn):
        if gd["orient"] == "m_on_x":
            return ttnn.CoreCoord(x0 + gm, y0 + gn)
        return ttnn.CoreCoord(x0 + gn, y0 + gm)

    mm_cores = [(gm, gn, mm_core(gm, gn)) for gm in range(grid_m) for gn in range(grid_n)]  # k = row-major index
    n_cores = len(mm_cores)
    if ctx["half_split"] == "n":
        half = lambda k: int(mm_cores[k][1] >= -(-grid_n // 2))
    elif ctx["half_split"] == "m":
        half = lambda k: int(mm_cores[k][0] >= -(-grid_m // 2))
    else:
        half = lambda k: 0 if k < -(-n_cores // 2) else 1
    link_of = lambda k: k % L
    mm_crs = ttnn.CoreRangeSet([ttnn.CoreRange(mm_core(0, 0), mm_core(grid_m - 1, grid_n - 1))])
    ports_of = {}  # per chip: {(d, l): logical port core}
    for grp in ctx["groups"]:
        for p, coord in enumerate(grp):
            peers = {
                "fwd": grp[(p + 1) % G] if (ring or p < G - 1) else None,
                "bwd": grp[(p - 1) % G] if (ring or p > 0) else None,
            }
            placement_mode = os.environ.get("MRS_PORT_PLACEMENT", PORT_PLACEMENT)  # env: perf experiments only
            if placement_mode == "eth" and not os.environ.get("TT_METAL_EMULE_MODE"):
                ports_of[coord] = _place_ports_near_eth(mesh, coord, gd["strip"], gx, gy, L, peers, node, ctx["links"])
            else:
                ports_of[coord] = _place_ports(gd["strip"], gx, gy, L)
    link_halves = {l: {half(k) for k in range(n_cores) if link_of(k) == l} for l in range(L)}
    # A is shared along an n-line (same gm), W along an m-line (same gn)
    a_shape_1d = ttnn.Mcast1DShape.PerColumn if gd["orient"] == "m_on_x" else ttnn.Mcast1DShape.PerRow
    w_shape_1d = ttnn.Mcast1DShape.PerRow if gd["orient"] == "m_on_x" else ttnn.Mcast1DShape.PerColumn

    nkb = Kt // kb
    regime = "resident" if resident else "stream"
    senders = int(os.environ.get("MRS_SENDERS", SENDERS_PER_LINE[regime]))  # env: perf experiments only
    a_split = -(-win_m // 2) if senders >= 2 and win_m >= 2 else win_m  # m rows of A part 0
    w_split = -(-kb // 2) if senders >= 2 and kb >= 2 else kb  # k rows of W part 0
    rA = int(resident and scatter_dim == -1)
    rW = int(resident and scatter_dim == -2)
    depth_a, depth_w = ctx["depths"]
    a_pages = core_m * Kt if rA else depth_a * win_m * kb
    w_pages = Kt * core_n if rW else depth_w * kb * win_n
    tA = int(a.buffer_aligned_page_size())
    tW = int(w.buffer_aligned_page_size())
    out_nt = block_n  # output tile columns (Nt for scatter -2, Nt / G for scatter -1)
    a_row_per_block = block_m if scatter_dim == -2 else 0
    w_col_per_block = block_n if scatter_dim == -1 else 0

    # chunks: every kernel walks num_chunks x (the per-chunk entries), chunk c -> (i_m, i_n) with the resident
    # operand's axis outer; a chunk's tile origin is (i_m * chunk_m, i_n * chunk_n) inside the block
    n_m, n_n, chunk_m, chunk_n = ctx["n_m"], ctx["n_n"], ctx["chunk_m"], ctx["chunk_n"]
    num_chunks = n_m * n_n
    m_outer = int(scatter_dim == -1)
    n_inner = n_n if m_outer else n_m
    zone_sets = ctx["zone_sets"]
    credit_chunks = max(0, num_chunks - zone_sets)  # chunks whose sends wait for a downstream credit
    chunk_ct = [num_chunks, n_inner, m_outer, chunk_m, chunk_n, block_m, block_n]
    ring_slots = {d: zone_sets * slots[d] for d in _DIRS}  # zone slots per direction (the arrival CBs' capacity)

    staging = res["staging"]
    staging_addr = int(staging.buffer_address())
    zone_fwd_off = DEPTH_HANDOFF * P * STAGE_TILE_BYTES
    zone_bwd_off = zone_fwd_off + ring_slots["fwd"] * P * STAGE_TILE_BYTES
    arr_fwd_addr, arr_bwd_addr, ready_addr, credit_addr = (
        int(ttnn.get_global_semaphore_address(s)) for s in res["sems"]
    )
    mailbox_addr = int(res["mailbox"].buffer_address())
    chunk_tiles = max(1, min(int(ttnn.get_tt_fabric_max_payload_size_bytes()) // STAGE_TILE_BYTES, P))

    a_ct = list(ttnn.TensorAccessorArgs(a).get_compile_time_args())
    w_ct = list(ttnn.TensorAccessorArgs(w).get_compile_time_args())
    o_ct = list(ttnn.TensorAccessorArgs(output).get_compile_time_args())
    a_addr, w_addr, o_addr = int(a.buffer_address()), int(w.buffer_address()), int(output.buffer_address())

    cfg = ctx["compute_kernel_config"]
    compute_cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=cfg.math_fidelity,
        math_approx_mode=cfg.math_approx_mode,
        fp32_dest_acc_en=True,  # the cross-device adds accumulate in fp32 DEST (a floor, never below the caller's)
        dst_full_sync_en=True,  # restores 8-tile fp32 subblocks
    )

    reader_ct = (
        [
            CB_ACT_K_WINDOW,
            CB_WEIGHT_K_WINDOW,
            CB_ARRIVALS_FWD,
            CB_ARRIVALS_BWD,
            win_m,
            win_n,
            kb,
            nkb,
            Kt,
            Nt,
            a_row_per_block,
            w_col_per_block,
            rA,
            rW,
            P,
            num_entries,
            tA,
            tW,
            a_pages // (win_m * kb),  # CB capacity in K-block windows
            w_pages // (kb * win_n),
            a_split,
            w_split,
            int(os.environ.get("MRS_PREFETCH", PREFETCH_STEPS)),  # env: perf experiments only
            int(os.environ.get("MRS_LOOKAHEAD_AFTER_BARRIER", LOOKAHEAD_AFTER_BARRIER[regime])),  # env: perf only
            int(os.environ.get("MRS_A_READ_NOC", READ_NOCS[regime][0])),  # env: perf experiments only
            int(os.environ.get("MRS_W_READ_NOC", READ_NOCS[regime][1])),  # env: perf experiments only
            W,
        ]
        + chunk_ct  # 27..33
        + a_ct
        + w_ct
    )
    compute_ct = [
        CB_ACT_K_WINDOW,
        CB_WEIGHT_K_WINDOW,
        CB_PARTIAL_HOLD,
        CB_FINISHED,
        CB_ARRIVALS_FWD,
        CB_ARRIVALS_BWD,
        win_n,  # in1 window row stride (the W window of one wave)
        kb,
        nkb,
        P,
        num_entries,
        W,
        ctx["super_block"],  # K-windows a subblock accumulates in DEST per pack (entry 0 packs per window)
        CB_ZERO,
        num_chunks,
        n_inner,
        int(resident),
    ]
    writer_ct = [CB_FINISHED, P, W, out_nt, DEPTH_HANDOFF, STAGE_TILE_BYTES, num_entries, CB_ZERO] + chunk_ct + o_ct
    max_port_cores = -(-n_cores // L)  # matmul cores one port serves (link(k) = k mod L)
    port_ct = [
        CB_PORT_CHUNKS,
        P,
        chunk_tiles,
        PORT_GROUP,
        STAGE_TILE_BYTES,
        DEPTH_HANDOFF,
        MAILBOX_STRIDE,
        max_port_cores,
        W,
        int(ARRIVAL_INC_PER_WAVE or W == 1),  # every wave's unit carries the increment (else only the last)
        num_chunks,
        num_entries,  # entries per chunk
        zone_sets,
    ]
    assert PORT_GROUP + 1 <= 8, "port sender headers: PORT_GROUP data headers + 1 credit header (pool of 8)"

    src = ttnn.KernelDescriptor.SourceType.FILE_PATH
    # Perf-only ablation switches (debug scaffolding; output is wrong by design when set): MRS_ABLATE=compute,dram
    ablate = set(filter(None, os.environ.get("MRS_ABLATE", "").split(",")))
    ablate_compute = [("SKIP_COMPUTE", "1"), ("CKL_ELTWISE_CHAIN_SKIP_COMPUTE", "1")] if "compute" in ablate else []
    ablate_reader = [("MRS_SKIP_DRAM", "1")] if "dram" in ablate else []
    ablate_reader += [("MRS_SKIP_MCAST", "1")] if "mcast" in ablate else []
    dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(
        processor=getattr(ttnn.DataMovementProcessor, risc), noc=noc
    )
    port_vcore_of = {coord: {key: virt(c) for key, c in ports.items()} for coord, ports in ports_of.items()}

    def unit_walks(sc, d, l):
        """Port (d, l)'s units on a chip with schedule sc: [(entry, {half: block}, mask, h1_first)]. A unit is one
        entry index; with mirrored orders the two halves may hold different blocks at that index."""
        out = []
        for e in range(len(sc["entries"][0])):
            blk = {h: sc["entries"][h][e] for h in (0, 1)}
            mask = sum(1 << h for h in link_halves[l] if sc["roles"][(blk[h], h)][0] == d)
            if not mask:
                continue
            # walk a chain-start half (ready right after its matmul) before a relay half (waits for its arrival)
            h1_first = int(mask == 3 and sc["roles"][(blk[1], 1)][1] and not sc["roles"][(blk[0], 0)][1])
            out.append((e, blk, mask, h1_first))
        return out

    sender_plan = plan_mcast_senders(
        os.environ.get("MRS_SENDER_PLACEMENT", MCAST_SENDER_PLACEMENT[regime]),  # env: perf experiments only
        grid_m,
        grid_n,
        a_split < win_m,
        w_split < kb,
        a_shape_1d,
        w_shape_1d,
    )

    mesh_desc = ttnn.MeshProgramDescriptor()
    for grp in ctx["groups"]:
        for p, coord in enumerate(grp):
            sc = ctx["sched"][p]
            roles, entries, arrivals, thresholds = sc["roles"], sc["entries"], sc["arrivals"], sc["thresholds"]
            nxt = grp[(p + 1) % G] if (ring or p < G - 1) else None
            prv = grp[(p - 1) % G] if (ring or p > 0) else None
            peers = {"fwd": nxt, "bwd": prv}
            ports = ports_of[coord]
            port_vcore = port_vcore_of[coord]
            all_crs = ttnn.CoreRangeSet(
                [ttnn.CoreRange(mm_core(0, 0), mm_core(grid_m - 1, grid_n - 1))]
                + [ttnn.CoreRange(c, c) for c in ports.values()]
            )
            program = ttnn.ProgramDescriptor()
            program.semaphores = [
                ttnn.SemaphoreDescriptor(id=0, core_ranges=all_crs, initial_value=0),  # consumed_fwd
                ttnn.SemaphoreDescriptor(id=1, core_ranges=all_crs, initial_value=0),  # consumed_bwd
                # freed (port cores): chunks' worth of arrivals this port's matmul cores consumed, in the direction
                # this port's peer sends into this chip (forwarded upstream as chunk credits)
                ttnn.SemaphoreDescriptor(id=2, core_ranges=all_crs, initial_value=0),
            ]
            # per (direction of the arrival, link): matmul cores whose half receives that direction's arrivals
            receivers = {(d, l): 0 for d in _DIRS for l in range(L)}
            for k in range(n_cores):
                for d in _DIRS:
                    receivers[(d, link_of(k))] += int(len(arrivals[(half(k), d)]) > 0)

            # ---- matmul cores: runtime args ----
            r_rt, c_rt, w_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for k, (gm, gn, core) in enumerate(mm_cores):
                h = half(k)
                l = link_of(k)
                ent_words, roles_k, kinds_k = [], [], []
                for b, (mask, lag) in zip(entries[h], entry_finish_plan(sc, h)):
                    r = roles[(b, h)]
                    thr_f = thresholds[(h, "fwd")].get(b, 0) if mask & 1 else 0
                    thr_b = thresholds[(h, "bwd")].get(b, 0) if mask & 2 else 0
                    # b is the virtual block (b * W + wave): the reader addresses the real block's operand slice
                    ent_words += [(b // W) | (mask << 16) | (lag << 20), thr_f | (thr_b << 16)]
                    roles_k.append([ROLE_COPY, ROLE_ADD_FWD, ROLE_ADD_BWD, ROLE_ADD_BOTH][mask] | (lag << 8))
                    kinds_k.append(KIND_FIN if r[0] == "fin" else (KIND_FWD if r[0] == "fwd" else KIND_BWD))
                pf = port_vcore.get(("fwd", l)) if peers["fwd"] is not None else None
                pb = port_vcore.get(("bwd", l)) if peers["bwd"] is not None else None
                n_arr = {d: len(arrivals[(h, d)]) for d in _DIRS}
                # a consumed chunk of d arrivals is reported to the port sending back upstream (opposite direction)
                rep = {"fwd": pb, "bwd": pf}
                for d in _DIRS:
                    assert n_arr[d] == 0 or rep[d] is not None, (p, k, d)
                r_rt[core.x][core.y] = (
                    [
                        a_addr,
                        w_addr,
                        gm * core_m,  # chunk-local origin of my per-core block
                        gn * core_n,
                        arr_fwd_addr,
                        arr_bwd_addr,
                        # arrival increments my counters receive per chunk
                        sum(inc_unit(b, W) == b for b in arrivals[(h, "fwd")]),
                        sum(inc_unit(b, W) == b for b in arrivals[(h, "bwd")]),
                        n_arr["fwd"],  # arrival slots per chunk (consumption reports count these)
                        n_arr["bwd"],
                        (rep["fwd"].x | (rep["fwd"].y << 16)) if n_arr["fwd"] else 0,
                        (rep["bwd"].x | (rep["bwd"].y << 16)) if n_arr["bwd"] else 0,
                        credit_chunks,  # chunks whose consumption is reported (the rest needs no credit)
                    ]
                    + wave_words
                    + ent_words
                )
                c_rt[core.x][core.y] = wave_mm_words + roles_k
                w_rt[core.x][core.y] = (
                    [
                        o_addr,
                        gm * core_m,
                        gn * core_n,
                        (pf.x | (pf.y << 16)) if pf is not None else 0,
                        (pb.x | (pb.y << 16)) if pb is not None else 0,
                        mailbox_addr + k * MAILBOX_STRIDE,
                    ]
                    + wave_words
                    + kinds_k
                )

            reader = ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "matmul_reduce_scatter_reader.cpp"),
                source_type=src,
                core_ranges=mm_crs,
                compile_time_args=reader_ct,
                runtime_args=r_rt,
                defines=ablate_reader,
                config=dm("RISCV_1", NOC0),
            )
            compute = ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "matmul_reduce_scatter_compute.cpp"),
                source_type=src,
                core_ranges=mm_crs,
                compile_time_args=compute_ct,
                runtime_args=c_rt,
                defines=ablate_compute,
                config=compute_cfg,
            )
            writer = ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "matmul_reduce_scatter_writer.cpp"),
                source_type=src,
                core_ranges=mm_crs,
                compile_time_args=writer_ct,
                runtime_args=w_rt,
                defines=[],
                config=dm("RISCV_0", NOC1),
            )
            mc_cfg = ttnn.McastConfig(noc=ttnn.NOC.NOC_0, handshake=True)
            kernels = [reader]
            for prefix, shape_1d, start, placement in sender_plan:
                if start is None:
                    ttnn.attach_absent(kernels[0], prefix)
                    continue
                snd = ttnn.Mcast1DFixedSenderConfig(starting_sender_index=start, sender_placement=placement)
                ttnn.Mcast1D(mesh, mm_crs, shape_1d, snd, mc_cfg).attach(program, prefix, kernels)
            reader = kernels[0]

            # ---- port cores ----
            pr_rt, ps_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            port_used = []
            for (d, l), pcore in sorted(ports.items(), key=lambda kv: (kv[0][1], kv[0][0])):
                peer = peers[d]
                if peer is None:
                    continue
                port_used.append(pcore)
                q = (p + 1) % G if d == "fwd" else (p - 1) % G
                down = ctx["sched"][q]
                table = [(k, c) for k, (_, _, c) in enumerate(mm_cores) if link_of(k) == l]
                core_words = []
                for k, c in table:
                    vc = virt(c)
                    core_words.append(vc.x | (vc.y << 8) | (half(k) << 16) | (k << 20))
                units = []
                for e, b, mask, h1_first in unit_walks(sc, d, l):
                    s0 = down["arrivals"][(0, d)].index(b[0]) if mask & 1 else 0
                    s1 = down["arrivals"][(1, d)].index(b[1]) if mask & 2 else 0
                    units.append(e | (mask << 8) | (h1_first << 10) | (s0 << 16) | (s1 << 24))
                consumed_id = 0 if d == "fwd" else 1
                pr_rt[pcore.x][pcore.y] = [staging_addr, mailbox_addr, consumed_id, len(core_words), len(units)] + (
                    wave_tiles + core_words + units
                )
                # ready fence: the peer's opposite port sends into this chip; tell it this chip has started
                peer_sched = ctx["sched"][q]
                opp = "bwd" if d == "fwd" else "fwd"
                send_ready = int(
                    any(peer_sched["roles"][(b, h)][0] == opp for b in range(num_entries) for h in link_halves[l])
                )
                rv = port_vcore_of[peer][(opp, l)]  # the peer chip's port that writes into this chip
                pn = node(peer)
                zone = zone_fwd_off if d == "fwd" else zone_bwd_off
                arr_addr = arr_fwd_addr if d == "fwd" else arr_bwd_addr
                # chunk credits: this port waits for one per chunk beyond the zone sets (downstream consumed chunk
                # c - zone_sets) and forwards one per chunk this chip's cores consumed of the peer's (opp) arrivals
                expect = receivers[(opp, l)]
                fwd_total = credit_chunks if expect > 0 else 0
                if credit_chunks:
                    assert bool(unit_walks(ctx["sched"][q], opp, l)) == (expect > 0), (p, d, l)
                args = (
                    [
                        staging_addr + zone,
                        arr_addr,
                        ready_addr,
                        send_ready,
                        rv.x,
                        rv.y,
                        int(pn.mesh_id),
                        int(pn.chip_id),
                        len(core_words),
                        len(units),
                        credit_addr,
                        credit_chunks if units else 0,  # credits this port waits for
                        expect,  # reports per consumed chunk (one per receiving core)
                        fwd_total,  # credits this port forwards upstream
                        ring_slots[d],  # the downstream zone ring (slots) of this direction
                        len(down["arrivals"][(0, d)]),  # downstream arrivals per chunk, half 0 / half 1
                        len(down["arrivals"][(1, d)]),
                    ]
                    + wave_tiles
                    + core_words
                    + units
                )
                args += list(
                    ttnn.setup_fabric_connection(node(coord), pn, ctx["links"][(coord, peer)][l], program, pcore)
                )
                ps_rt[pcore.x][pcore.y] = args
            used_port_crs = _crs(port_used)
            port_reader = ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "matmul_reduce_scatter_port_reader.cpp"),
                source_type=src,
                core_ranges=used_port_crs,
                compile_time_args=port_ct,
                runtime_args=pr_rt,
                defines=[],
                config=dm("RISCV_1", NOC0),
            )
            port_sender = ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "matmul_reduce_scatter_port_sender.cpp"),
                source_type=src,
                core_ranges=used_port_crs,
                compile_time_args=port_ct,
                runtime_args=ps_rt,
                defines=[],
                config=dm("RISCV_0", NOC1),
            )
            program.kernels = [reader, compute, writer, port_reader, port_sender]

            # ---- circular buffers ----
            cb = lambda idx, pages, page_bytes, fmt, crs: ttnn.CBDescriptor(
                total_size=pages * page_bytes,
                core_ranges=crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page_bytes)],
            )
            program.cbs = [
                cb(CB_ACT_K_WINDOW, a_pages, tA, a.dtype, mm_crs),
                cb(CB_WEIGHT_K_WINDOW, w_pages, tW, w.dtype, mm_crs),
                cb(CB_PARTIAL_HOLD, ctx["depth_hold"] * P, HOLD_TILE_BYTES, ttnn.float32, mm_crs),
                ttnn.cb_descriptor_from_sharded_tensor(
                    CB_FINISHED, staging, 0, DEPTH_HANDOFF * P * STAGE_TILE_BYTES, mm_crs
                ),
                ttnn.cb_descriptor_from_sharded_tensor(
                    CB_ARRIVALS_FWD, staging, zone_fwd_off, ring_slots["fwd"] * P * STAGE_TILE_BYTES, mm_crs
                ),
                ttnn.cb_descriptor_from_sharded_tensor(
                    CB_ARRIVALS_BWD, staging, zone_bwd_off, ring_slots["bwd"] * P * STAGE_TILE_BYTES, mm_crs
                ),
                cb(CB_PORT_CHUNKS, 2 * PORT_GROUP * chunk_tiles, STAGE_TILE_BYTES, ttnn.bfloat16, used_port_crs),
                cb(CB_ZERO, 1, STAGE_TILE_BYTES, ttnn.bfloat16, mm_crs),
            ]
            r, c = coord
            mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = program
    return mesh_desc
