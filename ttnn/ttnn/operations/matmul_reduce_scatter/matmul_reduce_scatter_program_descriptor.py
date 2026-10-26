# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""matmul_reduce_scatter host planner + MeshProgramDescriptor (see README.md).

Per chip, one program with two core roles:

  matmul cores (grid_m x grid_n, the compute grid minus a transport strip):
    reader  (NCRISC, NoC0): per entry, Kt/kb K-blocks of A (mcast channels "a", "a2") and W ("b", "b2"); the
                            cross-block operand is resident after entry 0 (re-credited, no NoC). Lands arrivals: polls
                            the global arrival counters, credits the zone slot the upstream port wrote over the fabric.
    compute (TRISCs):       the entry's matmul into a fp32 slot of cb_partial_hold, its finish (copy / hold + arrival
                            / hold + fwd + bwd) fused into the last super-block when the arrival has landed.
    writer  (BRISC, NoC1):  start / relay: signal the port's mailbox and pop once the port acks; fin: write the output.
  port cores (2 L, in the strip; one per (direction, link)):
    reader  (NCRISC, NoC0): per unit (entry, core halves), per core: wait the mailbox, NoC-read its finished sub-block
                            into cb_port_chunks, ack the core.
    sender  (BRISC, NoC1):  ready fence, then every chunk as one fabric packet into the downstream core's zone slot; a
                            fused increment of that core's arrival counter on its last chunk.
"""

from __future__ import annotations

from pathlib import Path

import ttnn
from ttnn.operations.ccl import Topology

from ._eth_placement import chip_eth_cores, worker_nearest_noc1

_mc = ttnn._ttnn.mcast_host  # kernel_lib/mcast host helper

KERNEL_DIR = Path(__file__).parent / "kernels"

# ---- circular buffers ------------------------------------------------------------------------------------------
CB_ACT_K_WINDOW = 0  # A K-block windows (resident: the whole core_m x Kt slice)
CB_WEIGHT_K_WINDOW = 1  # W K-block windows (resident: the whole Kt x core_n slice)
CB_PARTIAL_HOLD = 2  # fp32 matmul partial slots
CB_FINISHED = 3  # finished sub-block (bf16), staging offset 0; read remotely by the ports
CB_ARRIVALS_FWD = 4  # forward arrival zone slots (written by the upstream port over the fabric)
CB_ARRIVALS_BWD = 5  # backward arrival zone slots
CB_PORT_CHUNKS = 6  # port cores: gathered chunks waiting for the fabric sender
CB_ZERO = 7  # one bf16 zero tile (the writer fills it once): the fused finish's DEST += hold + 0 FPU add

# ---- knobs -------------------------------------------------------------------------------------------------------
DEPTH_IN0 = 3  # stream regime: A K-block windows
DEPTH_IN1 = 3  # stream regime: W K-block windows
RESIDENT_STREAM_DEPTHS = (3, 2)  # resident regime: the streamed operand's window depths, preferred first
SUPER_BLOCK = 2  # K-windows a subblock accumulates in fp32 DEST per pack (1 when Kt / kb is odd or it misses L1)
SUPER_BLOCK_EXTRA = 2  # streamed windows beyond a super-block (the loader's look-ahead)
DEPTH_HOLD = 2  # fp32 partial slots when some entry lags (else 1, see hold_depth)
DEPTH_HANDOFF = 2  # finished slots (a port gathers entry e while compute finishes e+1)
PREFER_PORT_ROW_STRIP = True  # ports in logical row 0, right under the Ethernet row
PORT_GROUP = 4  # chunks per port batch (cb_port_chunks holds 2 batches)
assert PORT_GROUP + 1 <= 8, "port sender headers: PORT_GROUP data headers + 1 credit header (pool of 8)"
ZONE_SETS = 2  # chunked walk: chunks whose arrivals the zones hold at once
MAX_AUTO_CHUNKS = 256  # the most chunks an auto split may have
KB_MAX_CHUNKED = 32  # K-block cap of a chunked plan (each chunk re-walks Kt / kb K-steps)
# chunk_cost_us() (ranks the auto split's candidates): FPU efficiency, per K-step cost (resident / stream), per handoff
# cost, DRAM bandwidth, program floor, clock
COST_FPU_EFF = 0.93
COST_STEP_US = 1.4
COST_STEP_STREAM_US = 2.1
COST_HANDOFF_US = 0.29
COST_DRAM_GBPS = 200
COST_FLOOR_US = 8.0
CLOCK_MHZ = 1350
KB_MIN_RESIDENT = 8  # smallest K-block the resident regime accepts (min(Kt, this))
KB_MAX = 16  # largest K-block of an unchunked plan
L1_HEADROOM = 64 * 1024  # bytes kept free per core beyond the op's own footprint
PREFETCH_STEPS = 1  # K-block windows a sender keeps reserved with DRAM reads in flight ahead of its multicast
# per regime ("resident" | "stream"):
LOOKAHEAD_AFTER_BARRIER = {"resident": 1, "stream": 0}  # look-ahead reads issued after the step's read barrier
READ_NOCS = {"resident": (0, 1), "stream": (0, 0)}  # (A, W) NoC of the matmul reader's DRAM reads
SENDERS_PER_LINE = {"resident": 2, "stream": 1}  # parts of a K-block window, each read + multicast by its own core
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

# The plan of the last call (regime, schedule, grid, ...), updated in place by every run(); the package exports it.
_LAST_PLAN = {}


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


def plan_transport(G, kind, mirror=False):
    """Per chip position: roles, entries[h], per (half, direction) arrival list (this chip's half-h entry order), the
    arrival thresholds (position in the upstream's half-h send order: one fused increment per arrival) and the
    upstream entry index that produces each arrival. slots[d] = zone slots per direction (max over chips and halves,
    so every chip shares one staging layout)."""
    ring = kind != "line"
    mirror = mirror and kind == "ring_balanced"
    sch = {p: build_schedule(p, G, kind, mirror) for p in range(G)}

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
                thresholds[(h, d)] = {b: i + 1 for i, b in enumerate(seq)}
                produced_at[(h, d)] = {b: sch[u][1][h].index(b) for b in seq}
                # deadlock freedom (mirrored orders): a relay waits only for an arrival produced at an earlier
                # upstream entry index (the shared descending-key order instead lags such a finish)
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
        lag = any(sc["produced_at"][(h, d)][b] >= e_idx for d, bit in (("fwd", 1), ("bwd", 2)) if mask & bit)
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


def _subblock(core_m, core_n, cap=DEST_TILES_FP32_FULL_SYNC):
    best = (1, 1)
    for h in range(1, core_m + 1):
        if core_m % h:
            continue
        for w in range(1, core_n + 1):
            if core_n % w or h * w > cap:
                continue
            if h * w > best[0] * best[1] or (h * w == best[0] * best[1] and w > best[1]):
                best = (h, w)
    return best


def plan_mcast_senders(grid_m, grid_n, a_two_parts, w_two_parts, disjoint):
    """[(channel, starting sender index | None (absent))] for "a", "a2", "b", "b2", every channel staggered: A lines
    are indexed by gm (length grid_n, sender at gn = (start + gm) % grid_n), W lines by gn (length grid_m, sender at
    gm = (start + gn) % grid_m). Part-1 senders start half a line further along. With `disjoint` each W channel's start
    is the smallest one whose sender cores overlap the senders already placed the least, so one NCRISC never
    serializes the DRAM reads + multicasts of both operands."""

    def cores(fam, start):
        if fam == "a":
            return {(gm, (start + gm) % grid_n) for gm in range(grid_m)}
        return {((start + gn) % grid_m, gn) for gn in range(grid_n)}

    taken, plan = set(), []
    for prefix, active, line_len in (
        ("a", True, grid_n),
        ("a2", a_two_parts, grid_n),
        ("b", True, grid_m),
        ("b2", w_two_parts, grid_m),
    ):
        if not active:
            plan.append((prefix, None))
            continue
        start = 0 if len(prefix) == 1 else line_len // 2
        if disjoint and prefix[0] == "b":
            start = min(range(line_len), key=lambda s0: (len(cores("b", s0) & taken), (s0 - start) % line_len))
        taken |= cores(prefix[0], start)
        plan.append((prefix, start))
    return plan


def choose_grid(gx, gy, block_m, block_n, L):
    """(strip, orientation, grid_m, grid_n, core_m, core_n, (sb_h, sb_w)) minimizing the tile-entries of the busiest
    core, then maximizing the subblock area."""
    best = None
    for strip in ("row", "col"):
        strip_len = gx if strip == "row" else gy
        if 2 * L > strip_len:
            continue
        rect = (gx, gy - 1) if strip == "row" else (gx - 1, gy)
        for orient in ("m_on_x", "m_on_y"):
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
            cand = dict(
                strip=strip,
                orient=orient,
                grid_m=grid_m,
                grid_n=grid_n,
                core_m=core_m,
                core_n=core_n,
                sb=sb,
            )
            if best is None or key < best[0]:
                best = (key, cand)
    if best is None:
        raise ValueError(f"matmul_reduce_scatter: no transport strip of the {gx}x{gy} grid fits {2 * L} ports")
    return best[1]


class _Grid:
    """The matmul grid of a plan: logical core and half of (gm, gn), and an entry's columns on an n-line."""

    def __init__(self, gd, half_split):
        self.gd = gd
        self.grid_m, self.grid_n = gd["grid_m"], gd["grid_n"]
        self.x0, self.y0 = (0, 1) if gd["strip"] == "row" else (0, 0)
        self.half_split = half_split
        self.n_cores = self.grid_m * self.grid_n

    def core(self, gm, gn):
        if self.gd["orient"] == "m_on_x":
            return ttnn.CoreCoord(self.x0 + gm, self.y0 + gn)
        return ttnn.CoreCoord(self.x0 + gn, self.y0 + gm)

    def cores(self):
        """[(gm, gn, logical core)] in row-major order: the list index is the core index k."""
        return [(gm, gn, self.core(gm, gn)) for gm in range(self.grid_m) for gn in range(self.grid_n)]

    def crs(self):
        return ttnn.CoreRangeSet([ttnn.CoreRange(self.core(0, 0), self.core(self.grid_m - 1, self.grid_n - 1))])

    def half(self, gm, gn):
        """With a resident cross-block operand the halves split along the streamed operand's multicast lines ("n":
        W lines, "m": A lines), so the two halves may compute different blocks; else the row-major split."""
        if self.half_split == "n":
            return int(gn >= -(-self.grid_n // 2))
        if self.half_split == "m":
            return int(gm >= -(-self.grid_m // 2))
        return 0 if gm * self.grid_n + gn < -(-self.n_cores // 2) else 1

    def line_columns(self, gn, b):
        """(chunk-local column, width) of block b on n-line gn."""
        return gn * self.gd["core_n"], self.gd["core_n"]


def window_depths(resident, depth):
    """(A, W) window depths in K-blocks: the stream regime's DEPTH_IN0 / DEPTH_IN1, or under the resident regime
    `depth` for the streamed operand (the resident operand's CB holds the whole Kt slice; its depth is unused)."""
    return (depth, depth) if resident else (DEPTH_IN0, DEPTH_IN1)


def footprint(
    kb, resident, *, depth=None, Kt, core_m, core_n, tA, tW, scatter_dim, slots, hold_depths, depths=None, zone_sets=1
):
    """Per matmul core L1 bytes: operand windows + fp32 hold + the staging slice + the zero tile. Per regime (keyed by
    `resident`): hold_depths (hold_depth() of the schedule that regime runs) and slots (its zone slots per direction).
    zone_sets: chunks whose arrivals the zones hold at once (1 unchunked)."""
    P = core_m * core_n
    rA = resident and scatter_dim == -1
    rW = resident and scatter_dim == -2
    dA, dW = depths or window_depths(resident, depth if depth is not None else RESIDENT_STREAM_DEPTHS[0])
    a = tA * (core_m * Kt if rA else dA * core_m * kb)
    w = tW * (Kt * core_n if rW else dW * kb * core_n)
    hold = HOLD_TILE_BYTES * hold_depths[resident] * P
    sl = slots[resident]
    staging = STAGE_TILE_BYTES * (DEPTH_HANDOFF + zone_sets * (sl["fwd"] + sl["bwd"])) * P
    return a + w + hold + staging + STAGE_TILE_BYTES


def choose_k_block(Kt, budget, kb_max=KB_MAX, **geo):
    """(resident, kb, depth): resident with the preferred kb | Kt (largest <= kb_max, then smaller, then larger), kb >=
    min(Kt, KB_MIN_RESIDENT), at the first RESIDENT_STREAM_DEPTHS depth where any such kb fits; else stream with the
    preferred kb | Kt that fits (depth: None, the stream regime's DEPTH_IN0 / DEPTH_IN1)."""

    def divisors(cap):
        # preference order: the largest divisor <= cap first, then smaller ones, then (no fit below the cap) larger
        out = [d for d in range(min(Kt, cap), 0, -1) if Kt % d == 0]
        return out + [d for d in range(cap + 1, Kt + 1) if Kt % d == 0]

    for depth in RESIDENT_STREAM_DEPTHS:
        for kb in divisors(kb_max):
            if kb < min(Kt, KB_MIN_RESIDENT):
                continue
            if footprint(kb, True, depth=depth, Kt=Kt, **geo) <= budget:
                return True, kb, depth
    for kb in divisors(kb_max):
        if footprint(kb, False, Kt=Kt, **geo) <= budget:
            return False, kb, None
    raise NotImplementedError("matmul_reduce_scatter: even kb=1 exceeds the L1 budget")


def super_block_plan(Kt, kb, resident, depth, budget, geo):
    """(S, (dA, dW)): the compute's DEST-resident super-block (K-windows per pack) and the operand window depths. S =
    SUPER_BLOCK when it divides Kt / kb and the streamed windows deepened to S + SUPER_BLOCK_EXTRA still fit L1; else
    1 at the regime's own depths."""
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
    for total in range(1, min(MAX_AUTO_CHUNKS, block_m * block_n) + 1):
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


def plan_chunk_geometry(n_m, n_n, *, G, kind, L, Kt, block_m, block_n, gx, gy, tA, tW, scatter_dim, budget):
    """Grid, regime and schedule for one chunk of chunk_m x chunk_n tiles (the whole block when n_m = n_n = 1). Raises
    NotImplementedError when the chunk's footprint misses the L1 budget."""
    chunk_m, chunk_n = -(-block_m // n_m), -(-block_n // n_n)
    n_m, n_n = -(-block_m // chunk_m), -(-block_n // chunk_n)  # no empty trailing chunk
    zone_sets = 1 if n_m * n_n == 1 else max(1, min(ZONE_SETS, n_m * n_n))
    gd = choose_grid(gx, gy, chunk_m, chunk_n, L)
    core_m, core_n = gd["core_m"], gd["core_n"]
    if kind == "ring_balanced" and gd["grid_m"] * gd["grid_n"] < 2:
        raise ValueError("matmul_reduce_scatter: a balanced ring needs >= 2 matmul cores (two halves)")
    # mirrored half orders (ring_balanced with a resident cross-block operand): the halves split along the streamed
    # operand's multicast lines (W lines = fixed gn for scatter -1, A lines = fixed gm for scatter -2), so they may
    # compute different blocks. Each regime's schedule fixes its hold depth, which the footprint (and so the regime
    # choice) needs.
    mirror_split = "k"
    if kind == "ring_balanced":
        if scatter_dim == -1 and gd["grid_n"] >= 2:
            mirror_split = "n"
        elif scatter_dim == -2 and gd["grid_m"] >= 2:
            mirror_split = "m"
    sched_of = {r: plan_transport(G, kind, mirror=r and mirror_split != "k") for r in (False, True)}
    hold_depths = {r: hold_depth(sched_of[r][0]) for r in (False, True)}
    geo = dict(
        core_m=core_m,
        core_n=core_n,
        tA=tA,
        tW=tW,
        scatter_dim=scatter_dim,
        slots={r: sched_of[r][1] for r in (False, True)},
        hold_depths=hold_depths,
        zone_sets=zone_sets,
    )
    resident, kb, depth = choose_k_block(Kt, budget, kb_max=KB_MAX if n_m * n_n == 1 else KB_MAX_CHUNKED, **geo)
    super_block, depths = super_block_plan(Kt, kb, resident, depth, budget, geo)
    chips_sched, slots = sched_of[resident]
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
        P=core_m * core_n,  # slot tiles: every hold / finished / zone slot holds one per-core block
        footprint=footprint(kb, resident, depth=depth, Kt=Kt, depths=depths, **geo),
    )


def chunk_cost_us(pl, *, G, L, Mt, Kt, Nt, tA, tW, scatter_dim):
    """Estimated traced latency (us) of plan `pl` (plan_chunk_geometry): the largest of compute, DRAM and transport
    (constants above) plus the program floor. Only ranks the auto split's candidates."""
    n_m, n_n = pl["n_m"], pl["n_n"]
    C = n_m * n_n
    cores = pl["gd"]["grid_m"] * pl["gd"]["grid_n"]
    steps = C * G * (Kt // pl["kb"])
    step_us = COST_STEP_US if pl["resident"] else COST_STEP_STREAM_US
    compute = C * G * pl["P"] * Kt * 32 / CLOCK_MHZ / COST_FPU_EFF + step_us * steps
    a_bytes, w_bytes = Mt * Kt * tA, Kt * Nt * tW
    if scatter_dim == -1:  # A shared by every block (resident: once), W columns per block (re-read per m chunk)
        traffic = (a_bytes if pl["resident"] else G * n_n * a_bytes) + n_m * w_bytes
    else:  # W shared by every block, A rows per block (re-read per n chunk)
        traffic = (w_bytes if pl["resident"] else G * n_m * w_bytes) + n_n * a_bytes
    dram = traffic / (COST_DRAM_GBPS * 1e3)
    transport = COST_HANDOFF_US * C * (G - 1) * cores / L
    return COST_FLOOR_US + max(compute, dram, transport)


_PLAN_CACHE = {}  # auto split per (shape, dtypes, group, mesh grid, budget): the search plans many candidates


# ======================================================================================================================
# Port placement
# ======================================================================================================================


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


_PORT_CACHE = {}


def _place_ports_near_eth(mesh, coord, strip, gx, gy, L, peers, node, links):
    """{(d, l): logical core} in the strip, each linked port at the strip cell with the fewest NoC1 hops to its link's
    Ethernet core (fabric rule: a sender right below its router). No probe dispatch: the router's Ethernet channel is
    the first runtime arg setup_fabric_connection emits (asked on a scratch descriptor); it indexes the chip's
    non-harvested Ethernet list. Falls back to _place_ports when the Ethernet map is unavailable. Cached per (mesh,
    chip, strip, L, peers)."""
    key = (id(mesh), coord, strip, L, peers["fwd"], peers["bwd"])
    hit = _PORT_CACHE.get(key)
    if hit is not None and hit[0] is mesh:
        return hit[1]
    allowed = {(x, 0) for x in range(gx)} if strip == "row" else {(gx - 1, y) for y in range(gy)}
    eth_list = chip_eth_cores(mesh, coord)
    cores = None if eth_list is None else {}
    taken = set()
    for l in range(L) if cores is not None else ():
        for d in _DIRS:
            peer = peers[d]
            if peer is None:
                continue
            link = links[(coord, peer)][l]
            scratch = ttnn.ProgramDescriptor()
            chan = list(ttnn.setup_fabric_connection(node(coord), node(peer), link, scratch, ttnn.CoreCoord(0, 0)))[0]
            core = worker_nearest_noc1(mesh, coord, eth_list[chan], taken, allowed) if chan < len(eth_list) else None
            if core is None:
                cores = None
                break
            cores[(d, l)] = core
            taken.add((core.x, core.y))
        if cores is None:
            break
    if cores is None:
        cores = _place_ports(strip, gx, gy, L)
    else:
        for k in _place_ports(strip, gx, gy, L):  # receive-only ends: any free strip cell
            if k not in cores:
                free = sorted(allowed - taken)
                cores[k] = ttnn.CoreCoord(*free[0])
                taken.add(free[0])
    _PORT_CACHE[key] = (mesh, cores)
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
    """The mesh's global semaphores and port mailbox (created once), and its staging tensor (cb_finished + zones),
    reallocated whenever this call needs a different size."""
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
    if res["staging_tiles"] != staging_tiles:
        if res["staging"] is not None:
            ttnn.deallocate(res["staging"])
            res["staging"] = None
        cfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, (staging_tiles * TILE, TILE), ttnn.ShardOrientation.ROW_MAJOR),
        )
        res["staging"] = ttnn.allocate_tensor_on_device(
            ttnn.Shape([gx * gy * staging_tiles * TILE, TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, cfg
        )
        res["staging_tiles"] = staging_tiles
    return res


def prepare_mesh(mesh):
    """Create the mesh's persistent semaphores and mailbox now, so a caller can place them among its other long-lived
    L1 allocations instead of wherever L1 is free at the first call."""
    _mesh_resources(mesh, 0)


def local_cb_layout(mesh, pl, scatter_dim, Kt, tA, tW):
    """{cb: (address, pages, page bytes)} of the matmul cores' CBs that are not backed by the staging tensor, in the
    order create_mesh_program_descriptor places them up from the L1 allocator base (A, hold, zero, W). `pl` is a plan
    (plan_chunk_geometry) or the program context, which carry the same keys."""
    gd, kb = pl["gd"], pl["kb"]
    depth_a, depth_w = pl["depths"]
    rA = pl["resident"] and scatter_dim == -1
    rW = pl["resident"] and scatter_dim == -2
    sizes = (
        (CB_ACT_K_WINDOW, gd["core_m"] * Kt if rA else depth_a * gd["core_m"] * kb, tA),
        (CB_PARTIAL_HOLD, pl["depth_hold"] * pl["P"], HOLD_TILE_BYTES),
        (CB_ZERO, 1, STAGE_TILE_BYTES),
        (CB_WEIGHT_K_WINDOW, Kt * gd["core_n"] if rW else depth_w * kb * gd["core_n"], tW),
    )
    align = ttnn.get_dram_alignment()
    addr = int(ttnn._ttnn.device.GetAllocatorBaseAddress(mesh, ttnn.BufferType.L1))
    layout = {}
    for cb, pages, page_bytes in sizes:
        addr = -(-addr // align) * align
        layout[cb] = (addr, pages, page_bytes)
        addr += pages * page_bytes
    return layout


def _l1_budget(mesh):
    """The CB budget: l1_bank_size (a worker core's allocatable L1, already measured from the allocator base), capped
    below the lowest live L1 buffer, minus L1_HEADROOM."""
    l1_top = int(ttnn._ttnn.reports.get_device_info(mesh).l1_bank_size)
    # Live L1 buffers grow down from the top of the bank; the CB region must end below the lowest of them.
    lowest = ttnn._ttnn.device.GetLowestOccupiedComputeL1Address(mesh)
    if lowest is not None:
        l1_top = min(l1_top, int(lowest) - int(ttnn._ttnn.device.GetAllocatorBaseAddress(mesh, ttnn.BufferType.L1)))
    return l1_top - L1_HEADROOM


# ======================================================================================================================
# Caller checks
# ======================================================================================================================


def _validate(a_shape, a_dtype, w_shape, w_dtype, cluster_axis, scatter_dim, topology):
    """The call's shapes, dtypes and keywords (scatter_dim already canonical: -1 or -2)."""
    a_shape, w_shape = list(a_shape), list(w_shape)
    if not 2 <= len(a_shape) <= 4 or any(s != 1 for s in a_shape[:-2]):
        raise ValueError(f"matmul_reduce_scatter: A must be rank 2-4 with leading dims 1, got {a_shape}")
    if not 2 <= len(w_shape) <= 4 or any(s != 1 for s in w_shape[:-2]):
        raise ValueError(f"matmul_reduce_scatter: W must be (K, N) with leading dims 1, got {w_shape}")
    M, K = a_shape[-2:]
    K2, N = w_shape[-2:]
    if K != K2:
        raise ValueError(f"matmul_reduce_scatter: K mismatch, A {a_shape} vs W {w_shape}")
    if M % TILE or K % TILE or N % TILE:
        raise ValueError(f"matmul_reduce_scatter: M, K and N must be multiples of {TILE}, got {M}, {K}, {N}")
    if a_dtype not in (ttnn.bfloat16, ttnn.bfloat8_b):
        raise NotImplementedError(f"matmul_reduce_scatter: input dtype {a_dtype} (bfloat16, bfloat8_b)")
    if w_dtype not in (ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b):
        raise NotImplementedError(f"matmul_reduce_scatter: weight dtype {w_dtype} (bfloat16, bfloat8_b, bfloat4_b)")
    if cluster_axis not in (0, 1):
        raise ValueError(f"matmul_reduce_scatter: cluster_axis must be 0 or 1, got {cluster_axis}")
    if scatter_dim not in (-1, -2):
        raise ValueError(f"matmul_reduce_scatter: scatter_dim must be the last or second-last dim, got {scatter_dim}")
    if topology not in (Topology.Linear, Topology.Ring):
        raise ValueError(f"matmul_reduce_scatter: topology must be Linear or Ring, got {topology}")


def _check_interleaved_tile(t, what):
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


def _plan_call(mesh, M, K, N, tA, tW, *, cluster_axis, scatter_dim, topology, num_links):
    """Group, links, schedule kind and the chunk / grid / K-block plan of a call (A: M x K, W: K x N, page bytes
    tA / tW), within the L1 that is free now minus L1_HEADROOM."""
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
    if ring and not _config_wraps(ttnn.get_fabric_config(), cluster_axis):
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
    elif 0 < num_links <= avail:
        L = num_links
    else:
        raise ValueError(f"matmul_reduce_scatter: num_links={num_links}, but a hop has only {avail} link(s)")

    # ---- geometry ----
    Mt, Kt, Nt = M // TILE, K // TILE, N // TILE
    block_m, block_n = (Mt // G, Nt) if scatter_dim == -2 else (Mt, Nt // G)
    gx, gy = _full_grid(mesh)
    budget = _l1_budget(mesh)
    cached = _MESH_CACHE.get(id(mesh))
    if cached is not None and cached["mesh"] is mesh and cached["staging"] is not None:
        budget += cached["staging_tiles"] * STAGE_TILE_BYTES  # our own staging is below the other L1 buffers
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
        budget=budget,
    )
    try:
        pl = plan_chunk_geometry(1, 1, **plan_args)  # whatever fits unchunked keeps its plan
    except NotImplementedError:
        key = tuple(sorted(plan_args.items()))
        if key not in _PLAN_CACHE:
            cost_args = dict(G=G, L=L, Mt=Mt, Kt=Kt, Nt=Nt, tA=tA, tW=tW, scatter_dim=scatter_dim)
            best = None
            for n_m, n_n in chunk_candidates(block_m, block_n, scatter_dim):
                if n_m * n_n == 1:
                    continue
                try:
                    cand = plan_chunk_geometry(n_m, n_n, **plan_args)
                except (NotImplementedError, ValueError):
                    continue
                cost = chunk_cost_us(cand, **cost_args)
                if best is None or cost < best[0]:
                    best = (cost, cand)
            if best is None:
                raise NotImplementedError(
                    f"matmul_reduce_scatter: no split of the {block_m}x{block_n}-tile block into <= "
                    f"{MAX_AUTO_CHUNKS} chunks fits L1"
                )
            _PLAN_CACHE[key] = (best[1]["n_m"], best[1]["n_n"])
        pl = plan_chunk_geometry(*_PLAN_CACHE[key], **plan_args)
    return dict(
        G=G,
        kind=kind,
        L=L,
        links=links,
        groups=groups,
        node=node,
        ring=ring,
        scatter_dim=scatter_dim,
        Mt=Mt,
        Kt=Kt,
        Nt=Nt,
        block_m=block_m,
        block_n=block_n,
        gx=gx,
        gy=gy,
        pl=pl,
    )


def run(
    a,
    w,
    *,
    cluster_axis,
    scatter_dim,
    topology,
    num_links,
    compute_kernel_config,
    memory_config,
    staging=None,
):
    mesh = a.device()
    staging = staging or "exact"
    if staging not in ("exact", "ephemeral"):
        raise ValueError(f"matmul_reduce_scatter: staging must be 'exact' or 'ephemeral', got {staging!r}")
    a_shape, w_shape = list(a.shape), list(w.shape)
    _validate(a_shape, a.dtype, w_shape, w.dtype, cluster_axis, scatter_dim, topology)
    M, K = a_shape[-2:]
    N = w_shape[-1]
    _check_interleaved_tile(a, "input_tensor")
    _check_interleaved_tile(w, "weight")
    if memory_config is not None and memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise NotImplementedError("matmul_reduce_scatter: sharded output memory_config")
    out_mem = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    tA = int(a.buffer_aligned_page_size())
    tW = int(w.buffer_aligned_page_size())
    call = _plan_call(
        mesh,
        M,
        K,
        N,
        tA,
        tW,
        cluster_axis=cluster_axis,
        scatter_dim=scatter_dim,
        topology=topology,
        num_links=num_links,
    )
    pl = call["pl"]
    gd = pl["gd"]
    _LAST_PLAN.clear()
    _LAST_PLAN.update(
        {
            "xblock": "resident" if pl["resident"] else "stream",
            "schedule": call["kind"],
            "kb": pl["kb"],
            "depths": pl["depths"],
            "super_block": pl["super_block"],
            "hold_depth": pl["depth_hold"],
            "grid": (gd["grid_m"], gd["grid_n"]),
            "strip": gd["strip"],
            "half_split": pl["half_split"],
            "chunks": (pl["n_m"], pl["n_n"]),
            "chunk_tiles": (pl["chunk_m"], pl["chunk_n"]),
            "zone_sets": pl["zone_sets"],
            "P": pl["P"],
            "footprint_kib": pl["footprint"] // 1024,
        }
    )

    slots = pl["slots"]
    res = _mesh_resources(mesh, (DEPTH_HANDOFF + pl["zone_sets"] * (slots["fwd"] + slots["bwd"])) * pl["P"])

    out_shape = list(a_shape)
    out_shape[-1] = N
    if scatter_dim == -2:
        out_shape[-2] = M // call["G"]
    else:
        out_shape[-1] = N // call["G"]
    output = ttnn.allocate_tensor_on_device(ttnn.Shape(out_shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, out_mem)

    ctx = dict(
        call,
        **pl,
        mesh=mesh,
        a=a,
        w=w,
        output=output,
        res=res,
        compute_kernel_config=compute_kernel_config,
    )
    result = ttnn.generic_op([a, w, output], create_mesh_program_descriptor(ctx))
    if staging == "ephemeral":
        # Free the L1 staging once the program is enqueued: later ops reuse it only after this one finishes, and
        # peers write into it only while this program runs. Keeps it from crowding other ops' static CBs.
        ttnn.deallocate(res["staging"])
        res["staging"], res["staging_tiles"] = None, 0
    return result


# ======================================================================================================================
# Program descriptors
# ======================================================================================================================


def _peers(grp, p, ring):
    G = len(grp)
    return {
        "fwd": grp[(p + 1) % G] if (ring or p < G - 1) else None,
        "bwd": grp[(p - 1) % G] if (ring or p > 0) else None,
    }


def _matmul_core_args(ctx, transport, grid, sc, k, gm, gn, port_vcore, peers):
    """(reader, compute, writer) runtime args of matmul core k on a chip with schedule sc."""
    h = grid.half(gm, gn)
    l = k % ctx["L"]
    roles, entries, arrivals, thresholds = sc["roles"], sc["entries"][h], sc["arrivals"], sc["thresholds"]
    res, core_m, core_n = ctx["res"], ctx["gd"]["core_m"], ctx["gd"]["core_n"]
    # per entry: the sub-block's offset from my origin (row0, col0 = my leftmost entry's column) and dense extents
    # (reader / writer), and its extents and subblock (compute)
    geo = [(0, *grid.line_columns(gn, b)) for b in entries]
    col0 = min(col for _, col, _ in geo)
    assert all(col - col0 < 256 for _, col, _ in geo), "8-bit entry column offset"
    geom_words = [(col - col0) << 8 | (core_m << 16) | (cols << 24) for _, col, cols in geo]
    mm_words = []
    for _, _, cols in geo:
        sb_h, sb_w = _subblock(core_m, cols)
        mm_words.append(core_m | (cols << 8) | (sb_h << 16) | (sb_w << 24))
    ent_words, roles_k, kinds_k = [], [], []
    for b, (mask, lag) in zip(entries, entry_finish_plan(sc, h)):
        r = roles[(b, h)]
        thr_f = thresholds[(h, "fwd")].get(b, 0) if mask & 1 else 0
        thr_b = thresholds[(h, "bwd")].get(b, 0) if mask & 2 else 0
        ent_words += [b | (mask << 16) | (lag << 20), thr_f | (thr_b << 16)]
        roles_k.append([ROLE_COPY, ROLE_ADD_FWD, ROLE_ADD_BWD, ROLE_ADD_BOTH][mask] | (lag << 8))
        kinds_k.append(KIND_FIN if r[0] == "fin" else (KIND_FWD if r[0] == "fwd" else KIND_BWD))
    pf = port_vcore.get(("fwd", l)) if peers["fwd"] is not None else None
    pb = port_vcore.get(("bwd", l)) if peers["bwd"] is not None else None
    n_arr = {d: len(arrivals[(h, d)]) for d in _DIRS}
    # a consumed chunk of d arrivals is reported to the port sending back upstream (opposite direction)
    rep = {"fwd": pb, "bwd": pf}
    for d in _DIRS:
        assert n_arr[d] == 0 or rep[d] is not None, (k, d)
    xy = lambda c: c.x | (c.y << 16)
    arr_fwd_addr, arr_bwd_addr = (int(ttnn.get_global_semaphore_address(s)) for s in res["sems"][:2])
    reader = [
        int(ctx["a"].buffer_address()),
        int(ctx["w"].buffer_address()),
        gm * core_m,  # chunk-local origin of my per-core block
        col0,
        arr_fwd_addr,
        arr_bwd_addr,
        n_arr["fwd"],  # arrivals per chunk: zone slots consumed and increments received
        n_arr["bwd"],
        xy(rep["fwd"]) if n_arr["fwd"] else 0,
        xy(rep["bwd"]) if n_arr["bwd"] else 0,
        transport["credit_chunks"],  # chunks whose consumption is reported (the rest needs no credit)
        *geom_words,
        *ent_words,
    ]
    writer = [
        int(ctx["output"].buffer_address()),
        gm * core_m,
        col0,
        xy(pf) if pf is not None else 0,
        xy(pb) if pb is not None else 0,
        int(res["mailbox"].buffer_address()) + k * MAILBOX_STRIDE,
        *geom_words,
        *kinds_k,
    ]
    return reader, mm_words + roles_k, writer


def _port_args(ctx, transport, receivers, grid, p, grp, d, l, ports_of, link_halves):
    """(port reader, port sender) runtime args of port (d, l) of chip grp[p], without the fabric connection.
    receivers: this chip's {(d, l): matmul cores receiving d arrivals on link l}."""
    G, L, mesh = ctx["G"], ctx["L"], ctx["mesh"]
    sc = ctx["sched"][p]
    coord, peer = grp[p], _peers(grp, p, ctx["ring"])[d]
    down = ctx["sched"][(p + 1) % G if d == "fwd" else (p - 1) % G]
    virt = mesh.worker_core_from_logical_core
    core_words = [
        virt(c).x | (virt(c).y << 8) | (grid.half(gm, gn) << 16) | (k << 20)
        for k, (gm, gn, c) in enumerate(grid.cores())
        if k % L == l
    ]
    units = []
    for e in range(G):
        blk = {h: sc["entries"][h][e] for h in (0, 1)}
        mask = sum(1 << h for h in link_halves[l] if sc["roles"][(blk[h], h)][0] == d)
        if not mask:
            continue
        # walk a chain-start half (ready right after its matmul) before a relay half (waits for its arrival)
        h1_first = int(mask == 3 and sc["roles"][(blk[1], 1)][1] and not sc["roles"][(blk[0], 0)][1])
        s0 = down["arrivals"][(0, d)].index(blk[0]) if mask & 1 else 0
        s1 = down["arrivals"][(1, d)].index(blk[1]) if mask & 2 else 0
        units.append(e | (mask << 8) | (h1_first << 10) | (s0 << 16) | (s1 << 24))
    res = ctx["res"]
    staging_addr = int(res["staging"].buffer_address())
    arr_addr, ready_addr, credit_addr = (
        int(ttnn.get_global_semaphore_address(res["sems"][i])) for i in ((0 if d == "fwd" else 1), 2, 3)
    )
    reader = [staging_addr, int(res["mailbox"].buffer_address()), 0 if d == "fwd" else 1, len(core_words), len(units)]
    reader += core_words + units
    # ready fence: the peer's opposite port sends into this chip; tell it this chip has started
    opp = "bwd" if d == "fwd" else "fwd"
    send_ready = int(any(down["roles"][(b, h)][0] == opp for b in range(G) for h in link_halves[l]))
    rv = virt(ports_of[peer][(opp, l)])  # the peer chip's port that writes into this chip
    pn = ctx["node"](peer)
    # chunk credits: this port waits for one per chunk beyond the zone sets (downstream consumed chunk c - zone_sets)
    # and forwards one per chunk this chip's cores consumed of the peer's (opp) arrivals
    expect = receivers[(opp, l)]
    credit_chunks = transport["credit_chunks"]
    sender = [
        staging_addr + transport["zone_off"][d],
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
        credit_chunks if expect > 0 else 0,  # credits this port forwards upstream
        transport["ring_slots"][d],  # the downstream zone ring (slots) of this direction
        len(down["arrivals"][(0, d)]),  # downstream arrivals per chunk, half 0 / half 1
        len(down["arrivals"][(1, d)]),
    ]
    sender += core_words + units
    return reader, sender


def create_mesh_program_descriptor(ctx):
    mesh, G, L, P, gd = ctx["mesh"], ctx["G"], ctx["L"], ctx["P"], ctx["gd"]
    core_m, core_n, kb, Kt = gd["core_m"], gd["core_n"], ctx["kb"], ctx["Kt"]
    grid_m, grid_n = gd["grid_m"], gd["grid_n"]
    scatter_dim, resident, slots = ctx["scatter_dim"], ctx["resident"], ctx["slots"]
    a, w, output, res = ctx["a"], ctx["w"], ctx["output"], ctx["res"]
    virt = mesh.worker_core_from_logical_core
    grid = _Grid(gd, ctx["half_split"])
    mm_cores = grid.cores()  # k = row-major index
    mm_crs = grid.crs()

    ports_of = {}  # per chip: {(d, l): logical port core}
    for grp in ctx["groups"]:
        for p, coord in enumerate(grp):
            ports_of[coord] = _place_ports_near_eth(
                mesh,
                coord,
                gd["strip"],
                ctx["gx"],
                ctx["gy"],
                L,
                _peers(grp, p, ctx["ring"]),
                ctx["node"],
                ctx["links"],
            )
    link_halves = {l: {grid.half(gm, gn) for k, (gm, gn, _) in enumerate(mm_cores) if k % L == l} for l in range(L)}

    nkb = Kt // kb
    regime = "resident" if resident else "stream"
    senders = SENDERS_PER_LINE[regime]
    a_split = -(-core_m // 2) if senders >= 2 and core_m >= 2 else core_m  # m rows of A part 0
    w_split = -(-kb // 2) if senders >= 2 and kb >= 2 else kb  # k rows of W part 0
    tA = int(a.buffer_aligned_page_size())
    tW = int(w.buffer_aligned_page_size())
    local_cbs = local_cb_layout(mesh, ctx, scatter_dim, Kt, tA, tW)
    a_pages, w_pages = local_cbs[CB_ACT_K_WINDOW][1], local_cbs[CB_WEIGHT_K_WINDOW][1]

    # chunks: every kernel walks num_chunks x (the per-chunk entries), chunk c -> (i_m, i_n) with the resident
    # operand's axis outer; a chunk's tile origin is (i_m * chunk_m, i_n * chunk_n) inside the block
    num_chunks = ctx["n_m"] * ctx["n_n"]
    m_outer = int(scatter_dim == -1)
    n_inner = ctx["n_n"] if m_outer else ctx["n_m"]
    zone_sets = ctx["zone_sets"]
    credit_chunks = max(0, num_chunks - zone_sets)  # chunks whose sends wait for a downstream credit
    chunk_ct = [num_chunks, n_inner, m_outer, ctx["chunk_m"], ctx["chunk_n"], ctx["block_m"], ctx["block_n"]]
    ring_slots = {d: zone_sets * slots[d] for d in _DIRS}  # the arrival CBs' slots
    zone_fwd_off = DEPTH_HANDOFF * P * STAGE_TILE_BYTES
    zone_off = {"fwd": zone_fwd_off, "bwd": zone_fwd_off + ring_slots["fwd"] * P * STAGE_TILE_BYTES}
    transport = dict(credit_chunks=credit_chunks, ring_slots=ring_slots, zone_off=zone_off)
    chunk_tiles = max(1, min(int(ttnn.get_tt_fabric_max_payload_size_bytes()) // STAGE_TILE_BYTES, P))

    # compile-time args, in each kernel's get_compile_time_arg_val order
    reader_ct = [
        CB_ACT_K_WINDOW,
        CB_WEIGHT_K_WINDOW,
        CB_ARRIVALS_FWD,
        CB_ARRIVALS_BWD,
        core_m,
        core_n,
        kb,
        nkb,
        Kt,
        ctx["Nt"],
        ctx["block_m"] if scatter_dim == -2 else 0,  # A rows per block
        ctx["block_n"] if scatter_dim == -1 else 0,  # W columns per block
        int(resident and scatter_dim == -1),
        int(resident and scatter_dim == -2),
        P,
        G,  # entries per chunk
        tA,
        tW,
        a_pages // (core_m * kb),  # CB capacity in K-block windows
        w_pages // (kb * core_n),
        a_split,
        w_split,
        PREFETCH_STEPS,
        LOOKAHEAD_AFTER_BARRIER[regime],
        READ_NOCS[regime][0],
        READ_NOCS[regime][1],
        *chunk_ct,  # 26..32
        *ttnn.TensorAccessorArgs(a).get_compile_time_args(),
        *ttnn.TensorAccessorArgs(w).get_compile_time_args(),
    ]
    compute_ct = [
        CB_ACT_K_WINDOW,
        CB_WEIGHT_K_WINDOW,
        CB_PARTIAL_HOLD,
        CB_FINISHED,
        CB_ARRIVALS_FWD,
        CB_ARRIVALS_BWD,
        core_n,  # in1 window row stride
        kb,
        nkb,
        P,
        G,  # entries per chunk
        ctx["super_block"],  # K-windows a subblock accumulates in DEST per pack (an operand fill packs per window)
        CB_ZERO,
        num_chunks,
        n_inner,
        int(resident),
    ]
    writer_ct = [CB_FINISHED, P, ctx["block_n"], DEPTH_HANDOFF, STAGE_TILE_BYTES, G, CB_ZERO, *chunk_ct]
    writer_ct += ttnn.TensorAccessorArgs(output).get_compile_time_args()
    port_ct = [
        CB_PORT_CHUNKS,
        P,
        chunk_tiles,
        PORT_GROUP,
        STAGE_TILE_BYTES,
        DEPTH_HANDOFF,
        MAILBOX_STRIDE,
        -(-grid.n_cores // L),  # matmul cores one port serves (link(k) = k mod L)
        num_chunks,
        G,  # entries per chunk
        zone_sets,
    ]

    src = ttnn.KernelDescriptor.SourceType.FILE_PATH
    dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(
        processor=getattr(ttnn.DataMovementProcessor, risc), noc=noc
    )
    cfg = ctx["compute_kernel_config"]
    compute_cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=cfg.math_fidelity,
        math_approx_mode=cfg.math_approx_mode,
        fp32_dest_acc_en=True,  # the cross-device adds accumulate in fp32 DEST (a floor, never below the caller's)
        dst_full_sync_en=True,  # restores 8-tile fp32 subblocks
    )
    port_vcore_of = {coord: {key: virt(c) for key, c in ports.items()} for coord, ports in ports_of.items()}
    # mcast channels: A is shared along an n-line (same gm), W along an m-line (same gn)
    order = {"m_on_x": (_mc.McastCoreOrder.ColumnMajor, _mc.McastCoreOrder.RowMajor)}.get(
        gd["orient"], (_mc.McastCoreOrder.RowMajor, _mc.McastCoreOrder.ColumnMajor)
    )
    mcast_lines = {"a": (grid_n, order[0]), "b": (grid_m, order[1])}  # (group size, core order)
    sender_plan = plan_mcast_senders(grid_m, grid_n, a_split < core_m, w_split < kb, disjoint=resident)
    mc_cfg = _mc.McastConfig(noc=ttnn.NOC.NOC_0, handshake=True)
    cb = lambda idx, pages, page_bytes, fmt, crs: ttnn.CBDescriptor(
        total_size=pages * page_bytes,
        core_ranges=crs,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page_bytes)],
    )
    # placed in local_cb_layout's order up from the allocator base
    local_formats = {
        CB_ACT_K_WINDOW: a.dtype,
        CB_PARTIAL_HOLD: ttnn.float32,
        CB_ZERO: ttnn.bfloat16,
        CB_WEIGHT_K_WINDOW: w.dtype,
    }
    staging = res["staging"]

    mesh_desc = ttnn.MeshProgramDescriptor()
    for grp in ctx["groups"]:
        for p, coord in enumerate(grp):
            sc = ctx["sched"][p]
            peers = _peers(grp, p, ctx["ring"])
            ports = ports_of[coord]
            all_crs = ttnn.CoreRangeSet([*mm_crs.ranges(), *(ttnn.CoreRange(c, c) for c in ports.values())])
            program = ttnn.ProgramDescriptor()
            program.semaphores = [
                ttnn.SemaphoreDescriptor(id=0, core_ranges=all_crs, initial_value=0),  # consumed_fwd
                ttnn.SemaphoreDescriptor(id=1, core_ranges=all_crs, initial_value=0),  # consumed_bwd
                # freed (port cores): chunks' worth of arrivals this port's matmul cores consumed, in the direction
                # this port's peer sends into this chip (forwarded upstream as chunk credits)
                ttnn.SemaphoreDescriptor(id=2, core_ranges=all_crs, initial_value=0),
            ]
            # per (direction of the arrival, link): matmul cores whose half receives that direction's arrivals
            receivers = {
                (d, l): sum(
                    int(len(sc["arrivals"][(grid.half(gm, gn), d)]) > 0)
                    for k, (gm, gn, _) in enumerate(mm_cores)
                    if k % L == l
                )
                for d in _DIRS
                for l in range(L)
            }

            # ---- matmul cores ----
            r_rt, c_rt, w_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for k, (gm, gn, core) in enumerate(mm_cores):
                r_rt[core.x][core.y], c_rt[core.x][core.y], w_rt[core.x][core.y] = _matmul_core_args(
                    ctx, transport, grid, sc, k, gm, gn, port_vcore_of[coord], peers
                )
            reader = ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "matmul_reduce_scatter_reader.cpp"),
                source_type=src,
                core_ranges=mm_crs,
                compile_time_args=reader_ct,
                runtime_args=r_rt,
                defines=[],
                config=dm("RISCV_1", NOC0),
            )
            compute = ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "matmul_reduce_scatter_compute.cpp"),
                source_type=src,
                core_ranges=mm_crs,
                compile_time_args=compute_ct,
                runtime_args=c_rt,
                defines=[],
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
            sem_id = len(program.semaphores)
            for prefix, start in sender_plan:
                if start is None:
                    _mc.attach_absent_mcast(reader, prefix)
                    continue
                group_size, core_order = mcast_lines[prefix[0]]
                snd = _mc.McastFixedSenderConfig(sender_index=start, placement=_mc.McastSenderPlacement.Staggered)
                m = _mc.Mcast(mesh, mc_cfg, mm_crs, group_size, snd, core_order)
                m.attach(program, prefix, [reader], sem_id)
                sem_id = m.next_semaphore_id()

            # ---- port cores ----
            pr_rt, ps_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            port_used = []
            for (d, l), pcore in sorted(ports.items(), key=lambda kv: (kv[0][1], kv[0][0])):
                peer = peers[d]
                if peer is None:
                    continue
                port_used.append(pcore)
                pr_rt[pcore.x][pcore.y], args = _port_args(
                    ctx, transport, receivers, grid, p, grp, d, l, ports_of, link_halves
                )
                if credit_chunks:
                    q = (p + 1) % G if d == "fwd" else (p - 1) % G
                    opp = "bwd" if d == "fwd" else "fwd"
                    assert any(
                        ctx["sched"][q]["roles"][(ctx["sched"][q]["entries"][h][e], h)][0] == opp
                        for e in range(G)
                        for h in link_halves[l]
                    ) == (receivers[(opp, l)] > 0), (p, d, l)
                args += list(
                    ttnn.setup_fabric_connection(
                        ctx["node"](coord), ctx["node"](peer), ctx["links"][(coord, peer)][l], program, pcore
                    )
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
            program.cbs = [
                *(
                    cb(idx, pages, page_bytes, local_formats[idx], mm_crs)
                    for idx, (_, pages, page_bytes) in local_cbs.items()
                ),
                ttnn.cb_descriptor_from_sharded_tensor(
                    CB_FINISHED, staging, 0, DEPTH_HANDOFF * P * STAGE_TILE_BYTES, mm_crs
                ),
                ttnn.cb_descriptor_from_sharded_tensor(
                    CB_ARRIVALS_FWD, staging, zone_off["fwd"], ring_slots["fwd"] * P * STAGE_TILE_BYTES, mm_crs
                ),
                ttnn.cb_descriptor_from_sharded_tensor(
                    CB_ARRIVALS_BWD, staging, zone_off["bwd"], ring_slots["bwd"] * P * STAGE_TILE_BYTES, mm_crs
                ),
                cb(CB_PORT_CHUNKS, 2 * PORT_GROUP * chunk_tiles, STAGE_TILE_BYTES, ttnn.bfloat16, used_port_crs),
            ]
            r, c = coord
            mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = program
    return mesh_desc
