# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""moe_fused_swiglu — everything that is a pure function of shapes, dtypes and grid.

The program descriptor next door assembles a `ProgramDescriptor`; this module decides WHAT to
assemble. Split out so the blocking arithmetic is testable on a host with no device, and so the
descriptor reads as a sequence rather than as an 800-line function.

Every measurement that justifies a constant here is recorded in
`perf_experiments/DESIGN_NOTES.md`, one section per decision. Comments below state the claim and
point there; they do not re-argue it.
"""

from __future__ import annotations

TILE = 32

# --------------------------------------------------------------------------------------------
# Tuning constants. Each shipped after a measurement; see DESIGN_NOTES.md for the A/B.
# --------------------------------------------------------------------------------------------
#: Token tile-rows per M-block — the CB SIZING bound, not the work. Power of two so every runtime
#: `m_eff` divides it and no shrunk reserve can straddle a CB's FIFO end.
M_BLOCK = 8

#: DEST tile budget for one matmul output sub-block, at half sync with fp32_dest_acc_en=False.
DEST_AUTO_LIMIT_TILES = 8

#: gate/up output sub-block height. 1, because its width is already HN_PAD against a budget of 8.
OUT_SUBBLOCK_H_GU = 1

#: Cap on the `down` sub-block height; the real value is derived against the DEST budget below.
#: Raising it measured a consistent small regression, so the cap is the identity.
OUT_SUBBLOCK_H_DN_MAX = 1

#: Eltwise DEST-window block size — tiles per acquire/commit/wait/release cycle. Worth 1.05-1.07x.
ELTWISE_BLK = DEST_AUTO_LIMIT_TILES

#: Buffer depths, in blocks. DEPTH_H = 3 so a late round's producer is not flow-controlled by its
#: own consumption; DEPTH_X = 2 lets the reader stage M-block b+1 during block b's phase 2.
DEPTH_W = 2
DEPTH_X = 2
DEPTH_H = 3
DEPTH_OUT = 2
DEPTH_XSTAGE = 1
XSTICK_ROWS = 1

#: W_down prefetch depth in phase-2 K-blocks. 1 measured best (227.8 us vs 228.7 at 4, 240.1 at 11).
WD_AHEAD = 1

#: Cross-M-block weight residency: every weight read is a pure function of this core's
#: kstart/hstart/jstart with no M-block index, so b > 0 re-reads bytes already in the CB slot.
#: gate/up -9.36 %, W_down a further -2.04 %.
W_RESIDENT = True
WD_RESIDENT = True

#: Hidden-axis chunks the gate/up weight stream is published and consumed in, so the matmul on
#: chunk c overlaps the DRAM read of c+1. Clamped below to a divisor of HN_PAD.
GU_CHUNKS = 3

#: Hold the writer's W_up stream until this core's reader has staged x, so the 3.67 MB activation
#: stream is not queued behind 16.5 MB of weights.
XPRIO = True

#: How many rounds' senders an h receiver acks in one reserve. THE round-cost lever; clamped to
#: DEPTH_H - 1 below.
HACK_AHEAD = 2

#: Eighths of every phase-2 W_down K-block read by the WRITER on NOC_1 instead of the reader's
#: NOC_0. A real interior optimum: -4.7 / -2.6 / -1.1 % at 3.
WD_SPLIT = 3

#: `/perf-measure` ablation hook: one transport stubbed, all CB scaffolding intact. NOT a
#: correctness mode — edit this to measure, never to ship. `+`-separated, cumulative.
ABLATE = ""
DM_ABLATIONS = ("no_reduce_xfer", "no_h_xfer", "no_x_xfer", "no_w_xfer", "no_xstage_xfer", "no_owrite")
COMPUTE_ABLATIONS = ("skip_compute", "skip_eltwise")

#: Mailbox handshake word: the reader publishes {count, M_t, m_blocks} plus this flag.
MAILBOX_MAGIC = 0xC0FFEE01
MAILBOX_WORDS = 16

# --------------------------------------------------------------------------------------------
# Circular buffers. The numeric slot is only the buffer index.
# --------------------------------------------------------------------------------------------
CB_X_IN = 0  # row-major x stick slices (bf16) or bfp8 tiles
CB_X_TILES = 1  # resident bfp8 in0 block, filled by the row multicast
CB_X_STAGE = 2  # tilized x tile-row awaiting its multicast turn
CB_W_GATE = 3
CB_W_UP = 4
CB_W_DOWN = 5
CB_H = 6  # gathered h, one phase-2 K-block per round
CB_IDX_SCRATCH = 7
CB_COUNTS_SCRATCH = 8
CB_GATHER_GATE = 9  # every contributor's gate slice, slot `row`
CB_GATHER_UP = 10
CB_SLICE_GATE = 11  # this worker's gate-slice accumulator (in-place)
CB_SLICE_UP = 12
CB_H_SLICE = 13  # this worker's finished h slice, unicast into the root's cb_h_local
CB_OUT_TILES = 14
CB_GATE_ACC = 15  # gate partial accumulator (matmul out + in-place reduce adds)
CB_UP_ACC = 16
CB_GATE_SILU = 17  # SiLU(sum(gate)) on this worker's slice
CB_H_LOCAL = 18  # column root: this column's assembled h block, awaiting its all-gather round
CB_OUT_INTERM = 19  # phase-2 packer-L1 accumulation region

# --------------------------------------------------------------------------------------------
# Semaphores. Every one is MONOTONE — never reset within a dispatch, always compared with
# wait_min against a running total. See RACE_AUDIT.md.
# --------------------------------------------------------------------------------------------
SEM_X_BASE = 0  # x row multicast (data_ready, consumer_ready)
SEM_H_BASE = 2  # h all-gather   (data_ready, consumer_ready)
SEM_GO = 4  # the peer INVITE: every core tells its column "my landing CBs are reserved"
SEM_DATA = 5  # contributor -> worker "my slice landed"
SEM_HSLICE = 6  # worker -> column root "my finished h slice landed in your cb_h_local"
SEM_XSTAGED = 7  # reader -> writer, SAME core: "x is staged" (XPRIO)
SEM_H_RDY_BASE = 8  # one VALID cell per cb_h slot
SEM_H_FREE = SEM_H_RDY_BASE + DEPTH_H  # receiver -> sender window ack
SEM_WDSPLIT = SEM_H_FREE + 1  # writer -> reader, SAME core: "my W_down share landed"
SEM_COUNT = SEM_WDSPLIT + 1
NUM_DEVICE_SEMAPHORES = 16

#: Per-core L1 available to this op's circular buffers, as a default for host-only use. The
#: descriptor overrides it with the device's own `get_max_worker_l1_unreserved_size()`.
L1_CB_BUDGET = 1_532_032

#: Headroom subtracted from that figure before deciding weight residency.
#:
#: NOT arbitrary. The allocator's "circular buffers grow to N B" is an ADDRESS, not a size: the CB
#: region starts above the kernel binaries, runtime args and semaphores, and that base is
#: PROGRAM-SPECIFIC, so no host-side call returns it. Measured here: a descriptor whose CBs sum to
#: 1 515 264 B threw at 1 626 752 B, i.e. a base of 111 488 B, while
#: `get_max_worker_l1_unreserved_size()` reported only 40 832 B of reservation. This margin is
#: exactly that difference, so the residency decision fails over cleanly instead of reaching the
#: allocator. It is measured on one program and the base is program-specific, so treat it as the
#: best available estimate rather than an exact bound — the allocator remains the final authority.
L1_CB_RESERVE = 70_656  # 111 488 measured base - 40 832 reported reserved


# --------------------------------------------------------------------------------------------
# Small numeric helpers
# --------------------------------------------------------------------------------------------
def pow2_ceil(v: int) -> int:
    p = 1
    while p < v:
        p <<= 1
    return p


def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def largest_divisor_le(n: int, cap: int) -> int:
    """The `scatter` worker count. HOST TWIN of `slice_workers()` in the shared kernel header —
    the kernel one is the definition; this only sizes the CBs it implies."""
    return max(d for d in range(1, min(n, cap) + 1) if n % d == 0)


def split(total: int, groups: int):
    """`base + (i < rem)` split. Returns (sizes, starts)."""
    base, rem = divmod(total, groups)
    sizes = [base + (1 if i < rem else 0) for i in range(groups)]
    starts, acc = [], 0
    for s in sizes:
        starts.append(acc)
        acc += s
    return sizes, starts


# --------------------------------------------------------------------------------------------
# The reduce-scatter slice plan
# --------------------------------------------------------------------------------------------
def scatter_plan(m_block: int, m_eff_min: int, hn_pad: int, kgroups: int):
    """CB sizing for the reduce-scatter, or (None, reason) if this geometry cannot express it.

    Enumerates EVERY runtime m_eff the kernels can reach and sizes the CBs for all of them at
    once, because the slice plan shrinks with m_eff while the CBs are allocated once.

    The precondition is `P % B == 0` for every CB cycled in blocks of B pages: a CB's write
    pointer wraps only at the CB END, so a block starting mid-CB and running past the end
    OVERRUNS INTO THE NEXT CB. A plan violating it measured PCC 0.709-0.886 where every legal
    plan scored >= 0.9955.
    """
    if kgroups < 2:
        return None, f"KGROUPS {kgroups} < 2: a column of one has no cross-column reduce to scatter"
    sizes, m = [], m_eff_min
    while m <= m_block:
        t = m * hn_pad
        sizes.append(t // largest_divisor_le(t, kgroups))
        m *= 2
    slice_pages = 1
    for a in sizes:
        slice_pages = slice_pages * a // gcd(slice_pages, a)
    gather_pages = kgroups * max(sizes)
    for a in sizes:
        if slice_pages % a or gather_pages % a:
            return None, (
                f"slice sizes {sorted(set(sizes))} over the reachable m_eff need slice CBs of "
                f"{slice_pages} and landing CBs of {gather_pages} pages, and {a} divides neither"
            )
    return {"slice_pages": slice_pages, "gather_pages": gather_pages, "sizes": sizes}, None


# --------------------------------------------------------------------------------------------
# The blocking model, resolved
# --------------------------------------------------------------------------------------------
class Blocking:
    """Every derived block factor for one (grid, emb, hidden, m_t_max). Pure arithmetic.

    Raises with a precise message for any geometry it cannot serve, rather than silently
    producing a program that computes the wrong thing.
    """

    def __init__(
        self,
        hgroups: int,
        kgroups: int,
        emb: int,
        hidden: int,
        m_t_max: int,
        *,
        w_tile: int = 576,
        bfp8_tile: int = 1088,
        bf16_tile: int = 2048,
        x_stick: int = 0,
        l1_budget: int = 0,
    ):
        if kgroups < 2:
            raise ValueError(
                f"moe_fused_swiglu: needs a grid at least 2 rows tall (got {hgroups}x{kgroups}); "
                f"a column of one has no cross-column reduce"
            )
        if emb % TILE or hidden % TILE:
            raise ValueError(f"moe_fused_swiglu: emb {emb} and hidden {hidden} must be tile-aligned")

        self.hgroups, self.kgroups = hgroups, kgroups
        self.num_cores = hgroups * kgroups
        self.emb, self.hidden = emb, hidden
        self.emb_t, self.hid_t = emb // TILE, hidden // TILE
        self.m_t_max = m_t_max

        if pow2_ceil(M_BLOCK) != M_BLOCK:
            raise ValueError(f"moe_fused_swiglu: M_BLOCK {M_BLOCK} must be a power of two")
        self.m_eff_min = pow2_ceil(OUT_SUBBLOCK_H_GU)
        if self.m_eff_min > M_BLOCK:
            raise ValueError(f"moe_fused_swiglu: OUT_SUBBLOCK_H_GU {OUT_SUBBLOCK_H_GU} exceeds M_BLOCK {M_BLOCK}")

        # Kg — the emb contraction, split across grid ROWS.
        self.kr_sizes, self.kr_starts = split(self.emb_t, kgroups)
        self.kr_pad = max(self.kr_sizes)

        # Hn — the hidden axis, split across grid COLUMNS. `hn_pad` is a PADDING choice, not
        # ceil(hid_t/hgroups): it must additionally decompose against the DEST budget and satisfy
        # the scatter plan's divisibility lattice, so it is searched.
        self.hn_pad, self.gu_chunks, self.plan = self._choose_hn_pad()
        self.gu_chunk_w = self.hn_pad // self.gu_chunks
        self.hn_sizes = [max(0, min(self.hn_pad, self.hid_t - x * self.hn_pad)) for x in range(hgroups)]
        self.hn_last = self.hid_t - (hgroups - 1) * self.hn_pad
        self.hn_block = self.gu_chunk_w
        self.gu_in1_subblocks = self.gu_chunk_w // self.hn_block

        # Ne — the emb output, split across ALL cores. EC_MAX is the phase-2 N *stride*: every
        # phase-2 CB reserves in EC_MAX-wide units, so its page count must be a multiple of it.
        self.ec_sizes, self.ec_starts = split(self.emb_t, self.num_cores)
        self.ec_max = max(self.ec_sizes)

        # `down` sub-block height: largest power of two still inside DEST, capped by the knob and
        # by M_BLOCK. Power of two so min(h, m_eff) divides m_eff for every runtime m_eff.
        h = 1
        while h * 2 <= min(OUT_SUBBLOCK_H_DN_MAX, M_BLOCK) and h * 2 * self.ec_max <= DEST_AUTO_LIMIT_TILES:
            h *= 2
        self.out_subblock_h_dn = h
        if M_BLOCK % OUT_SUBBLOCK_H_GU or M_BLOCK % h:
            raise ValueError(f"moe_fused_swiglu: M_BLOCK {M_BLOCK} must be a multiple of both sub-block heights")

        self.gather_pages = self.plan["gather_pages"]
        self.slice_pages = self.plan["slice_pages"]

        self.max_m_blocks = (m_t_max + M_BLOCK - 1) // M_BLOCK
        self.depth_x = DEPTH_X if self.max_m_blocks > 1 else 1
        self.depth_w = 1 if W_RESIDENT else DEPTH_W
        self.wd_ahead = max(1, min(WD_AHEAD, hgroups))
        self.hack_ahead = max(1, min(HACK_AHEAD, DEPTH_H - 1))

        # W_down residency wants the CB to hold the WHOLE phase-2 K stream, which is LINEAR IN N
        # and in the weight dtype: `hgroups * hn_pad * ec_max` tiles. At N 2048 / bfp4 that is
        # 111 KB and it fits; at 4x the hidden extent, or at bf16 weights (3.56x the bytes), it
        # does not. So residency is a BUDGET DECISION, not a constant — fall back to the smallest
        # legal depth rather than throwing at program build.
        self.w_tile, self.bfp8_tile, self.bf16_tile = w_tile, bfp8_tile, bf16_tile
        self.x_stick = x_stick or bfp8_tile
        self.l1_budget = l1_budget or L1_CB_BUDGET
        self.wd_resident = WD_RESIDENT
        self.depth_wd = self._choose_depth_wd()
        if self.l1_bytes(True) > self.l1_budget:
            self.wd_resident = False  # residency is what pins the depth to hgroups
            self.depth_wd = self._min_depth_wd()
            while self.l1_bytes(True) > self.l1_budget:
                nxt = self._next_smaller_depth_wd(self.depth_wd)
                if nxt == self.depth_wd:
                    break
                self.depth_wd = nxt
        self.wd_split = max(0, min(8, WD_SPLIT)) if self.depth_wd == hgroups else 0

    # -- hn_pad ---------------------------------------------------------------------------
    def _hn_pad_legal(self, hn_pad: int):
        """(gu_chunks, plan) if this hidden width can be served, else (None, reason).

        Four constraints, all of which have bitten:
          * it must COVER the hidden extent;
          * every column group must get a real column — a uniform width means the number of
            non-empty groups is `ceil(hid_t/hn_pad)`, so `hn_pad * (hgroups-1) < hid_t`;
          * it must decompose into chunks that fit the DEST budget, `OUT_SUBBLOCK_H_GU * chunk_w
            <= DEST_AUTO_LIMIT_TILES`;
          * the scatter plan's divisibility lattice must close (see `scatter_plan`).
        """
        if hn_pad * self.hgroups < self.hid_t:
            return None, "does not cover the hidden extent"
        if hn_pad * (self.hgroups - 1) >= self.hid_t:
            return None, f"leaves a column group with no real column (hid_t {self.hid_t})"
        # Prefer the tuned chunk count; widen it only if the DEST budget forces smaller chunks.
        for chunks in sorted(range(1, hn_pad + 1), key=lambda c: (abs(c - GU_CHUNKS), c)):
            if hn_pad % chunks:
                continue
            if OUT_SUBBLOCK_H_GU * (hn_pad // chunks) > DEST_AUTO_LIMIT_TILES:
                continue
            plan, why = scatter_plan(M_BLOCK, self.m_eff_min, hn_pad, self.kgroups)
            if plan is None:
                return None, why
            return (chunks, plan), None
        return None, (
            f"no chunk count divides {hn_pad} with a sub-block inside the DEST budget of " f"{DEST_AUTO_LIMIT_TILES}"
        )

    def _choose_hn_pad(self):
        """Smallest legal hidden width per column group.

        `ceil(hid_t/hgroups)` is only the FLOOR. The width also has to leave every column group a
        real column, decompose inside the DEST budget, and satisfy the scatter plan — and none of
        those holds for every N. Padding up costs a few pad columns, which `HnSteps` already
        narrows out of the last K-block's FMA.
        """
        floor = (self.hid_t + self.hgroups - 1) // self.hgroups
        ceiling = max(floor, (self.hid_t - 1) // max(1, self.hgroups - 1)) + 1
        reasons = []
        for hn_pad in range(floor, ceiling + 1):
            ok, why = self._hn_pad_legal(hn_pad)
            if ok:
                return hn_pad, ok[0], ok[1]
            reasons.append(f"  hn_pad {hn_pad}: {why}")
        # Actionable, not just a refusal: report the column counts that WOULD work here.
        workable = []
        for h in range(2, self.hgroups + 1):
            probe = Blocking.__new__(Blocking)
            probe.hgroups, probe.kgroups = h, self.kgroups
            probe.hid_t, probe.hidden, probe.m_eff_min = self.hid_t, self.hidden, self.m_eff_min
            f = (self.hid_t + h - 1) // h
            if any(probe._hn_pad_legal(v)[0] for v in range(f, f + DEST_AUTO_LIMIT_TILES + 2)):
                workable.append(h)
        raise ValueError(
            f"moe_fused_swiglu: hidden {self.hidden} ({self.hid_t} tiles) cannot be split across "
            f"{self.hgroups} grid columns:\n"
            + "\n".join(reasons)
            + (
                f"\n  workable column counts for this hidden: {workable} — pass a narrower core_grid"
                if workable
                else ""
            )
        )

    # -- depth_wd -------------------------------------------------------------------------
    def _depth_wd_legal(self, d: int) -> bool:
        """A cb_w_down depth must hold the pipeline: the `wd_ahead` blocks in flight, plus the
        reserved-but-unpublished one the deferred barrier carries across a round boundary, plus
        the one compute is consuming.

        It must additionally DIVIDE hgroups in exactly two cases, both of which make the reader's
        hgroups pushes per M-block have to land the write pointer back on the CB base:
          * residency, where slot r must hold K-block r on every M-block. Breaking this is
            silent-wrong-answer class — b > 0 matmuls against the wrong weight block, no hang;
          * `wd_ahead > 1`, where the multi-block BATCH reserve starts mid-CB and would otherwise
            straddle the FIFO end.
        With wd_ahead == 1 and no residency, single-block pushes can never straddle, so any depth
        above the floor is legal — which is what lets a prime hgroups fall back at all.
        """
        if d < self.wd_ahead + 2:
            return False
        if (self.wd_resident or self.wd_ahead > 1) and self.hgroups % d:
            return False
        return True

    def _min_depth_wd(self):
        for d in range(self.wd_ahead + 2, self.hgroups + 1):
            if self._depth_wd_legal(d):
                return d
        return self.hgroups

    def _next_smaller_depth_wd(self, d):
        for c in range(d - 1, self.wd_ahead + 1, -1):
            if self._depth_wd_legal(c):
                return c
        return d

    def _choose_depth_wd(self):
        """cb_w_down depth in phase-2 K-blocks.

        Needs >= wd_ahead + 2 (in flight, plus the reserved-not-published one the deferred
        barrier carries across a round boundary, plus the one compute is consuming) and must
        DIVIDE hgroups, so the reader's hgroups pushes per M-block bring the write pointer back
        to the CB base and K-block r always occupies slot r. Residency needs the full hgroups.
        """
        return self.hgroups if self.wd_resident else self._min_depth_wd()

    # -- the CB layout -------------------------------------------------------------------
    def cb_layout(self, x_is_rm: bool, out_tile: int = 0, idx_page: int = 64, counts_page: int = 64):
        """(cb_index, pages, page_bytes, format_key) for EVERY circular buffer, in order.

        THE one definition. The descriptor turns this into `CBDescriptor`s and `l1_bytes` sums it,
        so the residency decision is priced on exactly the bytes that get allocated — the two used
        to be separate arithmetic, and the copy undercounted by ~95 KB, which turned a clean
        "does not fit" into an allocator throw at program build.
        """
        b8, b16, w, out = self.bfp8_tile, self.bf16_tile, self.w_tile, (out_tile or self.bfp8_tile)
        gu = M_BLOCK * self.hn_pad
        outb = M_BLOCK * self.ec_max
        return [
            (CB_X_IN, XSTICK_ROWS * TILE if x_is_rm else 1, self.x_stick, "x_in"),
            (CB_X_TILES, self.depth_x * M_BLOCK * self.kr_pad, b8, "bfp8"),
            (CB_X_STAGE, DEPTH_XSTAGE * self.kr_pad if x_is_rm else 1, b8, "bfp8"),
            (CB_W_GATE, self.depth_w * self.kr_pad * self.hn_pad, w, "weight"),
            (CB_W_UP, self.depth_w * self.kr_pad * self.hn_pad, w, "weight"),
            (CB_W_DOWN, self.depth_wd * self.hn_pad * self.ec_max, w, "weight"),
            (CB_H, DEPTH_H * gu, b8, "bfp8"),
            (CB_IDX_SCRATCH, 1, idx_page, "u32"),
            (CB_COUNTS_SCRATCH, 1, counts_page, "u32"),
            (CB_GATHER_GATE, self.gather_pages, b8, "bfp8"),
            (CB_GATHER_UP, self.gather_pages, b8, "bfp8"),
            (CB_SLICE_GATE, self.slice_pages, b16, "bf16"),
            (CB_SLICE_UP, self.slice_pages, b16, "bf16"),
            (CB_H_SLICE, self.slice_pages, b8, "bfp8"),
            (CB_OUT_TILES, DEPTH_OUT * outb, out, "out"),
            (CB_GATE_ACC, gu, b8, "bfp8"),
            (CB_UP_ACC, gu, b8, "bfp8"),
            (CB_GATE_SILU, self.slice_pages, b16, "bf16"),
            (CB_H_LOCAL, gu, b8, "bfp8"),
            (CB_OUT_INTERM, outb, b16, "bf16"),
        ]

    def l1_bytes(self, x_is_rm: bool, out_tile: int = 0) -> int:
        return sum(pages * page for _, pages, page, _ in self.cb_layout(x_is_rm, out_tile))

    def describe(self) -> str:
        return (
            f"{self.hgroups}x{self.kgroups} grid, emb {self.emb}, hidden {self.hidden}: "
            f"kr_pad {self.kr_pad}, hn_pad {self.hn_pad} (floor "
            f"{(self.hid_t + self.hgroups - 1) // self.hgroups}), gu_chunks {self.gu_chunks}, "
            f"ec_max {self.ec_max}, depth_wd {self.depth_wd}, wd_split {self.wd_split}"
        )
