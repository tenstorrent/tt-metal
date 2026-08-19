# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Moe fused-SwiGLU blocking arithmetic and L1 tuning knobs."""

from __future__ import annotations

import os

TILE = 32


def _env_int(name: str, default: int) -> int:
    """Read a tuning override while keeping the checked-in value as the default."""
    return int(os.environ.get(f"MOE_FUSED_SWIGLU_{name}", str(default)), 0)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(f"MOE_FUSED_SWIGLU_{name}")
    return default if value is None else value.lower() not in ("0", "false", "no", "off")


# --------------------------------------------------------------------------------------------
# Tuning constants.
# --------------------------------------------------------------------------------------------
#: Token tile-rows per M-block — the CB SIZING bound, not the work. Power of two so every runtime
#: `m_eff` divides it and no shrunk reserve can straddle a CB's FIFO end.
M_BLOCK = _env_int("M_BLOCK", 8)

#: DEST tile budget for one matmul output sub-block, at half sync with fp32_dest_acc_en=False.
DEST_AUTO_LIMIT_TILES = 8

#: gate/up output sub-block height. 1, because its width is already HN_PAD against a budget of 8.
OUT_SUBBLOCK_H_GU = _env_int("OUT_SUBBLOCK_H_GU", 1)

#: Cap on the `down` sub-block height; the real value is derived against the DEST budget below.
#: Direct final-output packing reloads the bf16 partial before the last K-block. The host derives
#: the uniform-safe height against ec_max (2 at 11x8 / emb=7168); device cores with narrower real
#: ec may grow to this cap at runtime (4x2 exactly fills the eight-tile DEST).
OUT_SUBBLOCK_H_DN_MAX = _env_int("OUT_SUBBLOCK_H_DN_MAX", 4)

#: Eltwise DEST-window block size — tiles per acquire/commit/wait/release cycle. Worth 1.05-1.07x.
ELTWISE_BLK = DEST_AUTO_LIMIT_TILES

#: Buffer depths, in blocks. DEPTH_H = 3 so a late round's producer is not flow-controlled by its
#: own consumption; DEPTH_X = 2 lets the reader stage M-block b+1 during block b's phase 2.
DEPTH_W = _env_int("DEPTH_W", 2)
DEPTH_X = _env_int("DEPTH_X", 2)
DEPTH_H = _env_int("DEPTH_H", 3)
DEPTH_OUT = _env_int("DEPTH_OUT", 2)
XSTICK_ROWS = _env_int("XSTICK_ROWS", 1)

#: W_down prefetch depth in phase-2 K-blocks.
WD_AHEAD = _env_int("WD_AHEAD", 1)

#: Use full M tile rows for the phase-2 W_down schedule.
WD_MROW_ROUNDS = _env_bool("WD_MROW_ROUNDS", True)

#: Enable the grouped phase-2 schedule for sufficiently large M.
WD_MGROUPS = _env_bool("WD_MGROUPS", False)
WD_MGROUP_MIN_BLOCKS = _env_int("WD_MGROUP_MIN_BLOCKS", 4)
WD_MGROUP_ROWS = _env_int("WD_MGROUP_ROWS", M_BLOCK // 2)

#: Keep weights resident across M-blocks when L1 permits.
W_RESIDENT = _env_bool("W_RESIDENT", True)
WD_RESIDENT = _env_bool("WD_RESIDENT", True)

#: Hidden-axis chunks the gate/up weight stream is published and consumed in, so the matmul on
#: chunk c overlaps the DRAM read of c+1. Clamped below to a divisor of HN_PAD.
GU_CHUNKS = _env_int("GU_CHUNKS", 3)

#: Stage x before issuing the writer's W_up stream.
XPRIO = _env_bool("XPRIO", True)

#: Emit fine-grained device-profiler zones. Off for ordinary latency sweeps because each zone
#: writes profiler records on the kernel critical path.
STAGE_PROFILE = _env_bool("STAGE_PROFILE", False)

#: How many rounds' senders an h receiver acks in one reserve. THE round-cost lever; clamped to
#: DEPTH_H - 1 below.
HACK_AHEAD = _env_int("HACK_AHEAD", 2)

#: Bitmask selecting full-M h rounds sent by the writer on NoC1.
H_ROUND_NOC1_MASK = int(os.environ.get("MOE_H_ROUND_NOC1_MASK", "0"), 0) & ((1 << M_BLOCK) - 1)

#: Use one completion signal after both reduce-scatter payloads arrive.
SCATTER_ONE_SIGNAL = os.environ.get("MOE_SCATTER_ONE_SIGNAL", "1") not in ("0", "false", "False")

#: Eighths of each phase-2 W_down K-block read by the writer on NoC1.
WD_SPLIT = _env_int("WD_SPLIT", 3)

#: Whether phase-2 h all-gather payload multicasts are posted.
H_MCAST_POSTED = os.environ.get("MOE_H_MCAST_POSTED", "1") not in ("0", "false", "False")

#: Mailbox handshake word: the reader publishes {count, M_t, m_blocks} plus this flag.
MAILBOX_MAGIC = 0xC0FFEE01
MAILBOX_WORDS = 16

# --------------------------------------------------------------------------------------------
# Circular buffers. The numeric slot is only the buffer index.
# --------------------------------------------------------------------------------------------
CB_X_IN = 0  # row-major x stick slices (bf16) or bfp8 tiles
CB_X_TILES = 1  # resident bfp8 in0 block, filled by the row multicast
CB_X_STAGE = 2  # one-page compute-to-reader completion channel
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
CB_MAILBOX_WRITER = 20  # independent writer-ready FIFO over CB_X_STAGE's 64-byte allocation
CB_MAILBOX_COMPUTE = 21  # independent compute-ready FIFO over CB_X_STAGE's 64-byte allocation
MAILBOX_CB_ALIAS = (CB_X_STAGE, CB_MAILBOX_WRITER, CB_MAILBOX_COMPUTE)

# These three BFP8 views are live in strict phase order: reduce landing -> finished local h slice
# -> final output. They share one physical allocation when the caller's output is BFP8. The
# reader/writer phase-free semaphore below is the necessary cross-block edge: without it, a peer
# could write block b+1's gather payload while block b's output DMA still reads the same SRAM.
PHASE_CB_ALIAS = (CB_GATHER_GATE, CB_H_SLICE, CB_OUT_TILES)
# The SiLU result is consumed completely by the phase-1 multiply before phase 2 starts.  The
# W_down partial is likewise empty again before the next block enters phase 1, so these two BF16
# views can share one physical allocation while retaining independent CB FIFO state.
PHASE_BF16_ALIAS = (CB_GATE_SILU, CB_OUT_INTERM)

# --------------------------------------------------------------------------------------------
# Semaphores. All but one are MONOTONE — never reset within a dispatch, always compared with
# wait_min against a running total. The exception is SEM_H_RDY_BASE + s: those are the h
# all-gather's per-slot VALID FLAGS, set by the sender and cleared by each receiver every round.
# The reader and writer use a shared monotone protocol.
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
SEM_PHASE_FREE = SEM_WDSPLIT + 1  # writer -> reader, SAME core: aliased phase storage is reusable
SEM_HROW_FREE = SEM_PHASE_FREE + 1  # row aggregator -> workers: cb_h_local row slot is reusable
SEM_COUNT = SEM_HROW_FREE + 1
NUM_DEVICE_SEMAPHORES = 16

#: NOC_MAX_TRANSACTION_ID. The writer tags phase-2 W_down K-block r with transaction id r+1 so the
#: reader can be released block by block, which needs one distinct id per block.
NOC_MAX_TRANSACTION_ID = 15

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


def lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


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
        out_tile: int = 0,
        enable_phase_alias: bool = True,
        x_is_rm: bool = True,
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
        self.gu_chunks_target = GU_CHUNKS

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
        self.hn_pad, self.gu_chunks, self.plan, self.balanced_hn = self._choose_hn_pad()
        self.gu_chunk_w = self.hn_pad // self.gu_chunks
        if self.balanced_hn:
            self.hn_sizes, self.hn_starts = split(self.hid_t, hgroups)
        else:
            self.hn_starts = [x * self.hn_pad for x in range(hgroups)]
            self.hn_sizes = [max(0, min(self.hn_pad, self.hid_t - s)) for s in self.hn_starts]
        self.hn_block = self.gu_chunk_w
        self.gu_in1_subblocks = self.gu_chunk_w // self.hn_block
        self.wd_mrow_rounds = bool(WD_MROW_ROUNDS and kgroups == M_BLOCK)

        # Ne — the emb output, split across ALL cores. EC_MAX is the phase-2 N *stride*: every
        # phase-2 CB reserves in EC_MAX-wide units, so its page count must be a multiple of it.
        self.ec_sizes, self.ec_starts = split(self.emb_t, self.num_cores)
        self.ec_max = max(self.ec_sizes)
        self.mgroup_rows = WD_MGROUP_ROWS
        if self.mgroup_rows <= 0:
            raise ValueError(f"moe_fused_swiglu: WD_MGROUP_ROWS must be positive, got {self.mgroup_rows}")
        self.mgroup_cores = self.hgroups * self.mgroup_rows
        self.ec_group_sizes, self.ec_group_starts = split(self.emb_t, self.mgroup_cores)
        self.ec_group_max = max(self.ec_group_sizes)
        # The 12x8 Kimi-K3 geometry benefits strongly from the four-row down schedule: its
        # ordinary 8x2 output shard has the same 192-tile critical work as the larger tuned
        # models, while grouping changes that to 4x3 and halves the h all-gather. Keep the
        # experimental global override, but promote only the measured exact shapes by default.
        tuned_grouped_shape = (emb, hidden) in ((3584, 3072), (6144, 2048))
        enable_wd_mgroups = WD_MGROUPS or (hgroups == 12 and kgroups == 8 and tuned_grouped_shape)
        self.wd_mgroups = bool(
            enable_wd_mgroups
            and self.wd_mrow_rounds
            and M_BLOCK % self.mgroup_rows == 0
            and self.kgroups == M_BLOCK
            and self.kgroups % self.mgroup_rows == 0
            # Width five produced incorrect 7168x2048 output despite fitting DEST; the resident
            # W_down grouped path is currently validated only through width four.
            and self.ec_group_max <= min(4, DEST_AUTO_LIMIT_TILES)
        )
        # One physical resident W_down layout serves both runtime ownership modes. The reader
        # chooses which jstart/ec payload to place in its first columns; the row stride is the
        # larger grouped width in either mode.
        self.wd_ec_max = self.ec_group_max if self.wd_mgroups else self.ec_max

        # `down` sub-block height: largest power of two still inside DEST, capped by the knob and
        # by M_BLOCK. Power of two so min(h, m_eff) divides m_eff for every runtime m_eff.
        # Height 1 must be checked too, not just whether 2 fits: `ec_max` is the sub-block WIDTH
        # and grows as the grid narrows, so a narrow enough grid busts the budget at height 1.
        if self.ec_max > DEST_AUTO_LIMIT_TILES:
            raise ValueError(
                f"moe_fused_swiglu: the `down` sub-block is {self.ec_max} tiles wide (ec_max, = "
                f"emb tiles / {self.num_cores} cores) against a DEST budget of "
                f"{DEST_AUTO_LIMIT_TILES}. This grid is too small for emb {self.emb}; use more cores."
            )
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
        self.depth_h = DEPTH_H
        self.hack_ahead = max(1, min(HACK_AHEAD, self.depth_h - 1))

        # W_down residency wants the CB to hold the WHOLE phase-2 K stream, which is LINEAR IN N
        # and in the weight dtype: `hgroups * hn_pad * ec_max` tiles. At N 2048 / bfp4 that is
        # 111 KB and it fits; at 4x the hidden extent, or at bf16 weights (3.56x the bytes), it
        # does not. So residency is a BUDGET DECISION, not a constant — fall back to the smallest
        # legal depth rather than throwing at program build.
        self.w_tile, self.bfp8_tile, self.bf16_tile = w_tile, bfp8_tile, bf16_tile
        self.x_stick = x_stick or bfp8_tile
        self.out_tile = out_tile or bfp8_tile
        self.enable_phase_alias = enable_phase_alias
        self.x_is_rm = x_is_rm
        self.l1_budget = l1_budget or L1_CB_BUDGET
        self.wd_resident = WD_RESIDENT
        self.depth_wd = self._choose_depth_wd()
        # The two-group schedule cuts each h multicast from eight rounds over 88 cores to four
        # rounds over 44 cores.  If its doubled W_down shard misses L1, first spend one h slot:
        # 2 * HID_T remains a legal producer/consumer window for the shorter grouped schedule and
        # saves 64 BFP8 tiles here.  Ordinary programs keep the measured DEPTH_H=3 configuration.
        if (
            self.wd_mgroups
            and self.depth_h > 2
            and self.l1_bytes(True, self.out_tile, enable_phase_alias=self.enable_phase_alias) > self.l1_budget
        ):
            self.depth_h = 2
            self.hack_ahead = max(1, min(HACK_AHEAD, self.depth_h - 1))
        # The grouped layout is optional and is the first feature dropped when its duplicated
        # W_down shard does not fit. Recompute the ordinary width before testing the base mrow
        # schedule or residency.
        if (
            self.wd_mgroups
            and self.l1_bytes(True, self.out_tile, enable_phase_alias=self.enable_phase_alias) > self.l1_budget
        ):
            self.wd_mgroups = False
            self.wd_ec_max = self.ec_max
            self.depth_h = DEPTH_H
            self.hack_ahead = max(1, min(HACK_AHEAD, self.depth_h - 1))
        # The eight-row schedule needs both the complete resident W_down shard and its larger H
        # row buffers. If that combination does not fit, use the ordinary resident layout before
        # sacrificing residency for every M-block.
        if (
            self.wd_mrow_rounds
            and self.l1_bytes(True, self.out_tile, enable_phase_alias=self.enable_phase_alias) > self.l1_budget
        ):
            self.wd_mrow_rounds = False
        if self.l1_bytes(True, self.out_tile, enable_phase_alias=self.enable_phase_alias) > self.l1_budget:
            self.wd_resident = False  # residency is what pins the depth to hgroups
            self.depth_wd = self._min_depth_wd()
            while self.l1_bytes(True, self.out_tile, enable_phase_alias=self.enable_phase_alias) > self.l1_budget:
                nxt = self._next_smaller_depth_wd(self.depth_wd)
                if nxt == self.depth_wd:
                    break
                self.depth_wd = nxt
        # Balanced hidden blocks are packed contiguously only for the resident payload. That is
        # what lets the full-M row schedule retain its one K=HID_T matmul despite fixed-width CB
        # flow-control slots. A non-resident stream cannot keep absolute packed offsets live while
        # cycling a shallower FIFO, so it stays on the ordinary padded-block schedule.
        self.wd_packed = bool(self.balanced_hn and self.wd_resident)
        if self.balanced_hn and not self.wd_packed:
            self.wd_mrow_rounds = False
        if not (self.wd_mrow_rounds and self.wd_resident):
            self.wd_mgroups = False
            self.wd_ec_max = self.ec_max
        # A BF16 row-major prefetch needs only cb_x_in: one injector row per core is read during
        # block b, then tilized after block b's compute releases the sole resident-x slot.  The
        # old path reserved block b+1's complete tiled slot before phase 2; at depth 1 that reserve
        # formed a cycle with compute and hung exactly at the second M-block.  The reader now
        # delays that reserve, so large-hidden programs may reclaim the otherwise idle second slot.
        # Keep two slots whenever they fit (and for tiled input, whose prefetch lands directly in
        # cb_x_tiles); this is a pressure fallback, not a new default schedule.
        if (
            self.x_is_rm
            and self.depth_x > 1
            and self.l1_bytes(self.x_is_rm, self.out_tile, enable_phase_alias=self.enable_phase_alias) > self.l1_budget
        ):
            self.depth_x = 1
        # Two h slots are the smallest useful producer/consumer window.  Spend the measured third
        # slot only when it fits; N=3072 on 11x8 needs this last fallback after reclaiming x.
        if (
            self.depth_h > 2
            and self.l1_bytes(self.x_is_rm, self.out_tile, enable_phase_alias=self.enable_phase_alias) > self.l1_budget
        ):
            self.depth_h = 2
            self.hack_ahead = max(1, min(HACK_AHEAD, self.depth_h - 1))
        # The W_down NoC split needs BOTH: `depth_wd == hgroups` for the writer's address
        # derivation (K-block r at a fixed slot), and RESIDENCY, which is what confines every
        # W_down DRAM read to b == 0 where all slots are free from kernel start. Without residency
        # the writer would write slots that are live on b > 0 — a race, not a slowdown.
        self.wd_split = max(0, min(8, WD_SPLIT)) if (self.wd_resident and self.depth_wd == hgroups) else 0
        if self.wd_split and hgroups > NOC_MAX_TRANSACTION_ID:
            # Block r takes transaction id r+1, so above this the ids ALIAS and a block would be
            # published while its bytes are still in flight. Drop the split rather than the
            # correctness; it is a perf lever, not a requirement.
            self.wd_split = 0

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
        for chunks in sorted(range(1, hn_pad + 1), key=lambda c: (abs(c - self.gu_chunks_target), c)):
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
                return hn_pad, ok[0], ok[1], False
            reasons.append(f"  hn_pad {hn_pad}: {why}")
        # Some useful grids cannot express a uniform-start split even though a balanced split is
        # straightforward. Hidden=64 over 12 columns is the motivating case: six-wide uniform
        # slots would leave column 11 empty, while split() gives 6/6/6/6/5/.../5. Gate/up already
        # receives per-core (hstart, hn), and phase 2 keeps the fixed HN_PAD page stride while
        # narrowing every block's FMA steps to its real balanced width.
        if self.hgroups <= self.hid_t:
            hn_pad = floor
            for chunks in sorted(range(1, hn_pad + 1), key=lambda c: (abs(c - self.gu_chunks_target), c)):
                if hn_pad % chunks:
                    continue
                if OUT_SUBBLOCK_H_GU * (hn_pad // chunks) > DEST_AUTO_LIMIT_TILES:
                    continue
                plan, why = scatter_plan(M_BLOCK, self.m_eff_min, hn_pad, self.kgroups)
                if plan is not None:
                    return hn_pad, chunks, plan, True
                reasons.append(f"  balanced hn_pad {hn_pad}: {why}")
                break
        # Actionable, not just a refusal: report the column counts that WOULD work here.
        workable = []
        for h in range(2, self.hgroups + 1):
            candidate = Blocking.__new__(Blocking)
            candidate.hgroups, candidate.kgroups = h, self.kgroups
            candidate.hid_t, candidate.hidden, candidate.m_eff_min = self.hid_t, self.hidden, self.m_eff_min
            f = (self.hid_t + h - 1) // h
            if any(candidate._hn_pad_legal(v)[0] for v in range(f, f + DEST_AUTO_LIMIT_TILES + 2)):
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

        THE one definition of the LOGICAL views. `cb_allocations` groups phase-disjoint views into
        physical allocations, and both the descriptor and `l1_bytes` consume those groups. Thus the
        residency decision is still priced on exactly the bytes that get allocated — the two used
        to be separate arithmetic, and the copy undercounted by ~95 KB, which turned a clean "does
        not fit" into an allocator throw at program build.
        """
        b8, b16, w, out = self.bfp8_tile, self.bf16_tile, self.w_tile, (out_tile or self.bfp8_tile)
        gu = M_BLOCK * self.hn_pad
        h_fast = self.hid_t if self.wd_mrow_rounds else gu
        outb = max(M_BLOCK * self.ec_max, self.mgroup_rows * self.ec_group_max if self.wd_mgroups else 0)
        # Full blocks bypass the BF16 partial spill on the mrow path. The ordinary path can reach
        # at most M_BLOCK/2 rows, so do not reserve an unused full-M intermediate; this funds the
        # wider resident W_down shard without shrinking cb_h's legal divisor lattice.
        out_interm = (M_BLOCK // 2 if self.wd_mrow_rounds else M_BLOCK) * self.ec_max
        return [
            (CB_X_IN, XSTICK_ROWS * TILE if x_is_rm else 1, self.x_stick, "x_in"),
            (CB_X_TILES, self.depth_x * M_BLOCK * self.kr_pad, b8, "bfp8"),
            (CB_X_STAGE, 1, 64, "u32"),
            (CB_MAILBOX_WRITER, 1, 64, "u32"),
            (CB_MAILBOX_COMPUTE, 1, 64, "u32"),
            (CB_W_GATE, self.depth_w * self.kr_pad * self.hn_pad, w, "weight"),
            (CB_W_UP, self.depth_w * self.kr_pad * self.hn_pad, w, "weight"),
            (CB_W_DOWN, self.depth_wd * self.hn_pad * self.wd_ec_max, w, "weight"),
            (CB_H, self.depth_h * h_fast, b8, "bfp8"),
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
            (CB_H_LOCAL, max(gu, h_fast), b8, "bfp8"),
            (CB_OUT_INTERM, out_interm, b16, "bf16"),
        ]

    def phase_cb_alias(self, out_tile: int = 0) -> bool:
        """Whether all three phase views can share one physical allocation.

        The output view is caller-selectable, while gather and h-slice are always BFP8. Equal tile
        bytes is the geometry-side predicate; the descriptor additionally requires the actual
        output dtype to be bfloat8_b before constructing a multi-index CBDescriptor.
        """
        if (out_tile or self.bfp8_tile) != self.bfp8_tile:
            return False
        by_index = {entry[0]: entry for entry in self.cb_layout(True, out_tile)}
        entries = [by_index[index] for index in PHASE_CB_ALIAS]
        shared_pages = self.phase_cb_alias_pages(out_tile)
        return shared_pages < sum(entry[1] for entry in entries)

    def phase_cb_alias_pages(self, out_tile: int = 0) -> int:
        """Physical page capacity of the BFP8 phase alias, before its profitability check."""
        by_index = {entry[0]: entry for entry in self.cb_layout(True, out_tile)}
        shared_pages = 1
        for index in PHASE_CB_ALIAS:
            shared_pages = lcm(shared_pages, by_index[index][1])
        return shared_pages

    def cb_allocations(
        self,
        x_is_rm: bool,
        out_tile: int = 0,
        idx_page: int = 64,
        counts_page: int = 64,
        *,
        enable_phase_alias: bool = True,
    ):
        """(physical_bytes, logical_views) for every physical CB allocation.

        Each logical view is one tuple from `cb_layout`. Aliased views get independent FIFO state
        but the same base and capacity. The shared page count is therefore the LCM of the logical
        capacities: every whole-buffer cycle returns each view to a legal boundary, even when a
        future geometry stops making one capacity divide another.
        """
        layout = self.cb_layout(x_is_rm, out_tile, idx_page, counts_page)
        by_index = {entry[0]: entry for entry in layout}
        aliases = [MAILBOX_CB_ALIAS]
        if enable_phase_alias:
            if self.phase_cb_alias(out_tile):
                aliases.append(PHASE_CB_ALIAS)
            # Phase-disjointness is necessary but not sufficient: for capacities such as 18 and
            # 24 (hidden=3072), the 72-page LCM is larger than two separate allocations.  Never let
            # an alias consume more L1 than the views it replaces.
            bf16_entries = tuple(by_index[index] for index in PHASE_BF16_ALIAS)
            bf16_alias_pages = 1
            for _, capacity, _, _ in bf16_entries:
                bf16_alias_pages = lcm(bf16_alias_pages, capacity)
            if bf16_alias_pages < sum(entry[1] for entry in bf16_entries):
                aliases.append(PHASE_BF16_ALIAS)

        alias_by_index = {}
        alias_allocations = {}
        for alias in aliases:
            entries = tuple(by_index[index] for index in alias)
            page_bytes = entries[0][2]
            if any(entry[2] != page_bytes for entry in entries):
                raise ValueError(f"moe_fused_swiglu: aliased CB views have unequal page sizes: {entries}")
            pages = 1
            for _, capacity, _, _ in entries:
                pages = lcm(pages, capacity)
            alias_allocations[alias[0]] = (pages * page_bytes, entries)
            for index in alias:
                if index in alias_by_index:
                    raise ValueError(f"moe_fused_swiglu: CB {index} occurs in more than one alias group")
                alias_by_index[index] = alias[0]

        allocations = []
        for entry in layout:
            index, pages, page, _ = entry
            if index in alias_allocations:
                allocations.append(alias_allocations[index])
            elif index not in alias_by_index:
                allocations.append((pages * page, (entry,)))
        return allocations

    def l1_bytes(self, x_is_rm: bool, out_tile: int = 0, *, enable_phase_alias: bool = True) -> int:
        return sum(
            physical_bytes
            for physical_bytes, _ in self.cb_allocations(x_is_rm, out_tile, enable_phase_alias=enable_phase_alias)
        )

    def describe(self) -> str:
        return (
            f"{self.hgroups}x{self.kgroups} grid, emb {self.emb}, hidden {self.hidden}: "
            f"kr_pad {self.kr_pad}, hn_pad {self.hn_pad} (floor "
            f"{(self.hid_t + self.hgroups - 1) // self.hgroups}), gu_chunks {self.gu_chunks}, "
            f"ec_max {self.ec_max}, depth_wd {self.depth_wd}, wd_split {self.wd_split}, "
            f"wd_mrow {self.wd_mrow_rounds and self.wd_resident}, wd_mgroups {self.wd_mgroups}, "
            f"wd_ec_max {self.wd_ec_max}"
        )
