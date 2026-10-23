# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""distributed_rms_norm — host plan + MeshProgramDescriptor (regime R1 resident-flat, op_design.md).

Per chip, one program on two core roles:

  workers  (every grid core except the port cores; identical list on every chip). Worker k owns row-group
           rg = k // num_w_chunks and W-chunk wc = k % num_w_chunks: tile rows [rg*block_h, +block_h) x tile
           columns [wc*block_w, +block_w) of the device's x, resident in L1.
           reader  (RISCV_1, NoC0): x block + gamma slivers + scaler; leader: board gather, then inv_rms push.
           compute: sum of squares -> partial row vector; x*gamma (stall shadow); leader: combine -> inv_rms;
                    normalize.
           writer  (RISCV_0, NoC1): partial half-rows -> both ports of the link; leader: inv fan-out;
                    output tiles -> DRAM.
  ports    (2 per link: fwd toward p+1, bwd toward p-1). A persistent L1 "board" with G slots (one per group
           position) at the same address on every chip. Sender (RISCV_0, NoC1): ready fence, own slot once
           the local workers have written it, relays as they arrive (Fabric 2D unicast, fused write+inc of the
           peer port's arrival counter on each slot's last packet), then tells the link's leaders the board is
           complete and re-arms its counter.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# ---------------------------------------------------------------------------------------------------------------------
# Knobs (single source of truth; every dependent quantity derives from these + the solve)
# ---------------------------------------------------------------------------------------------------------------------
COMBINE_AHEAD = False  # leader combines round r+1 before normalizing round r (see combine_ahead)
MIN_BLOCK_W = 4  # smallest W-chunk (tiles) the solve may choose
DRAM_STAGGER = True  # tile row t's DRAM reads / stores start at column t % w_act (bank spread across cores)
STORE_FLUSH = False  # writer releases cb_out pages after a flush (data left L1), not a DRAM-ack barrier
OUT_DEPTH = 2  # cb_out_tiles depth in block rows (clamped to the core's rows: block_h * num_rounds)
HEADER_RING = 4  # Fabric packet headers in flight per port before a flush
BOARD_CAP_BYTES = 256 * 1024  # board scratch per port core
MIN_ROUNDS = 1  # exchange rounds (row-blocks) per core the solve must use at least; 1 = one block per core whenever
# that fits L1 (the default). A row-block of block_h tile rows is one exchange round; > 1 forces the multi-round
# schedule (the overlap lever: round r+1's exchange runs under round r's normalize/store on the gamma path).
AUTO_ROUNDS = True  # gamma path: split a core's work into several exchange rounds when that keeps its total tiles
# minimal (round r+1's load/sumsq/exchange then overlaps round r's normalize/store, and round r's store overlaps the
# later rounds' reads). Measured on the 1x4 ring (last-launched chip): 640x1792 21.6 -> 19.7 us (2 rounds of 1x6),
# 2048x1280 47 -> 38 us (5 rounds of 1x5). Without gamma the rounds run back to back (no overlap), so it stays off.
MIN_ROUND_TILES = 5  # smallest per-round block (tiles per core) AUTO_ROUNDS may pick: each round pays the exchange
# chain (gather, combine, fan-out) on the critical path; 4-tile rounds measured slower (640x1792: 3 rounds 26 us)
TILE_SLACK = 0.25  # AUTO_ROUNDS: plans whose per-core tiles are within this fraction of the fewest compete on exchange
# shape (one gather tile per row, then more rounds) instead of losing to the strict tile minimum. Measured 640x1792:
# 8 chunks x 7 tiles (14 tiles/core, G*C = 32: one gather tile) 18.1 us vs 10 x 6 (12 tiles, two gather tiles) 18.9 us.
PRE_COMBINE = None  # None = auto: the cross-chunk sum runs on the row-group leader before the exchange (one vector
# per row on the board instead of num_w_chunks) only when the per-chunk board would exceed BOARD_CAP_BYTES (WIDTH
# shards: num_w_chunks = shards, up to 86). True / False force it (tests).
GAMMA_STREAM = None  # None = auto: gamma stays resident with the fp32 x*gamma stall shadow whenever that fits L1;
# otherwise gamma slivers are streamed per tile row and normalize is one fused x*gamma*inv pass (wide HEIGHT shards).
GAMMA_MCAST = False  # regime R4: on the resident-gamma interleaved path with >= 2 row-slots, one worker per W-chunk
# column reads its chunk's gamma slivers into cb_gamma and multicasts the cb_gamma pages (mcast_pipe) down its column.
# Placement then is 2-D: worker (rg, wc) on grid core (1 + wc, rg), the ports in grid column 0, so each W-chunk column is
# one physical grid column (one rectangle) and each row-group one grid row. Applies when the plan fits that box;
# otherwise (and with False) every worker reads its own chunk's slivers, workers row-major after the row-0 ports.
GAMMA_BATCH = 8  # streamed gamma tiles per reader batch (the solve halves it when L1 is tight); cb_gamma holds two
L1_BUDGET = int((1536 - 256) * 1024 * 0.9)  # usable worker L1 for CBs (bytes): conservative host constant, not queried
# (hal max-unreserved is a static ceiling, not this device's allocator base, so it would over-promise)
VEC_BYTES = 128  # one 32-float fp32 partial / inv vector (two 64 B half-rows)
FP32_TILE = 4096
BF16_TILE = 2048

# CB indices (semantic names)
CB_X_BLOCK = 0
CB_GAMMA = 1
CB_SCALER = 2
CB_SUMSQ = 3  # alias group A
CB_SUMSQ_T = 4  # alias group A
CB_INV_ROW_OUT = 5  # alias group A
CB_PARTIAL_ROW = 6
CB_GATHER = 7
CB_INV_ROW = 8  # alias group B
CB_INV_COL = 9  # alias group B
CB_XG = 10
CB_OUT_TILES = 11
CB_PREGATHER = 12  # pre-combine: the leader's gather of its row-group's num_w_chunks partials (2 round halves)
CB_INV_FULL = 13  # streamed-gamma path: inv_rms broadcast over the tile's columns (alias group B)

# program semaphores
SEM_PORT_LOCAL = 0  # even rounds (multi-source counters are kept per round parity: a worker runs at most one
SEM_BOARD_READY = 1  # round ahead of its link's slowest, since publish(r+2) needs inv(r), so parity separates rounds)
SEM_INV_READY = 2
SEM_PORT_LOCAL_ODD = 3
SEM_BOARD_READY_ODD = 4
SEM_PRE_READY = 5  # pre-combine: the leader's count of chunk partials landed in cb_pregather (per round parity)
SEM_PRE_READY_ODD = 6
SEM_GATHER_DONE = 7  # ports: count of the link's leaders that have read the whole board this call
SEM_RX_FREE = 8  # ports: the opposite port's board is free (read + re-armed): return the peer's ready credit
SEM_INV_SIG = 9  # leader: scratch cell holding r + 1, multicast-SET into the members' SEM_INV_READY after the data

NOC0 = ttnn.NOC.RISCV_0_default
NOC1 = ttnn.NOC.RISCV_1_default


# ---------------------------------------------------------------------------------------------------------------------
# Groups, topology, links
# ---------------------------------------------------------------------------------------------------------------------


# reader's W-mask element width per x dtype (bfloat8_b: one mantissa byte per element behind a 64 B exponent header)
X_ELEM_BYTES = {ttnn.float32: 4, ttnn.bfloat16: 2, ttnn.bfloat8_b: 1}


def build_groups(mesh_shape, cluster_axis):
    """Ordered lists of mesh coords (group position order), one per group."""
    rows, cols = mesh_shape
    if cluster_axis == 0:
        return [[(r, c) for r in range(rows)] for c in range(cols)]
    return [[(r, c) for c in range(cols)] for r in range(rows)]


def _node(mesh_device, coord):
    return mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))


def _hops(G, ring):
    hops = [(p, p + 1) for p in range(G - 1)]
    if ring:
        hops.append((G - 1, 0))
    return hops


_LINK_CACHE = {}


def discovered_links(mesh_device, cluster_axis, ring):
    """Minimum over every hop (both directions) of every group of the usable forwarding links."""
    key = (id(mesh_device), ttnn.get_fabric_config(), cluster_axis, ring)
    if key in _LINK_CACHE:
        return _LINK_CACHE[key][1]
    groups = build_groups(tuple(mesh_device.shape), cluster_axis)
    G = len(groups[0])
    ring = ring and G > 2
    best = None
    for grp in groups:
        for a, b in _hops(G, ring):
            for s, d in ((grp[a], grp[b]), (grp[b], grp[a])):
                n = len(ttnn.get_forwarding_link_indices(_node(mesh_device, s), _node(mesh_device, d)))
                if n == 0:
                    raise ValueError(
                        f"distributed_rms_norm: no direct link {s} -> {d} along cluster_axis={cluster_axis} "
                        f"({'Ring' if ring else 'Linear'})"
                    )
                best = n if best is None else min(best, n)
    _LINK_CACHE[key] = (mesh_device, best)
    return best


def _schedule(p, G, ring):
    """(positions sent toward p+1, positions sent toward p-1), own first, in arrival order downstream."""
    if ring:
        kf = G // 2
        kb = G - 1 - kf
        return [(p - i) % G for i in range(kf)], [(p + i) % G for i in range(kb)]
    fwd = [p - i for i in range(p + 1)] if p < G - 1 else []
    bwd = [p + i for i in range(G - p)] if p > 0 else []
    return fwd, bwd


def plan_chips(mesh_device, cluster_axis, ring, num_links):
    groups = build_groups(tuple(mesh_device.shape), cluster_axis)
    G = len(groups[0])
    ring = ring and G > 2  # a 2-chip ring is the line
    chips = {}
    for grp in groups:
        for p, coord in enumerate(grp):
            nxt = grp[(p + 1) % G] if (ring or p < G - 1) else None
            prv = grp[(p - 1) % G] if (ring or p > 0) else None
            fwd, bwd = _schedule(p, G, ring)
            chips[coord] = dict(p=p, G=G, next=nxt, prev=prv, fwd=fwd, bwd=bwd)
    for coord, ch in chips.items():
        for d, peer in (("fwd", ch["next"]), ("bwd", ch["prev"])):
            if peer is None:
                ch[f"{d}_links"] = []
                continue
            links = ttnn.get_forwarding_link_indices(_node(mesh_device, coord), _node(mesh_device, peer))
            if len(links) < num_links:
                raise ValueError(
                    f"distributed_rms_norm: {coord} -> {peer} has {len(links)} link(s), {num_links} requested"
                )
            ch[f"{d}_links"] = list(links[:num_links])
    return chips, G, ring


_OPP = {"fwd": "bwd", "bwd": "fwd"}


def _needs_ready(chips, coord, d):
    """Port (d) of `coord` sends a ready to its peer when the peer's opposite-direction port sends into `coord`."""
    ch = chips[coord]
    peer = ch["next"] if d == "fwd" else ch["prev"]
    return peer is not None and bool(chips[peer][_OPP[d]])


# ---------------------------------------------------------------------------------------------------------------------
# Work-split solve (plan_blocks, op_design.md "Work Distribution")
# ---------------------------------------------------------------------------------------------------------------------


def inv_depth(num_rounds, stall_shadow):
    """cb_inv_row / cb_inv_col depth in rounds. On the gamma path round r+1's sumsq (and so its exchange and the
    leader's inv fan-out) runs before round r's normalize has consumed cb_inv_col, so the fan-out target alternates
    between two halves; one round per call (or the no-gamma order, where sumsq(r+1) follows normalize(r)) needs one."""
    return 2 if (num_rounds > 1 and stall_shadow) else 1


def combine_ahead(num_rounds, stall_shadow, pre_combine):
    """The leader combines round r+1 before normalizing round r (stall-shadow order, several rounds): round r+1's
    inv_rms fan-out then never waits on round r's inv arrival + normalize. cb_inv_row_out then holds two rounds in its
    own buffer: in alias group A, round r+2's sumsq would overwrite round r+1's inv before the writer fanned it out.
    Not with the pre-combine (its leader-side reduce of round r+1 follows the old order)."""
    return COMBINE_AHEAD and num_rounds > 1 and stall_shadow and not pre_combine


def out_rows(block_h, num_rounds):
    """cb_out_tiles depth in block rows: OUT_DEPTH, but never more rows than the core stores. Across rounds too: with
    one tile row per round (the auto-rounds plans) a one-row FIFO would make round r+1's normalize wait for round r's
    store to drain."""
    return min(OUT_DEPTH, block_h * num_rounds)


def gamma_pages(block_w, gamma_stream, gamma_batch=GAMMA_BATCH):
    """cb_gamma pages: the whole chunk (resident), or two batches of streamed slivers."""
    return 2 * min(gamma_batch, block_w) if gamma_stream else block_w


def worker_bytes(
    block_h,
    block_w,
    x_tile,
    g_tile,
    has_gamma,
    n_gather,
    num_rounds=1,
    *,
    x_in_cb=True,
    out_in_cb=True,
    n_pre=0,
    gamma_stream=False,
    gamma_batch=GAMMA_BATCH,
):
    """CB bytes per worker. A sharded x / out CB lives in the tensor's own L1 (x_in_cb / out_in_cb = False): the
    caller charges the shard against the budget instead."""
    stall = has_gamma and not gamma_stream
    b = block_h * block_w * x_tile if x_in_cb else 0  # cb_x_block
    b += gamma_pages(block_w, gamma_stream, gamma_batch) * g_tile if has_gamma else 0  # cb_gamma
    b += BF16_TILE  # cb_scaler
    b += block_h * FP32_TILE  # alias group A
    b += 2 * block_h * FP32_TILE if combine_ahead(num_rounds, stall, n_pre > 0) else 0  # own cb_inv_row_out
    b += block_h * FP32_TILE  # cb_partial_row
    b += n_gather * block_h * FP32_TILE  # cb_gather
    b += 2 * n_pre * block_h * FP32_TILE  # cb_pregather (two round halves)
    b += inv_depth(num_rounds, stall) * block_h * FP32_TILE  # alias group B
    b += block_h * block_w * FP32_TILE if stall else 0  # cb_xg
    b += out_rows(block_h, num_rounds) * block_w * x_tile if out_in_cb else 0  # cb_out_tiles
    return b


def board_slot_bytes(num_slots, block_h, chunks, num_links):
    """One round's slot: every row-slot of link l (block_h tile rows each) x num_w_chunks partial vectors."""
    slots_per_link = -(-num_slots // num_links)
    return slots_per_link * block_h * chunks * VEC_BYTES


def _candidate(th, tw, block_h, chunks, num_workers, G, num_links, x_tile, g_tile, has_gamma, min_rounds=1):
    """(key, plan) for one (block_h, num_w_chunks) choice, or None if it does not fit the board or L1.

    nrg row-blocks of block_h tile rows are dealt over `slots` row-slots of num_w_chunks workers each:
    row-block b -> slot b % slots, round b // slots (the ragged last row-block is its slot's last round)."""
    block_w = -(-tw // chunks)
    chunks = -(-tw // block_w)
    nrg = -(-th // block_h)
    slots = min(nrg, num_workers // chunks, -(-nrg // min_rounds))
    if slots < 1 or -(-nrg // slots) < min_rounds:
        return None
    rounds = -(-nrg // slots)
    slots = -(-nrg // rounds)  # balanced: the fewest row-slots that still take `rounds` rounds (no under-filled round)
    links = min(num_links, slots)
    if G * rounds * board_slot_bytes(slots, block_h, chunks, links) > BOARD_CAP_BYTES:
        return None
    n_gather = -(-(G * chunks) // 32)
    if worker_bytes(block_h, block_w, x_tile, g_tile, has_gamma, n_gather, rounds) > L1_BUDGET:
        return None
    plan = dict(
        block_h=block_h, block_w=block_w, num_w_chunks=chunks, num_row_groups=nrg, num_slots=slots, num_rounds=rounds
    )
    return plan


def plan_rounds(th, tw, num_workers, G, num_links, x_tile, g_tile, has_gamma):
    """Gamma path (stall shadow). Among the plans whose per-core tiles are within TILE_SLACK of the fewest: per-round
    blocks >= MIN_ROUND_TILES first (smaller rounds cannot hide the exchange chain), then the fewest gather tiles per
    row (ceil(G * num_w_chunks / 32): the leader's zero / board read / combine per round, exposed in the last round),
    then several rounds over one (round r+1's exchange overlaps round r's normalize/store), then the fewest tiles, then
    the most rounds (the fewest for small rounds). None if nothing fits (the caller falls back to the plain solve)."""
    cands = []
    for block_h in range(1, th + 1):
        for chunks in range(1, max(1, tw // MIN_BLOCK_W) + 1):
            c = _candidate(th, tw, block_h, chunks, num_workers, G, num_links, x_tile, g_tile, has_gamma, MIN_ROUNDS)
            if c is not None:
                cands.append(c)
    if not cands:
        return None
    work = lambda c: c["num_rounds"] * c["block_h"] * c["block_w"]
    limit = min(work(c) for c in cands) * (1 + TILE_SLACK)

    def key(c):
        R, tiles = c["num_rounds"], c["block_h"] * c["block_w"]
        big = tiles >= MIN_ROUND_TILES or R == 1
        n_gather = -(-(G * c["num_w_chunks"]) // 32)
        overlap = 0 if (R > 1 and big) else 1  # any exchange/compute overlap at all
        return (0 if big else 1, n_gather, overlap, work(c), -R if big else R, c["num_w_chunks"], c["block_h"])

    return min((c for c in cands if work(c) <= limit), key=key)


def plan_blocks(th, tw, num_workers, G, num_links, x_tile, g_tile, has_gamma):
    if AUTO_ROUNDS and has_gamma:
        plan = plan_rounds(th, tw, num_workers, G, num_links, x_tile, g_tile, has_gamma)
        if plan is not None:
            return plan
    # 1. one block per core (one exchange round): the Phase 0 solve, unchanged, whenever any candidate fits
    best = None
    for block_h in range(1, th + 1 if MIN_ROUNDS == 1 else 1):
        nrg = -(-th // block_h)
        if nrg > num_workers:
            continue
        links = min(num_links, nrg)
        chunks = max(1, min(num_workers // nrg, max(1, tw // MIN_BLOCK_W)))
        block_w = -(-tw // chunks)
        chunks = -(-tw // block_w)
        while chunks > 1 and G * board_slot_bytes(nrg, block_h, chunks, links) > BOARD_CAP_BYTES:
            chunks -= 1
            block_w = -(-tw // chunks)
            chunks = -(-tw // block_w)
        if G * board_slot_bytes(nrg, block_h, chunks, links) > BOARD_CAP_BYTES:
            continue
        n_gather = -(-(G * chunks) // 32)
        if worker_bytes(block_h, block_w, x_tile, g_tile, has_gamma, n_gather) > L1_BUDGET:
            continue
        key = (block_h * block_w, chunks, block_h)
        if best is None or key < best[0]:
            plan = dict(
                block_h=block_h, block_w=block_w, num_w_chunks=chunks, num_row_groups=nrg, num_slots=nrg, num_rounds=1
            )
            best = (key, plan)
    if best is not None:
        return best[1]
    # 2. several row-blocks per core (one exchange round per block): least per-core work, then fewest rounds
    for block_h in range(1, th + 1):
        for chunks in range(1, max(1, tw // MIN_BLOCK_W) + 1):
            c = _candidate(th, tw, block_h, chunks, num_workers, G, num_links, x_tile, g_tile, has_gamma, MIN_ROUNDS)
            if c is None:
                continue
            key = (c["num_rounds"] * block_h * c["block_w"], c["num_rounds"], c["num_w_chunks"], block_h)
            if best is None or key < best[0]:
                best = (key, c)
    if best is None:
        raise RuntimeError("distributed_rms_norm: working set exceeds L1 for every block (streaming regime deferred)")
    return best[1]


def _shard_cores(input_tensor):
    ss = input_tensor.memory_config().shard_spec
    return ttnn.corerange_to_cores(ss.grid, None, ss.orientation == ttnn.ShardOrientation.ROW_MAJOR)


L1_CB_MARGIN = 4 * 1024  # slack kept between the CB region and the lowest L1 buffer (sharded solve)


def free_cb_bytes(mesh_device, exclude_addr=None):
    """Bytes from the L1 allocator base to the lowest allocated L1 buffer: the room the CB region really has.
    Sharded tensors (the x / out shards themselves, the board, any live L1 tensor) are allocated top-down, so the
    static CBs must end below all of them (the runtime refuses a program whose CB region clashes).
    exclude_addr: an absolute L1 address whose block is ignored (the board, when the call may re-size it)."""
    mv = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1)
    base = int(ttnn.get_allocator_base_address(mesh_device, ttnn.BufferType.L1))
    lows = [
        int(b["address"])
        for b in mv.block_table
        if b.get("allocated") == "yes" and (exclude_addr is None or base + int(b["address"]) != exclude_addr)
    ]
    return min(lows) if lows else int(mv.total_bytes_per_bank)


def plan_sharded(input_tensor, th, tw, G, num_links, x_tile, g_tile, has_gamma, out_sharded, cb_budget):
    """Regimes R9 / R10: the shard spec fixes the core assignment and the per-core block extent; only block_h (rows
    per exchange round inside the shard), the pre-combine and the gamma regime are solved.

    HEIGHT: worker k = shard k holds tile rows [k*sh, (k+1)*sh) x every column (num_w_chunks = 1, one row-slot per
            shard, rounds walk the shard's rows contiguously).
    WIDTH : worker k = shard k holds every row x tile columns [k*sw, (k+1)*sw) (num_w_chunks = shards, one row-slot,
            rounds walk the rows). A padded last shard (Split.PAD) computes its whole width: the reader zero-fills its
            pad columns, so they add nothing to the sum of squares.
    cb_budget(board_need) -> (CB bytes available, board re-size needed): free L1 below every live buffer (x and out
    shards included) given the board this plan needs.
    Returns (plan, workers, geoms); plan["board_exact"] asks for the board to be re-sized to the plan's need."""
    mc = input_tensor.memory_config()
    ss = mc.shard_spec
    sh, sw = int(ss.shape[0]) // 32, int(ss.shape[1]) // 32
    cores = _shard_cores(input_tensor)
    if mc.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        if sw != tw:
            raise ValueError(f"distributed_rms_norm: HEIGHT shard width {sw} tiles != tensor width {tw} tiles")
        n = -(-th // sh)
        C, block_w, rows_total = 1, tw, sh
        geoms = [
            dict(rg=k, wc=0, row0=k * sh, row_end=min(th, (k + 1) * sh), col0=0, w_act=tw, w_valid=tw) for k in range(n)
        ]
    elif mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        if sh != th:
            raise ValueError(f"distributed_rms_norm: WIDTH shard height {sh} tiles != tensor height {th} tiles")
        n = -(-tw // sw)
        C, block_w, rows_total = n, sw, th
        geoms = [
            dict(rg=0, wc=k, row0=0, row_end=th, col0=k * sw, w_act=sw, w_valid=min(sw, tw - k * sw)) for k in range(n)
        ]
    else:
        raise ValueError(f"distributed_rms_norm: unsupported sharded memory layout {mc.memory_layout}")
    if n > len(cores):
        raise ValueError(f"distributed_rms_norm: shard grid has {len(cores)} cores, the tensor needs {n}")
    S = n // C  # row-slots
    L = min(num_links, S)
    pre_opts = [False, True] if PRE_COMBINE is None else [bool(PRE_COMBINE)]
    stream_opts = ([False, True] if GAMMA_STREAM is None else [bool(GAMMA_STREAM)]) if has_gamma else [False]
    max_h = -(-rows_total // MIN_ROUNDS)
    smallest = None
    for pre in pre_opts:
        cb_chunks = 1 if pre else C  # partial vectors per tile row on the board
        n_gather = -(-(G * cb_chunks) // 32)
        n_pre = -(-C // 32) if pre else 0
        modes = [(False, GAMMA_BATCH)] if False in stream_opts else []
        if True in stream_opts:
            gb = min(GAMMA_BATCH, block_w)
            while gb >= 1:
                modes.append((True, gb))
                gb //= 2
        for stream, gbatch in modes:
            for block_h in range(max_h, 0, -1):
                R = -(-rows_total // block_h)
                board_need = G * R * board_slot_bytes(S, block_h, cb_chunks, L)
                if board_need > BOARD_CAP_BYTES:
                    continue
                budget, board_exact = cb_budget(board_need)
                wb = worker_bytes(
                    block_h,
                    block_w,
                    x_tile,
                    g_tile,
                    has_gamma,
                    n_gather,
                    R,
                    x_in_cb=False,
                    out_in_cb=not out_sharded,
                    n_pre=n_pre,
                    gamma_stream=stream,
                    gamma_batch=gbatch,
                )
                smallest = wb if smallest is None else min(smallest, wb)
                if wb > budget:
                    continue
                plan = dict(
                    block_h=block_h,
                    block_w=block_w,
                    num_w_chunks=C,
                    num_row_groups=S,
                    num_slots=S,
                    num_rounds=R,
                    round_stride=block_h,
                    pre_combine=pre,
                    gamma_stream=stream,
                    gamma_batch=gbatch,
                    board_exact=board_exact,
                )
                return plan, cores[:n], geoms
    best_budget = cb_budget(0)[0]
    raise RuntimeError(
        "distributed_rms_norm: working set exceeds L1 next to the resident shards for every block (sharded): "
        f"CB room {best_budget} B, block_h=1 needs {smallest} B"
    )


# ---------------------------------------------------------------------------------------------------------------------
# Persistent resources (cached; created once before any call can be captured in a trace)
# ---------------------------------------------------------------------------------------------------------------------

_SEM_CACHE = {}
_BOARD_CACHE = {}


def _core_set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def _global_sems(mesh_device, port_cores):
    key = (id(mesh_device), tuple((c.x, c.y) for c in port_cores))
    if key not in _SEM_CACHE:
        # arrival counter (0 at rest) and ready credit (1 at rest: a peer may send into a board until it is consumed;
        # the board's owner returns it at the end of each call, once its leaders have read the board)
        sems = tuple(ttnn.create_global_semaphore(mesh_device, _core_set(port_cores), v) for v in (0, 1))
        # every chip's counters must be zero before any chip's first call can increment a neighbour's
        ttnn.synchronize_device(mesh_device)
        _SEM_CACHE[key] = (mesh_device, sems)
    return tuple(int(ttnn.get_global_semaphore_address(s)) for s in _SEM_CACHE[key][1])


def _board(mesh_device, port_cores, board_bytes, exact=False):
    # One persistent board per mesh. An L1 allocation reserves its address range on every core's bank, not only on
    # the port cores, so a second board (another port-core set: the sharded path's ports sit past the shard grid)
    # would cost every core up to BOARD_CAP_BYTES more. The board follows the current port set and grows on demand;
    # a port-set switch re-sizes it to the need. Keying on the size too would leak one board per size.
    words = max(4, -(-board_bytes // 16) * 4)  # 16 B multiple
    key = id(mesh_device)
    ports = tuple((c.x, c.y) for c in port_cores)
    cached = _BOARD_CACHE.get(key)
    if exact and cached is not None and cached[3] == ports and cached[2] != words:
        # re-size in place (the sharded solve needs the L1 a stale larger board holds): free first so the new board
        # takes the top of the freed range (after the host sync below: no call is in flight on either board).
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(cached[1])
        del _BOARD_CACHE[key]
        cached = None
    if cached is None or cached[3] != ports or cached[2] < words:
        # A peer sends into this chip's board on the credit returned at the end of the previous call, i.e. possibly
        # before this chip has finished the ops dispatched ahead of this call. That is safe only while the board's
        # range belongs to the board: a newly allocated range may still be in use by in-flight work (and a call in
        # flight on the old board must finish first), so every chip drains before the board moves. Host-side wait,
        # not a dispatch; it happens once per board size / port set.
        ttnn.synchronize_device(mesh_device)
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(_core_set(port_cores), (1, words), ttnn.ShardOrientation.ROW_MAJOR),
        )
        t = ttnn.allocate_tensor_on_device(
            ttnn.Shape([len(port_cores), words]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh_device, mem
        )
        # allocate the new board before freeing the old one, so the two never share an address: a call still in
        # flight on the old board cannot see writes aimed at the new one
        if cached is not None:
            ttnn.deallocate(cached[1])
        _BOARD_CACHE[key] = (mesh_device, t, words, ports)
    return _BOARD_CACHE[key][1]


def _board_bytes(mesh_device, port_cores):
    """(bytes, absolute L1 address) of the board a call on this port-core set can reuse ((0, None): none yet)."""
    cached = _BOARD_CACHE.get(id(mesh_device))
    if cached is None or cached[3] != tuple((c.x, c.y) for c in port_cores):
        return 0, None
    return 4 * cached[2], int(cached[1].buffer_address())


def _row_major_rects(cores):
    """Logical rectangles covering `cores` (a row-major run: full rows with ragged first / last rows), merging
    consecutive rows that span the same x range."""
    rows = {}
    for c in cores:
        rows.setdefault(c.y, []).append(c.x)
    rects = []
    for y in sorted(rows):
        xs = sorted(rows[y])
        if xs != list(range(xs[0], xs[-1] + 1)):
            return None
        if rects and rects[-1][0] == xs[0] and rects[-1][2] == xs[-1] and rects[-1][3] == y - 1:
            rects[-1][3] = y
        else:
            rects.append([xs[0], y, xs[-1], y])
    return rects


def _fanout_rects(mesh_device, members, leader):
    """inv_rms fan-out targets: logical rectangles covering exactly `members` (per-row contiguous x runs, consecutive
    rows with the same run merged), as flat (sx, sy, ex, ey, count, has_leader) in virtual coordinates with (sx, sy) =
    the high corner (NoC1 multicasts start there). Any core set decomposes this way; a 1-core rectangle is sent as an
    ordered unicast by the writer."""
    rows = {}
    for c in members:
        rows.setdefault(c.y, []).append(c.x)
    rects = []
    for y in sorted(rows):
        xs = sorted(rows[y])
        runs, a = [], xs[0]
        for i in range(1, len(xs) + 1):
            if i == len(xs) or xs[i] != xs[i - 1] + 1:
                runs.append((a, xs[i - 1]))
                if i < len(xs):
                    a = xs[i]
        for x0, x1 in runs:
            prev = next((r for r in rects if r[0] == x0 and r[2] == x1 and r[3] == y - 1), None)
            if prev is not None:
                prev[3] = y
            else:
                rects.append([x0, y, x1, y])
    out = []
    for x0, y0, x1, y1 in rects:
        lo = mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(x0, y0))
        hi = mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(x1, y1))
        incl = int(x0 <= leader.x <= x1 and y0 <= leader.y <= y1)
        out += [hi.x, hi.y, lo.x, lo.y, (x1 - x0 + 1) * (y1 - y0 + 1), incl]
    return out


def gamma_mcast_wanted(has_gamma, gamma_stream, num_slots):
    """R4 applies to the resident-gamma path (stall shadow) when a W-chunk column holds >= 2 row-slots."""
    return GAMMA_MCAST and has_gamma and not gamma_stream and num_slots >= 2


def column_placement(grid, S, C, num_links):
    """2-D placement for the gamma mcast: (port cores by key order, worker list k = rg * C + wc), or None when the
    plan does not fit the box (C columns right of the port column, S rows, 2 * num_links ports in column 0)."""
    if C > grid.x - 1 or S > grid.y or 2 * num_links > grid.y:
        return None
    ports = [ttnn.CoreCoord(0, i) for i in range(2 * num_links)]
    return ports, [ttnn.CoreCoord(1 + wc, rg) for rg in range(S) for wc in range(C)]


def gamma_mcast_family(mesh_device, used_workers, geoms, S, C):
    """McastFamily of the gamma broadcast: one group per W-chunk column (its S workers), sender = the column's
    middle row-slot. None if a column needs > 3 rectangles."""
    fam = ttnn.McastFamily(
        mesh_device, ttnn.McastConfig(noc=NOC0, handshake=False, data_ready=ttnn.McastDataReady.Counter)
    )
    for wc in range(C):
        col = [used_workers[rg * C + wc] for rg in range(S)]
        rects = _row_major_rects(col)
        if rects is None or len(rects) > 3:
            return None
        receivers = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(a, b), ttnn.CoreCoord(c, d)) for a, b, c, d in rects]
        )
        fam.add_group(receivers, [col[S // 2]])
    fam.prepare_arguments()
    return fam


def _f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def _xy(c):
    return (c.y << 16) | c.x


# ---------------------------------------------------------------------------------------------------------------------
# Descriptor
# ---------------------------------------------------------------------------------------------------------------------


def run_distributed_rms_norm(
    input_tensor, *, gamma, cluster_axis, epsilon, ring, num_links, compute_kernel_config, memory_config
):
    mesh_device = input_tensor.device()
    shape = list(input_tensor.shape)
    padded = list(input_tensor.padded_shape)
    th = math.prod(padded[:-2]) * (padded[-2] // 32)  # tensor_h_tiles
    tw = padded[-1] // 32  # tensor_w_tiles
    has_gamma = gamma is not None

    chips, G, ring = plan_chips(mesh_device, cluster_axis, ring, num_links)
    W_global = G * shape[-1]

    x_sharded = input_tensor.memory_config().is_sharded()
    out_sharded = memory_config.is_sharded()
    x_tile = int(input_tensor.tile.get_tile_size(input_tensor.dtype))
    g_tile = int(gamma.tile.get_tile_size(gamma.dtype)) if has_gamma else 0

    # ---- core roles ------------------------------------------------------------------------------------------------
    grid = mesh_device.compute_with_storage_grid_size()
    if grid.x * grid.y < 2 * num_links + 1:
        raise ValueError("distributed_rms_norm: core grid too small for the port cores")
    port_keys = [(d, l) for l in range(num_links) for d in ("fwd", "bwd")]
    if x_sharded:
        # workers = the shard cores (x is consumed where it lives); ports = the last 2*num_links cores of the grid
        # (row-major from the end), which a row-major shard grid from (0, 0) leaves free.
        tail = [ttnn.CoreCoord(x, y) for y in range(grid.y - 1, -1, -1) for x in range(grid.x - 1, -1, -1)]
        port_core = {k: tail[i] for i, k in enumerate(port_keys)}
        # the output (and its shard) exists before the solve, so the CB room it measures is the real one
        output = ttnn.allocate_tensor_on_device(
            ttnn.Shape(shape), input_tensor.dtype, ttnn.TILE_LAYOUT, mesh_device, memory_config
        )
        all_ports = [port_core[k] for k in port_keys]
        cur_board, cur_addr = _board_bytes(mesh_device, all_ports)
        room = free_cb_bytes(mesh_device)
        room_wo_board = free_cb_bytes(mesh_device, cur_addr) if cur_addr is not None else room

        def cb_budget(board_need):
            # keep the board (a larger one is allocated below everything else, so its growth is charged), or, when
            # that does not fit, re-size it to exactly this need (a stale large board from an earlier call would
            # otherwise hold L1 that the shards' CBs need)
            keep = room - L1_CB_MARGIN - (board_need if board_need > cur_board else 0)
            exact = room_wo_board - L1_CB_MARGIN - board_need
            return (keep, False) if keep >= exact else (exact, True)

        blk, used_workers, geoms = plan_sharded(
            input_tensor, th, tw, G, num_links, x_tile, g_tile, has_gamma, out_sharded, cb_budget
        )
        shard_xy = {(c.x, c.y) for c in _shard_cores(input_tensor)}
        if any((c.x, c.y) in shard_xy for c in port_core.values()):
            raise ValueError(
                "distributed_rms_norm: the shard grid leaves no free cores for the "
                f"{2 * num_links} port cores (they go on the last cores of the {grid.x}x{grid.y} grid)"
            )
    else:
        # 2 ports per link on logical row 0 ("simple" placement), workers = the rest; or (gamma mcast) the 2-D
        # column placement, same worker count
        if grid.x < 2 * num_links:
            raise ValueError("distributed_rms_norm: core grid too narrow for the port cores")
        blk = plan_blocks(th, tw, grid.x * grid.y - 2 * num_links, G, num_links, x_tile, g_tile, has_gamma)
        Ci, bw = blk["num_w_chunks"], blk["block_w"]
        placed = None
        if gamma_mcast_wanted(has_gamma, False, blk["num_slots"]):
            placed = column_placement(grid, blk["num_slots"], Ci, num_links)
        blk = dict(
            blk,
            round_stride=blk["num_slots"] * blk["block_h"],
            pre_combine=bool(PRE_COMBINE),
            gamma_stream=False,
            gamma_batch=GAMMA_BATCH,
            gamma_mcast=placed is not None,
        )
        if placed is not None:
            port_core = dict(zip(port_keys, placed[0]))
            used_workers = placed[1]
        else:
            port_core = {k: ttnn.CoreCoord(i, 0) for i, k in enumerate(port_keys)}
            port_xy = {(c.x, c.y) for c in port_core.values()}
            workers = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in port_xy]
            used_workers = workers[: blk["num_slots"] * Ci]
        geoms = []
        for k in range(len(used_workers)):
            rg, wc = divmod(k, Ci)
            w_act = min(bw, tw - wc * bw)
            geoms.append(
                dict(rg=rg, wc=wc, row0=rg * blk["block_h"], row_end=th, col0=wc * bw, w_act=w_act, w_valid=w_act)
            )
    block_h, block_w = blk["block_h"], blk["block_w"]
    C, S, R = blk["num_w_chunks"], blk["num_slots"], blk["num_rounds"]
    round_stride = blk["round_stride"]  # tile rows between a row-slot's consecutive rounds
    pre = blk["pre_combine"]
    gamma_stream = blk["gamma_stream"]
    gamma_batch = min(blk["gamma_batch"], block_w)
    stall = has_gamma and not gamma_stream  # x*gamma stall shadow (round r+1's exchange overlaps round r's normalize)
    C_board = 1 if pre else C  # partial vectors per tile row on the board
    n_gather = -(-(G * C_board) // 32)
    n_pre = -(-C // 32) if pre else 0
    L = min(num_links, S)  # links that carry at least one row-slot
    rg_per_link = -(-S // L)
    slot_bytes = board_slot_bytes(S, block_h, C_board, L)
    round_bytes = G * slot_bytes  # one round's board region; round r lives at board_addr + r * round_bytes
    inv_rounds = inv_depth(R, stall)
    lead_ahead = combine_ahead(R, stall, pre)
    used_ports = [(d, l) for (d, l) in port_keys if l < L]
    used_port_cores = [port_core[k] for k in used_ports]

    arrival_addr, ready_addr = _global_sems(mesh_device, [port_core[k] for k in port_keys])
    board = _board(mesh_device, [port_core[k] for k in port_keys], R * round_bytes, exact=blk.get("board_exact", False))
    board_addr = int(board.buffer_address())

    if not x_sharded:
        output = ttnn.allocate_tensor_on_device(
            ttnn.Shape(shape), input_tensor.dtype, ttnn.TILE_LAYOUT, mesh_device, memory_config
        )

    virt = lambda c: mesh_device.worker_core_from_logical_core(c)
    pvirt = {k: virt(port_core[k]) for k in used_ports}
    payload = int(ttnn.get_tt_fabric_max_payload_size_bytes())

    x_addr = int(input_tensor.buffer_address())
    g_addr = int(gamma.buffer_address()) if has_gamma else 0
    out_addr = int(output.buffer_address())
    x_page = int(input_tensor.buffer_aligned_page_size())
    g_page = int(gamma.buffer_aligned_page_size()) if has_gamma else 0
    g_face = g_tile // 4 if has_gamma else 0
    x_ct = list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    g_ct = list((ttnn.TensorAccessorArgs(gamma) if has_gamma else ttnn.TensorAccessorArgs()).get_compile_time_args())
    out_ct = list(ttnn.TensorAccessorArgs(output).get_compile_time_args())

    # per-worker geometry (identical on every chip): row-slot rg -> link + board row offset
    for g in geoms:
        g["link"] = g["rg"] // rg_per_link
        g["rt_base"] = (g["rg"] % rg_per_link) * block_h
    # row-group leader = its LAST row-major core: that is the high (NoC1-start) corner of its own fan-out rectangle,
    # where the inv_rms multicast is cheapest (a first-core leader routes every rectangle around the NoC1 ring; with
    # 10 concurrent leaders that measured slower than unicast, Perf 2 perf_experiments/inv_fanout_mcast)
    lead_wc = {}
    for g in geoms:
        if g["rg"] not in lead_wc:
            mem = [used_workers[g["rg"] * C + wc] for wc in range(C)]
            lead_wc[g["rg"]] = max(range(C), key=lambda wc: (mem[wc].y, mem[wc].x))
    is_lead = lambda g: g["wc"] == lead_wc[g["rg"]]
    leaders_of_link = {
        l: [used_workers[k] for k, g in enumerate(geoms) if is_lead(g) and g["link"] == l] for l in range(L)
    }
    # board publishers per link per round: every worker, or (pre-combine) the row-group leaders only
    workers_of_link = {l: sum(1 for g in geoms if g["link"] == l and (is_lead(g) or not pre)) for l in range(L)}
    leader_of_rg = {g["rg"]: used_workers[k] for k, g in enumerate(geoms) if is_lead(g)}

    # R4 gamma broadcast (interleaved x, column placement above)
    gmc = None
    if blk.get("gamma_mcast", False):
        gmc = gamma_mcast_family(mesh_device, used_workers, geoms, S, C)
    gamma_mcast = gmc is not None

    worker_set = _core_set(used_workers)
    all_set = _core_set(used_workers + used_port_cores)

    # ---- CBs (worker cores) -----------------------------------------------------------------------------------------
    fp32 = ttnn.float32
    fp32_dest = bool(compute_kernel_config.fp32_dest_acc_en)

    def cb(total, fmts):
        return ttnn.CBDescriptor(
            total_size=total,
            core_ranges=worker_set,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=i, data_format=f, page_size=p) for i, f, p in fmts
            ],
        )

    def sharded_cb(index, tensor):
        # zero-copy: the CB is the core's own shard (x consumed in place / out packed in place)
        return ttnn.cb_descriptor_from_sharded_tensor(index, tensor, core_ranges=worker_set)

    inv_group = [(CB_INV_ROW, fp32, FP32_TILE), (CB_INV_COL, fp32, FP32_TILE)]
    if gamma_stream:
        inv_group.append((CB_INV_FULL, fp32, FP32_TILE))
    cbs = [
        (
            sharded_cb(CB_X_BLOCK, input_tensor)
            if x_sharded
            else cb(block_h * block_w * x_tile, [(CB_X_BLOCK, input_tensor.dtype, x_tile)])
        ),
        cb(BF16_TILE, [(CB_SCALER, ttnn.bfloat16, BF16_TILE)]),
        cb(
            block_h * FP32_TILE,
            [(CB_SUMSQ, fp32, FP32_TILE), (CB_SUMSQ_T, fp32, FP32_TILE)]
            + ([] if lead_ahead else [(CB_INV_ROW_OUT, fp32, FP32_TILE)]),
        ),
        cb(block_h * FP32_TILE, [(CB_PARTIAL_ROW, fp32, FP32_TILE)]),
        cb(n_gather * block_h * FP32_TILE, [(CB_GATHER, fp32, FP32_TILE)]),
        cb(inv_rounds * block_h * FP32_TILE, inv_group),
        (
            sharded_cb(CB_OUT_TILES, output)
            if out_sharded
            else cb(out_rows(block_h, R) * block_w * x_tile, [(CB_OUT_TILES, input_tensor.dtype, x_tile)])
        ),
    ]
    if lead_ahead:
        cbs.append(cb(2 * block_h * FP32_TILE, [(CB_INV_ROW_OUT, fp32, FP32_TILE)]))
    if pre:
        cbs.append(cb(2 * n_pre * block_h * FP32_TILE, [(CB_PREGATHER, fp32, FP32_TILE)]))
    if has_gamma:
        cbs.append(cb(gamma_pages(block_w, gamma_stream, gamma_batch) * g_tile, [(CB_GAMMA, gamma.dtype, g_tile)]))
    if stall:
        # x*gamma is a DEST result: with a 16-bit DEST it is already rounded to bf16, so a Float32 page stores no
        # extra bits (ledger audit 3). The solve charges the fp32 page either way (upper bound, mode-independent plan).
        xg_fmt, xg_tile = (fp32, FP32_TILE) if fp32_dest else (ttnn.bfloat16, BF16_TILE)
        cbs.append(cb(block_h * block_w * xg_tile, [(CB_XG, xg_fmt, xg_tile)]))

    # ---- compute config --------------------------------------------------------------------------------------------
    ccfg = ttnn.ComputeConfigDescriptor()
    ccfg.math_fidelity = compute_kernel_config.math_fidelity
    ccfg.fp32_dest_acc_en = fp32_dest
    ccfg.math_approx_mode = bool(compute_kernel_config.math_approx_mode)
    modes = [ttnn.UnpackToDestMode.Default] * 64
    # lossless fp32 reloads: the sumsq transpose's input and the Accurate (SFPU) reduces' inputs. Only with a 32-bit
    # DEST: a 16-bit DEST cannot hold the unpacked fp32 tile, so those CBs then unpack through srcA (still Float32
    # pages: the wire format of the exchanged partials stays fp32 in both modes). The inv transpose (cb_inv_row) is
    # NOT tagged: its only consumer is a Col broadcast that reads srcB (Tf32), and fp32 -> Tf32 is idempotent, so the
    # unpacker's srcA transpose is bit-exact and cheaper than unpack-to-DEST + 32-bit transpose_dest (Perf 2,
    # perf_experiments/norm_no_transpose: worker normalize 1333 -> 1207 ns at h=1, w=7; -1.3 us at h=8).
    if fp32_dest:
        for idx in (CB_SUMSQ, CB_SUMSQ_T, CB_GATHER, CB_PREGATHER):
            modes[idx] = ttnn.UnpackToDestMode.UnpackToDestFp32
    ccfg.unpack_to_dest_mode = modes

    # every worker kernel derives round r's tile rows from R, round_stride and its runtime [row0, row_end)
    rounds_ct = [R, round_stride]
    modes_ct = [int(x_sharded), int(pre), n_pre, int(gamma_stream)]
    reader_ct = (
        [block_h, block_w, tw, int(has_gamma), x_page, g_page, g_face, n_gather, C_board, G, slot_bytes]
        + rounds_ct
        + modes_ct
        + [C, gamma_batch, int(DRAM_STAGGER)]
        + [shape[-1] % 32, X_ELEM_BYTES[input_tensor.dtype], int(has_gamma and gamma.layout == ttnn.ROW_MAJOR_LAYOUT)]
        + [int(gamma_mcast)]
        + x_ct
        + g_ct
    )
    writer_ct = (
        [block_h, block_w, tw, x_page, C_board, slot_bytes]
        + rounds_ct
        + [inv_rounds, round_bytes, int(out_sharded), int(pre), n_pre, int(STORE_FLUSH), int(DRAM_STAGGER)]
        + out_ct
    )
    compute_ct = [block_h, block_w, int(has_gamma), n_gather] + rounds_ct + modes_ct + [int(lead_ahead)]
    inv_w_bits, eps_bits = _f32_bits(1.0 / W_global), _f32_bits(epsilon)

    mesh_desc = ttnn.MeshProgramDescriptor()
    for coord, ch in chips.items():
        p = ch["p"]
        program = ttnn.ProgramDescriptor()
        program.semaphores = [
            ttnn.SemaphoreDescriptor(id=s, core_type=ttnn.CoreType.WORKER, core_ranges=all_set, initial_value=0)
            for s in (
                SEM_PORT_LOCAL,
                SEM_BOARD_READY,
                SEM_INV_READY,
                SEM_PORT_LOCAL_ODD,
                SEM_BOARD_READY_ODD,
                SEM_PRE_READY,
                SEM_PRE_READY_ODD,
                SEM_GATHER_DONE,
                SEM_RX_FREE,
                SEM_INV_SIG,
            )
        ]
        # which port on this chip holds slot g: own -> fwd port; arrivals from prev -> fwd; from next -> bwd
        fwd_recv = set(chips[ch["prev"]]["fwd"]) if ch["prev"] is not None else set()
        bwd_recv = set(chips[ch["next"]]["bwd"]) if ch["next"] is not None else set()
        src_dir = []
        for g in range(G):
            if g == p or g in fwd_recv:
                src_dir.append("fwd")
            elif g in bwd_recv:
                src_dir.append("bwd")
            else:
                raise AssertionError(f"distributed_rms_norm: slot {g} never reaches chip {coord}")

        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for k, core in enumerate(used_workers):
            gm = geoms[k]
            l = gm["link"]
            leader = is_lead(gm)
            src_ports = [_xy(pvirt[(src_dir[g], l)]) for g in range(G)]
            reader_rt[core.x][core.y] = [
                x_addr,
                g_addr,
                gm["row0"],
                gm["row_end"],
                gm["col0"],
                gm["w_act"],
                gm["w_valid"],
                int(leader),
                board_addr,
                gm["rt_base"],
                _xy(pvirt[("fwd", l)]),
                _xy(pvirt[("bwd", l)]),
            ] + src_ports
            fan = (
                _fanout_rects(mesh_device, [used_workers[gm["rg"] * C + wc] for wc in range(C)], core) if leader else []
            )
            writer_rt[core.x][core.y] = [
                out_addr,
                gm["row0"],
                gm["row_end"],
                gm["col0"],
                gm["w_act"],
                gm["wc"],
                _xy(virt(leader_of_rg[gm["rg"]])),
                p,
                board_addr,
                gm["rt_base"],
                _xy(pvirt[("fwd", l)]),
                _xy(pvirt[("bwd", l)]),
                int(leader),
                len(fan) // 6,
            ] + fan
            compute_rt[core.x][core.y] = [gm["row0"], gm["row_end"], gm["w_act"], int(leader), inv_w_bits, eps_bits]

        port_rt = ttnn.RuntimeArgs()
        for d, l in used_ports:
            core = port_core[(d, l)]
            peer = ch["next"] if d == "fwd" else ch["prev"]
            sends = ch[d] if peer is not None else []
            up = ch["prev"] if d == "fwd" else ch["next"]
            expect_in = len(chips[up][d]) if up is not None else 0
            send_ready = _needs_ready(chips, coord, d)
            args = [
                board_addr,
                slot_bytes,
                payload,
                workers_of_link[l],
                arrival_addr,
                ready_addr,
                expect_in,
                R,
                round_bytes,
            ]
            # the opposite-direction port: on the peer it is the sender into this chip (the credit goes there); here
            # it is the board whose release (rx_free) gates the credit, and vice versa (same core on every chip)
            rc = virt(port_core[(_OPP[d], l)])
            opp_recv = (
                (chips[ch["prev"]]["fwd"] if _OPP[d] == "fwd" else chips[ch["next"]]["bwd"])
                if (ch["prev"] if _OPP[d] == "fwd" else ch["next"]) is not None
                else []
            )
            assert send_ready == bool(opp_recv), "credit return must match the opposite port's receive"
            args += [int(send_ready), rc.x, rc.y]
            leaders = leaders_of_link[l]
            if peer is not None:
                pc = pvirt[(d, l)]  # same port key on the peer chip (identical placement on every chip)
                pn = _node(mesh_device, peer)
                args += [pc.x, pc.y, int(pn.mesh_id), int(pn.chip_id)]
            else:
                args += [0, 0, 0, 0]
            args += [len(leaders)] + [_xy(virt(c)) for c in leaders]
            args += [len(sends)] + list(sends)
            if peer is not None and (sends or send_ready):
                link_idx = ch[f"{d}_links"][l]
                args += list(ttnn.setup_fabric_connection(_node(mesh_device, coord), pn, link_idx, program, core))
            port_rt[core.x][core.y] = args

        program.cbs = cbs
        reader_kernel = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "distributed_rms_norm_reader.cpp"),
            core_ranges=worker_set,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=NOC0),
        )
        if gamma_mcast:
            gmc.attach(program, "gmc", [reader_kernel])
        else:
            ttnn.attach_absent(reader_kernel, "gmc")
        program.kernels = [
            reader_kernel,
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "distributed_rms_norm_writer.cpp"),
                core_ranges=worker_set,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=NOC1),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "distributed_rms_norm_compute.cpp"),
                core_ranges=worker_set,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=ccfg,
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "distributed_rms_norm_port_sender.cpp"),
                core_ranges=_core_set(used_port_cores),
                compile_time_args=[HEADER_RING],
                runtime_args=port_rt,
                config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=NOC1),
            ),
        ]
        r, c = coord
        mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = program

    io = [input_tensor] + ([gamma] if has_gamma else []) + [output]
    return ttnn.generic_op(io, mesh_desc)
