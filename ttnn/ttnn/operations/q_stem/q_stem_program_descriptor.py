# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""q_stem program descriptor: one program per chip, one ttnn.generic_op dispatch for the mesh.

Per-chip core roles (op_design.md, Blocking Model):
  port cores   row 0, 2 * num_links (direction x link): pull remote q head-rows from the stage-1 cores' L1 (NoC0),
               send them over the fabric into the destination core's landing scratch (NoC1), ready fence first.
  injector     one free row-0 core: reads x rows from DRAM in send order and multicasts them to the stage-1 rect.
  compute rect rows 1 .. gy-1: stage 1 (column split of q = x @ q_b, rect W1 x R1 anchored at (0, 1)) and stage 3
               (heads over every compute core: prefix = nope @ wkv_b1[h], rope pass-through, output write).

Knobs (single source each; every dependent quantity derives from them):
  S1_BLOCK_M_TILES   stage-1 row block (the exchange's send unit), clamped to a divisor of quarter_row_tiles
  S1_COLS_PER_CORE   stage-1 columns per core -> the stage-1 rect (smallest W1 x R1 holding ceil(Cq_t / it) cores)
  X_ROUND_TILES_CAP  x round size cap -> s1_kb (largest divisor of Q_t <= cap / s1_block_m)
  X_ROUND_DEPTH      cb_x_round depth (rounds)
  S3_BLOCK_M_TILES   stage-3 row block (1: the landing row layout interleaves nope / rope per row)
  S3_BLOCK_N_CAP     stage-3 N block cap in the streamed-wkv regime (R2)
  S3_LATE_REMOTE_ROWS stage-3 consumption order: remote rows < ts, local heads, remote rows >= ts (ts = rows - late)
  WKV_RESIDENT_BYTES wkv_b1 residency budget (R1 vs R2)
  WKV_DEPTH          cb_wkv depth (units: heads in R1, N-blocks in R2)
  RM_FULL_STICK      ROW_MAJOR out in R1: one untilized row block [prefix | rope] and one write per output stick
  OUT_DEPTH          output staging depth (row blocks): cb_prefix_out / cb_rope_out (TILE out) or
                     cb_prefix_rm / cb_rope_rm (ROW_MAJOR out; the tiled CBs then only feed the in-kernel untilize
                     and hold one block)
  FREE_WRITER_NOC    NoC of the stage-3-only cores' output writer (reader on the other NoC)
  PORT_BATCH_SEGMENTS port staging batch: segments per pull-read barrier (port reader) and per flush (sender);
                     the staging CB holds 2 batches
  QB_READ_BATCH      q_b slice reads per barrier
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# ---- knobs ------------------------------------------------------------------------------------------------------------
S1_BLOCK_M_TILES = 1
S1_COLS_PER_CORE = 2  # stage-1 columns per core: sizes the stage-1 rect; the rest of the grid starts stage 3 at t=0
# Head placement cost model (tile-MACs). Measured on the Kimi K3.1 Phase-0 call (HiFi4, fp32 DEST): a stage-3 row
# (1 x Dn_t @ Dn_t x N_t, per-row init + pack) costs ~2.5x a stage-1 tile-MAC, and cores outside the stage-1 rect
# get their first rows only after the q_b prologue + the first blocks (~0.35 of a full stage-1 core's run).
S3_COST_FACTOR = 2.5
FREE_CORE_START = 0.35
X_ROUND_TILES_CAP = 64
X_ROUND_DEPTH = 3
S3_BLOCK_M_TILES = 1
S3_BLOCK_N_CAP = 16
# Remote rows consumed AFTER the local heads (the last S3_LATE_REMOTE_ROWS rows of every remote head). The own quarter
# lands when stage 1 ends; the last remote rows land later still (they queue behind the fabric on the busiest link),
# so consuming local heads first hides their stage-3 rows + output writes behind that last landing. 0 -> local heads
# last (every remote row first). Clamped to rows_per_head - 1; dropped when it would push R1 out of residency.
S3_LATE_REMOTE_ROWS = 0
WKV_RESIDENT_BYTES = 512 * 1024  # per core: every remote head's weights resident (row-major consumption)
WKV_DEPTH = 2
OUT_DEPTH = 2
# ROW_MAJOR out, one N-block per row (R1): the rope tiles are packed behind the prefix into one row-wide tiled block,
# untilized once, and each output stick ((N + Dr) * 2 B, the whole page) leaves in ONE NoC write instead of one per
# segment (prefix | rope): half the writes per row, and no 128 B rope writes. False -> the segment path.
RM_FULL_STICK = True
PORT_BATCH_SEGMENTS = 4
# NoC of the output writer on the cores outside the stage-1 rect (their reader takes the other NoC), per region:
# "right" = the columns right of the stage-1 rect, "below" = the rows under it. Their output writes run while stage
# 1 + the stage-1 cores' tail keep NoC1 busy (hand-offs, port feeds, the stage-1 cores' own output writes): on NoC1
# a far-east free core's writes starved for tens of us (measured). 1 everywhere -> the default reader/writer split.
FREE_WRITER_NOC = {"right": 0, "below": 1}
S1_WRITER_NOC = 1  # the same for the stage-1 rect (its reader, incl. the x-multicast receive, on the other NoC)
QB_READ_BATCH = 8  # q_b slice reads per barrier (dm shortlist tie; rank-1 batch 4 is lamp L10)
MAX_HEADERS = 12  # PacketHeaderPool: <= 12 headers per RISC (passed to the port sender: one source)
HDRS_PER_DEST_CAP = 4

# ---- CB indices (semantic) ---------------------------------------------------------------------------------------------
CB_Q_SLICE = 0
CB_X_ROUND = 1
CB_QB_SLICE = 2
CB_Q_PARTIAL = 3
CB_LANDING = 4
CB_WKV = 5
CB_PREFIX_OUT = 6
CB_ROPE_OUT = 7
CB_PORT_STAGE = 8
CB_PREFIX_RM = 9
CB_ROPE_RM = 10

# ---- program semaphore ids (port cores; declared on every core the program uses) -------------------------------------
SEM_BLOCK_READY = 0

FWD, BWD = 0, 1


def _node(mesh, coord):
    return mesh.get_fabric_node_id(ttnn.MeshCoordinate(*coord))


def _crs(cores):
    """Cores -> a CoreRangeSet merged into rectangles (a per-core range list makes dispatch write every kernel
    binary / CB config core by core)."""
    out = ttnn.CoreRangeSet([ttnn.CoreRange(cores[0], cores[0])])
    for c in cores[1:]:
        out = out.merge(ttnn.CoreRangeSet([ttnn.CoreRange(c, c)]))
    return out


def _rect(x0, y0, x1, y1):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])


def _merge(*sets):
    out = sets[0]
    for s in sets[1:]:
        out = out.merge(s)
    return out


def _pack(c):
    return (int(c.x) << 16) | int(c.y)


def _largest_divisor_leq(n, cap):
    cap = max(1, min(n, cap))
    for d in range(cap, 0, -1):
        if n % d == 0:
            return d
    return 1


def _s1_rect(Cq_t, gx, gy_compute):
    """Smallest W1 x R1 rect (W1 <= gx, R1 <= gy_compute, every core >= 1 column) holding ceil(Cq_t / S1_COLS_PER_CORE)
    cores; ties -> fewer rows. Capped by the compute rect when Cq_t needs more."""
    want = min(max(1, -(-Cq_t // S1_COLS_PER_CORE)), gx * gy_compute)
    best = None
    for r in range(1, gy_compute + 1):
        w = min(gx, -(-want // r))
        if w * r < want or w * r > Cq_t:
            continue
        key = (w * r, r)
        if best is None or key < best[0]:
            best = (key, (w, r))
    if best is None:  # Cq_t smaller than any rect holding `want`: one row of Cq_t cores
        return min(gx, Cq_t), 1
    return best[1]


def _dest_limit(fp32_dest_acc_en):
    # half-sync DEST: 8 tiles bf16, 4 tiles fp32 (DEST_AUTO_LIMIT in dest_helpers.hpp)
    return 4 if fp32_dest_acc_en else 8


# ---------------------------------------------------------------------------------------------------------------------
# Geometry (all derived quantities, one place)
# ---------------------------------------------------------------------------------------------------------------------


class Geometry:
    def __init__(self, input_tensor, q_b, wkv, G, grid, fp32_dest_acc_en, output_layout=ttnn.TILE_LAYOUT):
        B, _, S, Q = (int(v) for v in input_tensor.shape)
        _, H, Dn, N = (int(v) for v in wkv.shape)
        Cq = int(q_b.shape[3])
        self.G, self.B, self.S, self.Q, self.H, self.Dn, self.N = G, B, S, Q, H, Dn, N
        self.Hl = H // G
        self.Dr = Cq // self.Hl - Dn
        self.Q_t, self.Dn_t, self.Dr_t, self.N_t = Q // 32, Dn // 32, self.Dr // 32, N // 32
        self.hw = self.Dn_t + self.Dr_t  # head_w_tiles
        self.Cq_t = self.Hl * self.hw
        self.St = S // 32  # row tiles per image
        self.qrt = (S // G) // 32  # quarter_row_tiles
        self.tile_bytes = ttnn.tile_size(ttnn.bfloat16)
        self.qb_tile_bytes = ttnn.tile_size(q_b.dtype)
        self.wkv_tile_bytes = ttnn.tile_size(wkv.dtype)
        self.x_tile_bytes = ttnn.tile_size(input_tensor.dtype)
        self.dest_limit = _dest_limit(fp32_dest_acc_en)
        # Stage-1 K-block partials: the DEST width. A Float32 page behind 16-bit DEST carries no extra bits.
        self.partial_dtype = ttnn.float32 if fp32_dest_acc_en else ttnn.bfloat16
        self.partial_tile_bytes = ttnn.tile_size(self.partial_dtype)

        # stage-1 blocking
        self.m = _largest_divisor_leq(self.qrt, S1_BLOCK_M_TILES)  # s1_block_m_tiles (no ragged block)
        self.s1_kb = _largest_divisor_leq(self.Q_t, max(1, X_ROUND_TILES_CAP // self.m))
        self.nkb = self.Q_t // self.s1_kb
        self.round_tiles = self.m * self.s1_kb
        self.blocks_per_image = (self.qrt // self.m) * G
        self.num_blocks = B * self.blocks_per_image

        # grid: row 0 = ports + injector; compute rect rows 1 .. gy-1
        self.gx, self.gy = int(grid.x), int(grid.y)
        self.C = self.gx * (self.gy - 1)
        self.W1, self.R1 = _s1_rect(self.Cq_t, self.gx, self.gy - 1)
        self.C1 = self.W1 * self.R1
        self.cols_base, self.cols_extra = divmod(self.Cq_t, self.C1)
        assert self.cols_base >= 1
        self.cols_max = self.cols_base + (1 if self.cols_extra else 0)

        # stage 3
        assert S3_BLOCK_M_TILES == 1, "the landing row layout [nope | rope] per row needs s3_block_m_tiles = 1"
        self.s3m = S3_BLOCK_M_TILES
        self.rows_per_head = B * self.qrt
        self.compute_core_list, self.heads_of = head_placement(self)
        self.hpc_max = max(len(hs) for hs in self.heads_of)
        s1_set = {(c.x, c.y) for c in self.s1_cores()}
        # R1 wkv units a core holds at once: all its remote heads (consumed row-major) + up to WKV_DEPTH local heads,
        # max over destination chips; per core set (stage-1 rect / the rest), so each gets its own cb_wkv size. With
        # late remote rows (consumed after the local heads) a core holding both keeps every head resident (FIFO).
        wkv_head_bytes = self.Dn_t * self.N_t * self.wkv_tile_bytes

        def wkv_units(late):
            s1u, freeu = 1, 1
            for k, hs in enumerate(self.heads_of):
                c = self.compute_core_list[k]
                for dest in range(G):
                    n_l = sum(1 for h in hs if h // self.Hl == dest)
                    n_r = len(hs) - n_l
                    need = len(hs) if (late and n_l and n_r) else n_r + min(WKV_DEPTH, n_l)
                    if (c.x, c.y) in s1_set:
                        s1u = max(s1u, need)
                    else:
                        freeu = max(freeu, need)
            return s1u, freeu, max(s1u, freeu) * wkv_head_bytes <= WKV_RESIDENT_BYTES

        self.late = min(S3_LATE_REMOTE_ROWS, self.rows_per_head - 1)
        self.wkv_units_s1, self.wkv_units_free, self.wkv_resident = wkv_units(self.late)
        if self.late and not self.wkv_resident and wkv_units(0)[2]:
            self.late = 0  # late rows would push R1 out of residency: keep R1
            self.wkv_units_s1, self.wkv_units_free, self.wkv_resident = wkv_units(0)
        self.ts = self.rows_per_head - self.late  # remote rows >= ts are consumed after the local heads
        self.s3n = self.N_t if self.wkv_resident else _largest_divisor_leq(self.N_t, S3_BLOCK_N_CAP)
        self.n_nb = self.N_t // self.s3n
        self.wkv_unit_tiles = self.Dn_t * self.s3n
        self.sbw3 = _largest_divisor_leq(self.s3n, self.dest_limit)
        self.landing_tiles = self.hpc_max * self.rows_per_head * self.hw
        self.out_w_tiles = self.N_t + self.Dr_t
        # output: ROW_MAJOR -> the compute untilizes each block (cb_prefix_out -> cb_prefix_rm, cb_rope_out ->
        # cb_rope_rm) and the writer writes sticks; the tiled CBs are then compute-internal (one block deep)
        self.out_rm = output_layout == ttnn.ROW_MAJOR_LAYOUT
        self.out_tile_depth = 1 if self.out_rm else OUT_DEPTH
        self.out_page_bytes = self.out_w_tiles * (self.tile_bytes // 32) if self.out_rm else self.tile_bytes
        # full-stick row block (RM_FULL_STICK): the rope tiles join the prefix in cb_prefix_out -> cb_prefix_rm
        self.full_stick = self.out_rm and self.n_nb == 1 and RM_FULL_STICK
        self.out_blk_tiles = self.s3n + self.Dr_t if self.full_stick else self.s3n  # cb_prefix_out/_rm block

    # stage-1 cores, row-major over the W1 x R1 rect anchored at (0, 1)
    def s1_cores(self):
        return [ttnn.CoreCoord(x, 1 + y) for y in range(self.R1) for x in range(self.W1)]

    def s1_cols(self, i):
        cols = self.cols_base + (1 if i < self.cols_extra else 0)
        c0 = i * self.cols_base + min(i, self.cols_extra)
        return c0, cols

    def compute_cores(self):
        return [ttnn.CoreCoord(x, y) for y in range(1, self.gy) for x in range(self.gx)]


# ---------------------------------------------------------------------------------------------------------------------
# Exchange plan (TRANSPORT): routes, ports, readies
# ---------------------------------------------------------------------------------------------------------------------


def _first_hop_dir(mesh, src, dst, nxt, prv):
    """FWD / BWD: the port whose connection carries the first hop of the fabric's route src -> dst."""
    d = ttnn.get_eth_forwarding_direction(_node(mesh, src), _node(mesh, dst))
    if d is None:
        return None
    for port, nb in ((FWD, nxt), (BWD, prv)):
        if d == ttnn.get_eth_forwarding_direction(_node(mesh, src), _node(mesh, nb)):
            return port
    return None


def port_cores(gx, num_links):
    """2 * num_links port cores, centred on logical row 0: {(direction, link): core}."""
    n = 2 * num_links
    x0 = max(0, gx // 2 - num_links)
    cores = [ttnn.CoreCoord(x0 + i, 0) for i in range(n)]
    return {(d, l): cores[2 * l + d] for l in range(num_links) for d in (FWD, BWD)}


def injector_core(gx, ports):
    busy = {c.x for c in ports.values()}
    free = sorted((x for x in range(gx) if x not in busy), key=lambda x: (abs(x - gx // 2), x))
    if not free:
        raise RuntimeError("q_stem: row 0 has no free core for the x injector")
    return ttnn.CoreCoord(free[0], 0)


def remote_order(p, G):
    """[p+1, p-1, p+2, p-2, ...] mod G without p and without repeats."""
    out = []
    for k in range(1, G):
        for d in ((p + k) % G, (p - k) % G):
            if d != p and d not in out:
                out.append(d)
    return out[: G - 1]


def exchange_plan(mesh, cluster_axis, num_links):
    rows, cols = tuple(mesh.shape)
    if cluster_axis != 1:
        raise ValueError("q_stem: only cluster_axis=1 groups are planned")
    G = cols
    if G < 3:
        raise ValueError(f"q_stem: a Ring needs at least 3 devices along cluster_axis, got {G}")
    chips = {}
    for r in range(rows):
        grp = [(r, c) for c in range(G)]
        nxt = [grp[(p + 1) % G] for p in range(G)]
        prv = [grp[(p - 1) % G] for p in range(G)]
        for p in range(G):
            for nb in (nxt[p], prv[p]):
                links = ttnn.get_forwarding_link_indices(_node(mesh, grp[p]), _node(mesh, nb))
                if len(links) < num_links:
                    raise ValueError(
                        f"q_stem: {grp[p]} -> {nb} has {len(links)} usable link(s), num_links={num_links} requested "
                        f"(fabric={ttnn.get_fabric_config()}); a Ring needs the wrap link along cluster_axis"
                    )
        route = {}
        for s in range(G):
            for t in range(G):
                if s != t:
                    d = _first_hop_dir(mesh, grp[s], grp[t], nxt[s], prv[s])
                    if d is None:
                        raise ValueError(f"q_stem: the fabric route {grp[s]} -> {grp[t]} leaves the ring")
                    route[(s, t)] = d
        for p in range(G):
            chips[grp[p]] = dict(p=p, grp=grp, next=nxt[p], prev=prv[p], route=route)
    return chips, G


# ---------------------------------------------------------------------------------------------------------------------
# Head placement (stage 3) and landing layout
# ---------------------------------------------------------------------------------------------------------------------


def head_placement(geo):
    """k3(h) = compute-core index (identical on every chip). Greedy on projected compute load (tile-matmuls): a
    stage-1 column costs num_blocks * m * Q_t, a head rows_per_head * Dn_t * N_t; each head goes to the least
    loaded core (ties -> the core with no stage-1 work, then grid order), so heads fill the cores outside the
    stage-1 rect first: there they run (and write their output) while stage 1 runs elsewhere.
    Returns (core_list, heads_of_core[k] = [h ...])."""
    import heapq

    s1 = {(c.x, c.y): i for i, c in enumerate(geo.s1_cores())}
    cores = geo.compute_cores()
    col_cost = geo.num_blocks * geo.m * geo.Q_t
    head_cost = S3_COST_FACTOR * geo.rows_per_head * geo.Dn_t * geo.N_t
    free_start = FREE_CORE_START * geo.cols_max * col_cost
    heap = []
    for k, c in enumerate(cores):
        i = s1.get((c.x, c.y))
        cols = 0 if i is None else geo.s1_cols(i)[1]
        heapq.heappush(heap, (cols * col_cost if cols else free_start, cols, k))
    heads_of = [[] for _ in cores]
    for h in range(geo.H):
        load, cols, k = heapq.heappop(heap)
        heads_of[k].append(h)
        heapq.heappush(heap, (load + head_cost, cols, k))
    return cores, heads_of


def landing_slot(heads, h, dest, geo):
    """(base, stride, jump) in tiles of head h's rows in its core's landing shard on chip `dest`: row t at
    base + t * stride + (jump if t >= geo.ts else 0). The landing layout IS the consumption order (device twin:
    q_stem_common.hpp row_of): remote rows t < ts row-major across the remote heads (all remote heads land with the
    same row profile), then the local heads head-major, then the late remote rows t >= ts row-major."""
    order = consumption_order(heads, dest, geo)
    n_r = sum(1 for hh in order if hh // geo.Hl != dest)
    n_l = len(order) - n_r
    ts = geo.ts if n_l else geo.rows_per_head
    idx = order.index(h)
    if idx < n_r:
        return idx * geo.hw, n_r * geo.hw, n_l * geo.rows_per_head * geo.hw
    return (n_r * ts + (idx - n_r) * geo.rows_per_head) * geo.hw, geo.hw, 0


def consumption_order(heads, dest, geo):
    """Landing slot order of a core's heads on destination chip `dest`, by expected completion: ring neighbours'
    heads first (their rows land during stage 1), then farther sources, the local heads last (the own quarter is
    computed last in the send order)."""
    G = geo.G

    def key(h):
        src = h // geo.Hl
        dist = min((src - dest) % G, (dest - src) % G)
        return (dist if dist > 0 else G, h)

    return sorted(heads, key=key)


# ---------------------------------------------------------------------------------------------------------------------
# Persistent resources (global semaphores + landing scratch), created once per mesh / layout
# ---------------------------------------------------------------------------------------------------------------------

_RESOURCES = {}


def _alloc_scratch(mesh, geo, tiles):
    shard_h = tiles * 32
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(_rect(0, 1, geo.gx - 1, geo.gy - 1), (shard_h, 32), ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.allocate_tensor_on_device(ttnn.Shape([geo.C * shard_h, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, mc)


def _resources(mesh, geo, ports, injector):
    """Per mesh (and core layout): G + 1 global semaphores (ready, arrive[0..G-1]) and ONE landing scratch (L1,
    height-sharded over the compute rect), grown when a call needs more rows than it holds. Created outside any
    trace on the first call (and after a mesh reopen); every call re-arms the counters, so they stay valid."""
    used = _merge(_rect(0, 1, geo.gx - 1, geo.gy - 1), _crs(list(ports.values()) + [injector]))
    key = (id(mesh), geo.gx, geo.gy, tuple(sorted((c.x, c.y) for c in ports.values())), geo.G)
    entry = _RESOURCES.get(key)
    if entry is not None and (entry["mesh"] is not mesh or not entry["scratch"].is_allocated()):
        _RESOURCES.pop(key)
        entry = None
    if entry is None:
        sems = [ttnn.create_global_semaphore(mesh, used, 0) for _ in range(1 + geo.G)]
        scratch = _alloc_scratch(mesh, geo, geo.landing_tiles)
        ttnn.synchronize_device(mesh)  # every chip's counters are zero before any chip's first call
        entry = dict(mesh=mesh, sems=sems, scratch=scratch, tiles=geo.landing_tiles)
        _RESOURCES[key] = entry
    elif entry["tiles"] < geo.landing_tiles:
        ttnn.deallocate(entry["scratch"])
        entry["scratch"] = _alloc_scratch(mesh, geo, geo.landing_tiles)
        entry["tiles"] = geo.landing_tiles
    sems, scratch = entry["sems"], entry["scratch"]
    ready_addr = int(ttnn.get_global_semaphore_address(sems[0]))
    arrive_addrs = [int(ttnn.get_global_semaphore_address(s)) for s in sems[1:]]
    return ready_addr, arrive_addrs, scratch


# ---------------------------------------------------------------------------------------------------------------------
# L1 fit (l1_ledger.md "Total per compute core"): every resident term, checked before the program is built
# ---------------------------------------------------------------------------------------------------------------------


def cb_l1_budget(mesh):
    """Bytes of L1 per core the CBs may occupy: CBs grow bottom-up from the allocator base, L1 buffers (the landing
    scratch, an L1 latent) top-down, so the budget is the lowest allocated L1 address."""
    view = ttnn.get_memory_view(mesh, ttnn.BufferType.L1)
    allocated = [int(b["address"]) for b in view.block_table if b["allocated"] == "yes"]
    return min(allocated) if allocated else int(view.total_bytes_per_bank)


def l1_terms(geo, wkv_depth):
    """{core set: {term: bytes}} of the statically allocated CBs (cb_landing lives in the scratch buffer)."""
    tb = geo.tile_bytes
    s3_out = (geo.out_tile_depth + (OUT_DEPTH if geo.out_rm else 0)) * geo.s3m * (geo.s3n + geo.Dr_t) * tb
    wkv_unit = geo.wkv_unit_tiles * geo.wkv_tile_bytes
    wkv_s1 = (geo.wkv_units_s1 if geo.wkv_resident else wkv_depth) * wkv_unit
    wkv_free = (geo.wkv_units_free if geo.wkv_resident else wkv_depth) * wkv_unit
    q_slice = geo.num_blocks * geo.m * geo.cols_max * tb
    stage1 = {
        "cb_q_slice": q_slice,
        "cb_x_round": X_ROUND_DEPTH * geo.round_tiles * geo.x_tile_bytes,
        "cb_qb_slice": geo.Q_t * geo.cols_max * geo.qb_tile_bytes,
        "cb_q_partial": geo.m * geo.cols_max * geo.partial_tile_bytes if geo.nkb > 1 else 0,
        "cb_wkv": wkv_s1,
        "cb_prefix/rope_out(+_rm)": s3_out,
    }
    return {
        "stage-1 core": stage1,
        "stage-3-only core": {"cb_wkv": wkv_free, "cb_prefix/rope_out(+_rm)": s3_out},
        "port core": {"cb_q_slice": q_slice, "cb_port_stage": 2 * PORT_BATCH_SEGMENTS * geo.hw * tb},
    }


def check_fits_l1(mesh, geo, wkv_depth):
    budget = cb_l1_budget(mesh)
    for role, terms in l1_terms(geo, wkv_depth).items():
        total = sum(terms.values())
        if total > budget:
            worst = max(terms, key=terms.get)
            raise RuntimeError(
                f"q_stem: per-core L1 does not fit on a {role}: {total} B of CBs > {budget} B free "
                f"(largest term {worst} = {terms[worst]} B; terms {terms}). Regime R3 (stage-1 streaming) "
                f"is not built."
            )


# ---------------------------------------------------------------------------------------------------------------------
# Program per chip
# ---------------------------------------------------------------------------------------------------------------------


def _dm(risc, noc):
    return ttnn.DataMovementConfigDescriptor(processor=getattr(ttnn.DataMovementProcessor, risc), noc=noc)


def _cb(index, dtype, page, pages, cores):
    return ttnn.CBDescriptor(
        total_size=page * pages,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
    )


def create_mesh_program_descriptor(x, q_b, wkv, out, compute_config, cluster_axis, num_links):
    output_layout = out.layout
    mesh = x.device()
    grid = mesh.compute_with_storage_grid_size()
    chips, G = exchange_plan(mesh, cluster_axis, num_links)
    geo = Geometry(x, q_b, wkv, G, grid, bool(compute_config.fp32_dest_acc_en), output_layout)
    L = num_links
    ports = port_cores(geo.gx, L)
    injector = injector_core(geo.gx, ports)
    ready_addr, arrive_addrs, scratch = _resources(mesh, geo, ports, injector)
    landing_addr = int(scratch.buffer_address())
    payload = int(ttnn.get_tt_fabric_max_payload_size_bytes())
    pkt_tiles = max(1, payload // geo.tile_bytes)
    virt = lambda c: mesh.worker_core_from_logical_core(c)

    s1_cores = geo.s1_cores()
    compute_cores, heads_of = geo.compute_core_list, geo.heads_of
    core_of_head = {}
    for k, hs in enumerate(heads_of):
        for h in hs:
            core_of_head[h] = k
    compute_rect = _rect(0, 1, geo.gx - 1, geo.gy - 1)
    s1_rect = _rect(0, 1, geo.W1 - 1, geo.R1)
    s1_set = {(c.x, c.y) for c in s1_cores}
    other_compute = [c for c in compute_cores if (c.x, c.y) not in s1_set]
    port_list = [ports[(d, l)] for l in range(L) for d in (FWD, BWD)]
    port_crs = _crs(port_list)
    inj_crs = _crs([injector])
    all_crs = _merge(compute_rect, port_crs, inj_crs)

    x_ct = list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
    qb_ct = list(ttnn.TensorAccessorArgs(q_b).get_compile_time_args())
    wkv_ct = list(ttnn.TensorAccessorArgs(wkv).get_compile_time_args())
    out_ct = list(ttnn.TensorAccessorArgs(out).get_compile_time_args())

    hdr_per_dest = max(1, min(HDRS_PER_DEST_CAP, MAX_HEADERS // max(1, G - 1)))
    wkv_units_total_max = geo.hpc_max if geo.wkv_resident else geo.hpc_max * geo.rows_per_head * geo.n_nb
    wkv_depth = max(1, min(WKV_DEPTH, wkv_units_total_max))
    check_fits_l1(mesh, geo, wkv_depth)

    mesh_desc = ttnn.MeshProgramDescriptor()
    for coord, ch in chips.items():
        p = ch["p"]
        grp = ch["grp"]
        route = ch["route"]
        rorder = remote_order(p, G)
        dirmap = [route[(p, d)] if d != p else 0 for d in range(G)]

        program = ttnn.ProgramDescriptor()
        if geo.wkv_resident:
            unit = geo.Dn_t * geo.N_t
            wkv_cbs = [_cb(CB_WKV, wkv.dtype, geo.wkv_tile_bytes, geo.wkv_units_s1 * unit, s1_rect)]
            if other_compute:
                wkv_cbs.append(
                    _cb(CB_WKV, wkv.dtype, geo.wkv_tile_bytes, geo.wkv_units_free * unit, _crs(other_compute))
                )
        else:
            wkv_cbs = [_cb(CB_WKV, wkv.dtype, geo.wkv_tile_bytes, wkv_depth * geo.wkv_unit_tiles, compute_rect)]
        key = "|".join(
            str(v)
            for v in (
                "q_stem",
                coord,
                G,
                L,
                payload,
                tuple(x.shape),
                tuple(q_b.shape),
                tuple(wkv.shape),
                x.dtype,
                q_b.dtype,
                wkv.dtype,
                x.memory_config().buffer_type,  # the injector's accessor differs for an L1 / DRAM latent
                output_layout,  # TILE pages vs in-kernel untilize + stick writer
                compute_config.math_fidelity,
                compute_config.fp32_dest_acc_en,
                compute_config.math_approx_mode,
                S1_BLOCK_M_TILES,
                X_ROUND_TILES_CAP,
                X_ROUND_DEPTH,
                S3_BLOCK_M_TILES,
                S3_BLOCK_N_CAP,
                S3_LATE_REMOTE_ROWS,
                WKV_RESIDENT_BYTES,
                WKV_DEPTH,
                OUT_DEPTH,
                RM_FULL_STICK,
                PORT_BATCH_SEGMENTS,
                tuple(sorted(FREE_WRITER_NOC.items())),
                S1_WRITER_NOC,
                os.environ.get("QSTEM_ABLATE", ""),
                tuple(sorted(route.items())),
                geo.gx,
                geo.gy,
            )
        )
        program.custom_program_hash = int.from_bytes(hashlib.sha1(key.encode()).digest()[:8], "little") >> 1
        program.semaphores = [ttnn.SemaphoreDescriptor(id=SEM_BLOCK_READY, core_ranges=all_crs, initial_value=0)]

        # ---------------- CBs (allocation order matters only for address identity of cb_q_slice / cb_x_round) ------
        bf16 = ttnn.bfloat16
        cbs = [
            _cb(CB_Q_SLICE, bf16, geo.tile_bytes, geo.num_blocks * geo.m * geo.cols_max, _merge(s1_rect, port_crs)),
            _cb(CB_X_ROUND, x.dtype, geo.x_tile_bytes, X_ROUND_DEPTH * geo.round_tiles, _merge(s1_rect, inj_crs)),
            _cb(CB_QB_SLICE, q_b.dtype, geo.qb_tile_bytes, geo.Q_t * geo.cols_max, s1_rect),
        ]
        if geo.nkb > 1:
            cbs.append(_cb(CB_Q_PARTIAL, geo.partial_dtype, geo.partial_tile_bytes, geo.m * geo.cols_max, s1_rect))
        cbs += [
            ttnn.cb_descriptor_from_sharded_tensor(CB_LANDING, scratch),
            *wkv_cbs,
            _cb(CB_PREFIX_OUT, bf16, geo.tile_bytes, geo.out_tile_depth * geo.s3m * geo.out_blk_tiles, compute_rect),
            _cb(CB_PORT_STAGE, bf16, geo.tile_bytes, 2 * PORT_BATCH_SEGMENTS * geo.hw, port_crs),
        ]
        if not geo.full_stick:  # full stick: the rope tiles live in cb_prefix_out / cb_prefix_rm
            cbs.append(_cb(CB_ROPE_OUT, bf16, geo.tile_bytes, geo.out_tile_depth * geo.s3m * geo.Dr_t, compute_rect))
        if geo.out_rm:  # untilized stick blocks (tile-sized pages: 32 sticks of block-width * 64 B)
            cbs.append(_cb(CB_PREFIX_RM, bf16, geo.tile_bytes, OUT_DEPTH * geo.s3m * geo.out_blk_tiles, compute_rect))
            if not geo.full_stick:
                cbs.append(_cb(CB_ROPE_RM, bf16, geo.tile_bytes, OUT_DEPTH * geo.s3m * geo.Dr_t, compute_rect))
        program.cbs = cbs

        # ---------------- per-compute-core assignment ------------------------------------------------------------
        s1_index = {(c.x, c.y): i for i, c in enumerate(s1_cores)}
        port_args = []
        for l in range(L):
            for d in (FWD, BWD):
                port_args += [_pack(virt(ports[(d, l)])), d]

        def head_slot(h, dest):
            return landing_slot(heads_of[core_of_head[h]], h, dest, geo)

        reader_rt_s1, reader_rt_other = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        compute_rt = ttnn.RuntimeArgs()
        writer_rt_s1, writer_rt_other = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for k, core in enumerate(compute_cores):
            i = s1_index.get((core.x, core.y))
            c0, cols = geo.s1_cols(i) if i is not None else (0, 0)
            heads = consumption_order(heads_of[k], p, geo)
            hpc = len(heads)
            n_r = sum(1 for h in heads if h // geo.Hl != p)
            # reader
            rargs = [int(q_b.buffer_address()), c0, cols, int(wkv.buffer_address()), p]
            rargs += arrive_addrs
            rargs += [hpc] + [v for h in heads for v in (h, h // geo.Hl)]
            (reader_rt_s1 if i is not None else reader_rt_other)[core.x][core.y] = rargs
            # compute
            sbw1 = _largest_divisor_leq(cols, geo.dest_limit) if cols else 1
            compute_rt[core.x][core.y] = [cols, sbw1, hpc, n_r]
            # writer: local-delivery pieces of my columns (own quarter), port list, heads for the output
            pieces = []
            if cols:
                c = c0
                while c < c0 + cols:
                    hl = c // geo.hw
                    cc = c % geo.hw
                    n = min(c0 + cols, (hl + 1) * geo.hw) - c
                    h = p * geo.Hl + hl
                    dest_core = compute_cores[core_of_head[h]]
                    base, stride, _ = head_slot(h, p)  # local heads: contiguous rows, no jump
                    pieces += [c - c0, n, _pack(virt(dest_core)), base + cc, stride]
                    c += n
            wargs = [p, cols, int(out.buffer_address()), landing_addr, arrive_addrs[p]]
            wargs += dirmap + rorder + port_args
            wargs += [len(pieces) // 5] + pieces
            wargs += [hpc, n_r] + heads
            (writer_rt_s1 if i is not None else writer_rt_other)[core.x][core.y] = wargs

        # ---------------- ports ---------------------------------------------------------------------------------
        core_table = [_pack(virt(c)) for c in s1_cores]
        port_reader_rt, port_sender_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for l in range(L):
            for d in (FWD, BWD):
                core = ports[(d, l)]
                my_heads = sorted(
                    (hl for hl in range(geo.Hl) if core_of_head[p * geo.Hl + hl] % L == l),
                    key=lambda hl: (core_of_head[p * geo.Hl + hl], hl),
                )
                sends_to = [t for t in range(G) if t != p and route[(p, t)] == d] if my_heads else []
                # readies this port sends: to every chip s whose port (route(s, p), l) sends to p, i.e. this
                # direction is route(p, s) and s's port has heads for link l
                ready_targets = []
                for s in range(G):
                    if s == p or route[(p, s)] != d:
                        continue
                    s_heads = [hl for hl in range(geo.Hl) if core_of_head[s * geo.Hl + hl] % L == l]
                    if s_heads:
                        ready_targets.append((s, route[(s, p)]))
                dests = sorted(set(sends_to) | {s for s, _ in ready_targets})
                assert len(dests) * hdr_per_dest <= MAX_HEADERS
                has_conn = bool(sends_to) or bool(ready_targets)

                port_reader_rt[core.x][core.y] = (
                    [p, d] + dirmap + rorder + [geo.cols_base, geo.cols_extra, len(my_heads)] + my_heads + core_table
                )
                sargs = [p, d, landing_addr, arrive_addrs[p], ready_addr, len(dests)]
                for t in dests:
                    n = _node(mesh, grp[t])
                    sargs += [int(n.chip_id), int(n.mesh_id)]
                sargs += [dests.index(t) if t in dests else 0xFF for t in range(G)]
                sargs += dirmap + rorder
                sargs += [len(my_heads)]
                for idx, hl in enumerate(my_heads):
                    h = p * geo.Hl + hl
                    k = core_of_head[h]
                    last_of_group = idx + 1 == len(my_heads) or core_of_head[p * geo.Hl + my_heads[idx + 1]] != k
                    group = sum(1 for hh in my_heads if core_of_head[p * geo.Hl + hh] == k)
                    inc = group * geo.hw if last_of_group else 0
                    sargs += [_pack(virt(compute_cores[k])), inc]
                    for t in range(G):
                        sargs += list(head_slot(h, t)) if t != p else [0, 0, 0]
                sargs += [len(ready_targets)]
                for s, sdir in ready_targets:
                    rc = virt(ports[(sdir, l)])
                    sargs += [dests.index(s), rc.x, rc.y]
                sargs += [len(sends_to), int(has_conn)]
                if has_conn:
                    peer = ch["next"] if d == FWD else ch["prev"]
                    link_ids = ttnn.get_forwarding_link_indices(_node(mesh, coord), _node(mesh, peer))
                    sargs += list(
                        ttnn.setup_fabric_connection(_node(mesh, coord), _node(mesh, peer), link_ids[l], program, core)
                    )
                port_sender_rt[core.x][core.y] = sargs

        # ---------------- kernels -------------------------------------------------------------------------------
        geo_ct = [geo.B, geo.qrt, geo.m, G, geo.hw]
        reader_ct = [
            CB_X_ROUND,
            CB_QB_SLICE,
            CB_LANDING,
            CB_WKV,
            geo.qb_tile_bytes,
            geo.wkv_tile_bytes,
            geo.Q_t,
            geo.Cq_t,
            geo.round_tiles,
            geo.nkb,
            geo.num_blocks,
            geo.Dn_t,
            geo.N_t,
            geo.s3n,
            geo.n_nb,
            int(geo.wkv_resident),
        ]

        def reader_ct_for(wkv_units):
            # wkv capacity of the core set in units (R1: remote heads + local depth; R2: wkv_depth)
            return reader_ct + [wkv_units, QB_READ_BATCH, *geo_ct, geo.ts, *qb_ct, *wkv_ct]

        reader_ct_s1 = reader_ct_for(geo.wkv_units_s1 if geo.wkv_resident else wkv_depth)
        reader_ct_other = reader_ct_for(geo.wkv_units_free if geo.wkv_resident else wkv_depth)
        reader_s1 = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "q_stem_reader.cpp"),
            core_ranges=s1_rect,
            compile_time_args=reader_ct_s1,
            runtime_args=reader_rt_s1,
            config=_dm("RISCV_1", ttnn.NOC.NOC_1 if S1_WRITER_NOC == 0 else ttnn.NOC.NOC_0),
        )
        inj_rt = ttnn.RuntimeArgs()
        inj_rt[injector.x][injector.y] = [int(x.buffer_address()), p] + rorder
        injector_reader_k = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "q_stem_injector_reader.cpp"),
            core_ranges=inj_crs,
            compile_time_args=[CB_X_ROUND, geo.x_tile_bytes, geo.Q_t, geo.s1_kb, geo.nkb, *geo_ct, *x_ct],
            runtime_args=inj_rt,
            config=_dm("RISCV_0", ttnn.NOC.NOC_1),
        )
        injector_k = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "q_stem_injector.cpp"),
            core_ranges=inj_crs,
            compile_time_args=[CB_X_ROUND, geo.x_tile_bytes, geo.round_tiles, geo.num_blocks * geo.nkb],
            runtime_args=ttnn.RuntimeArgs(),
            config=_dm("RISCV_1", ttnn.NOC.NOC_0),
        )
        kernels_tail = []
        # stage-3-only cores, grouped by their writer NoC (FREE_WRITER_NOC per region); one reader/writer pair each
        free_groups = {}
        for c in other_compute:
            free_groups.setdefault(FREE_WRITER_NOC["right" if c.x >= geo.W1 else "below"], []).append(c)

        def _rt_subset(rt_all, cores):
            rt = ttnn.RuntimeArgs()
            for c in cores:
                rt[c.x][c.y] = list(rt_all[c.x][c.y])
            return rt

        for wnoc, cores in sorted(free_groups.items()):
            reader_other = ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "q_stem_reader.cpp"),
                core_ranges=_crs(cores),
                compile_time_args=reader_ct_other,
                runtime_args=_rt_subset(reader_rt_other, cores),
                config=_dm("RISCV_1", ttnn.NOC.NOC_1 if wnoc == 0 else ttnn.NOC.NOC_0),
            )
            ttnn.attach_absent(reader_other, "mcast")
            kernels_tail.append(reader_other)

        compute_k = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "q_stem_compute.cpp"),
            core_ranges=compute_rect,
            compile_time_args=[
                CB_X_ROUND,
                CB_QB_SLICE,
                CB_Q_SLICE,
                CB_Q_PARTIAL,
                CB_LANDING,
                CB_WKV,
                CB_PREFIX_OUT,
                CB_ROPE_OUT,
                geo.Q_t,
                geo.s1_kb,
                geo.nkb,
                geo.m,
                geo.num_blocks,
                geo.rows_per_head,
                geo.Dn_t,
                geo.Dr_t,
                geo.N_t,
                geo.s3n,
                geo.n_nb,
                int(geo.wkv_resident),
                geo.sbw3,
                int(geo.out_rm),
                CB_PREFIX_RM,
                CB_ROPE_RM,
                int(geo.full_stick),
                geo.ts,
            ],
            runtime_args=compute_rt,
            config=compute_config,
        )
        writer_ct = [
            CB_Q_SLICE,
            CB_PREFIX_OUT,
            CB_ROPE_OUT,
            geo.tile_bytes,
            geo.num_blocks,
            geo.H,
            geo.N_t,
            geo.Dr_t,
            geo.s3n,
            geo.n_nb,
            2 * L,
            SEM_BLOCK_READY,
            *geo_ct,
            int(geo.out_rm),
            geo.out_page_bytes,
            CB_PREFIX_RM,
            CB_ROPE_RM,
            int(geo.full_stick),
            geo.ts,
            *out_ct,
        ]
        writer_k = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "q_stem_writer.cpp"),
            core_ranges=s1_rect,
            compile_time_args=writer_ct,
            runtime_args=writer_rt_s1,
            config=_dm("RISCV_0", ttnn.NOC.NOC_0 if S1_WRITER_NOC == 0 else ttnn.NOC.NOC_1),
        )
        for wnoc, cores in sorted(free_groups.items()):
            kernels_tail.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "q_stem_writer.cpp"),
                    core_ranges=_crs(cores),
                    compile_time_args=writer_ct,
                    runtime_args=_rt_subset(writer_rt_other, cores),
                    config=_dm("RISCV_0", ttnn.NOC.NOC_0 if wnoc == 0 else ttnn.NOC.NOC_1),
                )
            )
        port_reader_k = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "q_stem_port_reader.cpp"),
            core_ranges=port_crs,
            compile_time_args=[
                CB_PORT_STAGE,
                CB_Q_SLICE,
                geo.tile_bytes,
                geo.C1,
                SEM_BLOCK_READY,
                *geo_ct,
                PORT_BATCH_SEGMENTS,
            ],
            runtime_args=port_reader_rt,
            config=_dm("RISCV_1", ttnn.NOC.NOC_0),
        )
        port_sender_k = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "q_stem_port_sender.cpp"),
            core_ranges=port_crs,
            compile_time_args=[
                CB_PORT_STAGE,
                geo.tile_bytes,
                pkt_tiles,
                PORT_BATCH_SEGMENTS,
                hdr_per_dest,
                *geo_ct,
                MAX_HEADERS,
                geo.ts,
            ],
            runtime_args=port_sender_rt,
            config=_dm("RISCV_0", ttnn.NOC.NOC_1),
        )

        mc = ttnn.Mcast2D(
            mesh,
            s1_rect,
            ttnn.Mcast2DFixedSenderConfig(injector),
            ttnn.McastConfig(noc=ttnn.NOC.NOC_0, handshake=True),
        )
        mc.attach(program, "mcast", [injector_k, reader_s1])
        program.kernels = [
            injector_reader_k,
            injector_k,
            reader_s1,
            *kernels_tail,
            compute_k,
            writer_k,
            port_reader_k,
            port_sender_k,
        ]
        # Perf-breakdown hook (verification_report.md): QSTEM_ABLATE=<STAGE,...> stubs stage payloads (wrong output
        # by design). Kept for the perf refinements' per-stage breakdown; part of custom_program_hash.
        ablate = [a for a in os.environ.get("QSTEM_ABLATE", "").split(",") if a]
        if ablate:
            for kd in program.kernels:
                kd.defines = [(f"ABLATE_{a}", "1") for a in ablate]
        r, c = coord
        mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = program
    return mesh_desc, scratch


def run_q_stem(x, q_b, wkv, *, cluster_axis, num_links, compute_config, output_layout=ttnn.TILE_LAYOUT):
    mesh = x.device()
    B, _, S, _ = (int(v) for v in x.shape)
    _, H, Dn, N = (int(v) for v in wkv.shape)
    G = tuple(mesh.shape)[cluster_axis]
    Dr = int(q_b.shape[3]) // (H // G) - Dn
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([B, H, S // G, N + Dr]), ttnn.bfloat16, output_layout, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    desc, scratch = create_mesh_program_descriptor(x, q_b, wkv, out, compute_config, cluster_axis, num_links)
    return ttnn.generic_op([x, q_b, wkv, scratch, out], desc)
