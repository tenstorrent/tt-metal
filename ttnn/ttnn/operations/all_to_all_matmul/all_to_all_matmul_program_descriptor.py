# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""all_to_all_matmul — host plan and MeshProgramDescriptor (regime R1 head-resident, transport T1 direct).

Per chip p of a group (ring of G chips along cluster_axis):

  port cores (2 * num_links, one per (direction d, link l)):
    reader  (RISCV_1, NoC0)  reads each entry (source-local head j, destination chip t: the quarter t of head j's rows,
                             all batches, all K) from the input, one CB page = one fabric packet, batched.
    sender  (RISCV_0, NoC1)  sends readies to every port that writes into this chip; sends the entries in runs per
                             destination chip (nearest first; Refinement 4), waiting for a destination's ready just
                             before its first run, straight into the destination compute core's landing slot (same L1
                             address on every chip); full packets across the entries of a run, the packet that ends an
                             entry fused with an increment of that slot's arrival counter.
  compute cores (grid minus every chip's port cores; heads split across them, the same map on every chip):
    reader  (RISCV_1, NoC1)  per head slot (local-source heads first, then remote heads by source-local index j):
                             weight block 0 from DRAM, then the slot (local: DRAM read of the own quarter; remote:
                             wait for the arrival counter, re-arm it), then the remaining weight blocks.
    compute (TRISC)          matmul_block per (slot, n-block): A slot retained across n-blocks, popped after.
    writer  (RISCV_0, NoC0)  output row groups to the output pages.
  tail split (OVERLAP): up to TAIL_SPLIT_MAX_HEADS of a chip's last-landing remote heads are halved over N; the
                             primary's reader forwards the landed rows to a helper core's extra slot (+ arrival inc).

Blocking knobs (op_design.md "Blocking Model"), each defined once here: BLOCK_HEADS, WEIGHT_BLOCK_BYTES (-> block_n),
WEIGHT_DEPTH, WEIGHT_READ_CHUNK_BYTES, OUT_DEPTH, PORT_BATCH_BYTES, A_SLOT_BUDGET_BYTES; block_m = B * Mq_t and block_k = Kt (R1).
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# ---- Blocking knobs (single source of truth) ----
BLOCK_HEADS = 1  # heads per compute block: a head shares no operand with another head (op_design.md, Axes)
WEIGHT_BLOCK_BYTES = 256 * 1024  # bytes of one weight block (Kt x block_n tiles) -> block_n
WEIGHT_DEPTH = 2  # weight blocks in cb_weight
# Weight blocks of a slot read before its landing wait (capped at the cb_weight depth); the rest are read after it.
# 1 = only the first n-block before the landing (Refinements 0-4).
WEIGHT_PREFETCH_BLOCKS = 1
# Paced weight reads (Refinement 5): a core's first weight read starts WEIGHT_PACE_CYCLES x its head's estimated landing
# share (0 for local heads) after the program starts, so the cores' weight bursts are spread over the exchange instead
# of all hitting DRAM / NoC1 in its first ~25 us. 0 = off (every core starts at once).
WEIGHT_PACE_CYCLES = 0
# Weight bytes in flight per compute core (one read barrier per chunk). OVERLAP (Refinement 3): the whole 128 KiB block
# in flight on 96 cores stretched the fabric exchange from 49 to 69 us on Kimi; harness, Kimi model case, fastest chip:
# whole block 80.8, 64 KiB 76.4, 32 KiB 64.1, 16 KiB 64.2, 8 KiB 64.3, 2 KiB 63.6 us. 16 KiB sits mid-plateau whether
# the limit is bytes or reads in flight (bf16: 8 tiles, bfp8: 15 tiles). >= the block size disables the throttle.
WEIGHT_READ_CHUNK_BYTES = 16 * 1024
OUT_DEPTH = 2  # output row groups in cb_out
# Output bytes in flight per compute core (writer twin of WEIGHT_READ_CHUNK_BYTES, Refinement 5): one write barrier per
# chunk instead of one flush per row group. 0 = off (flush per row group).
OUT_WRITE_CHUNK_BYTES = 0
# Packets per DRAM-read barrier / per sender flush on the port cores = PORT_BATCH_BYTES // packet bytes (cb_port_chunks
# holds two batches). Phase 0: 8 beat 4 packets at 7232 B. Refinement 4 (grouped send order, Kimi model case,
# exchange-alone / whole op fastest chip): 8 packets 47.3 / 56.2, 12 packets 46.2 / 55.0-55.6 us. In bytes so the
# 14 KiB-payload cells get 6 packets (a smaller CB than the 8 x 14 KiB before).
PORT_BATCH_BYTES = 87 * 1024
# Port reader DRAM reads in flight (port_prefetch, Perf 2): the reads of the next ceil(PORT_READ_INFLIGHT_BYTES / packet
# bytes) packets (1..15, one NoC transaction id each) stay outstanding and each packet is pushed as soon as its own reads
# land, instead of one barrier per batch (the reads drained at every barrier, and under the compute cores' weight reads
# the heavy ports' senders waited ~11 us per call on the reader). Kimi model case, harness fastest / slowest chip:
# per-batch barrier 53.5 / 60.4; 2 / 3 / 4 / 6 / 8 / 12 packets 56.4 / 51.9 / 51.9 / 51.9 / 52.1 / 52.3 (slowest
# 67.7 / 58.4 / 57.2 / 57.4 / 57.4 / 57.7) us. A deeper cb_port_chunks (5 batches, or whole batches in flight) was null
# or worse: the reader is throughput-bound, not ring-bound.
PORT_READ_INFLIGHT_BYTES = 40 * 1024
# TRANSPORT (Refinement 4), port send schedule:
# - SEND_GROUPED_BY_DEST / SEND_RUN_ENTRIES: a port sends runs of SEND_RUN_ENTRIES consecutive entries per destination
#   chip, the destinations' runs alternating, nearest destination first (SEND_DEST_ORDER); else one entry per run (heads
#   outer, destinations interleaved, the Phase 0 order). Runs are what let packing act. Kimi model case, harness fastest
#   chip, whole op / exchange alone: run 1 60.4 / 50.5 (before), 2 56.9 / -, 3 55.7 / 45.8, 6 53.9 (slowest 65.6) /
#   46.5, all 56.2 / 47.3 us. 3 entries = 60 tiles = 20 full packets at 7232 B.
#   Path gate SEND_RUN_MAX_LOCAL_RATIO: runs change which heads land last; where the chip's own DRAM traffic (weight +
#   input + DRAM output) is large next to the exchanged input, the interleaved landing hides more of it and runs cost
#   1.3-1.5 us (Kimi Phase 0, DRAM output, ratio 8.2: run 1 87.6, run 2 / 3 / 6 / all 89.0 / 89.0 / 90.7 / 91.2 us), so
#   runs apply only up to this ratio (the model configs: 4.2).
# - FENCE_PER_DEST: a port waits only for the ready of the destination it is about to send to (one ready counter per
#   destination), so the 1-hop group starts after a 1-hop round trip and the 2-hop ready is awaited behind it (else
#   every ready before the first packet).
# - PACK_ACROSS_ENTRIES: a destination group is one stream of full packets; a packet straddling two entries is a fused
#   scatter write + increment (all_to_all_matmul_stream.hpp). Kimi 7232 B: 20 tiles per entry, 7 -> 6.67 packets per
#   entry. Needs packet_tiles <= slot_tiles (else off).
SEND_GROUPED_BY_DEST = True
SEND_RUN_ENTRIES = 3  # grouped: entries per destination run (runs of the destinations alternate); None = all
SEND_RUN_MAX_LOCAL_RATIO = 6.0
SEND_DEST_ORDER = "near"  # destination group order on a port: "near" (fewest hops first), "far", "legacy"
FENCE_PER_DEST = True
PACK_ACROSS_ENTRIES = True
# Tail split (OVERLAP, Refinement 3): the last-landing remote heads are multiplied half by their own core and half by a
# helper (an idle compute core, or a core whose heads are all local on that chip and so finish before the exchange
# does; idle cores are used first). The primary forwards the landed rows to the helper over the NoC. None = one split
# head per available helper, 0 = off (one core per head, the Phase 0 schedule), n = at most n split heads per chip.
# Harness, fastest chip, split heads 0 / 4 / 8 / 16 / all: Kimi model 64.1 / - / 60.5 / 60.3 / 60.5; Kimi Phase 0 (DRAM out,
# HiFi4) 91.0 / 87.2 / 87.6 / 92.0 / 91.5; DeepSeek 81.0 / - / 77.0 / 77.2 / 77.3; GLM 65.0 / - / 60.1 / 60.2 / 62.2.
# Only the last few landings are on the tail; past ~8 the extra helpers are local-head cores that finish their own
# head late (HiFi4) and become the new tail.
TAIL_SPLIT_MAX_HEADS = 8
# Grouped send path (local ratio <= SEND_RUN_MAX_LOCAL_RATIO): with the M-split taking the latest heads, more N-split
# heads help (tail_chain experiment, Kimi model case: 8 -> 12 with 4 M-split heads, 55.7 -> 54.4 us fastest). On the
# interleaved path (Kimi Phase 0, ratio 8.2, HiFi4) every extra helper slot on a local-head core costs (12 split heads:
# 87.6 -> 91.2 us), so that path keeps TAIL_SPLIT_MAX_HEADS and no M-split.
TAIL_SPLIT_MAX_HEADS_GROUPED = 12
# Landing-order estimate for the tail split: "share" = (index + 1) / entries of the source port (ports carry different
# entry counts, 12 or 24 at Kimi, and all finish together), "index" = raw index in the port's send list (Refinement 3).
TAIL_SPLIT_ESTIMATE = "share"
# Helper choice for a split head: "pool" = idle cores first, then local-only cores (Refinement 3); "nearest" = the free
# helper closest to the primary core (shortest forward path).
TAIL_SPLIT_HELPER_PICK = "pool"
# Tail M-split (OVERLAP, tail_chain experiment): the TAIL_MSPLIT_HEADS latest-landing remote heads of a chip are cut by
# output rows into TAIL_MSPLIT_PIECES pieces. The SOURCE port delivers each piece's rows straight into its core's slot
# (primary = piece 0, helpers = the rest) with its own arrival increment: no forward hop, and the earlier pieces start
# their matmul while the entry's later rows are still on the link. Needs packing (a piece is a sub-entry of the packet
# stream) and piece tiles >= packet tiles. Piece slots read all their weight n-blocks before the landing wait
# (TAIL_MSPLIT_PREFETCH, capped at the cb_weight depth). 0 = off.
TAIL_MSPLIT_HEADS = 4
TAIL_MSPLIT_PIECES = 3
TAIL_MSPLIT_PREFETCH = 2
A_SLOT_BUDGET_BYTES = 384 * 1024  # predicate A_RESIDENT (regime R1; else R3, deferred)
MAX_HEADS_PER_CORE = 8  # global arrival semaphores per compute core
MAX_PACKET_HEADERS = 12  # packet header pool per data-movement RISC

# ---- CB indices ----
CB_A_LANDED = 0
CB_WEIGHT = 1
CB_OUT = 2
CB_PORT_CHUNKS = 3
CB_RETILE_STAGE = 4  # regime R1' only: one image's source tile rows of a quarter (scratch)

FWD, BWD = 0, 1
NOC0 = ttnn.NOC.RISCV_1_default
NOC1 = ttnn.NOC.RISCV_0_default
# Compute cores: weight / local-row reads on NoC1, output writes on NoC0, which keeps NoC0's read-response path free
# for the port readers (NoC0) feeding the fabric. Measured on the Kimi Phase 0 case (QuietBox 2, 7232 B):
# 134.0 -> 99.4 us fastest chip (reader NoC0 / writer NoC1 before); with 8-packet port batches: 94.5 us.
COMPUTE_READER_NOC = NOC1
COMPUTE_WRITER_NOC = NOC0
# Output in L1 interleaved (Refinement 1 re-check, Kimi model case, harness): NoC1 fastest 81.01 -> 78.53 us but slowest
# 84.92 -> 86.15 us (beyond noise) -> rejected by eval.ccl_perf compare; kept on NoC0 as a live knob.
COMPUTE_WRITER_NOC_L1_OUT = NOC0
PORT_READER_NOC = NOC0  # NoC1 (shared with the sender) hung
PORT_SENDER_NOC = NOC1  # blackhole-fabric.md rule 7


def _node(mesh, coord):
    return mesh.get_fabric_node_id(ttnn.MeshCoordinate(*coord))


# ----------------------------------------------------------------------------------------------------------------------
# Groups, routes, ports
# ----------------------------------------------------------------------------------------------------------------------


def _groups(mesh_shape, cluster_axis):
    rows, cols = mesh_shape
    if cluster_axis == 1:
        return [[(r, c) for c in range(cols)] for r in range(rows)]
    return [[(r, c) for r in range(rows)] for c in range(cols)]


def _first_hop_dir(mesh, src, dst, nxt, prv):
    """FWD / BWD: the ring port whose connection carries the first hop of the fabric's route src -> dst."""
    d = ttnn.get_eth_forwarding_direction(_node(mesh, src), _node(mesh, dst))
    if d is None:
        return None
    for port, nb in ((FWD, nxt), (BWD, prv)):
        if d == ttnn.get_eth_forwarding_direction(_node(mesh, src), _node(mesh, nb)):
            return port
    return None


def _routes(mesh, cluster_axis):
    """{coord: dict(p, G, grp, next, prev, route{t_pos: dir})} for every chip."""
    groups = _groups(tuple(mesh.shape), cluster_axis)
    chips = {}
    for grp in groups:
        G = len(grp)
        for p, coord in enumerate(grp):
            nxt, prv = grp[(p + 1) % G], grp[(p - 1) % G]
            route = {}
            for t in range(G):
                if t == p:
                    continue
                d = _first_hop_dir(mesh, coord, grp[t], nxt, prv)
                if d is None:
                    raise RuntimeError(
                        f"all_to_all_matmul: the fabric route {coord} -> {grp[t]} leaves through neither ring neighbour"
                    )
                route[t] = d
            chips[coord] = dict(p=p, G=G, grp=grp, next=nxt, prev=prv, route=route)
    return chips


# Host-only port placement: the first runtime arg of a fabric connection is its Ethernet channel; the channel's
# physical NoC location comes from the arch SoC descriptor; the port core is the worker with the fewest NoC1 hops to
# it (fabric_all_gather's _worker_nearest_noc1, fed from host data instead of a probe program).
_SOC_CACHE = {}


def _soc(mesh):
    arch = str(mesh.arch()).split(".")[-1]
    if arch not in _SOC_CACHE:
        import yaml

        root = os.environ.get("TT_METAL_HOME", os.getcwd())
        arch_yaml = {"BLACKHOLE": "blackhole_140_arch.yaml", "WORMHOLE_B0": "wormhole_b0_80_arch.yaml"}[arch]
        soc = yaml.safe_load(open(os.path.join(root, "tt_metal", "soc_descriptors", arch_yaml)))
        eth = [tuple(int(v) for v in e.split("-")) for e in soc["eth"]]
        cols = sorted({int(w.split("-")[0]) for w in soc["functional_workers"]})
        grid = (int(soc["grid"]["x_size"]), int(soc["grid"]["y_size"]))
        _SOC_CACHE[arch] = (eth, cols, grid)
    return _SOC_CACHE[arch]


_HARVEST_CACHE = {}


def _chip_maps(mesh, coord):
    """(non-harvested Ethernet physical (x, y) list, {translated worker x: physical x}) of a chip, from the cluster
    descriptor + the arch SoC descriptor. Ethernet channel c is entry c of the non-harvested list (translated x =
    20 + c on Blackhole)."""
    if id(mesh) not in _HARVEST_CACHE:
        import yaml

        desc = yaml.safe_load(open(ttnn.cluster.serialize_cluster_descriptor()))
        arch_eth, tensix_cols, _ = _soc(mesh)
        maps = {}
        for chip, uid in desc.get("chip_unique_ids", {}).items():
            h = desc.get("harvesting", {}).get(chip, {})
            emask = int(h.get("eth_harvesting_mask", 0))
            translated = bool(h.get("noc_translation", False))
            tmask = int(h.get("harvest_mask", 0))
            live = [x for i, x in enumerate(tensix_cols) if not (tmask >> i) & 1]
            t2p = {tensix_cols[k]: px for k, px in enumerate(live)} if translated else {}
            eth = [e for i, e in enumerate(arch_eth) if not (emask >> i) & 1]
            maps[int(uid)] = (eth, t2p)
        _HARVEST_CACHE[id(mesh)] = (mesh, maps)
    node = _node(mesh, coord)
    uid = int(ttnn.cluster.get_chip_unique_id_from_fabric_node_id(int(node.mesh_id), int(node.chip_id)))
    return _HARVEST_CACHE[id(mesh)][1][uid]


def _eth_channel(mesh, src, dst, link):
    """The Ethernet channel of connection src -> dst on `link` (first word of its runtime args; host only)."""
    scratch = ttnn.ProgramDescriptor()
    args = ttnn.setup_fabric_connection(_node(mesh, src), _node(mesh, dst), link, scratch, ttnn.CoreCoord(0, 0))
    return int(list(args)[0])


def _nearest_worker_noc1(mesh, coord, eth_xy, taken, allowed):
    t2p = _chip_maps(mesh, coord)[1]
    _, _, (gx, gy) = _soc(mesh)
    ex, ey = eth_xy
    best = None
    for x, y in sorted(allowed):
        if (x, y) in taken:
            continue
        v = mesh.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        px = t2p.get(v.x, v.x)
        score = ((v.y - ey) % gy + (px - ex) % gx, v.y, px)
        if best is None or score < best[0]:
            best = (score, (x, y))
    return best[1] if best is not None else None


def _fallback_ports(grid, num_links):
    """2 * num_links cores of logical row 0, centred."""
    n = 2 * num_links
    x0 = max(0, grid.x // 2 - num_links)
    return {(d, l): (x0 + 2 * l + d, 0) for l in range(num_links) for d in (FWD, BWD)}


_PLAN_CACHE = {}


def _topology_plan(mesh, cluster_axis, num_links):
    """Routes, per-chip port cores {(d, l): (x, y)} and fabric link indices; cached per mesh (host only)."""
    key = (id(mesh), cluster_axis, num_links, str(ttnn.get_fabric_config()))
    if key in _PLAN_CACHE:
        return _PLAN_CACHE[key][1]
    chips = _routes(mesh, cluster_axis)
    grid = mesh.compute_with_storage_grid_size()
    allowed = {(x, y) for x in range(grid.x) for y in range(grid.y)}
    for coord, ch in chips.items():
        links = {}
        for d, peer in ((FWD, ch["next"]), (BWD, ch["prev"])):
            li = list(ttnn.get_forwarding_link_indices(_node(mesh, coord), _node(mesh, peer)))
            links[d] = li[:num_links]
        ch["links"] = links
        ports = None
        if not os.environ.get("TT_METAL_EMULE_MODE"):
            try:
                eth_list = _chip_maps(mesh, coord)[0]
                taken, ports = set(), {}
                for l in range(num_links):
                    for d, peer in ((FWD, ch["next"]), (BWD, ch["prev"])):
                        chan = _eth_channel(mesh, coord, peer, links[d][l])
                        core = _nearest_worker_noc1(mesh, coord, eth_list[chan], taken, allowed)
                        ports[(d, l)] = core
                        taken.add(core)
            except Exception:  # noqa: BLE001 - placement is a perf choice; fall back to the simple layout
                ports = None
        ch["ports"] = ports or _fallback_ports(grid, num_links)
    _PLAN_CACHE[key] = (mesh, chips)
    return chips


# ----------------------------------------------------------------------------------------------------------------------
# Persistent resources (landing scratch, arrival counters, ready counter): created once per mesh, grow-only
# ----------------------------------------------------------------------------------------------------------------------


def _rects(cells):
    """CoreRangeSet of rectangles covering a set of (x, y) cells (row runs merged vertically)."""
    rows = {}
    for x, y in sorted(cells, key=lambda c: (c[1], c[0])):
        runs = rows.setdefault(y, [])
        if runs and runs[-1][1] == x - 1:
            runs[-1][1] = x
        else:
            runs.append([x, x])
    rects = []  # [x0, x1, y0, y1]
    open_rects = {}
    for y in sorted(rows):
        nxt = {}
        for x0, x1 in rows[y]:
            r = open_rects.get((x0, x1))
            if r is not None and r[3] == y - 1:
                r[3] = y
            else:
                r = [x0, x1, y, y]
                rects.append(r)
            nxt[(x0, x1)] = r
        open_rects = nxt
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for x0, x1, y0, y1 in rects]
    )


class _Resources:
    def __init__(self):
        self.scratch = None
        self.cap_tiles = 0
        self.arrival = []
        self.ready = []


_RES_CACHE = {}


def _resources(mesh, compute_cells, port_cells, cap_tiles, n_arrival, n_ready, dtype):
    key = (id(mesh), tuple(sorted(compute_cells)), tuple(sorted(port_cells)), dtype)
    entry = _RES_CACHE.get(key)
    res = entry[1] if entry is not None else _Resources()
    changed = False
    compute_crs = _rects(compute_cells)
    while len(res.ready) < n_ready:  # one ready counter per destination chip of a port
        res.ready.append(ttnn.create_global_semaphore(mesh, _rects(port_cells), 0))
        changed = True
    while len(res.arrival) < n_arrival:
        res.arrival.append(ttnn.create_global_semaphore(mesh, compute_crs, 0))
        changed = True
    if res.cap_tiles < cap_tiles:
        n = len(compute_cells)
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(compute_crs, (cap_tiles * 32, 32), ttnn.ShardOrientation.ROW_MAJOR),
        )
        res.scratch = None
        res.scratch = ttnn.allocate_tensor_on_device(
            ttnn.Shape([n * cap_tiles * 32, 32]), dtype, ttnn.TILE_LAYOUT, mesh, mc
        )
        res.cap_tiles = cap_tiles
        changed = True
    if changed:
        # every chip's counters are zero before any chip's first increment can arrive
        ttnn.synchronize_device(mesh)
    _RES_CACHE[key] = (mesh, res)
    return res


# ----------------------------------------------------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------------------------------------------------


def _subblock(block_m, block_n, fp32_dest_acc_en):
    """(out_subblock_h, out_subblock_w): largest product within the DEST limit, h | block_m, w | block_n, wide first."""
    limit = 4 if fp32_dest_acc_en else 8
    best = (1, 1)
    for h in range(1, min(block_m, limit) + 1):
        if block_m % h:
            continue
        for w in range(1, min(block_n, limit) + 1):
            if block_n % w or h * w > limit:
                continue
            if (h * w, w) > (best[0] * best[1], best[1]):
                best = (h, w)
    return best


def _block_n(Kt, Nt, w_tile_bytes):
    if Kt * w_tile_bytes > WEIGHT_BLOCK_BYTES:
        raise RuntimeError(
            f"all_to_all_matmul: Kt * weight_tile_bytes = {Kt * w_tile_bytes} > WEIGHT_BLOCK_BYTES "
            f"({WEIGHT_BLOCK_BYTES}): regime R2 (K-blocked weight) is not built"
        )
    return max(d for d in range(1, Nt + 1) if Nt % d == 0 and Kt * d * w_tile_bytes <= WEIGHT_BLOCK_BYTES)


def _split_heads(compute_cores, H):
    """Contiguous heads per core (split_work_to_cores semantics): group 1 ceil(H/C), group 2 floor(H/C)."""
    C = len(compute_cores)
    q, rem = divmod(H, C)
    out, h = [], 0
    for i, core in enumerate(compute_cores):
        n = q + (1 if i < rem else 0)
        if n == 0:
            break
        out.append((core, list(range(h, h + n))))
        h += n
    return out


def _hash(key):
    return int.from_bytes(hashlib.sha1(key.encode()).digest()[:8], "little") >> 1


# ----------------------------------------------------------------------------------------------------------------------
# Program
# ----------------------------------------------------------------------------------------------------------------------


def run(input_tensor, weight, *, cluster_axis, num_links, memory_config, compute_kernel_config):
    mesh = input_tensor.device()
    B, Hl, S, K = tuple(input_tensor.shape)
    _, H, _, N = tuple(weight.shape)
    chips = _topology_plan(mesh, cluster_axis, num_links)
    G = next(iter(chips.values()))["G"]

    # ---- geometry (alignment-aware: R1 tile_aligned, R1' m_non_aligned re-tiles the quarters in the readers) ----
    Kt, Nt = K // 32, N // 32
    Mq = S // G
    Mq_t = (Mq + 31) // 32
    St = (S + 31) // 32
    retile = Mq % 32 != 0  # regime R1': the quarter starts mid-tile; kernels derive the same flag from Mq
    stage_tiles = (Mq_t + 1) * Kt  # source tile rows one image's quarter can span, all K
    block_m = B * Mq_t  # batch folded into the row axis; the whole head's rows in one block
    slot_tiles = block_m * Kt
    in_tile = int(input_tensor.buffer_page_size())
    w_tile = int(weight.buffer_page_size())
    block_n = _block_n(Kt, Nt, w_tile)
    num_n_blocks = Nt // block_n
    sbh, sbw = _subblock(block_m, block_n, bool(compute_kernel_config.fp32_dest_acc_en))
    payload = int(ttnn.get_tt_fabric_max_payload_size_bytes())
    packet_tiles = payload // in_tile
    assert packet_tiles >= 1
    packet_bytes = packet_tiles * in_tile
    assert BLOCK_HEADS == 1, "block_heads > 1 is a knob-turn not yet wired"

    # ---- per chip: who sends what (entries), readies ----
    L = num_links
    # Measurement hook (off by default): A2AMM_XP=group=0,fence=0,pack=0 overrides the TRANSPORT send-schedule knobs.
    _xp = dict(kv.split("=") for kv in os.environ.get("A2AMM_XP", "").split(",") if "=" in kv)
    in_bytes = B * Hl * St * Kt * in_tile  # per chip: the input is read once and (G-1)/G of it exchanged
    dram_bytes = in_bytes + H * Kt * Nt * w_tile
    if memory_config.buffer_type == ttnn.BufferType.DRAM:
        dram_bytes += B * H * Mq_t * Nt * in_tile
    local_ratio = dram_bytes / in_bytes
    grouped = bool(int(_xp.get("group", SEND_GROUPED_BY_DEST and local_ratio <= SEND_RUN_MAX_LOCAL_RATIO)))
    send_run = (None if _xp["run"] == "all" else int(_xp["run"])) if "run" in _xp else SEND_RUN_ENTRIES
    fence_per_dest = bool(int(_xp.get("fence", FENCE_PER_DEST)))
    pack = bool(int(_xp.get("pack", PACK_ACROSS_ENTRIES))) and packet_tiles <= slot_tiles
    dest_order = _xp.get("order", SEND_DEST_ORDER)
    batch_chunks = int(_xp.get("batch", max(1, PORT_BATCH_BYTES // packet_bytes)))
    port_window = int(_xp.get("pinfl", min(15, max(1, -(-PORT_READ_INFLIGHT_BYTES // packet_bytes)))))
    tail_est = _xp.get("est", TAIL_SPLIT_ESTIMATE)
    tail_pick = _xp.get("pick", TAIL_SPLIT_HELPER_PICK)
    tail_max = (
        int(_xp["tail"]) if "tail" in _xp else (TAIL_SPLIT_MAX_HEADS_GROUPED if grouped else TAIL_SPLIT_MAX_HEADS)
    )
    ms_heads = int(_xp.get("ms", TAIL_MSPLIT_HEADS if grouped else 0))
    ms_pieces = int(_xp.get("mp", TAIL_MSPLIT_PIECES))
    ms_pre = int(_xp.get("mpre", TAIL_MSPLIT_PREFETCH))
    rd_noc, wr_noc = COMPUTE_READER_NOC, (
        COMPUTE_WRITER_NOC_L1_OUT if memory_config.buffer_type == ttnn.BufferType.L1 else COMPUTE_WRITER_NOC
    )
    if _xp.get("cnoc") == "swap":  # measurement hook: the two compute data-movement RISCs exchange NoCs
        rd_noc, wr_noc = wr_noc, rd_noc
    assert rd_noc != wr_noc, "compute reader and writer share the NoC's transaction counters on one NoC"

    def _dest_key(p, t, G_):
        hops = min((t - p) % G_, (p - t) % G_)
        return {"near": (hops, t), "far": (-hops, t), "legacy": ((t - p) % G_, t)}[dest_order]

    sends = {}  # (coord, d, l) -> [(j, t_pos)]
    for coord, ch in chips.items():
        p, G_, grp = ch["p"], ch["G"], ch["grp"]
        for d in (FWD, BWD):
            dests = sorted((t for t, dd in ch["route"].items() if dd == d), key=lambda t: _dest_key(p, t, G_))
            for l in range(L):
                heads = list(range(l, Hl, L))
                run = len(heads) if grouped and send_run is None else (send_run if grouped else 1)
                run = max(1, run)
                sends[(coord, d, l)] = [
                    (j, t) for r0 in range(0, len(heads), run) for t in dests for j in heads[r0 : r0 + run]
                ]

    # ---- cores ----
    grid = mesh.compute_with_storage_grid_size()
    port_cells = {xy for ch in chips.values() for xy in ch["ports"].values()}
    compute_cells_list = [(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in port_cells]
    assign = _split_heads(compute_cells_list, H)  # [(core xy, [heads])], same on every chip
    core_of = {h: xy for xy, hs in assign for h in hs}
    heads_of = {xy: hs for xy, hs in assign}

    def slot_order(q, hs):
        local = [h for h in hs if h // Hl == q]
        remote = sorted((h for h in hs if h // Hl != q), key=lambda h: (h % Hl, h // Hl))
        return local + remote

    # slot index of head h on its core, as seen on chip position q (the sender's landing address; helper slots of the
    # tail split come after a core's own slots, so they never move these)
    slot_of = {}
    for q in range(G):
        for xy, hs in assign:
            for s, h in enumerate(slot_order(q, hs)):
                slot_of[(q, h)] = s

    # ---- estimated landing of every remote head: the share of its source port's entries sent up to and including
    #      it (ports finish together), in (0, 1]; local heads land at 0 ----
    def landing_share(coord, h):
        ch = chips[coord]
        q, grp = ch["p"], ch["grp"]
        if h // Hl == q:
            return 0.0
        j, src = h % Hl, grp[h // Hl]
        port_sends = sends[(src, chips[src]["route"][q], j % L)]
        return (port_sends.index((j, q)) + 1) / len(port_sends)

    # ---- tail split plan (per chip): late-landing remote head -> helper core ----
    def tail_plan(coord):
        ch = chips[coord]
        q, grp = ch["p"], ch["grp"]
        # (estimated landing: the share, or the raw index in its source port's list ("index"), slots of its core, h)
        est = []
        for xy, hs in assign:
            for h in hs:
                p = h // Hl
                if p == q:
                    continue
                j, src = h % Hl, grp[h // Hl]
                port_sends = sends[(src, chips[src]["route"][q], j % L)]
                idx = port_sends.index((j, q))
                est.append(((idx if tail_est == "index" else landing_share(coord, h)), len(hs), h))
        est.sort(reverse=True)
        idle = [xy for xy in compute_cells_list if xy not in heads_of]
        # a local-head helper gets one slot more than its heads: only where that still fits A_RESIDENT and the counters
        fits = lambda hs: (len(hs) + 1) * slot_tiles * in_tile <= A_SLOT_BUDGET_BYTES and len(hs) < MAX_HEADS_PER_CORE
        local_only = sorted(
            (xy for xy, hs in assign if all(h // Hl == q for h in hs) and fits(hs)), key=lambda xy: len(heads_of[xy])
        )
        helpers = idle + local_only
        # M-split first: the ms latest heads, (pieces - 1) helpers each (pool order)
        mplan = {}
        if m_rows:
            for _, _, h in est[:ms_heads]:
                need = len(m_rows) - 1
                if len(helpers) < need:
                    break
                mplan[h] = [None] + helpers[:need]  # piece 0 stays on the primary core
                helpers = helpers[need:]
        est = [e for e in est if e[2] not in mplan]
        n = len(helpers) if tail_max is None else min(len(helpers), tail_max)
        if tail_pick == "pool":
            return {h: helpers[i] for i, (_, _, h) in enumerate(est[:n])}, mplan
        # "nearest": latest-landing head first, each to the free helper with the shortest forward path
        free, plan = list(helpers), {}
        for _, _, h in est[:n]:
            px, py = core_of[h]
            hx = min(free, key=lambda c: (abs(c[0] - px) + abs(c[1] - py), helpers.index(c)))
            free.remove(hx)
            plan[h] = hx
        return plan, mplan

    # M-split row pieces (rows of block_m, multiples of the subblock height, each >= one packet): computed after the
    # final block_n below; the plan only needs the piece count, so fix it from block_m here.
    def _m_rows(pieces, sb_h):
        if ms_heads <= 0 or pieces < 2 or not pack:
            return []
        units = block_m // sb_h
        pieces = min(pieces, units)
        rows, u0 = [], 0
        for i in range(pieces):
            u1 = u0 + units // pieces + (1 if i < units % pieces else 0)
            rows.append((u0 * sb_h, u1 * sb_h))
            u0 = u1
        if any((r1 - r0) * Kt < packet_tiles for r0, r1 in rows):
            return []
        return rows

    # the subblock height under the halved block_n the split forces (same rule as below)
    _bn_split = block_n // 2 if (num_n_blocks == 1 and block_n % 2 == 0) else block_n
    m_rows = _m_rows(ms_pieces, _subblock(block_m, _bn_split, bool(compute_kernel_config.fp32_dest_acc_en))[0])
    _plans = {coord: tail_plan(coord) for coord in chips}
    split_of = {coord: v[0] for coord, v in _plans.items()}  # {coord: {h: helper xy}}
    msplit_of = {coord: v[1] for coord, v in _plans.items()}  # {coord: {h: [None, helper xy, ...]}}
    tail_split = any(split_of.values()) or any(msplit_of.values())
    if tail_split and num_n_blocks == 1:
        if block_n % 2:
            tail_split = False  # nothing to halve
        else:
            block_n //= 2
            num_n_blocks = 2
            sbh, sbw = _subblock(block_m, block_n, bool(compute_kernel_config.fp32_dest_acc_en))
    if not tail_split:
        split_of = {coord: {} for coord in chips}
        msplit_of = {coord: {} for coord in chips}
    assert (
        not any(msplit_of.values())
        or sbh == _subblock(block_m, _bn_split, bool(compute_kernel_config.fp32_dest_acc_en))[0]
    )
    if "sb" in _xp:  # measurement hook: out_subblock_h x out_subblock_w ("hxw"), must tile the block
        sbh, sbw = (int(v) for v in _xp["sb"].split("x"))
        assert block_m % sbh == 0 and block_n % sbw == 0
        assert not any(msplit_of.values()) or all(r % sbh == 0 for rr in m_rows for r in rr)
    nb_split = num_n_blocks // 2  # the primary keeps n-blocks [0, nb_split), the helper [nb_split, num_n_blocks)

    # slots per (coord, core): [(h, wait, nb0, nb1, helper xy or None, m0, m1, piece)]  piece = M-split piece index or -1
    slots = {}
    for coord, ch in chips.items():
        q, sp, mp = ch["p"], split_of[coord], msplit_of[coord]
        for xy in compute_cells_list:
            own = []
            for h in slot_order(q, heads_of.get(xy, [])):
                if h in mp:
                    own.append((h, True, 0, num_n_blocks, None, m_rows[0][0], m_rows[0][1], 0))
                else:
                    own.append((h, h // Hl != q, 0, nb_split if h in sp else num_n_blocks, sp.get(h), 0, block_m, -1))
            own += [(h, True, nb_split, num_n_blocks, None, 0, block_m, -1) for h, hx in sp.items() if hx == xy]
            own += [
                (h, True, 0, num_n_blocks, None, m_rows[i][0], m_rows[i][1], i)
                for h, hxs in mp.items()
                for i, hx in enumerate(hxs)
                if i > 0 and hx == xy
            ]
            if own:
                slots[(coord, xy)] = own
    helper_slot = {
        (coord, h): next(i for i, e in enumerate(slots[(coord, hx)]) if e[0] == h and e[2] == nb_split and e[7] < 0)
        for coord, sp in split_of.items()
        for h, hx in sp.items()
    }
    # (destination coord, h) -> [(core xy, slot index, tiles)] per piece, in stream order
    pieces_of = {}
    for coord, mp in msplit_of.items():
        for h, hxs in mp.items():
            lst = []
            for i, hx in enumerate(hxs):
                xy = core_of[h] if hx is None else hx
                si = next(k for k, e in enumerate(slots[(coord, xy)]) if e[0] == h and e[7] == i)
                lst.append((xy, si, (m_rows[i][1] - m_rows[i][0]) * Kt))
            pieces_of[(coord, h)] = lst
    slots_per_core_max = max(len(v) for v in slots.values())
    if slots_per_core_max > MAX_HEADS_PER_CORE:
        raise RuntimeError(f"all_to_all_matmul: {slots_per_core_max} head slots per core > {MAX_HEADS_PER_CORE}")
    # cb_weight depth: WEIGHT_DEPTH is the knob, but a core never holds more weight blocks than it processes per call
    # (Kimi: 1 head x 1 n-block -> one 128 KiB block; a second buffer would never be live).
    weight_depth = min(WEIGHT_DEPTH, max(sum(e[3] - e[2] for e in v) for v in slots.values()))
    weight_chunk_tiles = max(1, int(_xp.get("wchunk", WEIGHT_READ_CHUNK_BYTES)) // w_tile)
    w_prefetch = max(1, min(weight_depth, int(_xp.get("wpre", WEIGHT_PREFETCH_BLOCKS))))
    w_pace = int(_xp.get("wpace", WEIGHT_PACE_CYCLES))
    out_chunk_bytes = int(_xp.get("ochunk", OUT_WRITE_CHUNK_BYTES))
    out_chunk_tiles = out_chunk_bytes // in_tile if out_chunk_bytes else 0
    cap_tiles = slots_per_core_max * slot_tiles
    if cap_tiles * in_tile > A_SLOT_BUDGET_BYTES:
        raise RuntimeError(
            f"all_to_all_matmul: A_RESIDENT fails ({cap_tiles * in_tile} B of landed rows per core > "
            f"{A_SLOT_BUDGET_BYTES}): regime R3 (DRAM landing) is not built"
        )
    data_dests = {k: sorted({t for _, t in e}) for k, e in sends.items()}  # ready counter k <-> k-th data destination
    n_ready = max(1, max(len(v) for v in data_dests.values()))
    res = _resources(
        mesh, set(compute_cells_list), port_cells, cap_tiles, slots_per_core_max, n_ready, input_tensor.dtype
    )
    landing_addr = int(res.scratch.buffer_address())
    slot_bytes = slot_tiles * in_tile
    arrival_addrs = [int(ttnn.get_global_semaphore_address(s)) for s in res.arrival[:slots_per_core_max]]
    ready_addrs = [int(ttnn.get_global_semaphore_address(s)) for s in res.ready[:n_ready]]
    # kernels + compute CBs: the cores holding heads, plus every compute core when the tail split may use idle ones
    active_cells = list(compute_cells_list) if tail_split else [xy for xy, _ in assign]
    active_crs = _rects(active_cells)
    virt = lambda xy: mesh.worker_core_from_logical_core(ttnn.CoreCoord(*xy))

    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([B, H, Mq, N]), input_tensor.dtype, ttnn.TILE_LAYOUT, mesh, memory_config
    )
    in_addr, w_addr, out_addr = (int(t.buffer_address()) for t in (input_tensor, weight, output))
    in_ct = list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    w_ct = list(ttnn.TensorAccessorArgs(weight).get_compile_time_args())
    out_ct = list(ttnn.TensorAccessorArgs(output).get_compile_time_args())

    compute_cfg = ttnn.ComputeConfigDescriptor()
    compute_cfg.math_fidelity = compute_kernel_config.math_fidelity
    compute_cfg.fp32_dest_acc_en = bool(compute_kernel_config.fp32_dest_acc_en)
    compute_cfg.math_approx_mode = bool(compute_kernel_config.math_approx_mode)

    ready_out = {k: [] for k in sends}  # (coord, d, l) -> [(target coord, target port (d, l), ready counter index)]
    for (coord, d, l), dests_ in data_dests.items():
        ch = chips[coord]
        for g, t in enumerate(dests_):
            tc = ch["grp"][t]
            back = chips[tc]["route"][ch["p"]]
            ready_out[(tc, back, l)].append((coord, (d, l), g))

    cfg_key = "|".join(
        str(v)
        for v in (
            tuple(mesh.shape),
            cluster_axis,
            L,
            (B, Hl, S, K, H, N),
            input_tensor.dtype,
            weight.dtype,
            memory_config.buffer_type,
            compute_cfg.math_fidelity,
            compute_cfg.fp32_dest_acc_en,
            compute_cfg.math_approx_mode,
            packet_tiles,
            pack,
            fence_per_dest,
            grouped,
            send_run,
            dest_order,
            batch_chunks,
            tail_max,
            tail_est,
            weight_chunk_tiles,
            out_chunk_tiles,
            w_prefetch,
            w_pace,
            tuple(sorted(_xp.items())),
            block_n,
            sbh,
            sbw,
            res.cap_tiles,
            landing_addr,
            tuple(sorted((str(c), sorted(sp.items())) for c, sp in split_of.items())),
            tuple(sorted((str(c), sorted((h, str(v)) for h, v in mp.items())) for c, mp in msplit_of.items())),
            tuple(m_rows),
            sorted(port_cells),
        )
    )

    # Measurement hook (off by default): A2AMM_ABLATE=nowrite,noweight,nocompute,noportread,nosend,zones compiles the named
    # payload stubs / device zones into the kernels (ablation profiling; never set in production).
    _abl = [a for a in os.environ.get("A2AMM_ABLATE", "").split(",") if a]
    _defs = [("ABLATE_" + a.upper(), "1") for a in _abl]
    cfg_key += "|" + ",".join(_abl)
    mesh_desc = ttnn.MeshProgramDescriptor()
    for coord, ch in chips.items():
        q = ch["p"]
        program = ttnn.ProgramDescriptor()
        program.custom_program_hash = _hash(f"all_to_all_matmul|{cfg_key}|{coord}")

        # ---- CBs ----
        port_crs = _rects(set(ch["ports"].values()))
        cbs = [
            ttnn.cb_descriptor_from_sharded_tensor(CB_A_LANDED, res.scratch),
            ttnn.CBDescriptor(
                total_size=weight_depth * Kt * block_n * w_tile,
                core_ranges=active_crs,
                format_descriptors=[ttnn.CBFormatDescriptor(CB_WEIGHT, weight.dtype, w_tile)],
            ),
            ttnn.CBDescriptor(
                total_size=OUT_DEPTH * sbh * block_n * in_tile,
                core_ranges=active_crs,
                format_descriptors=[ttnn.CBFormatDescriptor(CB_OUT, input_tensor.dtype, in_tile)],
            ),
            ttnn.CBDescriptor(
                total_size=2 * batch_chunks * packet_bytes,
                core_ranges=port_crs,
                format_descriptors=[ttnn.CBFormatDescriptor(CB_PORT_CHUNKS, input_tensor.dtype, packet_bytes)],
            ),
        ]
        if retile:
            # One descriptor per core set: a single descriptor over the (non-rectangular) union of the compute and
            # port cores was placed inside cb_port_chunks on some chips (measured: DEVICE_PRINT of both addresses).
            cbs += [
                ttnn.CBDescriptor(
                    total_size=stage_tiles * in_tile,
                    core_ranges=crs,
                    format_descriptors=[ttnn.CBFormatDescriptor(CB_RETILE_STAGE, input_tensor.dtype, in_tile)],
                )
                for crs in (active_crs, port_crs)
            ]
        program.cbs = cbs

        # ---- compute cores ----
        reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for xy in active_cells:
            rd, cp, wr = [], [], []
            for h, wait, nb0, nb1, hx, m0, m1, piece in slots.get((coord, xy), []):
                # nbw packs this core's n-blocks [nb0, nb1) and rows [m0, m1) of the head (tail M-split), 8 bits each
                assert max(nb0, nb1, m0, m1) < 256
                nbw = nb0 | (nb1 << 8) | (m0 << 16) | (m1 << 24)
                fwd = [0, 0, 0, 0, 0]
                if hx is not None:
                    hv, hs_ = virt(hx), helper_slot[(coord, h)]
                    fwd = [1, hv.x, hv.y, landing_addr + hs_ * slot_bytes, arrival_addrs[hs_]]
                # paced weight start of a core's first slot (later slots follow the previous one's work)
                pace = int(w_pace * landing_share(coord, h)) if (not rd and nb0 == 0) else 0
                pre = max(1, min(weight_depth, ms_pre)) if piece >= 0 else w_prefetch
                rd += [h | (int(wait) << 16) | (pre << 24), nbw] + fwd + [pace]
                cp += [nbw]
                wr += [h, nbw]
            n = len(cp)
            reader_rt[xy[0]][xy[1]] = [n] + rd
            compute_rt[xy[0]][xy[1]] = [n] + cp
            writer_rt[xy[0]][xy[1]] = [n] + wr
        reader_ct = [CB_A_LANDED, CB_WEIGHT, Kt, Nt, block_n, num_n_blocks, B, Mq_t, St, Hl, in_tile, w_tile]
        reader_ct += [Mq, CB_RETILE_STAGE, weight_chunk_tiles, w_prefetch]
        reader_ct += in_ct + w_ct
        compute_ct = [CB_A_LANDED, CB_WEIGHT, CB_OUT, block_m, Kt, block_n, num_n_blocks, sbh, sbw]
        writer_ct = [CB_OUT, in_tile, block_m, Mq_t, Nt, block_n, num_n_blocks, sbh, H, out_chunk_tiles] + out_ct
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "all_to_all_matmul_reader.cpp"),
                core_ranges=active_crs,
                defines=_defs,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                common_runtime_args=[in_addr, w_addr, q] + arrival_addrs,
                config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=rd_noc),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "all_to_all_matmul_compute.cpp"),
                core_ranges=active_crs,
                defines=_defs,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=compute_cfg,
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "all_to_all_matmul_writer.cpp"),
                core_ranges=active_crs,
                defines=_defs,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                common_runtime_args=[out_addr],
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0,
                    noc=wr_noc,
                ),
            ),
        ]

        # ---- port cores ----
        port_reader_rt, port_sender_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for (d, l), (px, py) in ch["ports"].items():
            ents = sends[(coord, d, l)]
            readies = ready_out[(coord, d, l)]
            port_reader_rt[px][py] = [in_addr, len(ents)] + [j | (t << 16) for j, t in ents]
            dest_coords = sorted({ch["grp"][t] for _, t in ents} | {tc for tc, _, _ in readies})
            n_dest = len(dest_coords)
            has_conn = 1 if (ents or readies) else 0
            hdrs_per_dest = max(1, min(batch_chunks, MAX_PACKET_HEADERS // max(1, n_dest)))
            if n_dest > MAX_PACKET_HEADERS:
                raise RuntimeError(f"all_to_all_matmul: a port with {n_dest} destinations exceeds the header pool")
            data_coords = [ch["grp"][t] for t in data_dests[(coord, d, l)]]
            args = [has_conn, n_dest, hdrs_per_dest]
            for dc in dest_coords:  # route + ready counter (0: no data for that chip) per destination
                dn = _node(mesh, dc)
                ra = ready_addrs[data_coords.index(dc)] if dc in data_coords else 0
                args += [int(dn.chip_id), int(dn.mesh_id), ra]
            args += [len(readies)]
            for tc, (td, tl), g in readies:
                rc = virt(chips[tc]["ports"][(td, tl)])
                args += [dest_coords.index(tc), rc.x, rc.y, ready_addrs[g]]
            args += [len(ents)]
            subs = []
            for j, t in ents:
                h = q * Hl + j
                di = dest_coords.index(ch["grp"][t])
                pcs = pieces_of.get((ch["grp"][t], h))
                if pcs is None:
                    s = slot_of[(t, h)]
                    pcs = [(core_of[h], s, slot_tiles)]
                for pxy, s, nt in pcs:
                    cv = virt(pxy)
                    subs.append([di | (nt << 16), cv.x, cv.y, landing_addr + s * slot_bytes, arrival_addrs[s]])
            args[-1] = len(subs)
            for sub in subs:
                args += sub
            if has_conn:
                peer = ch["next"] if d == FWD else ch["prev"]
                args += list(
                    ttnn.setup_fabric_connection(
                        _node(mesh, coord), _node(mesh, peer), ch["links"][d][l], program, ttnn.CoreCoord(px, py)
                    )
                )
            port_sender_rt[px][py] = args
        port_reader_ct = [
            CB_PORT_CHUNKS,
            in_tile,
            packet_tiles,
            batch_chunks,
            B,
            Mq_t,
            Kt,
            St,
            Hl,
            Mq,
            CB_RETILE_STAGE,
            int(pack),
            port_window,
        ]
        port_reader_ct += in_ct
        port_sender_ct = [CB_PORT_CHUNKS, in_tile, packet_tiles, batch_chunks, slot_tiles, MAX_PACKET_HEADERS]
        port_sender_ct += [int(pack), int(fence_per_dest)]
        kernels += [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "all_to_all_matmul_port_reader.cpp"),
                core_ranges=port_crs,
                defines=_defs,
                compile_time_args=port_reader_ct,
                runtime_args=port_reader_rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1, noc=PORT_READER_NOC
                ),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "all_to_all_matmul_port_sender.cpp"),
                core_ranges=port_crs,
                defines=_defs,
                compile_time_args=port_sender_ct,
                runtime_args=port_sender_rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=PORT_SENDER_NOC
                ),
            ),
        ]
        program.kernels = kernels
        r, c = coord
        mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = program

    return ttnn.generic_op([input_tensor, weight, res.scratch, output], mesh_desc)
