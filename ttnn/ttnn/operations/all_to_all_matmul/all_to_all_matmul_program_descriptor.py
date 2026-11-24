# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""all_to_all_matmul — host plan and MeshProgramDescriptor (regime R1 head-resident, transport T1 direct).

Per chip p of a group (ring of G chips along cluster_axis):

  port cores (2 * num_links, one per (direction d, link l)):
    reader  (RISCV_1, NoC0)  reads each entry (source-local head j, destination chip t: the quarter t of head j's rows,
                             all batches, all K) from the input, one CB page = one fabric packet, batched.
    sender  (RISCV_0, NoC1)  sends readies to every port that writes into this chip, waits for one ready per
                             destination chip, then sends every entry's packets straight into the destination compute
                             core's landing slot (same L1 address on every chip), the last packet of an entry fused
                             with an increment of that slot's arrival counter.
  compute cores (grid minus every chip's port cores; heads split across them, the same map on every chip):
    reader  (RISCV_1, NoC0)  per head slot (local-source heads first, then remote heads by source-local index j):
                             weight block 0 from DRAM, then the slot (local: DRAM read of the own quarter; remote:
                             wait for the arrival counter, re-arm it), then the remaining weight blocks.
    compute (TRISC)          matmul_block per (slot, n-block): A slot retained across n-blocks, popped after.
    writer  (RISCV_0, NoC0)  output row groups to the output pages.

Blocking knobs (op_design.md "Blocking Model"), each defined once here: BLOCK_HEADS, WEIGHT_BLOCK_BYTES (-> block_n),
WEIGHT_DEPTH, OUT_DEPTH, PORT_BATCH_CHUNKS, A_SLOT_BUDGET_BYTES; block_m = B * Mq_t and block_k = Kt (R1).
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
OUT_DEPTH = 2  # output row groups in cb_out
PORT_BATCH_CHUNKS = 8  # packets per DRAM-read barrier / per sender flush on the port cores (8 beat 4 at 7232 B)
A_SLOT_BUDGET_BYTES = 384 * 1024  # predicate A_RESIDENT (regime R1; else R3, deferred)
MAX_HEADS_PER_CORE = 8  # global arrival semaphores per compute core
MAX_PACKET_HEADERS = 12  # packet header pool per data-movement RISC

# ---- CB indices ----
CB_A_LANDED = 0
CB_WEIGHT = 1
CB_OUT = 2
CB_PORT_CHUNKS = 3

FWD, BWD = 0, 1
NOC0 = ttnn.NOC.RISCV_1_default
NOC1 = ttnn.NOC.RISCV_0_default
# Compute cores: weight / local-row reads on NoC1, output writes on NoC0, which keeps NoC0's read-response path free
# for the port readers (NoC0) feeding the fabric. Measured on the Kimi Phase 0 case (QuietBox 2, 7232 B):
# 134.0 -> 99.4 us fastest chip (reader NoC0 / writer NoC1 before); with PORT_BATCH_CHUNKS 8: 94.5 us.
COMPUTE_READER_NOC = NOC1
COMPUTE_WRITER_NOC = NOC0
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
        self.ready = None


_RES_CACHE = {}


def _resources(mesh, compute_cells, port_cells, cap_tiles, n_arrival, dtype):
    key = (id(mesh), tuple(sorted(compute_cells)), tuple(sorted(port_cells)), dtype)
    entry = _RES_CACHE.get(key)
    res = entry[1] if entry is not None else _Resources()
    changed = False
    compute_crs = _rects(compute_cells)
    if res.ready is None:
        res.ready = ttnn.create_global_semaphore(mesh, _rects(port_cells), 0)
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

    # ---- geometry (alignment-aware; R1 is tile_aligned) ----
    Kt, Nt = K // 32, N // 32
    Mq = S // G
    Mq_t = (Mq + 31) // 32
    St = (S + 31) // 32
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

    # ---- cores ----
    grid = mesh.compute_with_storage_grid_size()
    port_cells = {xy for ch in chips.values() for xy in ch["ports"].values()}
    compute_cells_list = [(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in port_cells]
    assign = _split_heads(compute_cells_list, H)  # [(core xy, [heads])], same on every chip
    heads_per_core_max = max(len(hs) for _, hs in assign)
    if heads_per_core_max > MAX_HEADS_PER_CORE:
        raise RuntimeError(f"all_to_all_matmul: {heads_per_core_max} heads per core > {MAX_HEADS_PER_CORE}")
    cap_tiles = heads_per_core_max * slot_tiles
    if cap_tiles * in_tile > A_SLOT_BUDGET_BYTES:
        raise RuntimeError(
            f"all_to_all_matmul: A_RESIDENT fails ({cap_tiles * in_tile} B of landed rows per core > "
            f"{A_SLOT_BUDGET_BYTES}): regime R3 (DRAM landing) is not built"
        )
    res = _resources(mesh, set(compute_cells_list), port_cells, cap_tiles, heads_per_core_max, input_tensor.dtype)
    landing_addr = int(res.scratch.buffer_address())
    slot_bytes = slot_tiles * in_tile
    arrival_addrs = [int(ttnn.get_global_semaphore_address(s)) for s in res.arrival[:heads_per_core_max]]
    ready_addr = int(ttnn.get_global_semaphore_address(res.ready))
    core_of = {h: xy for xy, hs in assign for h in hs}
    heads_of = {xy: hs for xy, hs in assign}
    active_cells = [xy for xy, _ in assign]
    active_crs = _rects(active_cells)
    virt = lambda xy: mesh.worker_core_from_logical_core(ttnn.CoreCoord(*xy))

    def slot_order(q, hs):
        local = [h for h in hs if h // Hl == q]
        remote = sorted((h for h in hs if h // Hl != q), key=lambda h: (h % Hl, h // Hl))
        return local + remote

    # slot index of head h on its core, as seen on chip position q
    slot_of = {}
    for q in range(G):
        for xy, hs in assign:
            for s, h in enumerate(slot_order(q, hs)):
                slot_of[(q, h)] = s

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

    # ---- per chip: who sends what (entries), readies ----
    L = num_links
    sends = {}  # (coord, d, l) -> [(j, t_pos)]
    for coord, ch in chips.items():
        p, G_, grp = ch["p"], ch["G"], ch["grp"]
        for d in (FWD, BWD):
            dests = sorted((t for t, dd in ch["route"].items() if dd == d), key=lambda t: ((t - p) % G_, t))
            for l in range(L):
                sends[(coord, d, l)] = [(j, t) for j in range(l, Hl, L) for t in dests]
    ready_out = {k: [] for k in sends}  # (coord, d, l) -> [(target coord, target port (d, l))]
    ready_wait = {}
    for (coord, d, l), ents in sends.items():
        ch = chips[coord]
        dest_pos = sorted({t for _, t in ents})
        ready_wait[(coord, d, l)] = len(dest_pos)
        for t in dest_pos:
            tc = ch["grp"][t]
            back = chips[tc]["route"][ch["p"]]
            ready_out[(tc, back, l)].append((coord, (d, l)))

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
            block_n,
            sbh,
            sbw,
            res.cap_tiles,
            landing_addr,
            sorted(port_cells),
        )
    )

    # Measurement hook (off by default): A2AMM_ABLATE=nowrite,noweight,nocompute,noportread,zones compiles the named
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
                total_size=WEIGHT_DEPTH * Kt * block_n * w_tile,
                core_ranges=active_crs,
                format_descriptors=[ttnn.CBFormatDescriptor(CB_WEIGHT, weight.dtype, w_tile)],
            ),
            ttnn.CBDescriptor(
                total_size=OUT_DEPTH * sbh * block_n * in_tile,
                core_ranges=active_crs,
                format_descriptors=[ttnn.CBFormatDescriptor(CB_OUT, input_tensor.dtype, in_tile)],
            ),
            ttnn.CBDescriptor(
                total_size=2 * PORT_BATCH_CHUNKS * packet_bytes,
                core_ranges=port_crs,
                format_descriptors=[ttnn.CBFormatDescriptor(CB_PORT_CHUNKS, input_tensor.dtype, packet_bytes)],
            ),
        ]
        program.cbs = cbs

        # ---- compute cores ----
        reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for xy in active_cells:
            order = slot_order(q, heads_of[xy])
            words = [h | ((1 if h // Hl != q else 0) << 16) for h in order]
            reader_rt[xy[0]][xy[1]] = [len(order)] + words
            compute_rt[xy[0]][xy[1]] = [len(order)]
            writer_rt[xy[0]][xy[1]] = [len(order)] + order
        reader_ct = [CB_A_LANDED, CB_WEIGHT, Kt, Nt, block_n, num_n_blocks, B, Mq_t, St, Hl, in_tile, w_tile]
        reader_ct += in_ct + w_ct
        compute_ct = [CB_A_LANDED, CB_WEIGHT, CB_OUT, block_m, Kt, block_n, num_n_blocks, sbh, sbw]
        writer_ct = [CB_OUT, in_tile, block_m, Mq_t, Nt, block_n, num_n_blocks, sbh, H] + out_ct
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "all_to_all_matmul_reader.cpp"),
                core_ranges=active_crs,
                defines=_defs,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                common_runtime_args=[in_addr, w_addr, q] + arrival_addrs,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1, noc=COMPUTE_READER_NOC
                ),
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
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=COMPUTE_WRITER_NOC
                ),
            ),
        ]

        # ---- port cores ----
        port_reader_rt, port_sender_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for (d, l), (px, py) in ch["ports"].items():
            ents = sends[(coord, d, l)]
            readies = ready_out[(coord, d, l)]
            port_reader_rt[px][py] = [in_addr, len(ents)] + [j | (t << 16) for j, t in ents]
            dest_coords = sorted({ch["grp"][t] for _, t in ents} | {tc for tc, _ in readies})
            n_dest = len(dest_coords)
            has_conn = 1 if (ents or readies) else 0
            hdrs_per_dest = max(1, min(PORT_BATCH_CHUNKS, MAX_PACKET_HEADERS // max(1, n_dest)))
            if n_dest > MAX_PACKET_HEADERS:
                raise RuntimeError(f"all_to_all_matmul: a port with {n_dest} destinations exceeds the header pool")
            args = [ready_addr, ready_wait[(coord, d, l)], has_conn, n_dest, hdrs_per_dest]
            for dc in dest_coords:
                dn = _node(mesh, dc)
                args += [int(dn.chip_id), int(dn.mesh_id)]
            args += [len(readies)]
            for tc, (td, tl) in readies:
                rc = virt(chips[tc]["ports"][(td, tl)])
                args += [dest_coords.index(tc), rc.x, rc.y]
            args += [len(ents)]
            for j, t in ents:
                h = q * Hl + j
                cv = virt(core_of[h])
                s = slot_of[(t, h)]
                args += [dest_coords.index(ch["grp"][t]), cv.x, cv.y, landing_addr + s * slot_bytes, arrival_addrs[s]]
            if has_conn:
                peer = ch["next"] if d == FWD else ch["prev"]
                args += list(
                    ttnn.setup_fabric_connection(
                        _node(mesh, coord), _node(mesh, peer), ch["links"][d][l], program, ttnn.CoreCoord(px, py)
                    )
                )
            port_sender_rt[px][py] = args
        port_reader_ct = [CB_PORT_CHUNKS, in_tile, packet_tiles, PORT_BATCH_CHUNKS, B, Mq_t, Kt, St, Hl] + in_ct
        port_sender_ct = [CB_PORT_CHUNKS, in_tile, packet_tiles, PORT_BATCH_CHUNKS, slot_tiles, MAX_PACKET_HEADERS]
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
