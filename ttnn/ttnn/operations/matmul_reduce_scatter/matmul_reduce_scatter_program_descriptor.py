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

# ---- knobs (single source of truth) ------------------------------------------------------------------------------
DEPTH_IN0 = 3  # stream regime: A K-block windows (one computing, one landing, PREFETCH_STEPS read ahead by a sender)
DEPTH_IN1 = 3  # stream regime: W K-block windows
DEPTH_HOLD = 2  # fp32 partial slots (lag 1 + current)
DEPTH_HANDOFF = 2  # finished slots (a port gathers entry e while compute finishes e+1)
PREFER_PORT_ROW_STRIP = True  # ports in logical row 0, right under the Ethernet row (focus case: 150-160 us with
# the column strip -> 137-145 us with the row strip, despite 14 vs 12 tiles per core: the transport tail shrinks)
PORT_PLACEMENT = "eth"  # "eth": each port nearest its link's Ethernet core | "spread": evenly along the strip
PORT_GROUP = 4  # chunks per port batch (cb_port_chunks holds 2 batches); <= 8 (header pool)
ENTRIES_PER_BLOCK = 1  # entries per scatter block (overlap lamp; only 1 is built)
KB_MIN_RESIDENT = 8  # smallest K-block the resident regime accepts (min(Kt, this))
KB_MAX = 16  # largest K-block: the streamed operand needs several K-blocks per entry to overlap read / mcast / math
# (focus case, diagonal senders: kb 64 -> 212 us, 32 -> 160 us, 16 -> 159 us, 8 -> 164 us on the fastest chip)
L1_HEADROOM = 64 * 1024  # bytes kept free per core beyond the op's own footprint
LAG_POLICY = os.environ.get("MRS_LAG", "auto")  # "auto": lag a relay finish only when its arrival is produced late
# upstream; "always": the design's original rule (every add waits behind the next matmul). env: perf experiments only
PREFETCH_STEPS = 1  # K-block windows a sender keeps reserved with DRAM reads in flight ahead of its multicast
SENDERS_PER_LINE = 1  # each K-block window is split into this many contiguous parts (A by m rows, W by k rows), each
# read + multicast by a different core of the line, so no single RISC serializes a whole window's reads and mcast
MCAST_SENDER_PLACEMENT = (
    "diagonal"  # operand senders per line: "uniform" (first core: a column of DRAM readers) | "diagonal"
)
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


def build_schedule(p, G, kind):
    """roles[(b, h)] and this chip's entry order: the other blocks by descending distance-to-owner (ties: fwd first,
    then lower b), then its own block last."""
    roles = {(b, h): _role(p, b, h, G, kind) for b in range(G) for h in (0, 1)}

    def sort_key(b):
        key = max(roles[(b, h)][2] for h in (0, 1))
        return (-key, 0 if roles[(b, 0)][0] == "fwd" else 1, b)

    entries = sorted((b for b in range(G) if b != p), key=sort_key) + [p]
    return roles, entries


def plan_transport(G, kind):
    """Per chip position: roles, entries, per (half, direction) arrival list (this chip's entry order), and the
    arrival thresholds (position in the upstream's send order). slots[d] = zone slots per direction (max over chips
    and halves, so every chip shares one staging layout)."""
    ring = kind != "line"
    sch = {p: build_schedule(p, G, kind) for p in range(G)}

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
                for b in entries:
                    r = roles[(b, h)]
                    if r[0] == d and not r[1]:
                        assert u is not None and sch[u][0][(b, h)][0] == d, (p, b, h, d)
                        lst.append(b)
                    elif r[0] == "fin" and u is not None and sch[u][0][(b, h)][0] == d:
                        lst.append(b)
                seq = [b for b in sch[u][1] if sch[u][0][(b, h)][0] == d] if u is not None else []
                assert sorted(seq) == sorted(lst), (p, h, d, seq, lst)
                arrivals[(h, d)] = lst
                thresholds[(h, d)] = {b: i + 1 for i, b in enumerate(seq)}
                # the upstream's entry index that produces each arrival (drives the per-entry lag decision)
                produced_at[(h, d)] = {b: sch[u][1].index(b) for b in seq}
        chips[p] = dict(roles=roles, entries=entries, arrivals=arrivals, thresholds=thresholds, produced_at=produced_at)
    slots = {d: max(1, max(len(chips[p]["arrivals"][(h, d)]) for p in range(G) for h in (0, 1))) for d in _DIRS}
    return chips, slots


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


def footprint(kb, resident, *, Kt, core_m, core_n, tA, tW, scatter_dim, slots, staging_floor_tiles=0):
    """Per matmul core L1 bytes (l1_ledger.md): operand windows + fp32 hold + the staging slice. The staging tensor is
    cached grow-only per mesh, so it is counted at max(what this call needs, what is already allocated)."""
    P = core_m * core_n
    rA = resident and scatter_dim == -1
    rW = resident and scatter_dim == -2
    a = tA * (core_m * Kt if rA else DEPTH_IN0 * core_m * kb)
    w = tW * (Kt * core_n if rW else DEPTH_IN1 * kb * core_n)
    hold = HOLD_TILE_BYTES * DEPTH_HOLD * P
    staging = STAGE_TILE_BYTES * max(staging_floor_tiles, (DEPTH_HANDOFF + slots["fwd"] + slots["bwd"]) * P)
    return a + w + hold + staging


def choose_k_block(Kt, budget, force_resident, **geo):
    """(resident, kb): resident with the preferred kb | Kt (largest <= KB_MAX, then smaller, then larger), kb >=
    min(Kt, KB_MIN_RESIDENT), else stream with the preferred kb | Kt that fits. force_resident (test hook) replaces the
    residency decision."""
    # preference order: the largest divisor <= KB_MAX first, then smaller ones, then (no fit below the cap) larger ones
    divisors = [d for d in range(min(Kt, KB_MAX), 0, -1) if Kt % d == 0]
    divisors += [d for d in range(KB_MAX + 1, Kt + 1) if Kt % d == 0]
    if force_resident is not False:
        for kb in divisors:
            if kb < min(Kt, KB_MIN_RESIDENT):
                continue
            if footprint(kb, True, Kt=Kt, **geo) <= budget:
                return True, kb
        if force_resident is True:
            raise ValueError("matmul_reduce_scatter: the resident cross-block operand does not fit L1 for this shape")
    for kb in divisors:
        if footprint(kb, False, Kt=Kt, **geo) <= budget:
            return False, kb
    raise NotImplementedError(
        "matmul_reduce_scatter: even kb=1 exceeds the L1 budget (regime DRAM-landing is deferred)"
    )


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

        sems = [ttnn.create_global_semaphore(mesh, grid, 0) for _ in range(3)]  # arr_fwd, arr_bwd, ready
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
    if res["staging_tiles"] < staging_tiles:
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
    return int(ttnn._ttnn.reports.get_device_info(mesh).l1_bank_size) - L1_HEADROOM


# ======================================================================================================================
# Caller checks
# ======================================================================================================================


def _check_dram_interleaved_tile(t, what):
    mc = t.memory_config()
    if mc.buffer_type != ttnn.BufferType.DRAM or mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError(f"matmul_reduce_scatter: {what} must be DRAM interleaved")
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
    if len(w_shape) != 2:
        raise ValueError(f"matmul_reduce_scatter: W must be (K, N), got {w_shape}")
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
    gd = choose_grid(gx, gy, block_m, block_n, L)
    core_m, core_n = gd["core_m"], gd["core_n"]
    P = core_m * core_n
    n_cores = gd["grid_m"] * gd["grid_n"]
    if kind == "ring_balanced" and n_cores < 2:
        raise ValueError("matmul_reduce_scatter: a balanced ring needs >= 2 matmul cores (two halves)")
    assert ENTRIES_PER_BLOCK == 1, "entries_per_block > 1 (overlap lamp) is not built"

    chips_sched, slots = plan_transport(G, kind)
    tA = int(a.buffer_aligned_page_size())
    tW = int(w.buffer_aligned_page_size())
    cached = _MESH_CACHE.get(id(mesh))
    staging_floor = cached["staging_tiles"] if cached is not None and cached["mesh"] is mesh else 0
    geo = dict(
        core_m=core_m,
        core_n=core_n,
        tA=tA,
        tW=tW,
        scatter_dim=scatter_dim,
        slots=slots,
        staging_floor_tiles=staging_floor,
    )
    resident, kb = choose_k_block(Kt, _l1_budget(mesh), getattr(hooks, "_FORCE_XBLOCK_RESIDENCY", None), **geo)
    if os.environ.get("MRS_KB"):  # perf experiments only: cap the K-block (largest divisor of Kt <= MRS_KB)
        kb = max(d for d in range(1, kb + 1) if Kt % d == 0 and d <= int(os.environ["MRS_KB"]))
    if os.environ.get("MRS_SB_TALL"):  # perf experiments only: prefer taller subblocks on ties
        gd["sb"] = _subblock(core_m, core_n, prefer_tall=True)
    hooks._LAST_PLAN = {
        "xblock": "resident" if resident else "stream",
        "schedule": kind,
        "kb": kb,
        "grid": (gd["grid_m"], gd["grid_n"]),
        "strip": gd["strip"],
    }

    staging_tiles = (DEPTH_HANDOFF + slots["fwd"] + slots["bwd"]) * P
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
        resident=resident,
        slots=slots,
        sched=chips_sched,
        compute_kernel_config=compute_kernel_config,
        gx=gx,
        gy=gy,
    )
    desc = create_mesh_program_descriptor(ctx)
    return ttnn.generic_op([a, w, output], desc)


# ======================================================================================================================
# Program descriptors
# ======================================================================================================================


def create_mesh_program_descriptor(ctx):
    mesh = ctx["mesh"]
    G, L, kind, P = ctx["G"], ctx["L"], ctx["kind"], ctx["P"]
    gd = ctx["gd"]
    gx, gy = ctx["gx"], ctx["gy"]
    core_m, core_n, kb, Kt, Nt = gd["core_m"], gd["core_n"], ctx["kb"], ctx["Kt"], ctx["Nt"]
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
    senders = int(os.environ.get("MRS_SENDERS", SENDERS_PER_LINE))  # env: perf experiments only
    a_split = -(-core_m // 2) if senders >= 2 and core_m >= 2 else core_m  # m rows of A part 0
    w_split = -(-kb // 2) if senders >= 2 and kb >= 2 else kb  # k rows of W part 0
    sb_h, sb_w = gd["sb"]
    rA = int(resident and scatter_dim == -1)
    rW = int(resident and scatter_dim == -2)
    a_pages = core_m * Kt if rA else DEPTH_IN0 * core_m * kb
    w_pages = Kt * core_n if rW else DEPTH_IN1 * kb * core_n
    tA = int(a.buffer_aligned_page_size())
    tW = int(w.buffer_aligned_page_size())
    out_nt = block_n  # output tile columns (Nt for scatter -2, Nt / G for scatter -1)
    a_row_per_block = block_m if scatter_dim == -2 else 0
    w_col_per_block = block_n if scatter_dim == -1 else 0

    staging = res["staging"]
    staging_addr = int(staging.buffer_address())
    zone_fwd_off = DEPTH_HANDOFF * P * STAGE_TILE_BYTES
    zone_bwd_off = zone_fwd_off + slots["fwd"] * P * STAGE_TILE_BYTES
    arr_fwd_addr, arr_bwd_addr, ready_addr = (int(ttnn.get_global_semaphore_address(s)) for s in res["sems"])
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
            core_m,
            core_n,
            kb,
            nkb,
            Kt,
            Nt,
            a_row_per_block,
            w_col_per_block,
            rA,
            rW,
            P,
            G,
            tA,
            tW,
            a_pages // (core_m * kb),  # CB capacity in K-block windows
            w_pages // (kb * core_n),
            a_split,
            w_split,
            int(os.environ.get("MRS_PREFETCH", PREFETCH_STEPS)),  # env: perf experiments only
        ]
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
        core_m,
        core_n,
        kb,
        nkb,
        sb_h,
        sb_w,
        P,
        G,
    ]
    writer_ct = [CB_FINISHED, P, core_n, out_nt, DEPTH_HANDOFF, STAGE_TILE_BYTES, G] + o_ct
    port_ct = [CB_PORT_CHUNKS, P, chunk_tiles, PORT_GROUP, STAGE_TILE_BYTES, DEPTH_HANDOFF]

    src = ttnn.KernelDescriptor.SourceType.FILE_PATH
    # Perf-only ablation switches (debug scaffolding; output is wrong by design when set): MRS_ABLATE=compute,dram
    ablate = set(filter(None, os.environ.get("MRS_ABLATE", "").split(",")))
    ablate_compute = [("SKIP_COMPUTE", "1"), ("CKL_ELTWISE_CHAIN_SKIP_COMPUTE", "1")] if "compute" in ablate else []
    ablate_reader = [("MRS_SKIP_DRAM", "1")] if "dram" in ablate else []
    zones = [("MRS_ZONES", "1")] if os.environ.get("MRS_ZONES") else []
    ablate_compute = ablate_compute + zones
    ablate_reader = ablate_reader + zones
    dm = lambda risc, noc: ttnn.DataMovementConfigDescriptor(
        processor=getattr(ttnn.DataMovementProcessor, risc), noc=noc
    )
    port_vcore_of = {coord: {key: virt(c) for key, c in ports.items()} for coord, ports in ports_of.items()}

    def unit_walks(sc, d, l):
        """Port (d, l)'s units on a chip with schedule sc: [(entry, block, mask, h1_first, [k in walk order])]."""
        out = []
        for e, b in enumerate(sc["entries"]):
            mask = sum(1 << h for h in link_halves[l] if sc["roles"][(b, h)][0] == d)
            if not mask:
                continue
            # walk a chain-start half (ready right after its matmul) before a relay half (waits for its arrival)
            h1_first = int(mask == 3 and sc["roles"][(b, 1)][1] and not sc["roles"][(b, 0)][1])
            order = (1, 0) if h1_first else (0, 1)
            ks = [k for hh in order if mask >> hh & 1 for k in range(n_cores) if link_of(k) == l and half(k) == hh]
            out.append((e, b, mask, h1_first, ks))
        return out

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
            ]

            # ---- matmul cores: runtime args ----
            r_rt, c_rt, w_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for k, (gm, gn, core) in enumerate(mm_cores):
                h = half(k)
                l = link_of(k)
                m_valid = min(core_m, block_m - gm * core_m)
                n_valid = min(core_n, block_n - gn * core_n)
                ent_words, roles_k, kinds_k = [], [], []
                for b in entries:
                    r = roles[(b, h)]
                    mask = 0
                    if r[0] in _DIRS and not r[1]:
                        mask = 1 if r[0] == "fwd" else 2
                    elif r[0] == "fin":
                        mask = (1 if b in arrivals[(h, "fwd")] else 0) | (2 if b in arrivals[(h, "bwd")] else 0)
                    thr_f = thresholds[(h, "fwd")].get(b, 0) if mask & 1 else 0
                    thr_b = thresholds[(h, "bwd")].get(b, 0) if mask & 2 else 0
                    # lag (finish behind the next entry's matmul) only when an arrival is produced at the same or a
                    # later position of the upstream's walk; an arrival produced earlier has (most likely) landed
                    e_idx = len(roles_k)
                    lag = LAG_POLICY == "always" and mask != 0
                    lag = lag or any(
                        sc["produced_at"][(h, d)][b] >= e_idx for d, bit in (("fwd", 1), ("bwd", 2)) if mask & bit
                    )
                    lag = int(lag and e_idx + 1 < len(entries))
                    ent_words += [b | (mask << 16) | (lag << 20), thr_f | (thr_b << 16)]
                    roles_k.append([ROLE_COPY, ROLE_ADD_FWD, ROLE_ADD_BWD, ROLE_ADD_BOTH][mask] | (lag << 8))
                    kinds_k.append(KIND_FIN if r[0] == "fin" else (KIND_FWD if r[0] == "fwd" else KIND_BWD))
                r_rt[core.x][core.y] = [
                    a_addr,
                    w_addr,
                    gm * core_m,
                    gn * core_n,
                    m_valid,
                    n_valid,
                    arr_fwd_addr,
                    arr_bwd_addr,
                    len(arrivals[(h, "fwd")]),  # arrival increments my counters receive this call
                    len(arrivals[(h, "bwd")]),
                ] + ent_words
                c_rt[core.x][core.y] = roles_k
                pf = port_vcore.get(("fwd", l)) if peers["fwd"] is not None else None
                pb = port_vcore.get(("bwd", l)) if peers["bwd"] is not None else None
                w_rt[core.x][core.y] = [
                    o_addr,
                    gm * core_m,
                    gn * core_n,
                    m_valid,
                    n_valid,
                    (pf.x | (pf.y << 16)) if pf is not None else 0,
                    (pb.x | (pb.y << 16)) if pb is not None else 0,
                    mailbox_addr + k * MAILBOX_STRIDE,
                ] + kinds_k

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
                defines=zones,
                config=dm("RISCV_0", NOC1),
            )
            mc_cfg = ttnn.McastConfig(noc=ttnn.NOC.NOC_0, handshake=True)
            placement = os.environ.get("MRS_SENDER_PLACEMENT", MCAST_SENDER_PLACEMENT)
            snd = ttnn.Mcast1DFixedSenderConfig(
                starting_sender_index=0,
                sender_placement=ttnn.Mcast1DSenderPlacement.Diagonal
                if placement == "diagonal"
                else ttnn.Mcast1DSenderPlacement.Uniform,
            )
            kernels = [reader]
            # part 0 senders on the diagonal; part 1 senders half a line further along (a different core per line)
            for prefix, shape_1d, split, extent, line_len in (
                ("a", a_shape_1d, a_split, core_m, grid_n),
                ("b", w_shape_1d, w_split, kb, grid_m),
            ):
                ttnn.Mcast1D(mesh, mm_crs, shape_1d, snd, mc_cfg).attach(program, prefix, kernels)
                if split < extent:
                    snd2 = ttnn.Mcast1DFixedSenderConfig(
                        starting_sender_index=line_len // 2,
                        sender_placement=ttnn.Mcast1DSenderPlacement.Diagonal
                        if placement == "diagonal"
                        else ttnn.Mcast1DSenderPlacement.Uniform,
                    )
                    ttnn.Mcast1D(mesh, mm_crs, shape_1d, snd2, mc_cfg).attach(program, prefix + "2", kernels)
                else:
                    ttnn.attach_absent(kernels[0], prefix + "2")
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
                for e, b, mask, h1_first, _ in unit_walks(sc, d, l):
                    s0 = down["arrivals"][(0, d)].index(b) if mask & 1 else 0
                    s1 = down["arrivals"][(1, d)].index(b) if mask & 2 else 0
                    units.append(e | (mask << 8) | (h1_first << 10) | (s0 << 16) | (s1 << 24))
                consumed_id = 0 if d == "fwd" else 1
                pr_rt[pcore.x][pcore.y] = [staging_addr, mailbox_addr, consumed_id, len(core_words), len(units)] + (
                    core_words + units
                )
                # ready fence: the peer's opposite port sends into this chip; tell it this chip has started
                peer_sched = ctx["sched"][q]
                opp = "bwd" if d == "fwd" else "fwd"
                send_ready = int(any(peer_sched["roles"][(b, h)][0] == opp for b in range(G) for h in link_halves[l]))
                rv = port_vcore_of[peer][(opp, l)]  # the peer chip's port that writes into this chip
                pn = node(peer)
                zone = zone_fwd_off if d == "fwd" else zone_bwd_off
                arr_addr = arr_fwd_addr if d == "fwd" else arr_bwd_addr
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
                    ]
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
                defines=zones,
                config=dm("RISCV_1", NOC0),
            )
            port_sender = ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "matmul_reduce_scatter_port_sender.cpp"),
                source_type=src,
                core_ranges=used_port_crs,
                compile_time_args=port_ct,
                runtime_args=ps_rt,
                defines=zones,
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
                cb(CB_PARTIAL_HOLD, DEPTH_HOLD * P, HOLD_TILE_BYTES, ttnn.float32, mm_crs),
                ttnn.cb_descriptor_from_sharded_tensor(
                    CB_FINISHED, staging, 0, DEPTH_HANDOFF * P * STAGE_TILE_BYTES, mm_crs
                ),
                ttnn.cb_descriptor_from_sharded_tensor(
                    CB_ARRIVALS_FWD, staging, zone_fwd_off, slots["fwd"] * P * STAGE_TILE_BYTES, mm_crs
                ),
                ttnn.cb_descriptor_from_sharded_tensor(
                    CB_ARRIVALS_BWD, staging, zone_bwd_off, slots["bwd"] * P * STAGE_TILE_BYTES, mm_crs
                ),
                cb(CB_PORT_CHUNKS, 2 * PORT_GROUP * chunk_tiles, STAGE_TILE_BYTES, ttnn.bfloat16, used_port_crs),
            ]
            r, c = coord
            mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = program
    return mesh_desc
