# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""distributed_rms_norm — plan (groups, links, split, records) and the MeshProgramDescriptor.

Regime R1 (op_design.md): interleaved x resident in L1, rows x W-chunks over the worker rectangle, flat fp32
record all-gather over Fabric through port cores in logical row 0.

Per device:
  ports   : logical (2*l + d, 0), l < L_eff, d = 0 (fwd, toward p+1) / 1 (bwd, toward p-1). BRISC, NoC1.
  workers : logical rows 1.., row-major index k = r_g * S_w + w. reader NCRISC/NoC0, writer BRISC/NoC1, compute.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import ttnn
from ttnn.operations.ccl import Topology

KERNEL_DIR = Path(__file__).parent / "kernels"

# ---- circular buffers (worker cores) ---------------------------------------------------------------------------
CB_X_RESIDENT = 0
CB_GAMMA_TILES = 1
CB_SCALER = 2
CB_SQ_ACC = 3
CB_PARTIAL = 4
CB_XG = 5
CB_STATS_COMBINE = 6
CB_RSTD = 7
CB_OUTPUT_TILES = 8

# ---- program-local semaphores (declared over every core, before any fabric connection setup) ------------------
SEM_LOCAL_ARRIVALS = 0  # on ports: workers of the link increment once their records have landed
SEM_STATS_READY = 1  # on workers: every port multicasts +1 once its slots are complete

# ---- blocking knobs (single source of truth) -------------------------------------------------------------------
MIN_CORE_W_TILES = (
    8  # floor of the W chunk per worker; record pulls grow as S_w^2 (perf lamp "grid synchronization": 8 measured best)
)
X_BLOCKS_IN_FLIGHT = 1  # cb_x_resident depth in blocks (residency, not pipelining)
OUT_DEPTH = 2  # cb_output_tiles depth in out batches (double buffer)
HEADER_RING = 8  # fabric packet headers in flight per port (pool holds 12 per RISC)
RECORD_BYTES = 128  # one fp32 record: 32 per-row partial sums of one tile row
WORKER_L1_BUDGET = 1 << 20  # CB bytes per worker (P_resident)
SCRATCH_BUDGET = 1 << 20  # exchange scratch bytes per port core (P_scratch)
TILE = 32


def block_rows_knob(core_rows_max):
    """Rows per block. Phase 0 (R1): the whole per-core assignment, one block, one exchange round."""
    return core_rows_max


def out_batch_tiles_knob(core_w_tiles):
    """Tiles per writer barrier (Phase 0: one tile row of the core's W chunk). Must divide core_w_tiles: the writer
    walks each tile row in batches, and cb_output_tiles (OUT_DEPTH batches) must wrap on a batch boundary."""
    return core_w_tiles


# ----------------------------------------------------------------------------------------------------------------
# Groups, links, schedules (shape independent, cached)
# ----------------------------------------------------------------------------------------------------------------


def _groups(mesh_shape, cluster_axis):
    rows, cols = mesh_shape
    if cluster_axis == 0:
        return [[(r, c) for r in range(rows)] for c in range(cols)]
    return [[(r, c) for c in range(cols)] for r in range(rows)]


def _wraps_axis(cluster_axis):
    cfg = ttnn.get_fabric_config()
    fc = ttnn.FabricConfig
    wrap0 = {getattr(fc, n, None) for n in ("FABRIC_2D_TORUS_Y", "FABRIC_2D_TORUS_XY")}
    wrap1 = {getattr(fc, n, None) for n in ("FABRIC_2D_TORUS_X", "FABRIC_2D_TORUS_XY")}
    return cfg in (wrap0 if cluster_axis == 0 else wrap1)


def _schedule(p, G, ring):
    """(slots sent toward p+1, slots sent toward p-1), own slot first, in arrival order downstream."""
    if ring:
        kf = G // 2
        kb = G - 1 - kf
        return [(p - i) % G for i in range(kf)], [(p + i) % G for i in range(kb)] if kb > 0 else []
    fwd = [p - i for i in range(p + 1)] if p < G - 1 else []
    bwd = [p + i for i in range(G - p)] if p > 0 else []
    return fwd, bwd


_LINK_CACHE = {}


def _hop_links(mesh_device, cluster_axis, ring):
    """{(src_coord, dst_coord): [link indices]} for every directed hop of every group (cached)."""
    key = (id(mesh_device), tuple(mesh_device.shape), cluster_axis, ring, str(ttnn.get_fabric_config()))
    if key in _LINK_CACHE:
        return _LINK_CACHE[key][1]
    node = lambda coord: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))
    hops = {}
    for grp in _groups(tuple(mesh_device.shape), cluster_axis):
        G = len(grp)
        pairs = [(grp[p], grp[p + 1]) for p in range(G - 1)]
        if ring and G > 2:
            pairs.append((grp[G - 1], grp[0]))
        for a, b in pairs:
            hops[(a, b)] = list(ttnn.get_forwarding_link_indices(node(a), node(b)))
            hops[(b, a)] = list(ttnn.get_forwarding_link_indices(node(b), node(a)))
    _LINK_CACHE[key] = (mesh_device, hops)
    return hops


def caller_link_checks(mesh_device, cluster_axis, topology, num_links):
    """ValueError for link / topology requests the live cluster cannot satisfy."""
    mesh_shape = tuple(mesh_device.shape)
    G = mesh_shape[cluster_axis]
    ring = topology == Topology.Ring
    if ring:
        if not _wraps_axis(cluster_axis):
            raise ValueError(
                f"distributed_rms_norm: Ring along cluster_axis {cluster_axis} needs a fabric config that wraps it "
                f"(active: {ttnn.get_fabric_config()})"
            )
        node = lambda coord: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))
        for grp in _groups(mesh_shape, cluster_axis):
            if not list(ttnn.get_forwarding_link_indices(node(grp[-1]), node(grp[0]))):
                raise ValueError(f"distributed_rms_norm: no wrap link {grp[-1]} -> {grp[0]} for Ring")
    hops = _hop_links(mesh_device, cluster_axis, ring and G > 2)
    min_links = min(len(v) for v in hops.values())
    if min_links == 0:
        raise ValueError("distributed_rms_norm: a hop of the group has no usable fabric link")
    gx = mesh_device.compute_with_storage_grid_size().x
    if num_links is not None and 2 * num_links > gx:
        raise ValueError(
            f"distributed_rms_norm: num_links={num_links} needs {2 * num_links} port columns, grid has {gx}"
        )
    if num_links is not None and num_links > min_links:
        raise ValueError(
            f"distributed_rms_norm: num_links={num_links} exceeds the {min_links} usable link(s) per hop "
            f"(fabric {ttnn.get_fabric_config()})"
        )


# ----------------------------------------------------------------------------------------------------------------
# Split selection and plan
# ----------------------------------------------------------------------------------------------------------------


def _choose_split(tensor_row_tiles, tensor_w_tiles, num_workers, min_core_w_tiles=MIN_CORE_W_TILES):
    """(S_r, S_w, core_w_tiles): lexicographically smallest (max tiles per worker, S_w)."""
    best = None
    for s_w in range(1, max(1, min(num_workers, math.ceil(tensor_w_tiles / min_core_w_tiles))) + 1):
        s_r = min(tensor_row_tiles, num_workers // s_w)
        if s_r == 0:
            break
        cost = math.ceil(tensor_row_tiles / s_r) * math.ceil(tensor_w_tiles / s_w)
        if best is None or (cost, s_w) < best:
            best = (cost, s_w)
    s_w = best[1]
    core_w_tiles = math.ceil(tensor_w_tiles / s_w)
    s_w = math.ceil(tensor_w_tiles / core_w_tiles)  # no empty trailing chunk
    s_r = min(tensor_row_tiles, num_workers // s_w)
    return s_r, s_w, core_w_tiles


def _tile_bytes(dtype):
    return ttnn.tile_size(dtype)


def build_plan(input_tensor, *, cluster_axis, topology, num_links):
    mesh_device = input_tensor.device()
    mesh_shape = tuple(mesh_device.shape)
    ring = topology == Topology.Ring and mesh_shape[cluster_axis] > 2
    G = mesh_shape[cluster_axis]
    hops = _hop_links(mesh_device, cluster_axis, ring)
    grid = mesh_device.compute_with_storage_grid_size()
    gx, gy = grid.x, grid.y
    L = num_links if num_links is not None else min(len(v) for v in hops.values())
    L = min(L, gx // 2)  # ports live in logical row 0, columns 0..2L-1 (auto only; explicit L is checked by the caller)

    shape = tuple(input_tensor.shape)
    H, W = shape[-2], shape[-1]
    batch = math.prod(shape[:-2]) if len(shape) > 2 else 1
    tensor_row_tiles = batch * math.ceil(H / TILE)
    tensor_w_tiles = math.ceil(W / TILE)
    num_workers = gx * (gy - 1)
    S_r, S_w, core_w_tiles = _choose_split(tensor_row_tiles, tensor_w_tiles, num_workers)

    # balanced contiguous row groups (first rows % S_r groups take the ceil)
    base, extra = divmod(tensor_row_tiles, S_r)
    core_rows = [base + (1 if g < extra else 0) for g in range(S_r)]
    row_start = [sum(core_rows[:g]) for g in range(S_r)]
    L_eff = min(L, S_r)  # a link with no row group would carry no records
    link_of = [g % L_eff for g in range(S_r)]
    recs_link = [S_w * sum(core_rows[g] for g in range(S_r) if link_of[g] == l) for l in range(L_eff)]
    link_rec_base = []
    for g in range(S_r):
        link_rec_base.append(S_w * sum(core_rows[h] for h in range(g) if link_of[h] == link_of[g]))
    slot_bytes = RECORD_BYTES * max(recs_link)
    n_src_tiles = math.ceil(G * S_w / TILE)

    core_rows_max = max(core_rows)
    block_rows = block_rows_knob(core_rows_max)
    num_blocks = max(math.ceil(r / block_rows) for r in core_rows)
    assert num_blocks == 1, "multi-round exchange (R2) is a deferred regime: the port kernel runs one round"

    return dict(
        mesh_device=mesh_device,
        cluster_axis=cluster_axis,
        ring=ring,
        G=G,
        groups=_groups(mesh_shape, cluster_axis),
        hops=hops,
        gx=gx,
        gy=gy,
        L=L_eff,
        W=W,
        tensor_row_tiles=tensor_row_tiles,
        tensor_w_tiles=tensor_w_tiles,
        S_r=S_r,
        S_w=S_w,
        core_w_tiles=core_w_tiles,
        core_rows=core_rows,
        row_start=row_start,
        link_of=link_of,
        recs_link=recs_link,
        link_rec_base=link_rec_base,
        slot_bytes=slot_bytes,
        n_src_tiles=n_src_tiles,
        block_rows=block_rows,
    )


def _worker_footprint(plan, x_dtype, gamma_dtype, gamma_prescale):
    """Per-worker CB bytes (l1_ledger.md F_worker). cb_xg exists only on the gamma-prescale path."""
    T_x = _tile_bytes(x_dtype)
    T_f = _tile_bytes(ttnn.float32)
    br, cw = plan["block_rows"], plan["core_w_tiles"]
    has_gamma = gamma_dtype is not None
    f = X_BLOCKS_IN_FLIGHT * br * cw * T_x
    if has_gamma:
        f += cw * _tile_bytes(gamma_dtype)
    if has_gamma and gamma_prescale:
        f += br * cw * T_f
    f += _tile_bytes(ttnn.bfloat16)
    f += br * T_f * (3 + plan["n_src_tiles"])
    f += OUT_DEPTH * out_batch_tiles_knob(cw) * T_x
    return f


# ----------------------------------------------------------------------------------------------------------------
# Persistent per-mesh resources
# ----------------------------------------------------------------------------------------------------------------

_SEM_CACHE = {}


def _global_semaphores(mesh_device, port_cores):
    """(arrival, ready) global semaphores over the port cores; created once, synchronized once, cached."""
    key = (id(mesh_device), tuple(sorted((c.x, c.y) for c in port_cores)))
    if key not in _SEM_CACHE:
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in port_cores])
        sems = tuple(ttnn.create_global_semaphore(mesh_device, crs, 0) for _ in range(2))
        # every chip's counters must be zero before any chip's first increment can arrive
        ttnn.synchronize_device(mesh_device)
        _SEM_CACHE[key] = (mesh_device, sems)
    return _SEM_CACHE[key][1]


def _f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


def _crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


# ----------------------------------------------------------------------------------------------------------------
# MeshProgramDescriptor
# ----------------------------------------------------------------------------------------------------------------


def run_distributed_rms_norm(input_tensor, gamma, plan, *, epsilon, compute_kernel_config, memory_config):
    mesh_device = plan["mesh_device"]
    G, L, gx = plan["G"], plan["L"], plan["gx"]
    S_r, S_w, cw = plan["S_r"], plan["S_w"], plan["core_w_tiles"]
    Wt = plan["tensor_w_tiles"]
    has_gamma = gamma is not None

    g_dtype = gamma.dtype if has_gamma else None
    # Stall-shadow gamma prescale (cb_xg, a full fp32 block) when it fits; otherwise gamma is fused into normalize.
    gamma_prescale = has_gamma and _worker_footprint(plan, input_tensor.dtype, g_dtype, True) <= WORKER_L1_BUDGET
    footprint = _worker_footprint(plan, input_tensor.dtype, g_dtype, gamma_prescale)
    if footprint > WORKER_L1_BUDGET:
        raise RuntimeError(
            f"distributed_rms_norm: per-worker working set {footprint} B exceeds {WORKER_L1_BUDGET} B "
            "(multi-round regime R2 is deferred)"
        )
    if G * plan["slot_bytes"] > SCRATCH_BUDGET:
        raise RuntimeError(f"distributed_rms_norm: exchange scratch {G * plan['slot_bytes']} B exceeds budget")

    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(input_tensor.shape)), input_tensor.dtype, ttnn.TILE_LAYOUT, mesh_device, memory_config
    )

    # ---- cores ----
    port_cores = [ttnn.CoreCoord(2 * l + d, 0) for l in range(L) for d in (0, 1)]
    port_of = {(l, d): port_cores[2 * l + d] for l in range(L) for d in (0, 1)}
    n_used = S_r * S_w
    y_last = 1 + (n_used - 1) // gx  # worker rectangle: logical rows 1..y_last, full width
    worker_cores = [ttnn.CoreCoord(x, y) for y in range(1, y_last + 1) for x in range(gx)]
    worker_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(gx - 1, y_last))])
    port_set = _crs(port_cores)
    all_set = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(gx - 1, y_last))]
        + [ttnn.CoreRange(c, c) for c in port_cores]
    )
    virt = lambda c: mesh_device.worker_core_from_logical_core(c)
    rect0, rect1 = virt(ttnn.CoreCoord(0, 1)), virt(ttnn.CoreCoord(gx - 1, y_last))
    num_mcast_dests = len(worker_cores)

    # ---- exchange scratch: G slots per port core, L1 height-sharded, same address on every chip ----
    slot_bytes = plan["slot_bytes"]
    slot_words = G * slot_bytes // 4
    scratch_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(port_set, (1, slot_words), ttnn.ShardOrientation.ROW_MAJOR),
    )
    scratch = ttnn.allocate_tensor_on_device(
        ttnn.Shape([len(port_cores), slot_words]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh_device, scratch_mem
    )
    scratch_addr = int(scratch.buffer_address())
    arrival_sem, ready_sem = _global_semaphores(mesh_device, port_cores)
    arrival_addr = int(ttnn.get_global_semaphore_address(arrival_sem))
    ready_addr = int(ttnn.get_global_semaphore_address(ready_sem))
    packet_bytes = (ttnn.get_tt_fabric_max_payload_size_bytes() // RECORD_BYTES) * RECORD_BYTES

    # ---- formats / sizes ----
    x_dtype = input_tensor.dtype
    T_x, T_f = _tile_bytes(x_dtype), _tile_bytes(ttnn.float32)
    block_rows = plan["block_rows"]
    n_src_tiles = plan["n_src_tiles"]

    def cb(index, dtype, pages, cores):
        return ttnn.CBDescriptor(
            total_size=pages * _tile_bytes(dtype),
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=_tile_bytes(dtype))
            ],
        )

    cbs = [
        cb(CB_SCALER, ttnn.bfloat16, 1, worker_set),
        cb(CB_SQ_ACC, ttnn.float32, block_rows, worker_set),
        cb(CB_PARTIAL, ttnn.float32, block_rows, worker_set),
        cb(CB_STATS_COMBINE, ttnn.float32, n_src_tiles * block_rows, worker_set),
        cb(CB_RSTD, ttnn.float32, block_rows, worker_set),
    ]
    # CBs whose batch is a tile row of the W chunk are sized from the chunk the core actually owns: CB pointers
    # wrap only on an exact hit of the ring end, so every batch must divide the capacity. The last chunk is
    # ragged (tensor_w_tiles - (S_w - 1) * core_w_tiles), so its cores get their own CB descriptors.
    cw_last = Wt - (S_w - 1) * cw
    ragged = {(k % gx, 1 + k // gx) for k in range(n_used) if k % S_w == S_w - 1} if cw_last != cw else set()
    chunk_groups = [(cw, [c for c in worker_cores if (c.x, c.y) not in ragged])]
    if ragged:
        chunk_groups.append((cw_last, [c for c in worker_cores if (c.x, c.y) in ragged]))
    for cw_group, cores in chunk_groups:
        cores_set = _crs(cores)
        out_batch = out_batch_tiles_knob(cw_group)
        assert cw_group % out_batch == 0, "out_batch_tiles must divide the core's W chunk"
        cbs += [
            cb(CB_X_RESIDENT, x_dtype, X_BLOCKS_IN_FLIGHT * block_rows * cw_group, cores_set),
            cb(CB_OUTPUT_TILES, x_dtype, OUT_DEPTH * out_batch, cores_set),
        ]
        if has_gamma:
            cbs.append(cb(CB_GAMMA_TILES, gamma.dtype, cw_group, cores_set))
        if gamma_prescale:
            cbs.append(cb(CB_XG, ttnn.float32, block_rows * cw_group, cores_set))

    # ---- compile-time args ----
    x_acc = list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    g_acc = list(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if has_gamma
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
    o_acc = list(ttnn.TensorAccessorArgs(output).get_compile_time_args())
    gamma_elem = (4 if gamma.dtype == ttnn.float32 else 2) if has_gamma else 2
    reader_ct = (
        [
            CB_X_RESIDENT,
            CB_GAMMA_TILES,
            CB_SCALER,
            CB_STATS_COMBINE,
            int(has_gamma),
            SEM_STATS_READY,
            gamma_elem,
            int(gamma.buffer_aligned_page_size()) if has_gamma else 0,
            RECORD_BYTES,
            int(has_gamma and not gamma_prescale),
        ]
        + x_acc
        + g_acc
    )
    writer_ct = [CB_PARTIAL, CB_OUTPUT_TILES, SEM_LOCAL_ARRIVALS, RECORD_BYTES] + o_acc
    compute_ct = [
        CB_X_RESIDENT,
        CB_GAMMA_TILES,
        CB_SCALER,
        CB_SQ_ACC,
        CB_PARTIAL,
        CB_XG,
        CB_STATS_COMBINE,
        CB_RSTD,
        CB_OUTPUT_TILES,
        int(has_gamma),
        int(gamma_prescale),
    ]
    port_ct = [SEM_LOCAL_ARRIVALS, SEM_STATS_READY, HEADER_RING]

    inv_w_bits = _f32_bits(1.0 / (G * plan["W"]))
    eps_bits = _f32_bits(epsilon)
    x_addr = int(input_tensor.buffer_address())
    g_addr = int(gamma.buffer_address()) if has_gamma else 0
    o_addr = int(output.buffer_address())
    node = lambda coord: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))

    # workers per link and their (r_g, w)
    workers = []  # (core, r_g, w)
    for k in range(n_used):
        workers.append((ttnn.CoreCoord(k % gx, 1 + k // gx), k // S_w, k % S_w))
    workers_in_link = [sum(1 for _, g, _ in workers if plan["link_of"][g] == l) for l in range(L)]

    mesh_desc = ttnn.MeshProgramDescriptor()
    for grp in plan["groups"]:
        for p, coord in enumerate(grp):
            fwd, bwd = _schedule(p, G, plan["ring"])
            sends = {0: fwd, 1: bwd}
            nxt = grp[(p + 1) % G] if (plan["ring"] or p < G - 1) else None
            prv = grp[(p - 1) % G] if (plan["ring"] or p > 0) else None
            peer_of = {0: nxt, 1: prv}
            # what lands at my port (l, d): the upstream device's same-direction port sends it
            up_of = {0: prv, 1: nxt}
            up_pos = {0: (p - 1) % G, 1: (p + 1) % G}
            inbound = {}
            for d in (0, 1):
                inbound[d] = _schedule(up_pos[d], G, plan["ring"])[d] if up_of[d] is not None else []
            holds = {d: set(sends[d]) | set(inbound[d]) for d in (0, 1)}
            src_dir = [0 if q in holds[0] else 1 for q in range(G)]
            assert all(q in holds[src_dir[q]] for q in range(G)), "every slot must land at one of my ports"

            program = ttnn.ProgramDescriptor()
            program.cbs = cbs
            program.semaphores = [
                ttnn.SemaphoreDescriptor(id=SEM_LOCAL_ARRIVALS, core_ranges=all_set, initial_value=0),
                ttnn.SemaphoreDescriptor(id=SEM_STATS_READY, core_ranges=all_set, initial_value=0),
            ]

            # ---- worker runtime args ----
            reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            used = set()
            for core, g, w in workers:
                used.add((core.x, core.y))
                l = plan["link_of"][g]
                w_start = w * cw
                cw_here = min(cw, Wt - w_start)
                rows = plan["core_rows"][g]
                src_ports = []
                for q in range(G):
                    v = virt(port_of[(l, src_dir[q])])
                    src_ports += [v.x, v.y]
                reader_rt[core.x][core.y] = [
                    x_addr,
                    g_addr,
                    plan["row_start"][g],
                    rows,
                    block_rows,
                    w_start,
                    cw_here,
                    Wt,
                    scratch_addr,
                    slot_bytes,
                    plan["link_rec_base"][g],
                    S_w,
                    n_src_tiles,
                    G,
                    2 * L,
                ] + src_ports
                pub = [virt(port_of[(l, d)]) for d in (0, 1) if sends[d]]
                pub_args = []
                for v in pub:
                    pub_args += [v.x, v.y]
                writer_rt[core.x][core.y] = [
                    o_addr,
                    plan["row_start"][g],
                    rows,
                    block_rows,
                    w_start,
                    cw_here,
                    Wt,
                    scratch_addr + p * slot_bytes,
                    plan["link_rec_base"][g],
                    S_w,
                    w,
                    out_batch_tiles_knob(cw_here),
                    len(pub),
                ] + pub_args
                compute_rt[core.x][core.y] = [rows, block_rows, cw_here, n_src_tiles, inv_w_bits, eps_bits]
            for core in worker_cores:  # idle cores of the rectangle: zero rows (they still receive stats-ready)
                if (core.x, core.y) not in used:
                    reader_rt[core.x][core.y] = [
                        x_addr,
                        g_addr,
                        0,
                        0,
                        block_rows,
                        0,
                        0,
                        Wt,
                        scratch_addr,
                        slot_bytes,
                        0,
                        S_w,
                        n_src_tiles,
                        G,
                        2 * L,
                    ] + [0, 0] * G
                    writer_rt[core.x][core.y] = [o_addr, 0, 0, block_rows, 0, 0, Wt, scratch_addr, 0, S_w, 0, 1, 0]
                    compute_rt[core.x][core.y] = [0, block_rows, 0, n_src_tiles, inv_w_bits, eps_bits]

            # ---- port runtime args ----
            port_rt = ttnn.RuntimeArgs()
            for l in range(L):
                for d in (0, 1):
                    core = port_of[(l, d)]
                    peer = peer_of[d]
                    my_sends = sends[d] if peer is not None else []
                    # the peer's opposite-direction port writes into me: tell it I've started (ready fence)
                    send_ready = peer is not None and _peer_sends(p, d, G, plan["ring"])
                    has_conn = peer is not None and (bool(my_sends) or send_ready)
                    rv = virt(port_of[(l, 1 - d)])
                    pv = virt(port_of[(l, d)])
                    args = [
                        scratch_addr,
                        slot_bytes,
                        plan["recs_link"][l] * RECORD_BYTES,
                        packet_bytes,
                        arrival_addr,
                        ready_addr,
                        workers_in_link[l] if my_sends else 0,
                        len(inbound[d]),
                        int(send_ready),
                        rv.x,
                        rv.y,
                        int(has_conn),
                        pv.x,
                        pv.y,
                    ]
                    if peer is not None:
                        pn = node(peer)
                        args += [int(pn.mesh_id), int(pn.chip_id)]
                    else:
                        args += [0, 0]
                    args += [rect0.x, rect0.y, rect1.x, rect1.y, num_mcast_dests]
                    args += [len(my_sends)] + list(my_sends)
                    if has_conn:
                        link = plan["hops"][(coord, peer)][l]
                        args += list(ttnn.setup_fabric_connection(node(coord), node(peer), link, program, core))
                    port_rt[core.x][core.y] = args

            program.kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "distributed_rms_norm_reader.cpp"),
                    core_ranges=worker_set,
                    compile_time_args=reader_ct,
                    runtime_args=reader_rt,
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "distributed_rms_norm_writer.cpp"),
                    core_ranges=worker_set,
                    compile_time_args=writer_ct,
                    runtime_args=writer_rt,
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "distributed_rms_norm_compute.cpp"),
                    core_ranges=worker_set,
                    compile_time_args=compute_ct,
                    runtime_args=compute_rt,
                    config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=compute_kernel_config.math_fidelity,
                        fp32_dest_acc_en=bool(compute_kernel_config.fp32_dest_acc_en),
                        math_approx_mode=bool(compute_kernel_config.math_approx_mode),
                    ),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "distributed_rms_norm_port.cpp"),
                    core_ranges=port_set,
                    compile_time_args=port_ct,
                    runtime_args=port_rt,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_1_default
                    ),
                ),
            ]
            r, c = coord
            mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = program

    tensors = [input_tensor] + ([gamma] if has_gamma else []) + [scratch, output]
    return ttnn.generic_op(tensors, mesh_desc)


def _peer_sends(p, d, G, ring):
    """Whether the neighbour in direction d sends toward me (its opposite-direction port has entries)."""
    q = (p + 1) % G if d == 0 else (p - 1) % G
    fwd, bwd = _schedule(q, G, ring)
    return bool(bwd if d == 0 else fwd)
