# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""matmul_reduce_scatter — plan (blocking, schedule, placement) and the MeshProgramDescriptor.

One generic_op dispatch per call. Per chip:
  * compute rectangle (every grid row but the transport row(s)): a 2D-multicast matmul that walks the G scatter
    blocks in `compute_order`; each block is computed by the whole rectangle (per core `core_m_tiles x core_n_tiles`,
    K-blocked) and parked in the core's `cb_partial_handoff` slot (L1, backed by the `handoff_l1` sharded tensor).
      NCRISC: A operand (m-line injector + Mcast1D receivers) + hand-off ready/ack bookkeeping
      BRISC:  W operand (n-line injector + Mcast1D receivers)
      TRISC:  matmul_block (packer-L1 K accumulation, TileRowMajor hand-off)
  * transport row: per (direction, link) one port core, per (link, half) one final core — fabric_reduce_scatter's
    line transport, with "my partial" gathered from compute-core L1 instead of DRAM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32
BF16_TILE_BYTES = 2048

# ---- Blocking knobs (single source of truth; every dependent quantity derives from these) --------------------------
OPERAND_DEPTH = 2  # streamed operand K-blocks in flight (double buffering)
HANDOFF_DEPTH = 2  # hand-off slots per compute core
BLOCKS_IN_FLIGHT = 1  # scatter blocks per compute pass (lamp L3; >1 is a future knob-turn)
STREAM_BUDGET = 384 * 1024  # bytes of streamed operand K-blocks per compute core
K_MIN_RESIDENT = 4  # residency must not force a degenerate K-block
CORE_BLOCK_MAX = 64  # per-core block tiles (core_m_tiles * core_n_tiles); beyond -> R4 waves (deferred)
L1_RESERVE = 64 * 1024  # L1 kept free on compute cores beyond the CBs and the hand-off shard (semaphores, misc)
XPORT_CB_BYTES = 112 * 1024  # transport CB sizing (reference)
XPORT_GROUP_MAX = 8
INC_EVERY = 8  # arrival-counter increment cadence (blackhole-fabric rule 4)
DEST_TILES_16B = 8  # DEST capacity in 16-bit tiles (half-sync); a 32-bit DEST (fp32_dest_acc_en) holds half
XPORT_ADD_BLOCK_MAX = DEST_TILES_16B // 2  # transport add: tiles per CB handshake / DEST batch (always fp32 DEST)

NOC0 = ttnn.NOC.NOC_0
NOC1 = ttnn.NOC.NOC_1

# ---- CB indices -------------------------------------------------------------------------------------------------
CB_ACT_OPERAND = 0
CB_WEIGHT_OPERAND = 1
CB_PARTIAL_ACCUM = 2
CB_PARTIAL_HANDOFF = 3
CB_XPORT_PARTIAL = 4
CB_XPORT_ARRIVAL_A = 5
CB_XPORT_ARRIVAL_B = 6
CB_XPORT_SUM = 16


def _cdiv(a, b):
    return -(-a // b)


def _divisors_desc(n):
    return [d for d in range(n, 0, -1) if n % d == 0]


def _tile_bytes(dtype):
    return (
        int(ttnn.tile_size(dtype))
        if hasattr(ttnn, "tile_size")
        else {ttnn.float32: 4096, ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}[dtype]
    )


# ======================================================================================================================
# Blocking
# ======================================================================================================================


@dataclass
class Blocking:
    Mt: int
    Kt: int
    Nt: int
    G: int
    scatter_dim: int
    blk_m_tiles: int
    blk_n_tiles: int
    orientation: str  # "A": m-lines = grid rows; "B": m-lines = grid columns
    core_m_tiles: int
    core_n_tiles: int
    m_lines: int
    n_lines: int
    k_block_tiles: int
    num_k_blocks: int
    regime: str  # "R1" (invariant operand resident) or "R2" (both streamed)
    a_resident: bool
    w_resident: bool
    out_subblock_h: int
    out_subblock_w: int
    a_tile_bytes: int
    w_tile_bytes: int
    acc_dtype: object  # cb_partial_accum page format: follows the DEST width (fp32_dest_acc_en)
    acc_tile_bytes: int


def _plan_blocking(*, comp_rows, comp_cols, Mt, Kt, Nt, G, scatter_dim, a_dtype, w_dtype, fp32_acc, l1_cb_budget):
    """Grid factorization, K-block and regime (R1 resident invariant operand / R2 streamed) per the design."""
    blk_m, blk_n = (Mt // G, Nt) if scatter_dim == -2 else (Mt, Nt // G)
    best = None
    for orient, (m_avail, n_avail) in (("A", (comp_rows, comp_cols)), ("B", (comp_cols, comp_rows))):
        cm, cn = _cdiv(blk_m, m_avail), _cdiv(blk_n, n_avail)
        ml, nl = _cdiv(blk_m, cm), _cdiv(blk_n, cn)
        key = (cm * cn, -min(cm, cn), ml * nl)
        if best is None or key < best[0]:
            best = (key, orient, cm, cn, ml, nl)
    _, orient, cm, cn, ml, nl = best
    if cm * cn > CORE_BLOCK_MAX:
        raise ValueError(
            f"matmul_reduce_scatter: per-core block {cm}x{cn} tiles exceeds {CORE_BLOCK_MAX} "
            "(needs sub-block waves, regime R4, not built)"
        )

    a_tile, w_tile = _tile_bytes(a_dtype), _tile_bytes(w_dtype)
    acc_dtype = ttnn.float32 if fp32_acc else ttnn.bfloat16
    acc_tile = _tile_bytes(acc_dtype)
    accum = cm * cn * acc_tile
    budget = l1_cb_budget
    # R1: the block-invariant operand X resident for all G blocks, the other (Y) streamed.
    x_is_a = scatter_dim == -1
    core_x, tile_x = (cm, a_tile) if x_is_a else (cn, w_tile)
    core_y, tile_y = (cn, w_tile) if x_is_a else (cm, a_tile)
    resident = core_x * Kt * tile_x
    k_r1 = None
    for k in _divisors_desc(Kt):
        if k < min(K_MIN_RESIDENT, Kt):
            break
        if OPERAND_DEPTH * k * core_y * tile_y <= min(STREAM_BUDGET, budget - resident - accum):
            k_r1 = k
            break
    if k_r1 is not None:
        regime, kbt = "R1", k_r1
        a_res, w_res = x_is_a, not x_is_a
    else:
        regime, kbt = "R2", None
        for k in _divisors_desc(Kt):
            if OPERAND_DEPTH * k * (cm * a_tile + cn * w_tile) <= min(STREAM_BUDGET, budget - accum):
                kbt = k
                break
        if kbt is None:
            raise ValueError("matmul_reduce_scatter: per-core block does not fit L1 even with one-tile K-blocks")
        a_res = w_res = False

    dest_limit = DEST_TILES_16B // 2 if fp32_acc else DEST_TILES_16B
    sb_w = next(d for d in _divisors_desc(cn) if d <= dest_limit)
    sb_h = next(d for d in _divisors_desc(cm) if d * sb_w <= dest_limit)
    return Blocking(
        Mt=Mt,
        Kt=Kt,
        Nt=Nt,
        G=G,
        scatter_dim=scatter_dim,
        blk_m_tiles=blk_m,
        blk_n_tiles=blk_n,
        orientation=orient,
        core_m_tiles=cm,
        core_n_tiles=cn,
        m_lines=ml,
        n_lines=nl,
        k_block_tiles=kbt,
        num_k_blocks=Kt // kbt,
        regime=regime,
        a_resident=a_res,
        w_resident=w_res,
        out_subblock_h=sb_h,
        out_subblock_w=sb_w,
        a_tile_bytes=a_tile,
        w_tile_bytes=w_tile,
        acc_dtype=acc_dtype,
        acc_tile_bytes=acc_tile,
    )


@dataclass
class Transport:
    seg_tiles: int
    seg_bytes: int
    segs_per_row: int
    segs_per_block: int
    xport_group: int
    cap_segs: int


def _plan_transport(blk: Blocking):
    seg_tiles = max(1, min(int(ttnn.get_tt_fabric_max_payload_size_bytes()) // BF16_TILE_BYTES, blk.blk_n_tiles))
    seg_bytes = seg_tiles * BF16_TILE_BYTES
    segs_per_row = _cdiv(blk.blk_n_tiles, seg_tiles)
    group = max(1, min(XPORT_GROUP_MAX, XPORT_CB_BYTES // (2 * seg_bytes)))
    return Transport(
        seg_tiles=seg_tiles,
        seg_bytes=seg_bytes,
        segs_per_row=segs_per_row,
        segs_per_block=blk.blk_m_tiles * segs_per_row,
        xport_group=group,
        cap_segs=2 * group,
    )


# ======================================================================================================================
# Schedule
# ======================================================================================================================


def _schedule_mmrs(p, G, ring=False):
    """Per-port block lists (send order) and compute order for line position p.

    fwd: blocks sent toward p+1, farthest first; bwd: toward p-1. compute_order interleaves fwd/bwd one-by-one
    starting with fwd and ends with the own block p (the finals need it last)."""
    if ring:
        raise NotImplementedError("matmul_reduce_scatter: Ring schedule (R3) is not built")
    fwd = list(range(G - 1, p, -1))
    bwd = list(range(0, p))
    order = []
    for i in range(max(len(fwd), len(bwd))):
        if i < len(fwd):
            order.append(fwd[i])
        if i < len(bwd):
            order.append(bwd[i])
    order.append(p)
    return fwd, bwd, order


def _count_segs(first, stride, total):
    return _cdiv(total - first, stride) if first < total else 0


def _incs(n):
    return _cdiv(n, INC_EVERY) if n else 0


# ======================================================================================================================
# Placement
# ======================================================================================================================


@dataclass
class Placement:
    grid_x: int
    grid_y: int
    transport_rows: int
    fwd_ports: list = field(default_factory=list)  # logical CoreCoord per link
    bwd_ports: list = field(default_factory=list)
    finals: list = field(default_factory=list)  # [link 0 half 0, link 0 half 1, link 1 half 0, ...]
    vx: list = field(default_factory=list)  # virtual NoC x per logical column
    vy: list = field(default_factory=list)  # virtual NoC y per logical row


def _plan_placement(mesh_device, num_links):
    grid = mesh_device.compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    n_xport = 4 * num_links
    t_rows = _cdiv(n_xport, gx)
    if t_rows >= gy:
        raise ValueError("matmul_reduce_scatter: the core grid has no rows left for compute")
    xport = [ttnn.CoreCoord(i % gx, i // gx) for i in range(n_xport)]
    pl = Placement(grid_x=gx, grid_y=gy, transport_rows=t_rows)
    for l in range(num_links):
        pl.fwd_ports.append(xport[4 * l])
        pl.bwd_ports.append(xport[4 * l + 1])
        pl.finals += [xport[4 * l + 2], xport[4 * l + 3]]
    virt = lambda x, y: mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
    pl.vx = [int(virt(x, 0).x) for x in range(gx)]
    pl.vy = [int(virt(0, y).y) for y in range(gy)]
    # the hand-off gather addresses a compute core by (m-line coord, n-line coord): coordinates must be separable
    for y in range(gy):
        for x in range(gx):
            v = virt(x, y)
            assert (int(v.x), int(v.y)) == (pl.vx[x], pl.vy[y]), "non-separable worker coordinates"
    return pl


def compute_core(pl: Placement, blk: Blocking, ml, nl):
    """Logical core of (m-line, n-line)."""
    if blk.orientation == "A":
        return ttnn.CoreCoord(nl, pl.transport_rows + ml)
    return ttnn.CoreCoord(ml, pl.transport_rows + nl)


def compute_rect(pl: Placement, blk: Blocking):
    nx, ny = (blk.n_lines, blk.m_lines) if blk.orientation == "A" else (blk.m_lines, blk.n_lines)
    return ttnn.CoreRange(ttnn.CoreCoord(0, pl.transport_rows), ttnn.CoreCoord(nx - 1, pl.transport_rows + ny - 1))


# ======================================================================================================================
# Program descriptor
# ======================================================================================================================


def _cset(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def _cb(index, num_pages, page_bytes, dtype, core_ranges):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_bytes,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_bytes)],
    )


def _dm(risc, noc):
    return ttnn.DataMovementConfigDescriptor(processor=getattr(ttnn.DataMovementProcessor, risc), noc=noc)


def create_mesh_program_descriptor(
    mesh_device,
    *,
    a,
    w,
    scratch,
    handoff,
    output,
    sems,
    blk: Blocking,
    xp: Transport,
    pl: Placement,
    groups,
    cluster_axis,
    links,
    num_links,
    compute_config,
):
    """groups: {coord: (p, prev_coord|None, next_coord|None)}; links: {(coord, peer): [link ids]}."""
    sem_arr_fwd, sem_arr_bwd, sem_ready_fence, sem_block_ready = sems[:4]
    sem_block_ack = sems[4:7]  # one ack counter per consumer kind: fwd ports, bwd ports, finals
    G = blk.G
    L = num_links
    cm, cn = blk.core_m_tiles, blk.core_n_tiles
    block_tiles = cm * cn
    rect = compute_rect(pl, blk)
    rect_set = ttnn.CoreRangeSet([rect])
    n_cc = blk.m_lines * blk.n_lines
    node = lambda coord: mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))
    virt = lambda c: (pl.vx[c.x], pl.vy[c.y])
    packed = lambda c: (pl.vx[c.x] << 16) | pl.vy[c.y]

    a_ct = list(ttnn.TensorAccessorArgs(a).get_compile_time_args())
    w_ct = list(ttnn.TensorAccessorArgs(w).get_compile_time_args())
    scr_ct = list(ttnn.TensorAccessorArgs(scratch).get_compile_time_args())
    out_ct = list(ttnn.TensorAccessorArgs(output).get_compile_time_args())
    a_addr, w_addr = int(a.buffer_address()), int(w.buffer_address())
    scr_addr, out_addr = int(scratch.buffer_address()), int(output.buffer_address())
    handoff_base = int(handoff.buffer_address())

    a_pages = cm * blk.Kt if blk.a_resident else OPERAND_DEPTH * cm * blk.k_block_tiles
    w_pages = blk.Kt * cn if blk.w_resident else OPERAND_DEPTH * blk.k_block_tiles * cn
    xport_pages = xp.cap_segs * xp.seg_tiles
    # transport add block: the largest divisor of the transport CB ring <= the DEST batch (blocks never straddle the wrap)
    xport_add_block = next(d for d in _divisors_desc(xport_pages) if d <= XPORT_ADD_BLOCK_MAX)

    # compute-rectangle virtual bounds for the ack multicast (NoC0: start = min corner)
    x0, y0 = virt(rect.start)
    x1, y1 = virt(rect.end)
    m_on_y = 1 if blk.orientation == "A" else 0
    mcoords = [pl.vy[pl.transport_rows + ml] if m_on_y else pl.vx[ml] for ml in range(blk.m_lines)]
    ncoords = [pl.vx[nl] if m_on_y else pl.vy[pl.transport_rows + nl] for nl in range(blk.n_lines)]
    a_line_shape = ttnn.Mcast1DShape.PerRow if blk.orientation == "A" else ttnn.Mcast1DShape.PerColumn
    w_line_shape = ttnn.Mcast1DShape.PerColumn if blk.orientation == "A" else ttnn.Mcast1DShape.PerRow
    consumers = [packed(c) for c in pl.fwd_ports] + [packed(c) for c in pl.bwd_ports] + [packed(c) for c in pl.finals]

    mesh_desc = ttnn.MeshProgramDescriptor()
    for coord, (p, prev, nxt) in groups.items():
        fwd, bwd, order = _schedule_mmrs(p, G)
        cidx = {j: i for i, j in enumerate(order)}
        program = ttnn.ProgramDescriptor()
        kernels = []

        # ---------------- compute rectangle ----------------
        cbs = [
            _cb(CB_ACT_OPERAND, a_pages, blk.a_tile_bytes, a.dtype, rect_set),
            _cb(CB_WEIGHT_OPERAND, w_pages, blk.w_tile_bytes, w.dtype, rect_set),
            _cb(CB_PARTIAL_ACCUM, block_tiles, blk.acc_tile_bytes, blk.acc_dtype, rect_set),
            ttnn.cb_descriptor_from_sharded_tensor(CB_PARTIAL_HANDOFF, handoff),
        ]
        # per compute-order index: block, consumer kind (0 fwd ports, 1 bwd ports, 2 finals), cumulative acks
        order_rt, cum = [], [0, 0, 0]  # cumulative acks per consumer kind (each kind acks in order)
        for j in order:
            kind = 2 if j == p else (0 if j > p else 1)
            cum[kind] += 2 * L if kind == 2 else L
            order_rt += [j, kind, cum[kind]]

        a_groups = {1: [], 0: []}
        w_groups = {1: [], 0: []}
        rd_rt, wr_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for ml in range(blk.m_lines):
            for nl in range(blk.n_lines):
                c = compute_core(pl, blk, ml, nl)
                a_groups[int(nl == 0)].append(c)
                w_groups[int(ml == 0)].append(c)
                row0 = ml * cm
                col0 = nl * cn
                rd_rt[c.x][c.y] = (
                    [
                        a_addr,
                        blk.Kt,
                        row0,
                        min(cm, blk.blk_m_tiles - row0),
                        blk.blk_m_tiles if blk.scatter_dim == -2 else 0,
                        sem_block_ready,
                        *sem_block_ack,
                    ]
                    + order_rt
                    + consumers
                )
                wr_rt[c.x][c.y] = [
                    w_addr,
                    blk.Nt,
                    col0,
                    min(cn, blk.blk_n_tiles - col0),
                    blk.blk_n_tiles if blk.scatter_dim == -1 else 0,
                ] + list(order)

        reader_kernels, writer_kernels = [], []
        for sends, cores in a_groups.items():
            if not cores:
                continue
            rt = ttnn.RuntimeArgs()
            for c in cores:
                rt[c.x][c.y] = rd_rt[c.x][c.y]
            reader_kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "matmul_reduce_scatter_reader.cpp"),
                    core_ranges=_cset(cores),
                    compile_time_args=[
                        CB_ACT_OPERAND,
                        CB_PARTIAL_HANDOFF,
                        cm,
                        blk.k_block_tiles,
                        blk.num_k_blocks,
                        G,
                        block_tiles,
                        blk.a_tile_bytes,
                        sends,
                        int(blk.a_resident),
                        L,
                    ]
                    + a_ct,
                    runtime_args=rt,
                    config=_dm("RISCV_1", NOC0),
                )
            )
        for sends, cores in w_groups.items():
            if not cores:
                continue
            rt = ttnn.RuntimeArgs()
            for c in cores:
                rt[c.x][c.y] = wr_rt[c.x][c.y]
            writer_kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "matmul_reduce_scatter_writer.cpp"),
                    core_ranges=_cset(cores),
                    compile_time_args=[
                        CB_WEIGHT_OPERAND,
                        cn,
                        blk.k_block_tiles,
                        blk.num_k_blocks,
                        G,
                        blk.w_tile_bytes,
                        sends,
                        int(blk.w_resident),
                    ]
                    + w_ct,
                    runtime_args=rt,
                    config=_dm("RISCV_0", NOC1),
                )
            )
        mc_a = ttnn.Mcast1D(
            mesh_device,
            rect_set,
            a_line_shape,
            ttnn.Mcast1DFixedSenderConfig(),
            ttnn.McastConfig(noc=ttnn.NOC.NOC_0, handshake=True),
        )
        mc_w = ttnn.Mcast1D(
            mesh_device,
            rect_set,
            w_line_shape,
            ttnn.Mcast1DFixedSenderConfig(),
            ttnn.McastConfig(noc=ttnn.NOC.NOC_1, handshake=True),
        )
        mc_a.attach(program, "a", reader_kernels)
        mc_w.attach(program, "w", writer_kernels)
        kernels += reader_kernels + writer_kernels
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "matmul_reduce_scatter_compute.cpp"),
                core_ranges=rect_set,
                compile_time_args=[
                    CB_ACT_OPERAND,
                    CB_WEIGHT_OPERAND,
                    CB_PARTIAL_ACCUM,
                    CB_PARTIAL_HANDOFF,
                    cm // blk.out_subblock_h,
                    cn // blk.out_subblock_w,
                    blk.out_subblock_h,
                    blk.out_subblock_w,
                    blk.k_block_tiles,
                    blk.num_k_blocks,
                    G,
                ],
                runtime_args=[],
                config=compute_config,
            )
        )

        # ---------------- transport row ----------------
        line_rt = [m for m in mcoords] + [n for n in ncoords]

        def xport_reader_rt(first, stride, full, entries, arr_a, arr_b, ack_sem):
            return (
                [
                    scr_addr,
                    handoff_base,
                    block_tiles * BF16_TILE_BYTES,
                    xp.segs_per_block,
                    xp.segs_per_row,
                    blk.blk_n_tiles,
                    first,
                    stride,
                    full,
                    sem_block_ready,
                    n_cc,
                    arr_a,
                    arr_b,
                    ack_sem,
                    x0,
                    y0,
                    x1,
                    y1,
                    blk.m_lines,
                    blk.n_lines,
                    len(entries),
                ]
                + [v for e in entries for v in e]
                + line_rt
            )

        def xport_reader_kernel(cores, rt, target, has_a, has_b):
            return ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "matmul_reduce_scatter_xport_reader.cpp"),
                core_ranges=_cset(cores),
                compile_time_args=[
                    target,
                    CB_XPORT_ARRIVAL_A,
                    CB_XPORT_ARRIVAL_B,
                    has_a,
                    has_b,
                    xp.seg_tiles,
                    xp.xport_group,
                    INC_EVERY,
                    cm,
                    cn,
                    m_on_y,
                    xp.cap_segs,
                    BF16_TILE_BYTES,
                ]
                + scr_ct,
                runtime_args=rt,
                config=_dm("RISCV_1", NOC0),
            )

        def add_kernel(cores, rt, has_a, has_b):
            return ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "matmul_reduce_scatter_xport_add.cpp"),
                core_ranges=_cset(cores),
                compile_time_args=[
                    CB_XPORT_PARTIAL,
                    CB_XPORT_ARRIVAL_A,
                    CB_XPORT_ARRIVAL_B,
                    CB_XPORT_SUM,
                    has_a,
                    has_b,
                    xport_add_block,
                ],
                runtime_args=rt,
                # every cross-device addition accumulates in fp32 (requirement, independent of the user's config)
                config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
            )

        relay_ports, end_ports, finals = [], [], list(pl.finals)
        rd_relay, rd_end, add_relay, snd_rt = (
            ttnn.RuntimeArgs(),
            ttnn.RuntimeArgs(),
            ttnn.RuntimeArgs(),
            ttnn.RuntimeArgs(),
        )
        senders = []
        for d, blocks, peer, up, port_list in (
            ("fwd", fwd, nxt, prev, pl.fwd_ports),
            ("bwd", bwd, prev, nxt, pl.bwd_ports),
        ):
            if peer is None:
                continue  # no neighbour in this direction: nothing to send, nobody sends a ready to it
            relay = up is not None
            opp_ports = pl.bwd_ports if d == "fwd" else pl.fwd_ports
            for l in range(L):
                core = port_list[l]
                full = _count_segs(l, L, xp.segs_per_block)
                entries = [(cidx[j] % HANDOFF_DEPTH, j, 0) for j in blocks]
                rt = xport_reader_rt(
                    l, L, full, entries, sem_arr_fwd, sem_arr_fwd, sem_block_ack[0 if d == "fwd" else 1]
                )
                (rd_relay if relay else rd_end)[core.x][core.y] = rt
                (relay_ports if relay else end_ports).append(core)
                if relay:
                    add_relay[core.x][core.y] = [len(blocks) * full * xp.seg_tiles]
                # sender
                slots = [G if (d == "bwd" and j == p - 1) else j for j in blocks]
                rc = virt(opp_ports[l])  # peer chip's opposite-direction port (same placement on every chip)
                pc = virt(port_list[l])  # downstream port of the same (direction, link)
                f0, f1 = virt(pl.finals[2 * l]), virt(pl.finals[2 * l + 1])
                pn = node(peer)
                args = [
                    scr_addr,
                    xp.segs_per_block,
                    xp.segs_per_row,
                    blk.blk_n_tiles,
                    l,
                    L,
                    full,
                    sem_arr_fwd,
                    sem_arr_fwd if d == "fwd" else sem_arr_bwd,
                    _incs(len(blocks) * full) if relay else 0,
                    sem_ready_fence,
                    1,
                    rc[0],
                    rc[1],
                    pc[0],
                    pc[1],
                    f0[0],
                    f0[1],
                    f1[0],
                    f1[1],
                    _count_segs(l, 2 * L, xp.segs_per_block),
                    _count_segs(l + L, 2 * L, xp.segs_per_block),
                    int(pn.mesh_id),
                    int(pn.chip_id),
                    len(blocks),
                ] + slots
                args += list(ttnn.setup_fabric_connection(node(coord), pn, links[(coord, peer)][l], program, core))
                snd_rt[core.x][core.y] = args
                senders.append(core)

        has_fa, has_fb = int(prev is not None), int(nxt is not None)
        rd_final, add_final, wr_final = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, core in enumerate(finals):  # final (l, h): segments l + h L, stride 2 L
            first = i // 2 + (i % 2) * L
            n = _count_segs(first, 2 * L, xp.segs_per_block)
            rd_final[core.x][core.y] = xport_reader_rt(
                first, 2 * L, n, [(cidx[p] % HANDOFF_DEPTH, p, G)], sem_arr_fwd, sem_arr_bwd, sem_block_ack[2]
            )
            add_final[core.x][core.y] = [n * xp.seg_tiles]
            wr_final[core.x][core.y] = [
                out_addr,
                xp.segs_per_block,
                xp.segs_per_row,
                blk.blk_n_tiles,
                first,
                2 * L,
                sem_arr_fwd,
                _incs(n) if has_fa else 0,
                sem_arr_bwd,
                _incs(n) if has_fb else 0,
            ]

        xport_cores = relay_ports + end_ports + finals
        cbs.append(_cb(CB_XPORT_SUM, xport_pages, BF16_TILE_BYTES, ttnn.bfloat16, _cset(xport_cores)))
        if relay_ports + finals:
            cbs.append(_cb(CB_XPORT_PARTIAL, xport_pages, BF16_TILE_BYTES, ttnn.bfloat16, _cset(relay_ports + finals)))
        arr_a_cores = relay_ports + (finals if has_fa else [])
        if arr_a_cores:
            cbs.append(_cb(CB_XPORT_ARRIVAL_A, xport_pages, BF16_TILE_BYTES, ttnn.bfloat16, _cset(arr_a_cores)))
        if has_fb:
            cbs.append(_cb(CB_XPORT_ARRIVAL_B, xport_pages, BF16_TILE_BYTES, ttnn.bfloat16, _cset(finals)))

        if relay_ports:
            kernels.append(xport_reader_kernel(relay_ports, rd_relay, CB_XPORT_PARTIAL, 1, 0))
            kernels.append(add_kernel(relay_ports, add_relay, 1, 0))
        if end_ports:
            kernels.append(xport_reader_kernel(end_ports, rd_end, CB_XPORT_SUM, 0, 0))
        kernels.append(xport_reader_kernel(finals, rd_final, CB_XPORT_PARTIAL, has_fa, has_fb))
        kernels.append(add_kernel(finals, add_final, has_fa, has_fb))
        if senders:
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "matmul_reduce_scatter_port_sender.cpp"),
                    core_ranges=_cset(senders),
                    compile_time_args=[
                        CB_XPORT_SUM,
                        xp.seg_tiles,
                        xp.xport_group,
                        INC_EVERY,
                        xp.cap_segs,
                        BF16_TILE_BYTES,
                    ]
                    + scr_ct,
                    runtime_args=snd_rt,
                    config=_dm("RISCV_0", NOC1),
                )
            )
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "matmul_reduce_scatter_final_writer.cpp"),
                core_ranges=_cset(finals),
                compile_time_args=[CB_XPORT_SUM, xp.seg_tiles, xp.xport_group, xp.cap_segs, BF16_TILE_BYTES] + out_ct,
                runtime_args=wr_final,
                config=_dm("RISCV_0", NOC1),
            )
        )
        program.cbs = cbs
        program.kernels = kernels
        r, c = coord
        mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = program
    return mesh_desc
