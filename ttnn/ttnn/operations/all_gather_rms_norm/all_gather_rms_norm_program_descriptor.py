# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""all_gather_rms_norm — the MeshProgramDescriptor (one generic_op per call).

Scheme (op_design.md): stats-first, gather-normalized (SF-GN).
  * Port cores: 2L per chip, one per (direction, link), next to the link's Ethernet core. The link-0 ports are the
    stats HUBS (they hold the stats table and exchange it over the fabric, one fused increment per slot).
  * Workers (grid - ports): row-groups x k W members (select_split). Each reduces its block to per-row stats records,
    publishes them into both hubs' tables, waits for the stats round, combines the P = G*k partitions of its rows,
    normalizes its block into cb_out_block (resident) and writes it to its own output pages.
  * The ports pull the own normalized slice from the workers' L1 and send it into the peers' FINAL output pages
    (relay along the ring / line with arrival counters, the far slot of an even ring split between both directions).
  * W_local % 32 != 0 (R_U): the slices go to a slot-contiguous staging scratch instead, then every worker assembles
    its output tiles by element-shifted copies.

Every block knob is a single host value here (block_row_tiles = core_row_tiles, core_w_tiles, k, batch_chunks,
inc_every, scatter runs, balance threshold); CB page counts and loop trips derive from them.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import ttnn


class _LazyFag:
    """fabric_all_gather's host planning helpers (groups, schedules, port placement), imported on first use: the
    example module touches ttnn.Topology at import time, which is not bound while ttnn.operations imports."""

    def __getattr__(self, name):
        from ttnn.operations.examples.fabric_all_gather import program_descriptor_with_inline_kernels as mod

        return getattr(mod, name)


_fag = _LazyFag()

KERNEL_DIR = Path(__file__).parent / "kernels"

# ---- CB indices (kernels/agrn_common.hpp) ----
CB_X = 0
CB_GAMMA = 1
CB_BETA = 2
CB_SCALER = 3
CB_MASK = 4
CB_SUMSQ = 5
CB_LN_M = 6
CB_LN_SHIFT = 7
CB_LN_SQ = 8
CB_PARTIAL = 9
CB_RECORD_STAGING = 10
CB_RECORD_LANDING = 11
CB_GATHERED = 12
CB_STAT_CONSTS = 13
CB_INV = 14
CB_MEAN_COL = 15
CB_MEAN_B = 16
CB_RSTD_B = 17
CB_LN_TMP = 18
CB_OUT_BLOCK = 19
CB_ASSEMBLE = 20
CB_PORT_CHUNKS = 21
CB_X32 = 22
CB_GAMMA_ROWS = 23

# ---- program-local semaphores (declared before the fabric connections take theirs) ----
SEM_RECORDS_LOCAL = 0
SEM_STATS_READY = 1
SEM_OWN_READY = 2
SEM_LANDED = 3

RECORD_BYTES = 128
TILE = 32

# ---- knobs (single source; op_design.md Blocking Model) ----
P_CAP = 128  # stats partitions per row (G * k), <= 4 gather tiles
COMB_TILE_COST_FUSED = 1  # select_split cost model: one gather-tile reduce per row-tile
STATS_TABLE_CAP = 256 * 1024  # bytes of the stats table on each hub
BATCH_CHUNKS = 4  # port CB batch (TRANSPORT; catalog fabric_all_gather: 4 per batch best or within 5-16%)
INC_EVERY = 8  # arrival increment every N packets (+ each entry's last)
BALANCE_MIN_TILES = 64  # balance an even ring's far slot from this many tiles per slot (catalog)

NOC_READER = ttnn.NOC.RISCV_0_default  # NoC0 (fabric_all_gather's convention)
NOC_WRITER = ttnn.NOC.RISCV_1_default  # NoC1


# ---------------------------------------------------------------------------------------------------------------------
# Geometry and work split
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Geometry:
    W_local: int
    M: int
    num_images: int
    R: int  # tile-rows of the slice (leading dims folded)
    Wlt: int  # tile-columns of the local slice
    G: int
    W: int
    Wt: int  # tile-columns of the gathered output
    w_valid_last: int  # valid columns of the last local tile (32 if aligned)

    @property
    def aligned(self) -> bool:
        return self.W_local % TILE == 0


def derive_geometry(shape, G) -> Geometry:
    shape = [int(s) for s in shape]
    W_local, M = shape[-1], shape[-2]
    num_images = math.prod(shape[:-2]) if len(shape) > 2 else 1
    R = num_images * (-(-M // TILE))
    Wlt = -(-W_local // TILE)
    W = G * W_local
    return Geometry(W_local, M, num_images, R, Wlt, G, W, -(-W // TILE), W_local - TILE * (Wlt - 1))


@dataclass(frozen=True)
class Split:
    k: int
    n_groups: int
    rows_hi: int
    big: int  # groups with rows_hi tile-rows; the rest rows_hi - 1
    cw_nom: int

    @property
    def n_active(self):
        return self.n_groups * self.k

    def member_width(self, m, Wlt):
        return max(0, min(self.cw_nom, Wlt - m * self.cw_nom))


def select_split(R, Wlt, Nw, G, k_cap) -> Split:
    """op_design.md Work Distribution: cost = rows * (slice + COMB_TILE_COST_FUSED * ceil(G*k/32)), ties -> smaller k."""
    best = None
    for k in range(1, max(1, min(Wlt, k_cap)) + 1):
        groups = min(Nw // k, R)
        if groups == 0:
            break
        cost = (-(-R // groups)) * ((-(-Wlt // k)) + COMB_TILE_COST_FUSED * (-(-(G * k) // 32)))
        if best is None or cost < best[0]:
            best = (cost, k)
    k = best[1]
    k = -(-Wlt // (-(-Wlt // k)))  # no empty member
    n_groups = min(Nw // k, R)
    rows_hi = -(-R // n_groups)
    big = R - n_groups * (rows_hi - 1)
    return Split(k, n_groups, rows_hi, big, -(-Wlt // k))


# ---------------------------------------------------------------------------------------------------------------------
# Mesh plan: groups, schedules, ports (cached per mesh / axis / topology / links)
# ---------------------------------------------------------------------------------------------------------------------

_PLAN_CACHE = {}


def _node(mesh_device, coord):
    return mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))


def usable_links(mesh_device, cluster_axis, ring):
    """min over the group hops of the forwarding links (None if a needed hop has none)."""
    mesh_shape = tuple(mesh_device.shape)
    best = None
    for grp in _fag.build_groups(mesh_shape, cluster_axis):
        G = len(grp)
        hops = [(grp[i], grp[i + 1]) for i in range(G - 1)]
        if ring:
            hops.append((grp[G - 1], grp[0]))
        for a, b in hops:
            for x, y in ((a, b), (b, a)):
                n = len(ttnn.get_forwarding_link_indices(_node(mesh_device, x), _node(mesh_device, y)))
                best = n if best is None else min(best, n)
    return best


def _eth_physical(mesh_device, coord, channel):
    """Physical NoC (x, y) of the Ethernet core of a fabric channel (cluster descriptor + arch SoC descriptor)."""
    eth_list, _, _ = _fag._chip_maps(mesh_device, coord)
    if eth_list is None or channel >= len(eth_list):
        return None
    return eth_list[channel]


def build_plan(mesh_device, cluster_axis, topology, num_links):
    key = (id(mesh_device), cluster_axis, int(topology == ttnn.Topology.Ring), num_links)
    if key in _PLAN_CACHE:
        return _PLAN_CACHE[key][1]
    mesh_shape = tuple(mesh_device.shape)
    groups = _fag.build_groups(mesh_shape, cluster_axis)
    G = len(groups[0])
    ring = topology == ttnn.Topology.Ring and G > 2  # a 2-chip ring is the line
    grid = mesh_device.compute_with_storage_grid_size()
    allowed = {(x, y) for x in range(grid.x) for y in range(grid.y)}
    chips = {}
    for grp in groups:
        for p, coord in enumerate(grp):
            chips[coord] = dict(
                p=p,
                G=G,
                next=grp[(p + 1) % G] if (ring or p < G - 1) else None,
                prev=grp[(p - 1) % G] if (ring or p > 0) else None,
            )
    for coord, ch in chips.items():
        for d in ("fwd", "bwd"):
            peer = ch["next"] if d == "fwd" else ch["prev"]
            if peer is None:
                ch[f"{d}_links"] = []
                continue
            links = ttnn.get_forwarding_link_indices(_node(mesh_device, coord), _node(mesh_device, peer))
            if len(links) < num_links:
                raise ValueError(
                    f"all_gather_rms_norm: {coord} -> {peer} has {len(links)} usable link(s), {num_links} requested"
                )
            ch[f"{d}_links"] = list(links[:num_links])
    # Port placement: the free core with the fewest NoC1 hops to the link's Ethernet core (fabric_all_gather.plan);
    # the channel comes from the connection's own runtime args (no probe program).
    for coord, ch in chips.items():
        taken = set()
        ports = {}
        for d in ("fwd", "bwd"):
            peer = ch["next"] if d == "fwd" else ch["prev"]
            for l in range(num_links):
                core = None
                if peer is not None:
                    dummy = ttnn.ProgramDescriptor()
                    args = ttnn.setup_fabric_connection(
                        _node(mesh_device, coord),
                        _node(mesh_device, peer),
                        ch[f"{d}_links"][l],
                        dummy,
                        ttnn.CoreCoord(0, 0),
                    )
                    exy = _eth_physical(mesh_device, coord, int(args[0]))
                    if exy is not None:
                        core = _fag._worker_nearest_noc1(mesh_device, coord, exy, taken, allowed)
                ports[(d, l)] = core
                if core is not None:
                    taken.add((core.x, core.y))
        for key2, core in ports.items():
            if core is None:
                core = _fag._worker_below(mesh_device, coord, 0, taken, allowed)
                ports[key2] = core
                taken.add((core.x, core.y))
        ch["ports"] = ports
    plan = dict(chips=chips, G=G, ring=ring, num_links=num_links, grid=(grid.x, grid.y))
    _PLAN_CACHE[key] = (mesh_device, plan)
    return plan


# ---------------------------------------------------------------------------------------------------------------------
# Chunk walk (host replica of agrn_port_common.hpp for_each_chunk: counts only)
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Walk:
    nb: int
    rows_eff: int
    cols_eff: int
    row_stride: int
    slot_stride: int
    run_pages: int

    def chunks(self, q, part, first_bank, stride):
        if first_bank >= self.nb:
            return 0
        nb = self.nb
        nl = (nb - first_bank + stride - 1) // stride
        lo = nl // 2 if part == 2 else 0
        hi = nl // 2 if part == 1 else nl
        cnt = 0
        base = q * self.slot_stride
        for i in range(lo, hi):
            b = first_bank + i * stride
            for rr in range(self.rows_eff):
                rb = base + rr * self.row_stride
                jj = (b - rb) % nb
                if jj < self.cols_eff:
                    avail = (self.cols_eff - 1 - jj) // nb + 1
                    cnt += -(-avail // self.run_pages)
        return cnt


def _inc_sim(counts, inc_every):
    sent = incs = 0
    S0, I0 = [], []
    for n in counts:
        S0.append(sent)
        I0.append(incs)
        for i in range(n):
            sent += 1
            if sent % inc_every == 0 or i == n - 1:
                incs += 1
    return S0, I0, incs


# ---------------------------------------------------------------------------------------------------------------------
# Persistent resources (created once per mesh / port set, outside any trace)
# ---------------------------------------------------------------------------------------------------------------------

_RES_CACHE = {}


def _single_cores(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in sorted(cores)])


def global_semaphores(mesh_device, plan):
    port_cores = {(c.x, c.y) for ch in plan["chips"].values() for c in ch["ports"].values()}
    key = ("sems", id(mesh_device), tuple(sorted(port_cores)))
    if key not in _RES_CACHE:
        cores = _single_cores(port_cores)
        sems = tuple(ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(3))
        ttnn.synchronize_device(mesh_device)  # every chip's counters are 0 before any chip's first call
        _RES_CACHE[key] = (mesh_device, sems)
    return [int(ttnn.get_global_semaphore_address(s)) for s in _RES_CACHE[key][1]]  # ready, bulk, stats


def stats_table(mesh_device, plan, table_bytes):
    hub_cores = {
        (ch["ports"][(d, 0)].x, ch["ports"][(d, 0)].y) for ch in plan["chips"].values() for d in ("fwd", "bwd")
    }
    words = max(8, table_bytes // 4)
    key = ("table", id(mesh_device), tuple(sorted(hub_cores)), words)
    if key not in _RES_CACHE:
        import torch

        cores = _single_cores(hub_cores)
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(cores, (1, words), ttnn.ShardOrientation.ROW_MAJOR),
        )
        t = ttnn.from_torch(
            torch.zeros((len(hub_cores), words), dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=mem,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        _RES_CACHE[key] = (mesh_device, t)
    return _RES_CACHE[key][1]


def staging_buffer(mesh_device, geo: Geometry, dtype):
    key = ("staging", id(mesh_device), geo.G, geo.R, geo.Wlt, dtype)
    if key not in _RES_CACHE:
        t = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, geo.G * geo.R * TILE, geo.Wlt * TILE]),
            dtype,
            ttnn.TILE_LAYOUT,
            mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        _RES_CACHE[key] = (mesh_device, t)
    return _RES_CACHE[key][1]


# ---------------------------------------------------------------------------------------------------------------------
# Program descriptor
# ---------------------------------------------------------------------------------------------------------------------


def _f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def _datum_bytes(dtype):
    return {ttnn.float32: 4, ttnn.bfloat16: 2, ttnn.bfloat8_b: 1}[dtype]


def _cb(index, num_pages, page_size, dtype, core_ranges):
    return ttnn.CBDescriptor(
        total_size=int(num_pages * page_size),
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=int(page_size))],
    )


def _dm(processor, noc):
    return ttnn.DataMovementConfigDescriptor(processor=processor, noc=noc)


def _schedule(p, G, ring, balance):
    return _fag._schedule(p, G, ring, balance)


def create_mesh_program_descriptor(
    mesh_device,
    plan,
    geo: Geometry,
    input_tensor,
    output_tensor,
    gamma,
    beta,
    *,
    is_ln,
    epsilon,
    compute_kernel_config,
):
    chips, G, ring, L = plan["chips"], plan["G"], plan["ring"], plan["num_links"]
    gx, gy = plan["grid"]
    n_ports = 2 * L
    Nw = gx * gy - n_ports
    k_cap = P_CAP // G
    if is_ln:
        k_cap = max(1, min(k_cap, 32 // G))  # LN combine: one gather tile per stat (P <= 32)
    split = select_split(geo.R, geo.Wlt, Nw, G, k_cap)
    k = split.k
    nstats = 3 if is_ln else 1
    P = G * k
    gather_tiles = -(-P // 32)
    crt_max = split.rows_hi
    cw_max = split.cw_nom
    slot_bytes = geo.R * k * nstats * RECORD_BYTES
    table_bytes = G * slot_bytes
    if table_bytes > STATS_TABLE_CAP:
        raise RuntimeError(
            f"all_gather_rms_norm: stats table {table_bytes} B exceeds STATS_TABLE_CAP {STATS_TABLE_CAP} B "
            f"(R={geo.R}, k={k}, nstats={nstats}); a row-windowed stats round (R_WIN) is deferred"
        )

    x_dtype = input_tensor.dtype
    out_dtype = output_tensor.dtype
    is_ru = not geo.aligned
    fp32_dest = bool(compute_kernel_config.fp32_dest_acc_en)
    acc_dtype = ttnn.float32 if fp32_dest else ttnn.bfloat16
    T_in = ttnn.tile_size(x_dtype)
    out_page = int(output_tensor.buffer_aligned_page_size())
    T32 = ttnn.tile_size(ttnn.float32)
    has_gamma, has_beta = gamma is not None, beta is not None
    g_dtype = gamma.dtype if has_gamma else ttnn.bfloat16
    g_eb = _datum_bytes(g_dtype)
    gamma_page = 2 * 16 * 16 * g_eb  # faces 0/1 of a gamma tile (row 0 of each is read)
    gamma_is_tile = has_gamma and gamma.layout == ttnn.TILE_LAYOUT
    gamma_src_page = (TILE * TILE * g_eb) if gamma_is_tile else TILE * g_eb
    x_is_f32 = x_dtype == ttnn.float32

    # destination of the slices: final output pages (aligned) or the slot-contiguous staging scratch (R_U)
    staging = staging_buffer(mesh_device, geo, out_dtype) if is_ru else None
    dst_tensor = staging if is_ru else output_tensor
    dram = mesh_device.dram_grid_size()
    nb = dram.x * dram.y
    run_pages = max(1, ttnn.get_tt_fabric_max_payload_size_bytes() // out_page)
    payload_bytes = run_pages * out_page
    stats_payload = ttnn.get_tt_fabric_max_payload_size_bytes()
    if is_ru:
        walk = Walk(nb, 1, geo.R * geo.Wlt, geo.R * geo.Wlt, geo.R * geo.Wlt, run_pages)
    else:
        walk = Walk(nb, geo.R, geo.Wlt, geo.Wt, geo.Wlt, run_pages)
    balance = ring and G % 2 == 0 and G >= 4 and geo.R * geo.Wlt >= BALANCE_MIN_TILES

    # R_U assembly: max distinct source tiles per output tile
    src_max = 1
    if is_ru:
        for oc in range(geo.Wt):
            pages = set()
            for c in range(TILE):
                g_ = oc * TILE + c
                if g_ >= geo.W:
                    break
                q = g_ // geo.W_local
                pages.add((q, (g_ - q * geo.W_local) // TILE))
            src_max = max(src_max, len(pages))

    ready_addr, bulk_addr, stats_addr = global_semaphores(mesh_device, plan)
    table = stats_table(mesh_device, plan, table_bytes)
    table_addr = int(table.buffer_address())

    # ---- worker compile-time args (agrn_worker_args.hpp) ----
    worker_ct = [
        geo.R,
        geo.Wlt,
        k,
        split.n_groups,
        split.rows_hi,
        split.big,
        split.cw_nom,
        gx,
        G,
        nstats,
        gather_tiles,
        int(is_ln),
        int(has_gamma),
        int(has_beta),
        int(gamma_is_tile),
        T_in,
        out_page,
        g_eb,
        geo.w_valid_last,
        geo.W_local,
        geo.Wt,
        int(is_ru),
        int(x_is_f32),
        _datum_bytes(out_dtype),
        n_ports,
        src_max,
        gamma_page,
        gamma_src_page,
    ]
    acc_ct = list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    acc_ct += list(ttnn.TensorAccessorArgs(dst_tensor).get_compile_time_args())
    acc_ct += list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    acc_ct += list((ttnn.TensorAccessorArgs(gamma) if has_gamma else ttnn.TensorAccessorArgs()).get_compile_time_args())
    acc_ct += list((ttnn.TensorAccessorArgs(beta) if has_beta else ttnn.TensorAccessorArgs()).get_compile_time_args())

    # member widths in elements -> 1/n (LN local mean)
    inv_n = []
    for m in range(k):
        w_tiles = split.member_width(m, geo.Wlt)
        n = w_tiles * TILE - ((TILE - geo.w_valid_last) if m * split.cw_nom + w_tiles == geo.Wlt else 0)
        inv_n.append(_f32_bits(1.0 / max(1, n)))

    # ---- port compile-time args (agrn_port_common.hpp) ----
    port_ct = [
        CB_PORT_CHUNKS,
        out_page,
        run_pages,
        BATCH_CHUNKS,
        INC_EVERY,
        nb,
        walk.rows_eff,
        walk.cols_eff,
        walk.row_stride,
        walk.slot_stride,
        int(is_ru),
        geo.Wlt,
        k,
        split.n_groups,
        split.rows_hi,
        split.big,
        split.cw_nom,
        stats_payload,
    ] + list(ttnn.TensorAccessorArgs(dst_tensor).get_compile_time_args())

    # ---- compute config ----
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = compute_kernel_config.math_fidelity
    cfg.fp32_dest_acc_en = fp32_dest
    cfg.math_approx_mode = compute_kernel_config.math_approx_mode
    if is_ln:
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for cb in (CB_X if x_is_f32 else CB_X32, CB_GATHERED, CB_STAT_CONSTS, CB_LN_TMP, CB_MEAN_B):
            modes[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = modes

    full = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
    in_addr = int(input_tensor.buffer_address())
    out_addr = int(output_tensor.buffer_address())
    dst_addr = int(dst_tensor.buffer_address())
    g_addr = int(gamma.buffer_address()) if has_gamma else 0
    b_addr = int(beta.buffer_address()) if has_beta else 0

    mesh_desc = ttnn.MeshProgramDescriptor()
    for coord, ch in chips.items():
        p = ch["p"]
        ports = ch["ports"]
        port_list = [ports[(d, l)] for d in ("fwd", "bwd") for l in range(L)]
        port_set = {(c.x, c.y) for c in port_list}
        workers = [(x, y) for y in range(gy) for x in range(gx) if (x, y) not in port_set]
        active = workers[: split.n_active]
        virt = lambda c: mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(*c))
        pack = lambda v: (v.x & 0xFFFF) | (v.y << 16)
        active_noc = [pack(virt(c)) for c in active]
        hub = {d: virt((ports[(d, 0)].x, ports[(d, 0)].y)) for d in ("fwd", "bwd")}

        stats_sched = {}
        bulk_sched = {}
        for cc in chips:  # schedules of every chip (for upstream lookups)
            pp = chips[cc]["p"]
            f, b = _schedule(pp, G, ring, False)
            stats_sched[cc] = {"fwd": [q for q, _ in f], "bwd": [q for q, _ in b]}
            f, b = _schedule(pp, G, ring, balance)
            bulk_sched[cc] = {"fwd": f, "bwd": b}
        # which hub holds each slot
        up_of = {"fwd": ch["prev"], "bwd": ch["next"]}
        recv = {d: (stats_sched[up_of[d]][d] if up_of[d] is not None else []) for d in ("fwd", "bwd")}
        hub_mask = 0
        for q in range(G):
            if q == p:
                continue
            in_f, in_b = q in recv["fwd"], q in recv["bwd"]
            assert in_f or in_b, f"slot {q} never reaches chip {p}"
            if not in_f:
                hub_mask |= 1 << q

        program = ttnn.ProgramDescriptor()
        program.semaphores = [
            ttnn.SemaphoreDescriptor(id=sid, core_ranges=full, initial_value=0)
            for sid in (SEM_RECORDS_LOCAL, SEM_STATS_READY, SEM_OWN_READY, SEM_LANDED)
        ]

        # ---- CBs (all on the full program range: identical addresses on every core of the chip) ----
        cbs = [
            _cb(CB_X, crt_max * cw_max, T_in, x_dtype, full),
            _cb(CB_SCALER, 1, ttnn.tile_size(ttnn.bfloat16), ttnn.bfloat16, full),
            _cb(CB_PARTIAL, crt_max * nstats, T32, ttnn.float32, full),
            _cb(CB_RECORD_STAGING, crt_max * nstats, RECORD_BYTES, ttnn.float32, full),
            _cb(CB_RECORD_LANDING, G * k * nstats, RECORD_BYTES, ttnn.float32, full),
            _cb(CB_GATHERED, crt_max * nstats * gather_tiles, T32, ttnn.float32, full),
            _cb(CB_INV, crt_max, T32, ttnn.float32, full),
            _cb(CB_OUT_BLOCK, crt_max * cw_max, out_page, out_dtype, full),
            _cb(CB_PORT_CHUNKS, 2 * BATCH_CHUNKS * run_pages, out_page, out_dtype, full),
        ]
        if has_gamma:
            cbs.append(_cb(CB_GAMMA, cw_max, gamma_page, g_dtype, full))
            cbs.append(_cb(CB_GAMMA_ROWS, 1, (cw_max + 2) * 128 + 64, ttnn.float32, full))
        if has_beta:
            cbs.append(_cb(CB_BETA, cw_max, gamma_page, g_dtype, full))
        if is_ln:
            cbs += [
                # Float32 like cb_ln_sq: the reduce helper's masked fold unpacks the mask as srcB without a format reconfig
                _cb(CB_MASK, 1, T32, ttnn.float32, full),
                _cb(CB_LN_M, crt_max, T32, ttnn.float32, full),
                _cb(CB_LN_SHIFT, crt_max, T32, ttnn.float32, full),
                _cb(CB_LN_SQ, crt_max * cw_max, T32, ttnn.float32, full),
                _cb(CB_STAT_CONSTS, 1, T32, ttnn.float32, full),
                _cb(CB_MEAN_COL, crt_max, T32, ttnn.float32, full),
                _cb(CB_MEAN_B, crt_max, T32, ttnn.float32, full),
                _cb(CB_RSTD_B, crt_max, T32, ttnn.float32, full),
                _cb(CB_LN_TMP, crt_max, T32, ttnn.float32, full),
            ]
            if not x_is_f32:
                cbs.append(_cb(CB_X32, crt_max * cw_max, T32, ttnn.float32, full))
        else:
            cbs.append(_cb(CB_SUMSQ, crt_max, ttnn.tile_size(acc_dtype), acc_dtype, full))
        if is_ru:
            cbs.append(_cb(CB_ASSEMBLE, src_max + 1, out_page, out_dtype, full))
        program.cbs = cbs

        # ---- worker kernels (common runtime args only) ----
        worker_set = full.subtract(_single_cores(port_set))
        common = [
            in_addr,
            dst_addr,
            out_addr,
            g_addr,
            b_addr,
            table_addr,
            _f32_bits(epsilon),
            _f32_bits(1.0 / geo.W),
            p,
            hub_mask,
            pack(hub["fwd"]),
            pack(hub["bwd"]),
        ]
        common += [c.y * gx + c.x for c in port_list]
        common += [pack(virt((c.x, c.y))) for c in port_list]
        common += inv_n
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "all_gather_rms_norm_reader.cpp"),
                core_ranges=worker_set,
                compile_time_args=worker_ct + acc_ct,
                common_runtime_args=common,
                config=_dm(ttnn.DataMovementProcessor.RISCV_1, NOC_READER),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "all_gather_rms_norm_writer.cpp"),
                core_ranges=worker_set,
                compile_time_args=worker_ct + acc_ct,
                common_runtime_args=common,
                config=_dm(ttnn.DataMovementProcessor.RISCV_0, NOC_WRITER),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "all_gather_rms_norm_compute.cpp"),
                core_ranges=worker_set,
                compile_time_args=worker_ct,
                common_runtime_args=common,
                config=cfg,
            ),
        ]

        # ---- port kernels (per-core runtime args) ----
        reader_rt, sender_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        sig = []
        for d in ("fwd", "bwd"):
            peer = ch["next"] if d == "fwd" else ch["prev"]
            up = up_of[d]
            for l in range(L):
                core = ports[(d, l)]
                sends = bulk_sched[coord][d] if peer is not None else []
                up_sends = bulk_sched[up][d] if up is not None else []
                up_counts = [walk.chunks(q, part, l, L) for q, part in up_sends]
                S0, I0, expect_in = _inc_sim(up_counts, INC_EVERY)
                entries = []
                for kk, (q, part) in enumerate(sends):
                    if kk == 0:
                        entries += [q | (part << 16), 0, 0]
                        continue
                    uq, upart = up_sends[kk - 1]
                    assert uq == q and (upart == part or upart == 0), (uq, q, upart, part)
                    off = walk.chunks(q, 1, l, L) if (part == 2 and upart == 0) else 0
                    e = kk - 1
                    entries += [q | (part << 16), I0[e] - S0[e] // INC_EVERY + 1, S0[e] + off]
                r_args = [dst_addr, l, L, bulk_addr, expect_in, split.n_active, int(is_ru), len(sends)]
                r_args += entries + active_noc
                reader_rt[core.x][core.y] = r_args

                is_hub = l == 0
                st_sends = stats_sched[coord][d] if (peer is not None and is_hub) else []
                st_expect = len(stats_sched[up][d]) if (up is not None and is_hub) else 0
                send_ready = peer is not None
                rc = (
                    virt((chips[peer]["ports"][(_fag._OPP[d], l)].x, chips[peer]["ports"][(_fag._OPP[d], l)].y))
                    if send_ready
                    else None
                )
                pc = (
                    virt((chips[peer]["ports"][(d, l)].x, chips[peer]["ports"][(d, l)].y)) if peer is not None else None
                )
                pn = _node(mesh_device, peer) if peer is not None else None
                s_args = [
                    dst_addr,
                    l,
                    L,
                    bulk_addr,
                    ready_addr,
                    int(send_ready),
                    pack(rc) if rc is not None else 0,
                    pack(pc) if pc is not None else 0,
                    int(pn.mesh_id) if pn is not None else 0,
                    int(pn.chip_id) if pn is not None else 0,
                    int(peer is not None),
                    int(is_hub),
                    table_addr,
                    slot_bytes,
                    stats_addr,
                    st_expect,
                    split.n_active,
                    len(st_sends),
                ]
                s_args += st_sends
                s_args += [len(sends)]
                for q, part in sends:
                    s_args += [q | (part << 16), walk.chunks(q, part, l, L)]
                if is_hub:
                    s_args += active_noc
                sig.append((d, l, core.x, core.y, tuple(r_args[1:3] + r_args[4:]), tuple(s_args[1:3] + s_args[5:])))
                if peer is not None:
                    s_args += list(
                        ttnn.setup_fabric_connection(_node(mesh_device, coord), pn, ch[f"{d}_links"][l], program, core)
                    )
                sender_rt[core.x][core.y] = s_args
        port_cores = _single_cores(port_set)
        kernels += [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "all_gather_rms_norm_port_reader.cpp"),
                core_ranges=port_cores,
                compile_time_args=port_ct,
                runtime_args=reader_rt,
                config=_dm(ttnn.DataMovementProcessor.RISCV_1, NOC_READER),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "all_gather_rms_norm_port_sender.cpp"),
                core_ranges=port_cores,
                compile_time_args=port_ct,
                runtime_args=sender_rt,
                config=_dm(ttnn.DataMovementProcessor.RISCV_0, NOC_WRITER),
            ),
        ]
        program.kernels = kernels
        # Program cache key: everything that shapes the program except the per-call tensor addresses.
        h = hashlib.sha256(
            repr(
                (
                    coord,
                    worker_ct,
                    acc_ct,
                    port_ct,
                    common[5:],
                    sig,
                    [(c.x, c.y) for c in port_list],
                    is_ln,
                    fp32_dest,
                    str(cfg.math_fidelity),
                    bool(cfg.math_approx_mode),
                    str(x_dtype),
                    str(out_dtype),
                    str(g_dtype),
                    crt_max,
                    cw_max,
                )
            ).encode()
        ).digest()
        program.custom_program_hash = int.from_bytes(h[:8], "little") & ((1 << 63) - 1)
        r, c = coord
        mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = program
    return mesh_desc, [t for t in (staging, table) if t is not None]
