# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""chunk_gated_delta_rule_fwd — ProgramDescriptor (regime R1 of op_design.md).

ONE `ttnn.generic_op` dispatch.  Three stages live inside the same three kernel binaries and are
sequenced by SEGMENTED semaphore handoffs:

    P  item-parallel prep over (bh, chunk) on `num_item_cores = min(G, NI)` cores (strided,
       chunk-major item order wi = i*BH + bh)
    S  one scan unit per (bh, v_block) on core G-1-(u mod G), walking the chunks sequentially
    E  item-parallel output assembly on the P cores (the same items, same order)

P -> S: after the write barrier of item (bh, i) the P writer adds +1 to sem_ready[seg(i)] on each of
the NV scan cores of bh.  S -> E: after the write barrier of segment j a scan writer adds +1 to
sem_done[j] on every core holding an E item of (bh, segment j).  Every core runs all its P items,
then its scan units, then its E items (in all three kernels), which is what makes the rendezvous
deadlock-free.

BLOCKING KNOBS (single source of truth; every dependent quantity derives from these):

    BLOCK_HEADS          heads per item                                    [1]  (R3 would change it)
    BLOCK_CHUNKS         chunks per item                                   [1]
    GATHER_STAGE_TOKENS  tokens per face-row staging window                [16]
    GATHER_DEPTH         staging windows in flight                         [2]
    SCAN_STREAM_DEPTH    scan operand prefetch depth                       [2]
    EGRESS_DEPTH         compute -> writer blocks in flight                [2]
    ACCUM_DEPTH          in-place-updated CB depth (NOT tunable: 1 hangs)  [2]
    BLOCK_DEPTH          per-item working-set CB depth                     [1]
    READY_SEGMENTS_MAX   handoff segments cap (2 semaphores each, <= 8)    [4]
    item_block_val_tiles (Vi)  solved: largest divisor of Vt that fits L1
    scan_block_val_tiles (Vs)  solved: occupancy-first V split of the scan
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# --------------------------------------------------------------------------------------------
# Blocking knobs
# --------------------------------------------------------------------------------------------
BLOCK_HEADS = 1
BLOCK_CHUNKS = 1
GATHER_STAGE_TOKENS = 16
GATHER_DEPTH = 2
SCAN_STREAM_DEPTH = 2
EGRESS_DEPTH = 2
ACCUM_DEPTH = 2
BLOCK_DEPTH = 1
READY_SEGMENTS_MAX = 4

# L1 that `get_max_worker_l1_unreserved_size()` still counts but the kernel-config ring buffer
# occupies (measured on gated_delta_net_backward's three binaries).
L1_KERNEL_CONFIG_RESERVE = 80 * 1024

# A tile row spans two faces 256 elements apart: one 272-element span read covers it.  Each
# staging line carries 64 B of slack so it can start at (dram_offset % 64).
ROW_SPAN_ELEMS = 272
ROW_SPAN_SLACK = 64
SCALAR_STAGE_BYTES = 32 * 16  # one 16-byte slot per token of a tile

MAX_HEADS = 32  # page index assumes ceil(H / 32) == 1
MAX_RT_ARGS = 341

# --------------------------------------------------------------------------------------------
# CB indices (mirror kernels/cgdr_common.hpp)
# --------------------------------------------------------------------------------------------
CB_CONST = 0
CB_GATHER_STAGE = 1
CB_SCALAR_STAGE = 2
CB_Q_IN = 3
CB_K_IN = 4
CB_VBLOCK_IN = 5
CB_GATE_IN = 6
CB_VEC = 7
CB_QS = 8
CB_KB = 9
CB_KW = 10
CB_L = 11
CB_CC_A = 12
CB_CC_B = 13
CB_T = 14
CB_POW = 15
CB_VMAT = 16
CB_INTRA_IN = 17
CB_VNEW_IN = 18
CB_KMAT_IN = 20
CB_SCAN_PT = 21
CB_SCAN_VCORR = 22
CB_SCAN_GAMMA = 23
CB_STATE = 25
CB_SCAN_VNEW = 26
CB_SCRATCH_EGRESS = 27
CB_OUT_EGRESS = 28

_IN_DTYPE_CBS = (CB_Q_IN, CB_K_IN, CB_VBLOCK_IN, CB_GATE_IN, CB_OUT_EGRESS)
_RAW_CBS = (CB_GATHER_STAGE, CB_SCALAR_STAGE)


def _ceil_div(a, b):
    return (a + b - 1) // b


def _round_up(a, m):
    return _ceil_div(a, m) * m


def default_compute_kernel_config():
    """The ONE definition of `compute_kernel_config=None` (a fresh descriptor per call)."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )


class _Geometry:
    """Every derived extent in one place."""

    def __init__(self, B, T, H, K, V, chunk_size, elem_bytes, num_cores):
        self.B, self.T, self.H, self.K, self.V = B, T, H, K, V
        self.C = chunk_size
        self.Ct = chunk_size // 32
        self.Kt = K // 32
        self.Vt = V // 32
        self.NC = _ceil_div(T, chunk_size * BLOCK_CHUNKS) * BLOCK_CHUNKS
        self.Tt = _ceil_div(T, 32)
        self.BH = B * H
        self.NI = self.BH * self.NC
        self.G = num_cores
        self.neumann_steps = max(1, int(math.ceil(math.log2(chunk_size))))
        self.elem_bytes = elem_bytes
        self.row_span_stride = _round_up(ROW_SPAN_ELEMS * elem_bytes + ROW_SPAN_SLACK, 64)

        # Scan split (occupancy-first): NV = largest divisor of Vt with BH * NV <= G, else 1.
        self.NV = 1
        for d in range(self.Vt, 0, -1):
            if self.Vt % d == 0 and self.BH * d <= self.G:
                self.NV = d
                break
        self.Vs = self.Vt // self.NV
        self.NU = self.BH * self.NV

        # Handoff segments: NS = min(NC, READY_SEGMENTS_MAX), chunks per segment derived, then the
        # segment count re-derived so no segment is empty.
        ns = min(self.NC, READY_SEGMENTS_MAX)
        self.seg_chunks = _ceil_div(self.NC, ns)
        self.NS = _ceil_div(self.NC, self.seg_chunks)

        self.num_item_cores = min(self.G, self.NI)


def _quanta(geo, Vi):
    """The ONE uniform push size of every CB reused across stage roles (ring-wrap invariant)."""
    Ct, Kt, Vs = geo.Ct, geo.Kt, geo.Vs
    return {
        "QV": max(Ct * Vi, Kt * Vs, Kt * Vi),  # P v[:, vb] | S initial_state | E h_i[:, vb]
        "QF": Ct * max(Kt, Ct, Vi, Vs, 1),  # nkcd, Q, intra, P^T, Gamma, v_corr | v_new
        "QO": max(Ct, Ct * Ct, Kt * Vs, Ct * Vi),  # decay, Tinv | h_i, final_state | o, v_new
    }


def _num_const_pages(geo):
    return 4 * geo.Ct * geo.Ct + geo.Ct + 1  # EYE, LT, SL, SU [C,C]; ONES [1,C]; E_ROW0


def _vec_pages(geo):
    return 2 * geo.Ct + 1  # gamma, w (column tiles); Gamma_full (one tile)


def _cb_pages(geo, Vi):
    """Page count per CB — the single source for the footprint solve and the descriptor."""
    Ct, Kt, Vs = geo.Ct, geo.Kt, geo.Vs
    q = _quanta(geo, Vi)
    return {
        CB_CONST: _num_const_pages(geo),
        CB_GATHER_STAGE: GATHER_DEPTH,
        CB_SCALAR_STAGE: 1,
        CB_Q_IN: BLOCK_DEPTH * Ct * Kt,
        CB_K_IN: BLOCK_DEPTH * Ct * Kt,
        CB_VBLOCK_IN: BLOCK_DEPTH * q["QV"],
        CB_GATE_IN: BLOCK_DEPTH * 2 * Ct,
        CB_VEC: _vec_pages(geo),
        CB_QS: BLOCK_DEPTH * Ct * Kt,
        CB_KB: ACCUM_DEPTH * Ct * Kt,
        CB_KW: BLOCK_DEPTH * Ct * Kt,
        CB_L: BLOCK_DEPTH * Ct * Ct,
        CB_CC_A: BLOCK_DEPTH * Ct * Ct,
        CB_CC_B: BLOCK_DEPTH * Ct * Ct,
        CB_T: ACCUM_DEPTH * Ct * Ct,
        CB_POW: ACCUM_DEPTH * Ct * Ct,
        CB_VMAT: BLOCK_DEPTH * Ct * Vi,
        CB_INTRA_IN: BLOCK_DEPTH * Ct * Ct,
        CB_VNEW_IN: BLOCK_DEPTH * Ct * Vi,
        CB_KMAT_IN: SCAN_STREAM_DEPTH * Ct * Kt,
        CB_SCAN_PT: SCAN_STREAM_DEPTH * Kt * Ct,
        CB_SCAN_VCORR: SCAN_STREAM_DEPTH * Ct * Vs,
        CB_SCAN_GAMMA: SCAN_STREAM_DEPTH,
        CB_STATE: ACCUM_DEPTH * Kt * Vs,
        CB_SCAN_VNEW: BLOCK_DEPTH * Ct * Vs,
        CB_SCRATCH_EGRESS: EGRESS_DEPTH * q["QF"],
        CB_OUT_EGRESS: EGRESS_DEPTH * q["QO"],
    }


def _page_bytes(geo, idx, in_tile, f32_tile):
    if idx == CB_GATHER_STAGE:
        return GATHER_STAGE_TOKENS * geo.row_span_stride + 64
    if idx == CB_SCALAR_STAGE:
        return SCALAR_STAGE_BYTES
    if idx in _IN_DTYPE_CBS:
        return in_tile
    return f32_tile


def _footprint(geo, Vi, in_tile, f32_tile):
    return sum(n * _page_bytes(geo, idx, in_tile, f32_tile) for idx, n in _cb_pages(geo, Vi).items())


def _solve_item_block_val_tiles(geo, in_tile, f32_tile, budget):
    """Largest divisor of Vt whose closed-form L1 footprint fits (l1_ledger.md)."""
    for vi in range(geo.Vt, 0, -1):
        if geo.Vt % vi == 0 and _footprint(geo, vi, in_tile, f32_tile) <= budget:
            return vi
    raise ValueError(
        "chunk_gated_delta_rule_fwd: no item_block_val_tiles fits L1 "
        f"(Ct={geo.Ct}, Kt={geo.Kt}, Vt={geo.Vt}, Vs={geo.Vs}, budget={budget} B, "
        f"footprint at Vi=1: {_footprint(geo, 1, in_tile, f32_tile)} B)"
    )


class _ScratchMap:
    """Tile-index bases inside the flat fp32 DRAM scratch (per-item strides, item index wi)."""

    def __init__(self, geo):
        Ct, Kt, Vt, NI = geo.Ct, geo.Kt, geo.Vt, geo.NI
        strides = (
            ("nkcd", Ct * Kt),
            ("pt", Kt * Ct),
            ("gam", 1),
            ("vcorr", Ct * Vt),
            ("qd", Ct * Kt),
            ("intra", Ct * Ct),
            ("vnew", Ct * Vt),
        )
        off = 0
        self.base = {}
        for name, stride in strides:
            self.base[name] = off
            off += NI * stride
        self.tiles = max(off, 1)


def _f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _resolve_compute_config(compute_kernel_config, in_dtype):
    cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()
    desc = ttnn.ComputeConfigDescriptor()
    desc.math_fidelity = cfg.math_fidelity
    desc.fp32_dest_acc_en = bool(cfg.fp32_dest_acc_en)
    desc.math_approx_mode = bool(cfg.math_approx_mode)
    desc.dst_full_sync_en = bool(getattr(cfg, "dst_full_sync_en", False))
    if desc.dst_full_sync_en:
        dest_limit = 8 if desc.fp32_dest_acc_en else 16
    else:
        dest_limit = 4 if desc.fp32_dest_acc_en else 8
    return desc, dest_limit


def build_program(q, k, v, g, beta, *, initial_state, chunk_size, scale, compute_kernel_config, memory_config):
    device = q.device()
    B, T, H, K = (int(d) for d in q.shape)
    V = int(v.shape[-1])
    in_dtype = q.dtype
    in_tile = ttnn.tile_size(in_dtype)
    f32_tile = ttnn.tile_size(ttnn.float32)
    elem_bytes = q.element_size()

    grid = device.compute_with_storage_grid_size()
    G = grid.x * grid.y
    geo = _Geometry(B, T, H, K, V, chunk_size, elem_bytes, G)

    budget = ttnn.get_max_worker_l1_unreserved_size() - L1_KERNEL_CONFIG_RESERVE
    Vi = _solve_item_block_val_tiles(geo, in_tile, f32_tile, budget)
    NVI = geo.Vt // Vi
    quanta = _quanta(geo, Vi)
    smap = _ScratchMap(geo)

    if scale is None:
        scale = float(K) ** -0.5
    has_h0 = initial_state is not None
    out_mem = memory_config if memory_config is not None else q.memory_config()

    def alloc(shape, dtype=in_dtype, mem=None):
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(shape)), dtype, ttnn.TILE_LAYOUT, device, mem if mem is not None else out_mem
        )

    o = alloc((B, T, H, V))
    final_state = alloc((B, H, K, V))
    h = alloc((B, geo.NC, H, K, V))
    v_new = alloc((B, T, H, V))
    g_cumsum = alloc((B, T, H))
    A = alloc((B, T, H, chunk_size))
    sc = alloc((smap.tiles * 32, 32), dtype=ttnn.float32, mem=ttnn.DRAM_MEMORY_CONFIG)

    # ---------------- work distribution --------------------------------------------------
    def core_of(c):
        return ttnn.CoreCoord(c % grid.x, c // grid.x)

    nic = geo.num_item_cores
    items_of = {c: list(range(c, geo.NI, nic)) for c in range(nic)}
    units_of = {}
    for u in range(geo.NU):
        units_of.setdefault(G - 1 - (u % G), []).append(u)
    unit_core = {u: G - 1 - (u % G) for u in range(geo.NU)}
    active = sorted(set(items_of) | set(units_of))

    phys = {}

    def noc_xy(c):
        if c not in phys:
            pc = device.worker_core_from_logical_core(core_of(c))
            phys[c] = (pc.x, pc.y)
        return phys[c]

    def seg_of(i):
        return i // geo.seg_chunks

    def seg_chunks_of(j):
        return range(j * geo.seg_chunks, min(geo.NC, (j + 1) * geo.seg_chunks))

    # E cores of (bh, segment j): the item cores holding any (bh, i in segment j).
    e_cores = {}
    for c, wis in items_of.items():
        for wi in wis:
            i, bh = wi // geo.BH, wi % geo.BH
            e_cores.setdefault((bh, seg_of(i)), set()).add(c)

    compute_desc, dest_limit = _resolve_compute_config(compute_kernel_config, in_dtype)

    # ---------------- circular buffers -------------------------------------------------------
    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(core_of(c), core_of(c)) for c in active])
    cbs = []
    for idx, n in sorted(_cb_pages(geo, Vi).items()):
        page = _page_bytes(geo, idx, in_tile, f32_tile)
        fmt = in_dtype if idx in _IN_DTYPE_CBS else ttnn.float32
        cbs.append(
            ttnn.CBDescriptor(
                total_size=n * page,
                core_ranges=core_ranges,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page)],
            )
        )

    # ---------------- compile-time args (identical common block; cgdr_common.hpp) --------------
    sem_ready_base = 0
    sem_done_base = geo.NS
    common_ct = [
        T,
        H,
        geo.Ct,
        geo.Kt,
        geo.Vt,
        Vi,
        NVI,
        geo.Vs,
        geo.NV,
        geo.NC,
        geo.BH,
        geo.NS,
        geo.seg_chunks,
        geo.neumann_steps,
        elem_bytes,
        geo.row_span_stride,
        GATHER_STAGE_TOKENS,
        GATHER_DEPTH,
        dest_limit,
        int(has_h0),
        geo.Tt,
        quanta["QV"],
        quanta["QF"],
        quanta["QO"],
        smap.base["nkcd"],
        smap.base["pt"],
        smap.base["gam"],
        smap.base["vcorr"],
        smap.base["qd"],
        smap.base["intra"],
        smap.base["vnew"],
        sem_ready_base,
        sem_done_base,
        BLOCK_CHUNKS,
        BLOCK_HEADS,
        _num_const_pages(geo),
        _vec_pages(geo),
    ]
    assert len(common_ct) == 37  # CT_ACC_BASE in cgdr_common.hpp

    def acc_args(t):
        return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    reader_ct = list(common_ct)
    for t in (q, k, v, g, beta, initial_state if has_h0 else q, sc, h):
        reader_ct += acc_args(t)
    writer_ct = list(common_ct)
    for t in (sc, o, final_state, h, v_new, g_cumsum, A):
        writer_ct += acc_args(t)
    compute_ct = list(common_ct)

    # ---------------- runtime args -----------------------------------------------------------
    scale_bits = _f32_bits(scale)
    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    addr = lambda t: t.buffer_address()  # noqa: E731
    h0_addr = initial_state.buffer_address() if has_h0 else 0

    for c in active:
        cc = core_of(c)
        wis = items_of.get(c, [])
        units = units_of.get(c, [])
        n_items = len(wis)
        first, stride = (c, nic) if n_items else (0, 1)

        exp_ready = [0] * geo.NS
        for u in units:
            for j in range(geo.NS):
                exp_ready[j] += len(seg_chunks_of(j))
        exp_done = [0] * geo.NS
        for j in range(geo.NS):
            bhs = {wi % geo.BH for wi in wis if seg_of(wi // geo.BH) == j}
            exp_done[j] = geo.NV * len(bhs)

        reader_args = (
            [addr(q), addr(k), addr(v), addr(g), addr(beta), h0_addr, addr(sc), addr(h)]
            + [n_items, first, stride, len(units)]
            + units
            + exp_ready
            + exp_done
        )

        ready_targets = []
        for wi in wis:
            bh = wi % geo.BH
            for vb in range(geo.NV):
                ready_targets += list(noc_xy(unit_core[bh * geo.NV + vb]))
        done_targets = []
        for u in units:
            bh = u // geo.NV
            for j in range(geo.NS):
                cores = sorted(e_cores.get((bh, j), ()))
                done_targets.append(len(cores))
                for ec in cores:
                    done_targets += list(noc_xy(ec))
        writer_args = (
            [addr(sc), addr(o), addr(final_state), addr(h), addr(v_new), addr(g_cumsum), addr(A)]
            + [n_items, first, stride, len(units)]
            + units
            + ready_targets
            + done_targets
        )
        for name, args in (("reader", reader_args), ("writer", writer_args)):
            if len(args) > MAX_RT_ARGS:
                raise ValueError(
                    f"chunk_gated_delta_rule_fwd: {name} runtime args of core {c} ({len(args)}) exceed "
                    f"{MAX_RT_ARGS} (handoff target lists too long for this shape)"
                )
        reader_rt[cc.x][cc.y] = reader_args
        writer_rt[cc.x][cc.y] = writer_args
        compute_rt[cc.x][cc.y] = [n_items, len(units), scale_bits]

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "chunk_gated_delta_rule_fwd_reader.cpp"),
            core_ranges=core_ranges,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "chunk_gated_delta_rule_fwd_writer.cpp"),
            core_ranges=core_ranges,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "chunk_gated_delta_rule_fwd_compute.cpp"),
            core_ranges=core_ranges,
            compile_time_args=compute_ct,
            runtime_args=compute_rt,
            config=compute_desc,
        ),
    ]

    semaphores = [
        ttnn.SemaphoreDescriptor(id=sem_ready_base + j, core_ranges=core_ranges, initial_value=0) for j in range(geo.NS)
    ] + [
        ttnn.SemaphoreDescriptor(id=sem_done_base + j, core_ranges=core_ranges, initial_value=0) for j in range(geo.NS)
    ]

    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)

    io_tensors = [q, k, v, g, beta]
    if has_h0:
        io_tensors.append(initial_state)
    io_tensors += [sc, final_state, h, v_new, g_cumsum, A, o]
    ttnn.generic_op(io_tensors, program)

    debug = {"sc": sc, "smap": smap, "geo": geo, "Vi": Vi, "active_cores": len(active)}
    return (o, final_state, h, v_new, g_cumsum, A), debug
