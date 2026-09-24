# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""gated_delta_net_backward — ProgramDescriptor.

ONE `ttnn.generic_op` dispatch.  Three stages (P: prep, S: scan, G: gradient
assembly) live inside the same three kernel binaries and are separated by two
per-(b,h)-group semaphore rendezvous.  Host-side output/scratch allocation and
the `block_val_tiles` extent solve are not dispatches.

BLOCKING KNOBS (single source of truth — every dependent quantity derives from
these; never restate one as a literal):

    BLOCK_CHUNKS        chunks per (bh) work item                    [1]
    GATHER_STAGE_TOKENS tokens per face-row staging window           [16]
    GATHER_DEPTH        staging windows in flight (software pipeline) [2]
    EGRESS_DEPTH        blocks in flight to the writer               [2]
    BLOCK_DEPTH         depth of a resident block CB                 [1]
    ACCUM_DEPTH         depth of an in-place-updated block CB        [2]
    block_val_tiles     V-extent of one compute pass  (solved below)

`block_val_tiles` is the only shape-varying extent: it is the largest divisor
of `Vt` whose closed-form L1 footprint fits the worker budget.
"""

from __future__ import annotations

import math
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# --------------------------------------------------------------------------
# Blocking knobs (see module docstring)
# --------------------------------------------------------------------------
BLOCK_CHUNKS = 1
GATHER_STAGE_TOKENS = 16
GATHER_DEPTH = 2
EGRESS_DEPTH = 2
BLOCK_DEPTH = 1
ACCUM_DEPTH = 2

# L1 below the circular-buffer region that `get_max_worker_l1_unreserved_size()`
# does not account for: the kernel-config ring buffer (70656 B on this part)
# plus slack.  The `block_val_tiles` solve must budget against the remainder.
L1_KERNEL_CONFIG_RESERVE = 80 * 1024

# A 32x32 tile row spans two faces: 16 elements in face f, 16 in face f+1, with
# the two runs exactly 256 elements apart.  One NoC read therefore covers the
# whole row as a 272-element span.
ROW_SPAN_ELEMS = 272
# Slack so a staging line can start at (dram_byte_offset % 64), which is what
# the NoC requires of a DRAM read's L1 destination.
ROW_SPAN_SLACK = 64

# Named-block counts inside the three multi-slot CBs.  Defined HERE only: the
# host sizes the CB from them and the kernel derives every slot offset from the
# same value arriving as a compile-time arg, so a slot added on one side cannot
# silently disagree with the other.
NUM_CONST_MASKS = 6  # [C,C] mask blocks in cb_const (LT, NSTRICT, UT, EYE, BIAS, SUT)
NUM_VECA_SLOTS = 5  # per-item decay column slots  (decay, gamma, w, beta, dc1)
NUM_VECB_SLOTS = 9  # stage-G column accumulator slots

# --------------------------------------------------------------------------
# Circular-buffer indices (semantic names; 0..31)
# --------------------------------------------------------------------------
CB_CONST = 0  # 6 constant [C,C] mask blocks + 1 scale tile (f32)
CB_COLONES = 1  # column-of-ones tiles; the row-sum matmul operand (f32)
CB_GATHER = 2  # reader-local face-row staging (raw bytes)
CB_QIN = 3  # gathered / compact q                      (in dtype)
CB_KIN = 4  # gathered / compact k                      (in dtype)
CB_VIN = 5  # gathered / compact v   slice              (in dtype)
CB_DOIN = 6  # gathered / compact do  slice              (in dtype)
CB_GATEIN = 7  # gathered g, beta column tiles             (in dtype)
CB_LOAD_ITEM = 8  # per-item scratch loads (Tinv, vec)       (f32)
CB_LOAD_VB = 9  # per-V-block scratch loads                (f32)
CB_KA = 10  # [C,K] working blocks
CB_KB = 11
CB_KC = 12
CB_KD = 13
CB_KE = 14
CB_KF = 15
CB_KTR = 16  # [K,C] transposed block
CB_VA = 17  # [C,Vb] working blocks
CB_VB_ = 18
CB_VC = 19
CB_CA = 20  # [C,C] working blocks
CB_CB_ = 21  # L (never updated in place)
CB_CC = 22
CB_CD = 23
CB_CE = 24
CB_SA = 25  # [K,Vb] state blocks
CB_SB = 26
CB_VECA = 27  # decay / gamma / w / beta / dc1 columns
CB_VECB = 28  # stage-G column accumulators
CB_VECC = 29  # column scratch
CB_EGR = 30  # compute -> writer, f32 scratch blocks
CB_GEGR = 31  # compute -> writer, gradient blocks (in dtype)

SEM_PREP = 0
SEM_SCAN = 1


def _ceil_div(a, b):
    return (a + b - 1) // b


def _round_up(a, m):
    return ((a + m - 1) // m) * m


class _Geometry:
    """Every derived count in one place."""

    def __init__(self, B, T, H, K, V, chunk_size, elem_bytes):
        self.B, self.T, self.H, self.K, self.V = B, T, H, K, V
        self.chunk_size = chunk_size
        self.Ct = chunk_size // 32
        self.Kt = _ceil_div(K, 32)
        self.Vt = _ceil_div(V, 32)
        self.Tt = _ceil_div(T, 32)
        self.NC = _ceil_div(T, chunk_size * BLOCK_CHUNKS) * BLOCK_CHUNKS
        self.BH = B * H
        self.NI = self.BH * self.NC
        self.neumann_steps = max(1, int(math.ceil(math.log2(chunk_size))))
        self.elem_bytes = elem_bytes
        self.row_span_stride = _round_up(ROW_SPAN_ELEMS * elem_bytes + ROW_SPAN_SLACK, 64)


def _cb_blocks(geo, Vb):
    """The ONE uniform push size of every CB whose block size is not a plain
    product of the block extents.

    Single source of truth: the descriptor sizes each CB from these and passes
    the same values to the kernels as compile-time args (`_BLOCK_CT_ORDER`), so
    the device never recomputes a sizing formula the host already solved.  A CB
    with two different push sizes would wrap its fifo, which is a hang — hence
    "uniform": where a stage needs fewer tiles than the block, it leaves the
    tail unused.
    """
    Ct, Kt = geo.Ct, geo.Kt
    return {
        "nconst": NUM_CONST_MASKS * Ct * Ct + 2 + Ct,  # masks + scale + zero + Ct row-of-ones
        "ncol": max(Ct, Kt, Vb),
        "maxv": max(Ct, Kt) * Vb,
        "lvb": max(
            2 * Ct * Kt + Ct * Vb + Kt * Vb + 1,  # stage S reverse step
            2 * Ct * Vb + 2 * Kt * Vb,  # stage G V-slice
        ),
        "litem": Ct * Ct + 4 * Ct,
        "maxblk": max(Ct * Ct, Ct * Kt, Ct * Vb, Kt * Vb, 4 * Ct),
        "maxblk_g": max(Ct * Kt, Ct * Vb, Ct),
    }


# The compile-time-arg order the kernels' `gdn_common.hpp` reads these in.
_BLOCK_CT_ORDER = ("nconst", "ncol", "maxv", "lvb", "litem", "maxblk", "maxblk_g")


def _cb_pages(geo, Vb):
    """Page count per CB index — the single source for both the footprint
    solve and the descriptor build."""
    Ct, Kt = geo.Ct, geo.Kt
    blk = _cb_blocks(geo, Vb)
    nmax, maxblk, maxblk_g, maxv, lvb = (
        blk["ncol"],
        blk["maxblk"],
        blk["maxblk_g"],
        blk["maxv"],
        blk["lvb"],
    )
    return {
        CB_CONST: blk["nconst"],
        CB_COLONES: nmax,
        CB_GATHER: GATHER_DEPTH,  # one page per in-flight staging window
        CB_QIN: BLOCK_DEPTH * Ct * Kt,
        CB_KIN: BLOCK_DEPTH * Ct * Kt,
        CB_VIN: BLOCK_DEPTH * maxv,
        CB_DOIN: BLOCK_DEPTH * Ct * Vb,
        CB_GATEIN: BLOCK_DEPTH * 2 * Ct,
        CB_LOAD_ITEM: blk["litem"],
        CB_LOAD_VB: lvb,
        CB_KA: ACCUM_DEPTH * Ct * Kt,
        CB_KB: ACCUM_DEPTH * Ct * Kt,
        CB_KC: ACCUM_DEPTH * Ct * Kt,
        CB_KD: ACCUM_DEPTH * Ct * Kt,
        CB_KE: ACCUM_DEPTH * Ct * Kt,
        CB_KF: ACCUM_DEPTH * Ct * Kt,
        CB_KTR: ACCUM_DEPTH * Ct * Kt,
        CB_VA: ACCUM_DEPTH * Ct * Vb,
        CB_VB_: ACCUM_DEPTH * Ct * Vb,
        CB_VC: ACCUM_DEPTH * Ct * Vb,
        CB_CA: ACCUM_DEPTH * Ct * Ct,
        CB_CB_: Ct * Ct,
        CB_CC: ACCUM_DEPTH * Ct * Ct,
        CB_CD: ACCUM_DEPTH * Ct * Ct,
        CB_CE: ACCUM_DEPTH * Ct * Ct,
        CB_SA: ACCUM_DEPTH * Kt * Vb,
        CB_SB: ACCUM_DEPTH * Kt * Vb,
        CB_VECA: NUM_VECA_SLOTS * Ct,  # exactly the per-item column count
        CB_VECB: NUM_VECB_SLOTS * Ct,  # exactly the stage-G column-list length
        CB_VECC: ACCUM_DEPTH * Ct,  # single-block groups, in-place capable
        CB_EGR: EGRESS_DEPTH * maxblk,
        CB_GEGR: EGRESS_DEPTH * maxblk_g,
    }


_IN_DTYPE_CBS = (CB_QIN, CB_KIN, CB_VIN, CB_DOIN, CB_GATEIN, CB_GEGR)


def _cb_bytes(geo, Vb, in_tile, f32_tile):
    pages = _cb_pages(geo, Vb)
    total = 0
    for idx, n in pages.items():
        if idx == CB_GATHER:
            total += GATHER_DEPTH * (GATHER_STAGE_TOKENS * geo.row_span_stride + 64)
        elif idx in _IN_DTYPE_CBS:
            total += n * in_tile
        else:
            total += n * f32_tile
    return total


def _solve_block_val_tiles(geo, in_tile, f32_tile, budget):
    """Largest divisor of Vt whose closed-form L1 footprint fits."""
    best = None
    for vb in range(geo.Vt, 0, -1):
        if geo.Vt % vb:
            continue
        if _cb_bytes(geo, vb, in_tile, f32_tile) <= budget:
            best = vb
            break
    if best is None:
        raise RuntimeError(
            "gated_delta_net_backward: no block_val_tiles fits L1 "
            f"(Ct={geo.Ct}, Kt={geo.Kt}, Vt={geo.Vt}, budget={budget}, "
            f"min footprint={_cb_bytes(geo, 1, in_tile, f32_tile)})"
        )
    return best


class _ScratchMap:
    """Tile-index bases inside the two flat scratch buffers."""

    def __init__(self, geo, Vb):
        Ct, Kt, Vt, NI = geo.Ct, geo.Kt, geo.Vt, geo.NI
        # float32 scratch: per-item strides
        self.st_attn = Ct * Ct
        self.st_kcd = Ct * Kt
        self.st_p = Ct * Kt
        self.st_vec = 4 * Ct  # decay, beta, dc1, rmg
        self.st_vcorr = Ct * Vt
        self.st_u = Ct * Vt
        self.st_c = Kt * Vt
        self.st_s = Kt * Vt
        self.st_ds = Kt * Vt
        self.st_vnew = Ct * Vt
        self.st_dvnew = Ct * Vt

        off = 0
        self.base_attn = off
        off += NI * self.st_attn
        self.base_kcd = off
        off += NI * self.st_kcd
        self.base_p = off
        off += NI * self.st_p
        self.base_vec = off
        off += NI * self.st_vec
        self.base_vcorr = off
        off += NI * self.st_vcorr
        self.base_u = off
        off += NI * self.st_u
        self.base_c = off
        off += NI * self.st_c
        self.base_s = off
        off += NI * self.st_s
        self.base_ds = off
        off += NI * self.st_ds
        self.base_vnew = off
        off += NI * self.st_vnew
        self.base_dvnew = off
        off += NI * self.st_dvnew
        self.f32_tiles = max(off, 1)

        # in-dtype compact scratch
        off = 0
        self.base_cq = off
        off += NI * Ct * Kt
        self.base_ck = off
        off += NI * Ct * Kt
        self.base_cv = off
        off += NI * Ct * Vt
        self.base_cdo = off
        off += NI * Ct * Vt
        self.in_tiles = max(off, 1)


def _compute_config(compute_kernel_config):
    # Defaults, overridable ONLY through the caller's compute_kernel_config —
    # never through the environment.  Math fidelity changes the answer, so it
    # has to travel with the call, not with the shell.
    fidelity = ttnn.MathFidelity.HiFi4
    fp32_dest_acc = True
    math_approx = False
    dst_full_sync = False
    if compute_kernel_config is not None:
        fidelity = getattr(compute_kernel_config, "math_fidelity", fidelity)
        fp32_dest_acc = bool(getattr(compute_kernel_config, "fp32_dest_acc_en", fp32_dest_acc))
        math_approx = bool(getattr(compute_kernel_config, "math_approx_mode", math_approx))
        dst_full_sync = bool(getattr(compute_kernel_config, "dst_full_sync_en", dst_full_sync))
    desc = ttnn.ComputeConfigDescriptor()
    desc.math_fidelity = fidelity
    desc.fp32_dest_acc_en = fp32_dest_acc
    desc.math_approx_mode = math_approx
    desc.dst_full_sync_en = dst_full_sync
    if dst_full_sync:
        dest_limit = 8 if fp32_dest_acc else 16
    else:
        dest_limit = 4 if fp32_dest_acc else 8
    return desc, dest_limit


def _accessor_args(tensor, fallback):
    """CT args for one accessor.  Absent optional tensors borrow `fallback`'s
    descriptor so the CT-arg layout is identical in both program variants; the
    kernel guards every use behind the `has_h0` / `has_dht` compile-time flag,
    so the placeholder accessor is declared but never dereferenced."""
    return list(ttnn.TensorAccessorArgs(tensor if tensor is not None else fallback).get_compile_time_args())


def _f32_bits(value):
    import struct

    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def build_program(q, k, v, g, beta, do, *, dht, initial_state, chunk_size, scale, compute_kernel_config, memory_config):
    device = q.device()
    qs = [int(d) for d in q.shape]
    vs = [int(d) for d in v.shape]
    B, T, H, K = qs[0], qs[1], qs[2], qs[3]
    V = vs[3]

    in_dtype = q.dtype
    in_tile = ttnn.tile_size(in_dtype)
    f32_tile = ttnn.tile_size(ttnn.float32)
    elem_bytes = q.element_size()

    geo = _Geometry(B, T, H, K, V, chunk_size, elem_bytes)
    # `get_max_worker_l1_unreserved_size()` measures from the kernel-config base,
    # so it still includes the kernel-config ring buffer that this program's five
    # kernel binaries live in.  Subtract it (plus slack) or the extent solve picks
    # a block_val_tiles that overflows L1 at program build.
    budget = ttnn.get_max_worker_l1_unreserved_size() - L1_KERNEL_CONFIG_RESERVE
    Vb = _solve_block_val_tiles(geo, in_tile, f32_tile, budget)
    NVB = geo.Vt // Vb
    smap = _ScratchMap(geo, Vb)

    if scale is None:
        scale = float(K) ** -0.5

    out_mem = memory_config if memory_config is not None else q.memory_config()

    def alloc(shape, dtype=in_dtype, mem=None):
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(shape)), dtype, ttnn.TILE_LAYOUT, device, mem if mem is not None else out_mem
        )

    dq = alloc((B, T, H, K))
    dk = alloc((B, T, H, K))
    dv = alloc((B, T, H, V))
    dg = alloc((B, T, H))
    dbeta = alloc((B, T, H))
    has_h0 = initial_state is not None
    has_dht = dht is not None
    dh0 = alloc((B, H, K, V)) if has_h0 else None

    dram = ttnn.DRAM_MEMORY_CONFIG
    sc = alloc((smap.f32_tiles * 32, 32), dtype=ttnn.float32, mem=dram)
    scin = alloc((smap.in_tiles * 32, 32), dtype=in_dtype, mem=dram)

    # ---------------- work distribution -----------------------------------
    grid = device.compute_with_storage_grid_size()
    (num_cores, all_cores, group1, group2, items1, items2) = ttnn.split_work_to_cores(grid, geo.NI, row_wise=True)

    core_list = []
    for group, n in ((group1, items1), (group2, items2)):
        if n == 0:
            continue
        for core in ttnn.corerange_to_cores(group, row_wise=True):
            core_list.append((core, n))
    cursor = 0
    assign = []  # (core, start, n)
    for core, n in core_list:
        assign.append((core, cursor, n))
        cursor += n
    assert cursor == geo.NI, f"work split lost items: {cursor} != {geo.NI}"

    # owner core of group bh == core holding item bh*NC
    owner_of = {}
    for core, start, n in assign:
        for wi in range(start, start + n):
            if wi % geo.NC == 0:
                owner_of[wi // geo.NC] = core
    assert len(owner_of) == geo.BH

    # cores that hold items of group bh (stage-G release targets)
    members_of = {bh: [] for bh in range(geo.BH)}
    for core, start, n in assign:
        bhs = sorted({wi // geo.NC for wi in range(start, start + n)})
        for bh in bhs:
            members_of[bh].append(core)

    compute_desc, dest_limit = _compute_config(compute_kernel_config)

    # ---------------- circular buffers ------------------------------------
    blocks = _cb_blocks(geo, Vb)
    pages = _cb_pages(geo, Vb)
    cbs = []
    for idx in sorted(pages):
        n = pages[idx]
        if idx == CB_GATHER:
            page_size = GATHER_STAGE_TOKENS * geo.row_span_stride + 64
            fmt = ttnn.float32
        elif idx in _IN_DTYPE_CBS:
            page_size = in_tile
            fmt = in_dtype
        else:
            page_size = f32_tile
            fmt = ttnn.float32
        cbs.append(
            ttnn.CBDescriptor(
                total_size=n * page_size,
                core_ranges=all_cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page_size)],
            )
        )

    # ---------------- compile-time args -----------------------------------
    common_ct = (
        [
            B,  # 0
            T,  # 1
            H,  # 2
            K,  # 3
            V,  # 4
            geo.Ct,  # 5  block_chunk_tiles
            geo.Kt,  # 6  block_key_tiles
            geo.Vt,  # 7
            Vb,  # 8  block_val_tiles
            NVB,  # 9  num_v_blocks
            geo.NC,  # 10 tensor_chunks
            geo.BH,  # 11
            geo.neumann_steps,  # 12
            elem_bytes,  # 13
            geo.row_span_stride,  # 14
            GATHER_STAGE_TOKENS,  # 15
            GATHER_DEPTH,  # 16
            dest_limit,  # 17
            int(has_h0),  # 18
            int(has_dht),  # 19
            BLOCK_CHUNKS,  # 20
            EGRESS_DEPTH,  # 21
            geo.Tt,  # 22
            smap.base_attn,  # 23
            smap.base_kcd,  # 24
            smap.base_p,  # 25
            smap.base_vec,  # 26
            smap.base_vcorr,  # 27
            smap.base_u,  # 28
            smap.base_c,  # 29
            smap.base_s,  # 30
            smap.base_ds,  # 31
            smap.base_vnew,  # 32
            smap.base_dvnew,  # 33
            smap.base_cq,  # 34
            smap.base_ck,  # 35
            smap.base_cv,  # 36
            smap.base_cdo,  # 37
            SEM_PREP,  # 38
            SEM_SCAN,  # 39
        ]
        + [blocks[name] for name in _BLOCK_CT_ORDER]
        + [  # 40..46 uniform CB block sizes
            NUM_CONST_MASKS,  # 47
            NUM_VECA_SLOTS,  # 48
            NUM_VECB_SLOTS,  # 49
        ]
    )

    reader_ct = list(common_ct)
    for tensor in (q, k, v, g, beta, do, initial_state, dht, sc, scin):
        reader_ct.extend(_accessor_args(tensor, q))

    writer_ct = list(common_ct)
    for tensor in (sc, dq, dk, dv, dg, dbeta, dh0):
        writer_ct.extend(_accessor_args(tensor, dq))

    compute_ct = list(common_ct)

    # ---------------- runtime args ----------------------------------------
    scale_bits = _f32_bits(scale)
    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()

    addr = lambda t: (t.buffer_address() if t is not None else 0)  # noqa: E731

    for core, start, n in assign:
        bhs = sorted({wi // geo.NC for wi in range(start, start + n)})
        # A core is the scan owner of exactly the groups whose chunk-0 item it
        # holds; every group has exactly one owner.
        scan_bh_list = [bh for bh in range(geo.BH) if owner_of[bh] == core]

        reader_rt[core.x][core.y] = [
            addr(q),
            addr(k),
            addr(v),
            addr(g),
            addr(beta),
            addr(do),
            addr(initial_state),
            addr(dht),
            addr(sc),
            addr(scin),
            start,
            n,
            int(len(scan_bh_list) > 0),
            len(bhs),
            scale_bits,
            len(scan_bh_list),
        ] + scan_bh_list

        compute_rt[core.x][core.y] = [
            start,
            n,
            int(len(scan_bh_list) > 0),
            scale_bits,
            len(scan_bh_list),
        ] + scan_bh_list

        # groups this core produces prep for -> (bh, owner_noc_x, owner_noc_y)
        group_tail = []
        for bh in bhs:
            oc = device.worker_core_from_logical_core(owner_of[bh])
            group_tail.extend([bh, oc.x, oc.y])
        # for each owned group, the release targets (physical NoC coords)
        release_tail = []
        for bh in scan_bh_list:
            targets = members_of[bh]
            release_tail.append(len(targets))
            for tc in targets:
                pc = device.worker_core_from_logical_core(tc)
                release_tail.extend([pc.x, pc.y])

        writer_rt[core.x][core.y] = (
            [
                addr(sc),
                addr(dq),
                addr(dk),
                addr(dv),
                addr(dg),
                addr(dbeta),
                addr(dh0),
                start,
                n,
                int(len(scan_bh_list) > 0),
                scale_bits,
                len(bhs),
            ]
            + group_tail
            + [len(scan_bh_list)]
            + scan_bh_list
            + release_tail
        )

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "gated_delta_net_backward_reader.cpp"),
            core_ranges=all_cores,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
            compiler_include_paths=[str(KERNEL_DIR)],
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "gated_delta_net_backward_writer.cpp"),
            core_ranges=all_cores,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
            compiler_include_paths=[str(KERNEL_DIR)],
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "gated_delta_net_backward_compute.cpp"),
            core_ranges=all_cores,
            compile_time_args=compute_ct,
            runtime_args=compute_rt,
            config=compute_desc,
            compiler_include_paths=[str(KERNEL_DIR)],
        ),
    ]

    semaphores = [
        ttnn.SemaphoreDescriptor(id=SEM_PREP, core_ranges=all_cores, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_SCAN, core_ranges=all_cores, initial_value=0),
    ]

    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)

    io_tensors = [q, k, v, g, beta, do]
    if has_h0:
        io_tensors.append(initial_state)
    if has_dht:
        io_tensors.append(dht)
    io_tensors += [sc, scin, dq, dk, dv, dg, dbeta]
    if has_h0:
        io_tensors.append(dh0)

    ttnn.generic_op(io_tensors, program)

    scratch = {
        "sc": sc,
        "scin": scin,
        "map": smap,
        "geo": geo,
        "Vb": Vb,
        "NVB": NVB,
        "num_cores": num_cores,
    }
    return (dq, dk, dv, dg, dbeta, dh0), scratch
