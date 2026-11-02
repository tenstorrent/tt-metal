# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm "narrow-flat" regime: ROW_MAJOR WIDTH / BLOCK_SHARDED shards whose width w_s divides a 16-datum face row.

Graduation hook (rms_norm_program_descriptor.create_program_descriptor, first statement):

    from .rms_norm_narrow_flat_program_descriptor import create_narrow_flat_program_descriptor  # lazy: import cycle
    desc = create_narrow_flat_program_descriptor(
        input_tensor, gamma, output_tensor, epsilon=epsilon, compute_kernel_config=compute_kernel_config
    )
    if desc is not None:
        return desc

The regime replaces, for these shards, the stick staging -> tilize -> per-tile-row owner combine -> untilize ->
stick write-back path (select_sharded_blocking with RMS_RM + RMS_SHARDED). It returns None (op keeps its path) only
where it is inexpressible: shard width not dividing 16 datums, shard pitch != w_s * datum bytes, or the plan's CBs
do not fit the L1 budget.

Mechanism. A shard of N sticks x w_s datums is ONE contiguous L1 region (shard page = w_s * datum bytes = the L1
alignment granule). Read it as `Layout::TILE` tiles of 1024 consecutive datums ("flat tiles"): datum p sits in face
p // 256, face row (p % 256) // 16, face column p % 16 — so every stick (w_s consecutive datums, w_s | 16) lies in w_s
consecutive LOGICAL columns c0..c0+w_s-1 (c0 % w_s == 0) of ONE logical row of the tile:

  pass 1   sq_f = x_f * x_f (FPU); P_u = sum_{j<R} sq_{uR+j} @ S_j (matmul), R = w_s flat tiles per unit, where
           S_j[i][c] = 1 iff c == j * (32 / w_s) + i // w_s: the compressed unit tile holds ONE value per stick (1024
           sticks = R flat tiles). W tail (last member, v < w_s valid columns): rows i % w_s >= v of S_j are zero (mask
           folded into the constant — free; the caller's shard is never written).
  combine  the unit tile is cut into Q sub-blocks (1024 / Q consecutive datums); sub-unit v = u * Q + s is owned by
           member v % k. Q = smallest power of two <= 16 with 2 * U * Q >= k (and Q <= k), so ~k/2..k members own one
           sub-unit and an owner's incast is k contributions of T / Q bytes. Member m's contribution lands in sub-block
           m % Q of gathered tile m // Q; the owner sums the ceil(k / Q) tiles (FPU), folds the Q sub-blocks and
           finalizes rsqrt(sum / W + eps) on the SFPU, forwards the result to the unit's leader (owner of sub-unit u * Q),
           which multicasts the whole inv tile to the group (<= 3 member rectangles; else unicasts) + a flag.
           The owner combine is pipelined into pass 1 (owned sub-unit v combined after pass 1 of unit v // Q + LAG; the
           last LAG units' inside pass 2).
  pass 2   out_f = (inv_u @ E_j) * x_f with E_j = S_j^T carrying gamma: E_j[c][i] = gamma[col0 + i % w_s] where S_j^T
           is 1 (gamma pad = 0) — one matmul_block (inv tile reused) + one DEST-reuse multiply per flat tile.
  cb_x / cb_out are backed on the input / output shards with tile-sized pages (zero copies). The last flat tile may
  run past the shard end: x through a 1-tile scratch (local copy + zero fill), out through a 1-tile scratch + a
  tail_bytes local write. No tilize / untilize, no stick staging. Every member processes its whole ALLOCATED shard
  (uniform F per program; padded sticks of a ragged BLOCK row are computed and written into the output's padding).
  Assumption shared with the op's TILE-sharded W-tail mask: the input shard padding (W-tail column, ragged-row
  sticks) holds finite values (FPU 0 * x); ttnn's host-to-device sharding writes zeros there.

Measured BH p150b (perf_experiments/narrow_rm_flat), device ns old -> new: (5,3,928,544) WIDTH bf16 850k -> 40k,
(13,777,1023) WIDTH 679k -> 37k, (1024,1024) WIDTH fp32 DEST 117k -> 16k, (4,1,512,512) fp32 WIDTH 227k -> 14k,
(99991,64) BLOCK 513k -> 29k.
"""

from __future__ import annotations

import struct
from pathlib import Path

import ttnn

from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_ELEMS = 1024
FACE_ROW = 16

# CB indices — must match kernels/rms_norm_narrow_flat_common.hpp.
CB_X = 0  # shard-backed input, tile-sized pages (F_full flat tiles)
CB_XTAIL = 1  # last partial flat tile of x (local copy, zero-filled)
CB_MMS = 3  # pass-1 0/1 matrices S_j, j < NB = min(R, F)
CB_SQ = 4  # x^2 flat tiles (two units: squares of unit u + 1 computed before the compress of unit u)
CB_PARTIAL = 5  # per-unit compressed partials (ring)
CB_GATHERED = 6  # owner: KT tiles per owned sub-unit
CB_INV_OWNED = 7  # owner: inv of the owned sub-units
CB_INV = 8  # inv of every unit (multicast by the unit's leader)
CB_OUT = 9  # shard-backed output, tile-sized pages
CB_OUTTAIL = 10  # last partial flat tile of out (scratch)
CB_MME = 11  # pass-2 expand matrices E_j (gamma folded in)
CB_FLAGS = 12  # unit flags (at u * Q) + leaders' collect counters + multicast flag source, 16 B each

SEM_PARTIALS = 0
WRITER_NOC_XY_ARG_BASE = 2
MAX_RECTS = 3  # multicast rectangles per member group (inv return); more -> unicast return
READER_RECT_ARG_BASE = 6  # n_rect, self mask, MAX_RECTS x (x_start, y_start, x_end, y_end, cores) in NoC coordinates
READER_NOC_XY_ARG_BASE = READER_RECT_ARG_BASE + 2 + 5 * MAX_RECTS
FLAG_BYTES = 16
# Exchange sub-blocks per unit tile: the smallest power of two <= Q_MAX with 2 * U * Q >= k. BH p150b: (13,777,1023)
# k = 128, U = 10: Q = 8 34.3 us vs Q = 16 36.7 us vs whole-tile units (Q = 1, 10 owners with a 128-tile incast each)
# 61.8 us; (5,3,928,544) k = 68, U = 14: Q = 4..8 44.5 us vs Q = 1 59.4 us. Q_MAX: a sub-block must be >= 64 datums
# (4 face rows = whole SFPU row groups) for the owner's fold.
Q_MAX = 16
# Owner combine pipelined into pass 1: owned sub-unit v is combined after pass 1 of unit v // Q + PIPE_LAG.
# BH p150b: lag 4 vs 2: (5,3,928,544) 43.5 vs 44.6 us, (13,777,1023) 37.2 vs 38.9 us, (99991,64) 31.7 vs 33.2 us.
PIPE_LAG = 4
PARTIAL_RING = 16  # cb_partial pages


def _f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def _div_up(a, b):
    return (a + b - 1) // b


def narrow_flat_applies(input_tensor) -> bool:
    """ROW_MAJOR WIDTH / BLOCK_SHARDED bf16 / fp32, shard width w_s | 16 datums, shard sticks packed back to back."""
    if input_tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
        return False
    mc = input_tensor.memory_config()
    if not mc.is_sharded() or mc.memory_layout not in (
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ):
        return False
    if input_tensor.dtype not in (ttnn.bfloat16, ttnn.float32):
        return False
    sw = int(mc.shard_spec.shape[1])
    return FACE_ROW % sw == 0 and input_tensor.buffer_aligned_page_size() == sw * input_tensor.element_size()


def _members(input_tensor, geo):
    """[[(core, m, col0, valid_cols)] per group] in select_sharded_blocking's shard-grid conventions."""
    mc = input_tensor.memory_config()
    ss = mc.shard_spec
    sh, sw = int(ss.shape[0]), int(ss.shape[1])
    row_major = ss.orientation == ttnn.ShardOrientation.ROW_MAJOR
    n_h, n_w = _div_up(geo.num_images * geo.H, sh), _div_up(geo.W, sw)
    if mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        order = pd._ordered_shard_cores(ss.grid, row_major)

        def core_of(g, i):
            return order[i]

    else:
        bb = ss.grid.bounding_box()
        x0, y0 = bb.start.x, bb.start.y

        def core_of(g, i):
            return (x0 + i, y0 + g) if row_major else (x0 + g, y0 + i)

    groups = []
    for g in range(n_h):
        mem = []
        for i in range(n_w):
            x, y = core_of(g, i)
            mem.append((ttnn.CoreCoord(x, y), i, i * sw, min(sw, geo.W - i * sw)))
        groups.append(mem)
    return groups, sh, sw


def _mcast_rects(device, cores):
    """[(logical CoreRange, [x_start, y_start, x_end, y_end, count])] covering `cores` exactly (NoC coordinates, min
    corner first; the kernel swaps for NoC1), or [] (unicast return) beyond MAX_RECTS rectangles. BH translated
    coordinates skip the non-Tensix columns inside a box: the multicast reaches the `count` Tensix cores."""
    merged = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores]).merge(ttnn.CoreRangeSet([]))
    ranges = list(merged.ranges())
    if len(ranges) > MAX_RECTS:
        return []
    out = []
    for r in ranges:
        a = device.worker_core_from_logical_core(r.start)
        b = device.worker_core_from_logical_core(r.end)
        n = (r.end.x - r.start.x + 1) * (r.end.y - r.start.y + 1)
        out.append((r, [min(a.x, b.x), min(a.y, b.y), max(a.x, b.x), max(a.y, b.y), n]))
    return out


def create_narrow_flat_program_descriptor(input_tensor, gamma, output_tensor, *, epsilon, compute_kernel_config):
    """The narrow-flat program, or None when the regime is inexpressible / does not fit L1 (the op keeps its path)."""
    if not narrow_flat_applies(input_tensor):
        return None
    if gamma is not None and gamma.layout != ttnn.ROW_MAJOR_LAYOUT:
        return None  # the reader reads the member's w_s datums from the ONE RM gamma page (TILE gamma: EXCLUSIONS)
    device = input_tensor.device()
    geo = pd.derive_geometry(input_tensor.shape)
    x_dtype = input_tensor.dtype
    has_gamma = gamma is not None
    g_dtype = gamma.dtype if has_gamma else x_dtype
    fp32_dest = bool(compute_kernel_config.fp32_dest_acc_en)
    t_acc = pd.acc_dtype(fp32_dest)
    eb = input_tensor.element_size()
    g_eb = gamma.element_size() if has_gamma else eb
    T_x = ttnn.tile_size(x_dtype)
    T_acc = ttnn.tile_size(t_acc)
    # pass-2 expand matrices carry gamma: fp32 when gamma or the accumulation is fp32
    t_mme = ttnn.float32 if (t_acc == ttnn.float32 or (has_gamma and g_dtype == ttnn.float32)) else ttnn.bfloat16
    T_mme = ttnn.tile_size(t_mme)

    groups, sh, sw = _members(input_tensor, geo)
    k = len(groups[0])
    elems = sh * sw  # every member processes its whole allocated shard (uniform F; padded sticks are don't-care)
    F_full, tail_elems = divmod(elems, TILE_ELEMS)
    F = F_full + (1 if tail_elems else 0)
    tail_bytes = tail_elems * eb
    R = sw
    U = _div_up(F, R)
    NB = min(R, F)
    Q = 1
    while Q < Q_MAX and 2 * U * Q < k:
        Q *= 2
    while Q > 1 and Q > k:  # a sub-unit's throttle predecessor (v - k) must sit in an earlier unit
        Q //= 2
    V = U * Q
    KT = _div_up(k, Q)
    o_max = _div_up(V, k)
    dest_tiles = 4 if fp32_dest else 8
    lag = max(0, min(PIPE_LAG, U, PARTIAL_RING - 1))

    plan = {  # cb: (pages, page bytes, dtype)
        CB_MMS: (NB, T_acc, t_acc),
        CB_MME: (NB, T_mme, t_mme),
        CB_FLAGS: (V + U + 1, FLAG_BYTES, ttnn.uint32),
        CB_SQ: (2 * max(R, dest_tiles), T_acc, t_acc),
        CB_PARTIAL: (min(U, PARTIAL_RING), T_acc, t_acc),
        CB_GATHERED: (o_max * KT, T_acc, t_acc),
        CB_INV_OWNED: (min(o_max, 2), T_acc, t_acc),
        CB_INV: (U, T_acc, t_acc),
    }
    if tail_elems:
        plan[CB_XTAIL] = (1, T_x, x_dtype)
        plan[CB_OUTTAIL] = (1, T_x, x_dtype)
    if sum(p * b for p, b, _ in plan.values()) > pd.cb_l1_budget(device):
        return None

    all_cores = [c for grp in groups for (c, *_rest) in grp]
    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in all_cores])

    def shard_cb(index, tensor):
        d = ttnn.cb_descriptor_from_sharded_tensor(index, tensor, 0, F_full * T_x, core_ranges)
        d.format_descriptors = [ttnn.CBFormatDescriptor(buffer_index=index, data_format=x_dtype, page_size=T_x)]
        return d

    cbs = [
        ttnn.CBDescriptor(
            total_size=pages * page_bytes,
            core_ranges=core_ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page_bytes)],
        )
        for idx, (pages, page_bytes, dt) in plan.items()
    ]
    if F_full:
        cbs += [shard_cb(CB_X, input_tensor), shard_cb(CB_OUT, output_tensor)]

    defines = [("RMS_HAS_GAMMA", "1")] if has_gamma else []
    g_acc_args = ttnn.TensorAccessorArgs(gamma) if has_gamma else ttnn.TensorAccessorArgs()
    g_page = gamma.buffer_aligned_page_size() if has_gamma else 0
    reader_ct = [F_full, tail_bytes, T_x, sw, R, 1 if fp32_dest else 0, g_eb, g_page, NB]
    reader_ct += [1 if t_mme == ttnn.float32 else 0, U, k, T_acc, SEM_PARTIALS, FLAG_BYTES, READER_NOC_XY_ARG_BASE]
    reader_ct += [Q, KT, READER_RECT_ARG_BASE]
    reader_ct += g_acc_args.get_compile_time_args()
    compute_ct = [F_full, 1 if tail_elems else 0, R, U, k, lag, Q, KT]
    writer_ct = [F_full, tail_bytes, T_x, U, k, Q, T_acc, SEM_PARTIALS, FLAG_BYTES, WRITER_NOC_XY_ARG_BASE, KT]

    x_addr = input_tensor.buffer_address()
    out_addr = output_tensor.buffer_address()
    g_addr = gamma.buffer_address() if has_gamma else 0
    inv_w_bits = _f32_bits(1.0 / geo.W)
    eps_bits = _f32_bits(epsilon)

    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for grp in groups:
        noc_xy = []
        for c, *_rest in grp:
            v = device.worker_core_from_logical_core(c)
            noc_xy += [v.x, v.y]
        rects = _mcast_rects(device, [c for c, *_r in grp])
        rect_flat = [a for (_lr, nr) in rects for a in nr]
        rect_flat += [0] * (5 * MAX_RECTS - len(rect_flat))
        for c, m, col0, valid in grp:
            owned = len(range(m, V, k))
            mask = sum(
                1 << i
                for i, (lr, _nr) in enumerate(rects)
                if lr.start.x <= c.x <= lr.end.x and lr.start.y <= c.y <= lr.end.y
            )
            reader_rt[c.x][c.y] = [x_addr, valid, col0, g_addr, m, owned, len(rects), mask] + rect_flat + noc_xy
            compute_rt[c.x][c.y] = [m, owned, inv_w_bits, eps_bits]
            writer_rt[c.x][c.y] = [out_addr, m] + noc_xy

    def kernel(name, ct, rt, config):
        return ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / f"rms_norm_narrow_flat_{name}.cpp"),
            core_ranges=core_ranges,
            compile_time_args=ct,
            defines=defines,
            runtime_args=rt,
            config=config,
        )

    kernels = [
        kernel("reader", reader_ct, reader_rt, ttnn.ReaderConfigDescriptor()),
        kernel("writer", writer_ct, writer_rt, ttnn.WriterConfigDescriptor()),
        kernel("compute", compute_ct, compute_rt, compute_kernel_config),
    ]
    semaphores = [ttnn.SemaphoreDescriptor(id=SEM_PARTIALS, core_ranges=core_ranges, initial_value=0)]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)
