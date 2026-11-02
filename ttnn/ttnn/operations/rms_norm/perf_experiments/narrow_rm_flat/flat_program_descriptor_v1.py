# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm "flat" regime: ROW_MAJOR WIDTH / BLOCK_SHARDED shards whose shard width w_s divides a 16-datum face row.

A shard of N sticks x w_s datums is ONE contiguous L1 region (shard page = w_s * elem bytes = the L1 alignment
granule). Read it as `Layout::TILE` tiles of 1024 consecutive datums ("flat tiles"): datum p of a flat tile sits in
face p // 256, face row (p % 256) // 16, face column p % 16 — so every stick (w_s consecutive datums, w_s | 16) lies
in w_s consecutive LOGICAL columns c0..c0+w_s-1 (c0 % w_s == 0) of ONE logical row of the tile. Per-stick reductions
are therefore a right-multiplication by a 0/1 matrix, and every elementwise stage runs on the flat tiles as-is:

  pass 1   sq_f = x_f * x_f (FPU);  cmp: P_u = sum_{j<R} sq_{uR+j} @ S_j   (R = w_s flat tiles per unit)
                                     bc : P_f = sq_f @ B                  (B = blockdiag(ones(w_s x w_s)))
           S_j[i][c] = 1 iff c == j * (32 / w_s) + i // w_s  -> the compressed unit tile holds ONE value per stick
           (32 x 32 = 1024 sticks = R flat tiles); B puts the stick sum on each of its w_s datums (broadcast form).
           W tail (last member, v < w_s valid columns): rows i with i % w_s >= v of S_j / B are zero (mask folded
           into the constant — free).
  combine  owner(u) = u % k; every member writes its unit-u partial into slot (u // k) * k + m of the owner's
           cb_gathered; the owner sums the k tiles (reduce AccumulateViaAdd, Skip) and finalizes FULL-tile
           rsqrt(sum / W + eps), then returns the inv tile into slot u of every member's cb_inv (unicast).
  pass 2   cmp: inv_f = inv_u @ E_j (E_j = S_j^T: expand back to the flat layout), out_f = inv_f * x_f * gamma_f
           bc : out_f = x_f * inv_f * gamma_f.   gamma_f = this member's w_s gamma datums repeated 1024 / w_s times.
  cb_x / cb_out are backed on the input / output shards with tile-sized pages (zero copies). The last flat tile
  may run past the shard end: x goes through a 1-tile scratch (local copy + zero fill), out through a 1-tile
  scratch + partial local write. No tilize / untilize, no stick staging.
"""

from __future__ import annotations

import struct

import ttnn

from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

from pathlib import Path

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_ELEMS = 1024
FACE_ROW = 16

# CB indices — must match kernels/rms_norm_flat_common.hpp.
CB_X = 0  # shard-backed input, tile-sized pages (F_full flat tiles)
CB_XTAIL = 1  # last partial flat tile of x (local copy, zero-filled)
CB_GAMMA = 2  # gamma_flat: this member's w_s gamma datums repeated over one tile
CB_MM = 3  # 0/1 constants: cmp S_0..S_{R-1}, E_0..E_{R-1}; bc B
CB_SQ = 4  # x^2 flat tiles (pass 1 intermediate)
CB_PARTIAL = 5  # this member's per-unit partials (ring)
CB_GATHERED = 6  # owner: k partials per owned unit
CB_INV_OWNED = 7  # owner: inv of the owned units
CB_INV = 8  # inv of every unit (written by the owners)
CB_OUT = 9  # shard-backed output, tile-sized pages
CB_OUTTAIL = 10  # last partial flat tile of out (scratch)

SEM_PARTIALS = 0
SEM_INV = 1
WRITER_NOC_XY_ARG_BASE = 3

PARTIAL_RING = 16  # cb_partial pages (bc form: U = F can be large)


def _f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def _div_up(a, b):
    return (a + b - 1) // b


def flat_applies(input_tensor) -> bool:
    """ROW_MAJOR WIDTH / BLOCK_SHARDED, shard width w_s | 16 datums, shard sticks packed back to back in L1."""
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
    eb = input_tensor.element_size()
    return FACE_ROW % sw == 0 and input_tensor.buffer_aligned_page_size() == sw * eb


def _members(input_tensor, geo):
    """[(group, [(core, m, col0, valid_cols)])] in the op's shard-grid conventions (select_sharded_blocking)."""
    mc = input_tensor.memory_config()
    ss = mc.shard_spec
    sh, sw = int(ss.shape[0]), int(ss.shape[1])
    row_major = ss.orientation == ttnn.ShardOrientation.ROW_MAJOR
    rows_ext = geo.num_images * geo.H
    n_h, n_w = _div_up(rows_ext, sh), _div_up(geo.W, sw)
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


def create_flat_program_descriptor(
    input_tensor, gamma, output_tensor, *, epsilon, compute_kernel_config, form="cmp", extra_defines=()
):
    assert form in ("cmp", "bc")
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
    T_g = ttnn.tile_size(g_dtype)

    groups, sh, sw = _members(input_tensor, geo)
    k = len(groups[0])
    elems = sh * sw  # every member processes its whole allocated shard (uniform F; padded sticks are don't-care)
    F_full, tail_elems = divmod(elems, TILE_ELEMS)
    F = F_full + (1 if tail_elems else 0)
    tail_bytes = tail_elems * eb
    R = sw if form == "cmp" else 1
    U = _div_up(F, R)
    o_max = _div_up(U, k)
    dest_tiles = 4 if fp32_dest else 8

    all_cores = [c for grp in groups for (c, *_rest) in grp]
    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in all_cores])

    def cb(index, pages, page_bytes, dtype):
        return ttnn.CBDescriptor(
            total_size=pages * page_bytes,
            core_ranges=core_ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_bytes)],
        )

    def shard_cb(index, tensor, pages, page_bytes, dtype):
        d = ttnn.cb_descriptor_from_sharded_tensor(index, tensor, 0, pages * page_bytes, core_ranges)
        d.format_descriptors = [ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_bytes)]
        return d

    cbs = []
    if F_full:
        cbs.append(shard_cb(CB_X, input_tensor, F_full, T_x, x_dtype))
        cbs.append(shard_cb(CB_OUT, output_tensor, F_full, T_x, x_dtype))
    if tail_elems:
        cbs.append(cb(CB_XTAIL, 1, T_x, x_dtype))
        cbs.append(cb(CB_OUTTAIL, 1, T_x, x_dtype))
    if has_gamma:
        cbs.append(cb(CB_GAMMA, 1, T_g, g_dtype))
    n_mm = 2 * R if form == "cmp" else 1
    cbs.append(cb(CB_MM, n_mm, T_acc, t_acc))
    cbs.append(cb(CB_SQ, max(R, dest_tiles), T_acc, t_acc))
    cbs.append(cb(CB_PARTIAL, min(U, PARTIAL_RING), T_acc, t_acc))
    cbs.append(cb(CB_GATHERED, o_max * k, T_acc, t_acc))
    cbs.append(cb(CB_INV_OWNED, o_max, T_acc, t_acc))
    cbs.append(cb(CB_INV, U, T_acc, t_acc))

    defines = [("FLAT_CMP", "1")] if form == "cmp" else []
    if has_gamma:
        defines.append(("RMS_HAS_GAMMA", "1"))
    defines.extend((d, "1") for d in extra_defines)

    g_acc_args = ttnn.TensorAccessorArgs(gamma) if has_gamma else ttnn.TensorAccessorArgs()
    g_page = gamma.buffer_aligned_page_size() if has_gamma else 0
    reader_ct = [F_full, tail_bytes, T_x, sw, eb, R, 1 if fp32_dest else 0, g_eb, g_page, n_mm]
    reader_ct.extend(g_acc_args.get_compile_time_args())
    compute_ct = [F_full, 1 if tail_elems else 0, R, U, k]
    writer_ct = [F_full, tail_bytes, T_x, U, k, o_max, T_acc, SEM_PARTIALS, SEM_INV, WRITER_NOC_XY_ARG_BASE]

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
            noc_xy.extend([v.x, v.y])
        for c, m, col0, valid in grp:
            owned = len(range(m, U, k))
            reader_rt[c.x][c.y] = [x_addr, valid, col0, g_addr]
            compute_rt[c.x][c.y] = [m, owned, inv_w_bits, eps_bits]
            writer_rt[c.x][c.y] = [out_addr, m, owned] + noc_xy

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "rms_norm_flat_reader.cpp"),
            core_ranges=core_ranges,
            compile_time_args=reader_ct,
            defines=defines,
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "rms_norm_flat_writer.cpp"),
            core_ranges=core_ranges,
            compile_time_args=writer_ct,
            defines=defines,
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "rms_norm_flat_compute.cpp"),
            core_ranges=core_ranges,
            compile_time_args=compute_ct,
            defines=defines,
            runtime_args=compute_rt,
            config=compute_kernel_config,
        ),
    ]
    semaphores = [
        ttnn.SemaphoreDescriptor(id=SEM_PARTIALS, core_ranges=core_ranges, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_INV, core_ranges=core_ranges, initial_value=0),
    ]
    info = dict(k=k, sw=sw, sh=sh, F=F, F_full=F_full, tail=tail_elems, R=R, U=U, o_max=o_max)
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs), info
