# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm program descriptor: CBs, kernels, per-core args, and the blocking-model solver."""

import struct
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE = 32
BF16_DEST_MAX_WT = 64  # 16-bit DEST accumulation is accurate enough up to W=2048
G_CAP = 16  # max tiles in a compute window (Rb * Sb)
X_BUDGET = 512 * 1024  # bytes of cb_x
GAMMA_BUDGET = 128 * 1024  # bytes of resident gamma tiles
RM_DEPTH = 2  # tile-rows of RM sticks buffered in cb_x_rm
OUT_DEPTH = 2  # windows buffered in cb_out_tiles / cb_out_rm

# Circular buffer slots (single source; passed to kernels as compile-time args)
CB_X_RM = 0
CB_X = 1
CB_SCALER = 2
CB_GAMMA = 3
CB_SQ = 4
CB_STAT = 5
CB_XN = 6
CB_OUT_TILES = 7
CB_OUT_RM = 8
CB_GAMMA_RM = 9  # reader-private scratch for the gamma stick segment
CB_PSEND = 10  # W-split: this core's partial Sum x^2 tiles (compute -> writer, sent to every peer)
CB_PART = 11  # W-split: gathered partials of a row group (writer <- peers' NoC writes, -> compute)
SEM_PART = (0, 1)  # W-split: one arrival semaphore per row-block parity
W_SPLIT_MIN_GAIN = 0.75  # split only if the estimated per-core work drops below this fraction
W_SPLIT_OVERHEAD = 2  # per peer exchange cost, in tile-equivalents of per-core work
PART_BUDGET = 256 * 1024  # bytes of cb_part


def _div_up(a, b):
    return (a + b - 1) // b


def _fp32_bits(v):
    return struct.unpack("I", struct.pack("f", float(v)))[0]


@dataclass
class Blocking:
    Rb: int  # tile-rows per compute block
    Wb: int  # resident / chunk width in tiles of cb_x
    Sb: int  # segment (compute/DM granule) in tiles; divides Wt
    x_depth: int
    resident: bool  # R1 (whole row resident, x read once) vs R2 (two-pass streamed)
    gamma_resident: bool


def pick_blocking(Wt, rows_core_max, in_tile_bytes, gamma_tile_bytes, has_gamma):
    """Single source of every block knob; CB sizes and trip counts derive from this."""
    # Sb: largest divisor of Wt <= G_CAP (segments are uniform; no ragged tile-column tail).
    Sb = max(d for d in range(1, min(Wt, G_CAP) + 1) if Wt % d == 0)
    resident = Wt * in_tile_bytes <= X_BUDGET
    if resident:
        Wb = Wt
    else:
        Wb = max(Sb, (X_BUDGET // (2 * in_tile_bytes)) // Sb * Sb)
        Wb = min(Wb, Wt)
    Rb = max(1, min(G_CAP // min(Wb, G_CAP), rows_core_max))
    if Rb > 1:
        assert Sb == Wt == Wb
    x_depth = 2 if (not resident or 2 * Rb * Wb * in_tile_bytes <= X_BUDGET) else 1
    gamma_resident = has_gamma and Wt * gamma_tile_bytes <= GAMMA_BUDGET
    return Blocking(Rb, Wb, Sb, x_depth, resident, gamma_resident)


def pick_w_split(Wt, total_rows, num_grid_cores, im_tile_bytes):
    """Host predicate for the W-split regime. Returns (n_w, n_r); n_w == 1 means the row-split path.

    n_w must divide Wt (uniform compile-time slice). The split is taken only when it lowers the
    estimated per-core work (rows * slice width + exchange overhead) well below the unsplit one.
    """
    base_work = _div_up(total_rows, num_grid_cores) * Wt
    best = (1, min(total_rows, num_grid_cores), base_work)
    for n_w in range(2, min(Wt, num_grid_cores) + 1):
        if Wt % n_w:
            continue
        n_r = min(total_rows, num_grid_cores // n_w)
        rows = _div_up(total_rows, n_r)
        if 2 * rows * n_w * im_tile_bytes > PART_BUDGET:
            continue
        work = rows * (Wt // n_w) + W_SPLIT_OVERHEAD * n_w
        if work < best[2] and work <= W_SPLIT_MIN_GAIN * base_work:
            best = (n_w, n_r, work)
    return best[0], best[1]


def _cb(index, dtype, page_size, num_pages, core_ranges):
    return ttnn.CBDescriptor(
        total_size=page_size * num_pages,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
    )


def create_program_descriptor(input_tensor, output_tensor, gamma, epsilon, compute_kernel_config):
    shape = list(input_tensor.shape)
    W = shape[-1]
    H = shape[-2]
    images = 1
    for d in shape[:-2]:
        images *= d
    Wt = _div_up(W, TILE)
    Ht_img = _div_up(H, TILE)
    # Accumulating Wt squared tiles sequentially in a 16-bit DEST loses ~6 mantissa bits once the running
    # sum is >> one tile's partial (measured: 1.0 -> 0.859 output at W=5120, error grows with Wt, and is
    # independent of segmenting / CB dtype / fidelity). Wide rows therefore use an fp32 DEST for the
    # Sigma x^2 accumulation even when the caller asked for a 16-bit DEST.
    if (not compute_kernel_config.fp32_dest_acc_en) and Wt > BF16_DEST_MAX_WT:
        compute_kernel_config = ttnn.ComputeConfigDescriptor(
            math_fidelity=compute_kernel_config.math_fidelity,
            fp32_dest_acc_en=True,
            math_approx_mode=compute_kernel_config.math_approx_mode,
        )
    total_tile_rows = images * Ht_img

    is_rm = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    has_gamma = gamma is not None
    in_dtype = input_tensor.dtype
    out_dtype = output_tensor.dtype
    es_x = 2 if in_dtype == ttnn.bfloat8_b else input_tensor.element_size()
    in_tile = ttnn.tile_size(in_dtype)
    out_tile = ttnn.tile_size(out_dtype)
    bf16_tile = ttnn.tile_size(ttnn.bfloat16)
    g_dtype = gamma.dtype if has_gamma else ttnn.bfloat16
    g_is_tile = has_gamma and gamma.layout == ttnn.TILE_LAYOUT
    es_g = 2 if (not has_gamma or g_dtype == ttnn.bfloat8_b) else gamma.element_size()
    g_tile = ttnn.tile_size(g_dtype)

    device = input_tensor.device()
    grid = device.compute_with_storage_grid_size()
    # Intermediates follow the DEST width: fp32 only when fp32 DEST accumulation is on.
    im_dtype = ttnn.float32 if compute_kernel_config.fp32_dest_acc_en else ttnn.bfloat16
    im_tile = ttnn.tile_size(im_dtype)
    n_w, n_r = pick_w_split(Wt, total_tile_rows, grid.x * grid.y, im_tile)
    Wt_full = Wt
    Wt = Wt_full // n_w  # per-core slice width in tiles (every per-core CB / loop below is slice-sized)
    if n_w == 1:
        num_cores, all_cores, group1, group2, rows_g1, rows_g2 = ttnn.split_work_to_cores(
            grid, total_tile_rows, row_wise=True
        )
        core_plan = []  # (core, first_row, rows, slice index)
        first = 0
        for group, rows in ((group1, rows_g1), (group2, rows_g2)):
            for core in ttnn.corerange_to_cores(group, row_wise=True):
                core_plan.append((core, first, rows, 0))
                first += rows
        rows_max = rows_g1
    else:
        # Row group g (g < n_r) owns a contiguous run of tile-rows; its n_w cores each own one W slice.
        # Linear core index g * n_w + j, row-major over the grid.
        base_rows, extra = divmod(total_tile_rows, n_r)
        rows_max = base_rows + (1 if extra else 0)
        core_plan = []
        first = 0
        for g in range(n_r):
            rows = base_rows + (1 if g < extra else 0)
            for j in range(n_w):
                lin = g * n_w + j
                core_plan.append((ttnn.CoreCoord(lin % grid.x, lin // grid.x), first, rows, j))
            first += rows
        all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(c[0], c[0]) for c in core_plan])
    blk = pick_blocking(Wt, rows_max, in_tile, g_tile, has_gamma)
    Rb, Wb, Sb = blk.Rb, blk.Wb, blk.Sb
    G = Rb * Sb
    num_seg = Wt // Sb

    # ------------------------------------------------------------------ CBs
    cbs = []
    if is_rm:
        cbs.append(_cb(CB_X_RM, in_dtype, Sb * TILE * es_x, RM_DEPTH * TILE, all_cores))
    cbs.append(_cb(CB_X, in_dtype, in_tile, blk.x_depth * Rb * Wb, all_cores))
    cbs.append(_cb(CB_SCALER, ttnn.bfloat16, bf16_tile, 1, all_cores))
    if has_gamma:
        g_pages = Wt if blk.gamma_resident else 2 * Sb
        cbs.append(_cb(CB_GAMMA, g_dtype, g_tile, g_pages, all_cores))
        if not g_is_tile:
            cbs.append(_cb(CB_GAMMA_RM, g_dtype, Sb * TILE * es_g, 1, all_cores))
        cbs.append(_cb(CB_XN, im_dtype, im_tile, G, all_cores))
    cbs.append(_cb(CB_SQ, im_dtype, im_tile, G, all_cores))
    cbs.append(_cb(CB_STAT, im_dtype, im_tile, Rb, all_cores))
    if n_w > 1:
        cbs.append(_cb(CB_PSEND, im_dtype, im_tile, Rb, all_cores))
        cbs.append(_cb(CB_PART, im_dtype, im_tile, 2 * Rb * n_w, all_cores))
    cbs.append(_cb(CB_OUT_TILES, out_dtype, out_tile, OUT_DEPTH * G, all_cores))
    if is_rm:
        cbs.append(_cb(CB_OUT_RM, out_dtype, out_tile, OUT_DEPTH * Sb, all_cores))

    # ------------------------------------------------------------ CT args
    reader_ct = [
        int(is_rm), es_x, Wt, Sb, Rb, int(blk.resident), int(has_gamma), es_g, int(blk.gamma_resident),
        W, H, Ht_img, CB_X_RM, CB_X, CB_SCALER, CB_GAMMA, CB_GAMMA_RM, int(g_is_tile), Wt_full,
    ]  # fmt: skip
    reader_ct.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if has_gamma
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    writer_ct = [
        int(is_rm), es_x, Wt, Sb, Rb, W, H, Ht_img, CB_OUT_TILES, CB_OUT_RM, Wt_full, n_w, CB_PSEND, CB_PART,
        SEM_PART[0], SEM_PART[1],
    ]  # fmt: skip
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    compute_ct = [
        int(is_rm), int(has_gamma), int(blk.resident), int(blk.gamma_resident), Wt, Sb, Rb, num_seg,
        CB_X_RM, CB_X, CB_SCALER, CB_GAMMA, CB_SQ, CB_STAT, CB_XN, CB_OUT_TILES, CB_OUT_RM,
        n_w, CB_PSEND, CB_PART,
    ]  # fmt: skip

    # ------------------------------------------------------------ RT args
    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    x_addr = input_tensor.buffer_address()
    o_addr = output_tensor.buffer_address()
    g_addr = gamma.buffer_address() if has_gamma else 0
    # With a 16-bit DEST the running sum of squares loses accuracy once it grows large (measured: >4096
    # gives up to 20% error), so fold 1/W into the reduce scaler (partials are then O(mean)) and leave
    # nothing to multiply in compute. With fp32 DEST the scaler stays 1.0 and compute applies 1/W.
    prescale = not compute_kernel_config.fp32_dest_acc_en
    scaler_bits = _fp32_bits(1.0 / W if prescale else 1.0)
    inv_w_bits = _fp32_bits(1.0 if prescale else 1.0 / W)
    eps_bits = _fp32_bits(epsilon)

    group_cores = {}  # row group -> its cores' physical coords, in slice order
    if n_w > 1:
        for core, first_row, rows, j in core_plan:
            phys = device.worker_core_from_logical_core(core)
            group_cores.setdefault(first_row, []).append((phys.x, phys.y))
    covered = 0
    for core, first_row, rows, j in core_plan:
        slice_tile0 = j * Wt
        is_last_slice = int(j == n_w - 1)
        reader_rt[core.x][core.y] = [x_addr, g_addr, first_row, rows, scaler_bits, slice_tile0, is_last_slice]
        writer_rt[core.x][core.y] = [o_addr, first_row, rows, slice_tile0, j] + [
            v for xy in group_cores.get(first_row, []) for v in xy
        ]
        compute_rt[core.x][core.y] = [rows, inv_w_bits, eps_bits]
        covered += rows if j == 0 else 0
    assert covered == total_tile_rows

    semaphores = (
        [ttnn.SemaphoreDescriptor(id=sid, core_ranges=all_cores, initial_value=0) for sid in SEM_PART]
        if n_w > 1
        else []
    )

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=compute_kernel_config,
    )
    return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=semaphores, cbs=cbs)
