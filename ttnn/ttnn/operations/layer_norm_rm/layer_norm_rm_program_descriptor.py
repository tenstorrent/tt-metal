# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm - Program Descriptor

Defines the ProgramDescriptor: circular buffers, kernels, runtime args, and
work distribution for row-major layer normalization.

Work unit: tile-row (32 RM sticks spanning full width = Wt tiles).
Grid: 1-D linear, up to compute_with_storage_grid_size().
"""

import struct
from pathlib import Path
from functools import reduce
from operator import mul

import ttnn

# Kernel source files are in the kernels/ subdirectory
KERNEL_DIR = Path(__file__).parent / "kernels"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _float_to_uint32(value: float) -> int:
    """Convert a float to its IEEE-754 uint32 bit pattern."""
    return int.from_bytes(struct.pack("f", value), byteorder="little")


def _product(shape, start=0, end=None):
    """Compute the product of shape dimensions from start to end."""
    if end is None:
        end = len(shape)
    result = 1
    for i in range(start, end):
        result *= shape[i]
    return result


# ---------------------------------------------------------------------------
# CB index constants (from op_design.md)
# ---------------------------------------------------------------------------
CB_INPUT_RM = 0  # c_0: RM sticks for tilize
CB_TILIZED = 1  # c_1: Tilized input tiles
CB_GAMMA_RM = 2  # c_2: Gamma RM stick (optional)
CB_BETA_RM = 3  # c_3: Beta RM stick (optional)
CB_SCALER = 8  # c_8: Reduce scaler (1/W)
CB_EPS = 9  # c_9: Epsilon tile
CB_OUTPUT_RM = 16  # c_16: Untilized RM output
CB_MEAN = 24  # c_24: Row mean
CB_CENTERED = 25  # c_25: x - mean
CB_CENTERED_SQ = 26  # c_26: (x - mean)^2
CB_VAR = 27  # c_27: Row variance
CB_RSQRT = 28  # c_28: rsqrt(var + eps)
CB_GAMMA_TILED = 29  # c_29: Gamma tilized (optional)
CB_BETA_TILED = 30  # c_30: Beta tilized (optional)
CB_NORMALIZED = 31  # c_31: Normalized output


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
    epsilon: float = 1e-5,
) -> ttnn.ProgramDescriptor:
    """
    Create the ProgramDescriptor for layer_norm_rm.

    Args:
        input_tensor: Input RM bfloat16 tensor on device.
        output_tensor: Pre-allocated output RM bfloat16 tensor on device.
        gamma: Optional scale tensor (1,1,1,W) RM bfloat16 on device.
        beta: Optional shift tensor (1,1,1,W) RM bfloat16 on device.
        epsilon: Variance stabilizer.

    Returns:
        ProgramDescriptor ready for ttnn.generic_op.
    """
    device = input_tensor.device()
    has_gamma = gamma is not None
    has_beta = beta is not None

    # ===== 1. TENSOR METADATA =====
    shape = input_tensor.shape
    rank = len(shape)
    W = shape[rank - 1]
    H = shape[rank - 2]
    # Batch dims: product of all dims except last two
    batch = _product(shape, 0, rank - 2) if rank > 2 else 1

    Wt = W // 32  # tiles per row
    nblocks = (batch * H) // 32  # total tile-rows

    # Page sizes
    tile_size = ttnn.tile_size(input_tensor.dtype)  # bf16 tile = 2*32*32 = 2048
    stick_size = W * input_tensor.element_size()  # W * 2 bytes for bf16

    # ===== 2. CORE GRID AND WORK DISTRIBUTION =====
    grid = device.compute_with_storage_grid_size()
    max_x = grid.x - 1
    max_y = grid.y - 1
    max_core = ttnn.CoreCoord(max_x, max_y)
    all_possible_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), max_core)])

    (
        num_cores,
        core_grid,
        core_group_1,
        core_group_2,
        nblocks_per_core_g1,
        nblocks_per_core_g2,
    ) = ttnn.split_work_to_cores(all_possible_cores, nblocks)

    # core_grid is the full set of active cores (union of group_1 and group_2)
    # core_group_1: cores that process nblocks_per_core_g1 tile-rows
    # core_group_2: cores that process nblocks_per_core_g2 tile-rows (cliff)

    # ===== 3. CIRCULAR BUFFER DESCRIPTORS =====
    cbs = []

    # c_0: RM sticks for tilize (Wt pages of tile_size)
    cbs.append(_make_cb(CB_INPUT_RM, Wt * tile_size, tile_size, input_tensor.dtype, core_grid))

    # c_1: Tilized input tiles (Wt pages)
    cbs.append(_make_cb(CB_TILIZED, Wt * tile_size, tile_size, input_tensor.dtype, core_grid))

    # c_8: Reduce scaler 1/W (1 page)
    cbs.append(_make_cb(CB_SCALER, 1 * tile_size, tile_size, input_tensor.dtype, core_grid))

    # c_9: Epsilon tile (1 page)
    cbs.append(_make_cb(CB_EPS, 1 * tile_size, tile_size, input_tensor.dtype, core_grid))

    # c_16: Untilized RM output (Wt pages of tile_size)
    cbs.append(_make_cb(CB_OUTPUT_RM, Wt * tile_size, tile_size, output_tensor.dtype, core_grid))

    # c_24: Row mean (1 page)
    cbs.append(_make_cb(CB_MEAN, 1 * tile_size, tile_size, input_tensor.dtype, core_grid))

    # c_25: Centered x - mean (Wt pages)
    cbs.append(_make_cb(CB_CENTERED, Wt * tile_size, tile_size, input_tensor.dtype, core_grid))

    # c_26: Centered squared (Wt pages)
    cbs.append(_make_cb(CB_CENTERED_SQ, Wt * tile_size, tile_size, input_tensor.dtype, core_grid))

    # c_27: Row variance (1 page)
    cbs.append(_make_cb(CB_VAR, 1 * tile_size, tile_size, input_tensor.dtype, core_grid))

    # c_28: rsqrt(var + eps) (1 page)
    cbs.append(_make_cb(CB_RSQRT, 1 * tile_size, tile_size, input_tensor.dtype, core_grid))

    # c_31: Normalized output (Wt pages)
    cbs.append(_make_cb(CB_NORMALIZED, Wt * tile_size, tile_size, input_tensor.dtype, core_grid))

    # Optional gamma/beta CBs
    if has_gamma:
        # c_2: Gamma RM stick (1 page of stick_size)
        cbs.append(_make_cb(CB_GAMMA_RM, 1 * stick_size, stick_size, input_tensor.dtype, core_grid))
        # c_29: Gamma tilized (Wt pages)
        cbs.append(_make_cb(CB_GAMMA_TILED, Wt * tile_size, tile_size, input_tensor.dtype, core_grid))

    if has_beta:
        # c_3: Beta RM stick (1 page of stick_size)
        cbs.append(_make_cb(CB_BETA_RM, 1 * stick_size, stick_size, input_tensor.dtype, core_grid))
        # c_30: Beta tilized (Wt pages)
        cbs.append(_make_cb(CB_BETA_TILED, Wt * tile_size, tile_size, input_tensor.dtype, core_grid))

    # ===== 4. KERNEL DESCRIPTORS =====

    # --- Reader kernel ---
    reader_ct_args = [
        stick_size,  # 0: stick_size (W * sizeof(bf16))
        1 if has_gamma else 0,  # 1: has_gamma
        1 if has_beta else 0,  # 2: has_beta
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    reader_rt_args = _build_reader_rt_args(
        input_tensor,
        gamma,
        beta,
        Wt,
        nblocks,
        epsilon,
        W,
        grid,
        core_group_1,
        core_group_2,
        nblocks_per_core_g1,
        nblocks_per_core_g2,
    )

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Compute kernel ---
    compute_ct_args = [
        Wt,  # 0: Wt (tiles per row)
        1 if has_gamma else 0,  # 1: has_gamma
        1 if has_beta else 0,  # 2: has_beta
    ]

    compute_rt_args = _build_compute_rt_args(
        grid,
        core_group_1,
        core_group_2,
        nblocks_per_core_g1,
        nblocks_per_core_g2,
    )

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_compute.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=False,
            math_approx_mode=False,
        ),
    )

    # --- Writer kernel ---
    writer_ct_args = [
        stick_size,  # 0: stick_size
        Wt,  # 1: Wt (tiles per row)
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = _build_writer_rt_args(
        output_tensor,
        nblocks,
        grid,
        core_group_1,
        core_group_2,
        nblocks_per_core_g1,
        nblocks_per_core_g2,
    )

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ===== 5. ASSEMBLE PROGRAM DESCRIPTOR =====
    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, compute_kernel, writer_kernel],
        cbs=cbs,
        semaphores=[],
    )


# ---------------------------------------------------------------------------
# CB helper
# ---------------------------------------------------------------------------


def _make_cb(
    buffer_index: int,
    total_size: int,
    page_size: int,
    dtype,
    core_ranges,
) -> ttnn.CBDescriptor:
    """Create a single CBDescriptor."""
    return ttnn.CBDescriptor(
        total_size=total_size,
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(
                buffer_index=buffer_index,
                data_format=dtype,
                page_size=page_size,
            )
        ],
    )


# ---------------------------------------------------------------------------
# Runtime args builders
# ---------------------------------------------------------------------------


def _iter_cores_in_range_set(core_range_set):
    """Yield (x, y) for every core in the CoreRangeSet, row-major order."""
    for cr in core_range_set.ranges():
        for y in range(cr.start.y, cr.end.y + 1):
            for x in range(cr.start.x, cr.end.x + 1):
                yield (x, y)


def _core_in_range_set(x, y, core_range_set):
    """Check if (x, y) is in any CoreRange of the set."""
    for cr in core_range_set.ranges():
        if cr.start.x <= x <= cr.end.x and cr.start.y <= y <= cr.end.y:
            return True
    return False


def _build_reader_rt_args(
    input_tensor,
    gamma,
    beta,
    Wt,
    nblocks,
    epsilon,
    W,
    grid,
    core_group_1,
    core_group_2,
    nblocks_per_core_g1,
    nblocks_per_core_g2,
):
    """Build per-core runtime args for the reader kernel."""
    rt_args = ttnn.RuntimeArgs()

    src_addr = input_tensor.buffer_address()
    gamma_addr = gamma.buffer_address() if gamma is not None else 0
    beta_addr = beta.buffer_address() if beta is not None else 0

    scaler_value = _float_to_uint32(1.0 / W)
    eps_value = _float_to_uint32(epsilon)

    stick_id = 0  # running counter for start_stick_id

    # Iterate over all possible cores in the grid to set args (or empty for idle)
    for y in range(grid.y):
        for x in range(grid.x):
            in_g1 = _core_in_range_set(x, y, core_group_1)
            in_g2 = len(core_group_2.ranges()) > 0 and _core_in_range_set(x, y, core_group_2)

            if in_g1:
                nblocks_this_core = nblocks_per_core_g1
            elif in_g2:
                nblocks_this_core = nblocks_per_core_g2
            else:
                rt_args[x][y] = []
                continue

            num_sticks = nblocks_this_core * 32

            rt_args[x][y] = [
                src_addr,  # 0: src_addr
                num_sticks,  # 1: num_sticks
                Wt,  # 2: Wt
                scaler_value,  # 3: scaler_value (1/W as float bits)
                eps_value,  # 4: eps_value
                stick_id,  # 5: start_stick_id
                gamma_addr,  # 6: gamma_addr (0 if absent)
                beta_addr,  # 7: beta_addr (0 if absent)
            ]
            stick_id += num_sticks

    return rt_args


def _build_compute_rt_args(
    grid,
    core_group_1,
    core_group_2,
    nblocks_per_core_g1,
    nblocks_per_core_g2,
):
    """Build per-core runtime args for the compute kernel."""
    rt_args = ttnn.RuntimeArgs()

    for y in range(grid.y):
        for x in range(grid.x):
            in_g1 = _core_in_range_set(x, y, core_group_1)
            in_g2 = len(core_group_2.ranges()) > 0 and _core_in_range_set(x, y, core_group_2)

            if in_g1:
                rt_args[x][y] = [nblocks_per_core_g1]
            elif in_g2:
                rt_args[x][y] = [nblocks_per_core_g2]
            else:
                rt_args[x][y] = []

    return rt_args


def _build_writer_rt_args(
    output_tensor,
    nblocks,
    grid,
    core_group_1,
    core_group_2,
    nblocks_per_core_g1,
    nblocks_per_core_g2,
):
    """Build per-core runtime args for the writer kernel."""
    rt_args = ttnn.RuntimeArgs()
    dst_addr = output_tensor.buffer_address()

    stick_id = 0  # running counter for start_stick_id

    for y in range(grid.y):
        for x in range(grid.x):
            in_g1 = _core_in_range_set(x, y, core_group_1)
            in_g2 = len(core_group_2.ranges()) > 0 and _core_in_range_set(x, y, core_group_2)

            if in_g1:
                nblocks_this_core = nblocks_per_core_g1
            elif in_g2:
                nblocks_this_core = nblocks_per_core_g2
            else:
                rt_args[x][y] = []
                continue

            rt_args[x][y] = [
                dst_addr,  # 0: dst_addr
                nblocks_this_core,  # 1: nblocks
                stick_id,  # 2: start_stick_id
            ]
            stick_id += nblocks_this_core * 32

    return rt_args
