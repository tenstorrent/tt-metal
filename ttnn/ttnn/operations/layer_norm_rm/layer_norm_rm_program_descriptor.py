# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
layer_norm_rm — Program Descriptor

Defines the ProgramDescriptor: circular buffers, kernels, and runtime args.

Work unit: one tile-row = 32 RM sticks spanning the full width (Wt tiles).
Work distribution: nblocks_per_core tile-rows assigned to each core,
with a cliff core getting the remainder.

CB layout (all bfloat16, tile_size = 2048 bytes):
  0  cb_in_rm      — input RM sticks (tile-sized pages, Wt pages/block)
  1  cb_tilized    — tilized input (Wt tiles/block, persists across reduce+sub)
  2  cb_mean       — row-reduced mean (1 tile/block)
  3  cb_centered   — x - mean (Wt tiles/block, persists across square+mul)
  4  cb_var_input  — (x-mean)^2, streaming (1 tile at a time)
  5  cb_var        — variance (1 tile/block)
  6  cb_gamma      — gamma tiles (Wt tiles/block, optional)
  7  cb_beta        — beta tiles (Wt tiles/block, optional)
  8  cb_eps        — epsilon scalar tile (1 tile, filled once by compute)
  9  cb_scaler     — 1/W reduce scaler (1 tile, program-lifetime)
  16 cb_normed     — normalized output (Wt tiles, streaming 1 per tile)
  17 cb_out_rm     — untilized output (tile-sized pages, Wt pages/block)
  24 cb_inv_std    — 1/sqrt(var+eps) (1 tile/block)
"""

from pathlib import Path
import struct
import ttnn

# Kernel files are relative to the tt-metal repo root
_REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent  # ttnn/ttnn/operations/layer_norm_rm -> repo root
KERNEL_DIR = Path("ttnn/ttnn/operations/layer_norm_rm/kernels")

# CB index constants matching the design
CB_IN_RM = 0
CB_TILIZED = 1
CB_MEAN = 2
CB_CENTERED = 3
CB_VAR_INPUT = 4
CB_VAR = 5
CB_GAMMA = 6
CB_BETA = 7
CB_EPS = 8
CB_SCALER = 9
CB_NORMED = 16
CB_OUT_RM = 17
CB_INV_STD = 24


def _float_to_uint32(f: float) -> int:
    """Pack a Python float as its IEEE-754 single-precision bit pattern."""
    return struct.unpack("I", struct.pack("f", f))[0]


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    gamma: ttnn.Tensor = None,
    beta: ttnn.Tensor = None,
    epsilon: float = 1e-5,
) -> ttnn.ProgramDescriptor:
    """
    Build the ProgramDescriptor for layer_norm_rm.

    Args:
        input_tensor:  Input tensor (bfloat16, ROW_MAJOR, INTERLEAVED)
        output_tensor: Pre-allocated output tensor (same spec)
        gamma:         Optional (1,1,1,W) gamma tensor
        beta:          Optional (1,1,1,W) beta tensor
        epsilon:       Numerical stability constant

    Returns:
        ProgramDescriptor for ttnn.generic_op
    """
    # =========================================================
    # 1. TENSOR METADATA
    # =========================================================
    shape = input_tensor.shape
    ndim = len(shape)

    # Last two dims are H and W (both divisible by 32, validated upstream)
    W = shape[-1]  # width in elements
    H = shape[-2]  # height in elements
    Wt = W // 32  # tiles per row
    Ht = H // 32  # tile-rows per H dimension

    # Total tile-rows across all batch dims
    batch = 1
    for i in range(ndim - 2):
        batch *= shape[i]
    total_nblocks = batch * Ht  # total tile-rows

    # Byte sizes
    # For ROW_MAJOR tensors, buffer_page_size() == stick_size (W * sizeof(dtype))
    stick_size = input_tensor.buffer_page_size()
    tile_size = ttnn.tile_size(input_tensor.dtype)  # 2048 for bf16 32x32

    has_gamma = gamma is not None
    has_beta = beta is not None

    # =========================================================
    # 2. WORK DISTRIBUTION
    # =========================================================
    device = input_tensor.device()
    compute_grid = device.compute_with_storage_grid_size()

    (
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        nblocks_per_core_group_1,
        nblocks_per_core_group_2,
    ) = ttnn.split_work_to_cores(compute_grid, total_nblocks)

    # =========================================================
    # 3. CIRCULAR BUFFER DESCRIPTORS
    # =========================================================
    cbs = []

    # --- CB 0: cb_in_rm — input RM sticks, tile-sized pages ---
    # IMPORTANT: page_size MUST be tile_size (not stick_size).
    # The tilize helper reads face/tile dims from CB metadata.
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_IN_RM,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # --- CB 1: cb_tilized — tilized input ---
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_TILIZED,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # --- CB 2: cb_mean — 1 tile per block ---
    cbs.append(
        ttnn.CBDescriptor(
            total_size=tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_MEAN,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # --- CB 3: cb_centered — Wt tiles per block ---
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_CENTERED,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # --- CB 4: cb_var_input — streaming, 1 tile ---
    cbs.append(
        ttnn.CBDescriptor(
            total_size=tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_VAR_INPUT,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # --- CB 5: cb_var — 1 tile per block ---
    cbs.append(
        ttnn.CBDescriptor(
            total_size=tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_VAR,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # --- CB 6: cb_gamma (only if gamma present) ---
    if has_gamma:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=Wt * tile_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_GAMMA,
                        data_format=ttnn.bfloat16,
                        page_size=tile_size,
                    )
                ],
            )
        )

    # --- CB 7: cb_beta (only if beta present) ---
    if has_beta:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=Wt * tile_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_BETA,
                        data_format=ttnn.bfloat16,
                        page_size=tile_size,
                    )
                ],
            )
        )

    # --- CB 8: cb_eps — 1 tile (epsilon scalar, filled by compute) ---
    cbs.append(
        ttnn.CBDescriptor(
            total_size=tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_EPS,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # --- CB 9: cb_scaler — 1/W scaler (program-lifetime, filled by reader) ---
    cbs.append(
        ttnn.CBDescriptor(
            total_size=tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_SCALER,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # --- CB 16: cb_normed — normalized output, Wt tiles streaming ---
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_NORMED,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # --- CB 17: cb_out_rm — output RM sticks, tile-sized pages ---
    cbs.append(
        ttnn.CBDescriptor(
            total_size=Wt * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_OUT_RM,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # --- CB 24: cb_inv_std — 1/sqrt(var+eps), 1 tile ---
    cbs.append(
        ttnn.CBDescriptor(
            total_size=tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_INV_STD,
                    data_format=ttnn.bfloat16,
                    page_size=tile_size,
                )
            ],
        )
    )

    # =========================================================
    # 4. RUNTIME ARGS AND KERNEL DESCRIPTORS
    # =========================================================
    input_addr = input_tensor.buffer_address()
    output_addr = output_tensor.buffer_address()
    gamma_addr = gamma.buffer_address() if has_gamma else 0
    beta_addr = beta.buffer_address() if has_beta else 0
    epsilon_packed = _float_to_uint32(epsilon)

    reader_ct_args = [
        stick_size,  # index 0: bytes per RM stick
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    writer_ct_args = [
        stick_size,  # index 0: output stick size (same as input)
        Wt,  # index 1: tiles per row
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # Build per-core runtime args
    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()

    # Accumulate start sticks per core (linearized assignment)
    # CoreRangeSet.ranges() returns a list of CoreRange objects
    start_stick = 0
    core_list = []  # list of (x, y, nblocks)

    def _collect_cores(core_ranges_set, nblocks_per_core):
        for cr in core_ranges_set.ranges():
            for x in range(cr.start.x, cr.end.x + 1):
                for y in range(cr.start.y, cr.end.y + 1):
                    core_list.append((x, y, nblocks_per_core))

    _collect_cores(core_group_1, nblocks_per_core_group_1)
    if nblocks_per_core_group_2 > 0:
        _collect_cores(core_group_2, nblocks_per_core_group_2)

    for x, y, nblocks in core_list:
        num_sticks = nblocks * 32  # 32 RM sticks per tile-row
        reader_rt_args[x][y] = [
            input_addr,  # 0: src_addr
            start_stick,  # 1: start_stick_id
            num_sticks,  # 2: number of RM sticks to read
            gamma_addr,  # 3: gamma buffer address (0 if absent)
            beta_addr,  # 4: beta buffer address (0 if absent)
        ]
        writer_rt_args[x][y] = [
            output_addr,  # 0: dst_addr
            start_stick,  # 1: start_stick_id
            nblocks,  # 2: number of tile-row blocks
        ]
        compute_rt_args[x][y] = []
        start_stick += num_sticks

    # Set empty args for all idle cores in the grid
    used_cores = {(x, y) for (x, y, _) in core_list}
    grid_x = compute_grid.x
    grid_y = compute_grid.y
    for x in range(grid_x):
        for y in range(grid_y):
            if (x, y) not in used_cores:
                reader_rt_args[x][y] = []
                writer_rt_args[x][y] = []
                compute_rt_args[x][y] = []

    # =========================================================
    # 5. KERNEL DESCRIPTORS
    # =========================================================
    # Compute compile-time args per core group (nblocks_per_core varies)
    # For simplicity with stub kernels, we use the same CT args for all cores
    # and pass nblocks_per_core as a compile-time arg per core group.
    # Since generic_op supports a single KernelDescriptor per kernel type,
    # we use the larger of the two groups as the compile-time value, and
    # the kernel will read actual count from runtime args (not CT args).
    # For the stub, we just use nblocks_per_core_group_1.
    compute_nblocks = nblocks_per_core_group_1 if nblocks_per_core_group_1 > 0 else 1

    compute_ct_args = [
        Wt,  # 0: tiles per row
        compute_nblocks,  # 1: nblocks_per_core (stub uses this)
        1 if has_gamma else 0,  # 2: has_gamma
        1 if has_beta else 0,  # 3: has_beta
        epsilon_packed,  # 4: epsilon as uint32 bits
    ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "layer_norm_rm_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=False,
            math_approx_mode=False,
        ),
    )

    # =========================================================
    # 6. ASSEMBLE PROGRAM DESCRIPTOR
    # =========================================================
    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
