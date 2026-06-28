# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Program descriptor for scaled_dot_product_attention (Flash Attention).

Implements the Flash Attention v2 recurrence with online softmax.
See op_design.md for the full algorithm and CB layout.
"""

from pathlib import Path
import math
import struct

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE_DIM = 32

# Block sizes (op_design.md tile block sizing).
B_Q_T = 4  # Q-block tile rows (128 rows)
B_KV_T = 4  # KV-block tile cols (128 cols)


def create_program_descriptor(
    query: ttnn.Tensor,
    key: ttnn.Tensor,
    value: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    attn_mask: ttnn.Tensor | None = None,
    is_causal: bool = False,
    scale: float | None = None,
    math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en: bool = True,
    math_approx_mode: bool = False,
) -> ttnn.ProgramDescriptor:
    """Build the program descriptor for Flash Attention."""
    q_shape = list(query.shape)
    k_shape = list(key.shape)
    B = q_shape[0]
    H_q = q_shape[1]
    H_kv = k_shape[1]
    S_q = q_shape[2]
    S_kv = k_shape[2]
    D = q_shape[-1]
    D_t = D // TILE_DIM
    S_q_tiles = S_q // TILE_DIM
    S_kv_tiles = S_kv // TILE_DIM

    tile_size = ttnn.tile_size(query.dtype)  # bf16 → 2048 bytes

    # Resolve scale: explicit value or 1/sqrt(D).
    resolved_scale = scale if scale is not None else (1.0 / math.sqrt(D))
    scale_bits = struct.unpack("I", struct.pack("f", resolved_scale))[0]

    # Work distribution: one (B,H) pair per work unit.
    num_work_units = B * H_q
    grid_size = query.device().compute_with_storage_grid_size()
    num_cores, all_cores, core_group_1, core_group_2, units_per_core_1, units_per_core_2 = ttnn.split_work_to_cores(
        grid_size, num_work_units, row_wise=True
    )

    has_mask = attn_mask is not None

    # Mask shape: (B, 1, S_q, S_kv) or (B, H_q, S_q, S_kv)
    if has_mask:
        mask_h = attn_mask.shape[1]
        mask_is_per_head = mask_h == H_q
    else:
        mask_is_per_head = False

    num_q_blocks = (S_q_tiles + B_Q_T - 1) // B_Q_T
    num_kv_blocks = (S_kv_tiles + B_KV_T - 1) // B_KV_T
    num_o_tiles = B_Q_T * D_t
    num_score_tiles = B_Q_T * B_KV_T

    # --- Circular Buffers ---
    # CB indices: 0-7 inputs, 8-15 special, 16-23 outputs, 24-31 intermediates.
    CB_Q = 0
    CB_K = 1
    CB_V = 2
    CB_MASK = 3
    CB_SCALER_MAX = 6
    CB_SCALER_SUM = 7
    CB_ALPHA = 8
    CB_SCALE_FACTOR = 5
    CB_O = 16
    CB_OUT = 17
    CB_SCORES = 24
    CB_SCORES_MASKED = 25
    CB_MAX_NEW = 26
    CB_MAX_OLD = 27
    CB_EXP_SCORES = 28
    CB_SUM_NEW = 29
    CB_SUM_OLD = 30
    CB_O_ACCUM = 31

    cbs = [
        # Input CBs
        ttnn.CBDescriptor(
            total_size=num_o_tiles * tile_size,  # B_q_t * D_t
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_Q, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=2 * B_KV_T * D_t * tile_size,  # double-buffered: 2 * B_kv_t * D_t
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_K, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=2 * B_KV_T * D_t * tile_size,  # double-buffered: 2 * B_kv_t * D_t
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_V, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=2 * num_score_tiles * tile_size,  # double-buffered: 2 * B_q_t * B_kv_t
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_MASK, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        # Scaler CBs (1 page each, pushed once by reader, never popped by reduce)
        ttnn.CBDescriptor(
            total_size=1 * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_SCALER_MAX, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=1 * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_SCALER_SUM, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        # Scale factor CB (1 page, pushed once by reader)
        ttnn.CBDescriptor(
            total_size=1 * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_SCALE_FACTOR, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        # Alpha CB (B_q_t pages)
        ttnn.CBDescriptor(
            total_size=B_Q_T * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_ALPHA, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        # Output CBs
        ttnn.CBDescriptor(
            total_size=num_o_tiles * tile_size,  # B_q_t * D_t
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_O, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=num_o_tiles * tile_size,  # B_q_t * D_t (writer drains)
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_OUT, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        # Intermediate CBs
        ttnn.CBDescriptor(
            total_size=num_score_tiles * tile_size,  # B_q_t * B_kv_t
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_SCORES, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=num_score_tiles * tile_size,  # B_q_t * B_kv_t
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_SCORES_MASKED, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=B_Q_T * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_MAX_NEW, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=B_Q_T * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_MAX_OLD, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=num_score_tiles * tile_size,  # B_q_t * B_kv_t
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_EXP_SCORES, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=B_Q_T * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_SUM_NEW, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=B_Q_T * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_SUM_OLD, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=num_o_tiles * tile_size,  # B_q_t * D_t
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_O_ACCUM, data_format=query.dtype, page_size=tile_size)
            ],
        ),
    ]

    # --- Reader kernel ---
    # CT args: [has_mask, H_q, H_kv, mask_is_per_head,
    #           ...Q_accessor, ...K_accessor, ...V_accessor, ...mask_accessor]
    reader_ct_args = [1 if has_mask else 0, H_q, H_kv, 1 if mask_is_per_head else 0]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(query).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(key).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(value).get_compile_time_args())
    if has_mask:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(attn_mask).get_compile_time_args())
    else:
        reader_ct_args.extend(ttnn.TensorAccessorArgs().get_compile_time_args())

    # RT args per core: [num_work_units, B_q_t, B_kv_t, D_t, S_q_tiles, S_kv_tiles,
    #                    b0, h0, b1, h1, ...,
    #                    q_addr, k_addr, v_addr, scale_bits, mask_addr]
    reader_rt_args = ttnn.RuntimeArgs()
    cores = ttnn.grid_to_cores(num_cores, grid_size.x, grid_size.y, row_wise=True)
    work_unit_assigned = 0
    for core_idx, core in enumerate(cores):
        if units_per_core_2 == 0:
            units_this_core = units_per_core_1
        else:
            group1_count = (num_work_units - num_cores * units_per_core_2) // (units_per_core_1 - units_per_core_2)
            if core_idx < group1_count:
                units_this_core = units_per_core_1
            else:
                units_this_core = units_per_core_2

        rt = [units_this_core, B_Q_T, B_KV_T, D_t, S_q_tiles, S_kv_tiles]
        for i in range(units_this_core):
            bh = work_unit_assigned + i
            b = bh // H_q
            h = bh % H_q
            rt.append(b)
            rt.append(h)
        work_unit_assigned += units_this_core
        rt.append(query.buffer_address())
        rt.append(key.buffer_address())
        rt.append(value.buffer_address())
        rt.append(scale_bits)
        rt.append(attn_mask.buffer_address() if has_mask else 0)
        reader_rt_args[core.x][core.y] = rt

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Writer kernel ---
    # CT args: [num_o_tiles_per_q_block, ...TensorAccessorArgs(output)]
    writer_ct_args = [num_o_tiles]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # RT args per core: [output_addr, total_tiles]
    # total_tiles = num_work_units * num_q_blocks * num_o_tiles
    writer_rt_args = ttnn.RuntimeArgs()
    for core_idx, core in enumerate(cores):
        if units_per_core_2 == 0:
            units_this_core = units_per_core_1
        else:
            group1_count = (num_work_units - num_cores * units_per_core_2) // (units_per_core_1 - units_per_core_2)
            if core_idx < group1_count:
                units_this_core = units_per_core_1
            else:
                units_this_core = units_per_core_2
        total_tiles = units_this_core * num_q_blocks * num_o_tiles
        writer_rt_args[core.x][core.y] = [output_tensor.buffer_address(), total_tiles]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # --- Compute kernel ---
    # CT args: [has_mask, B_q_t, B_kv_t, D_t, S_q_tiles, S_kv_tiles]
    compute_ct_args = [1 if has_mask else 0, B_Q_T, B_KV_T, D_t, S_q_tiles, S_kv_tiles]

    # RT args per core: [num_work_units]
    compute_rt_args = ttnn.RuntimeArgs()
    for core_idx, core in enumerate(cores):
        if units_per_core_2 == 0:
            units_this_core = units_per_core_1
        else:
            group1_count = (num_work_units - num_cores * units_per_core_2) // (units_per_core_1 - units_per_core_2)
            if core_idx < group1_count:
                units_this_core = units_per_core_1
            else:
                units_this_core = units_per_core_2
        compute_rt_args[core.x][core.y] = [units_this_core]

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=fp32_dest_acc_en,
            math_approx_mode=math_approx_mode,
        ),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
