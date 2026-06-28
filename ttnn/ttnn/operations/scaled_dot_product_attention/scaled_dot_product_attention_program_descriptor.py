# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
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

# Max block sizes (op_design.md tile block sizing). Actual sizes are clamped
# to the tensor dimensions at runtime.
MAX_B_Q_T = 4  # Max Q-block tile rows (128 rows)
MAX_B_KV_T = 4  # Max KV-block tile cols (128 cols)


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

    # Clamp block sizes to actual tensor dimensions.
    B_q_t = min(MAX_B_Q_T, S_q_tiles)
    B_kv_t = min(MAX_B_KV_T, S_kv_tiles)

    tile_size = ttnn.tile_size(query.dtype)

    resolved_scale = scale if scale is not None else (1.0 / math.sqrt(D))
    scale_bits = struct.unpack("I", struct.pack("f", resolved_scale))[0]

    num_work_units = B * H_q
    grid_size = query.device().compute_with_storage_grid_size()
    num_cores, all_cores, core_group_1, core_group_2, units_per_core_1, units_per_core_2 = \
        ttnn.split_work_to_cores(grid_size, num_work_units, row_wise=True)

    has_mask = attn_mask is not None

    # --- Circular Buffers ---
    CB_Q = 0
    CB_K = 1
    CB_V = 2
    CB_MASK = 3
    CB_SCALER_REDUCE = 4
    CB_SCALE_FACTOR = 5
    CB_ALPHA = 8
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

    num_q_tiles = B_q_t * D_t
    num_k_tiles = B_kv_t * D_t
    num_v_tiles = B_kv_t * D_t
    num_mask_tiles = B_q_t * B_kv_t
    num_o_tiles = B_q_t * D_t
    num_score_tiles = B_q_t * B_kv_t

    cbs = [
        ttnn.CBDescriptor(
            total_size=num_q_tiles * tile_size,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB_Q, data_format=query.dtype, page_size=tile_size)],
        ),
        ttnn.CBDescriptor(
            total_size=2 * num_k_tiles * tile_size,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB_K, data_format=query.dtype, page_size=tile_size)],
        ),
        ttnn.CBDescriptor(
            total_size=2 * num_v_tiles * tile_size,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB_V, data_format=query.dtype, page_size=tile_size)],
        ),
        ttnn.CBDescriptor(
            total_size=2 * num_mask_tiles * tile_size,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB_MASK, data_format=query.dtype, page_size=tile_size)],
        ),
        ttnn.CBDescriptor(
            total_size=2 * tile_size,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB_SCALER_REDUCE, data_format=query.dtype, page_size=tile_size)],
        ),
        ttnn.CBDescriptor(
            total_size=1 * tile_size,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB_SCALE_FACTOR, data_format=query.dtype, page_size=tile_size)],
        ),
        ttnn.CBDescriptor(
            total_size=B_q_t * tile_size,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB_ALPHA, data_format=query.dtype, page_size=tile_size)],
        ),
        ttnn.CBDescriptor(
            total_size=num_o_tiles * tile_size,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB_O, data_format=query.dtype, page_size=tile_size)],
        ),
        ttnn.CBDescriptor(
            total_size=num_o_tiles * tile_size,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB_OUT, data_format=query.dtype, page_size=tile_size)],
        ),
        ttnn.CBDescriptor(
            total_size=num_score_tiles * tile_size,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB_SCORES, data_format=query.dtype, page_size=tile_size)],
        ),
        ttnn.CBDescriptor(
            total_size=num_score_tiles * tile_size,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB_SCORES_MASKED, data_format=query.dtype, page_size=tile_size)],
        ),
        ttnn.CBDescriptor(
            total_size=B_q_t * tile_size,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB_MAX_NEW, data_format=query.dtype, page_size=tile_size)],
        ),
        ttnn.CBDescriptor(
            total_size=B_q_t * tile_size,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB_MAX_OLD, data_format=query.dtype, page_size=tile_size)],
        ),
        ttnn.CBDescriptor(
            total_size=num_score_tiles * tile_size,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB_EXP_SCORES, data_format=query.dtype, page_size=tile_size)],
        ),
        ttnn.CBDescriptor(
            total_size=B_q_t * tile_size,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB_SUM_NEW, data_format=query.dtype, page_size=tile_size)],
        ),
        ttnn.CBDescriptor(
            total_size=B_q_t * tile_size,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB_SUM_OLD, data_format=query.dtype, page_size=tile_size)],
        ),
        ttnn.CBDescriptor(
            total_size=num_o_tiles * tile_size,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB_O_ACCUM, data_format=query.dtype, page_size=tile_size)],
        ),
    ]

    # --- Reader kernel ---
    reader_ct_args = [1 if has_mask else 0, H_q, H_kv]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(query).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(key).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(value).get_compile_time_args())
    if has_mask:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(attn_mask).get_compile_time_args())
    else:
        reader_ct_args.extend(ttnn.TensorAccessorArgs().get_compile_time_args())

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

        rt = [units_this_core, B_q_t, B_kv_t, D_t, S_q_tiles, S_kv_tiles]
        for i in range(units_this_core):
            bh = work_unit_assigned + i
            b = bh // H_q
            h = bh % H_q
            rt.append(b)
            rt.append(h)
        work_unit_assigned += units_this_core
        rt.append(query.buffer_address())
        rt.append(key.buffer_address())
        rt.append(scale_bits)
        rt.append(value.buffer_address())
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
    writer_ct_args = [0]  # placeholder (total_tiles is now RT arg)
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # RT args per core: [output_addr, total_tiles]
    # total_tiles = num_work_units * num_o_tiles (each work unit produces num_o_tiles output tiles)
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
        total_tiles = units_this_core * num_o_tiles
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
    compute_ct_args = [1 if has_mask else 0, B_q_t, B_kv_t, D_t, S_q_tiles, S_kv_tiles]
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=[],
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
