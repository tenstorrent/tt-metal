# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Program descriptor for scaled_dot_product_attention (Flash Attention)."""

from pathlib import Path
import math
import struct

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE_DIM = 32
B_Q_T = 4
B_KV_T = 4


def create_program_descriptor(
    query,
    key,
    value,
    output_tensor,
    *,
    attn_mask=None,
    is_causal=False,
    scale=None,
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    math_approx_mode=False,
):
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
    tile_size = ttnn.tile_size(query.dtype)

    resolved_scale = scale if scale is not None else (1.0 / math.sqrt(D))
    scale_bits = struct.unpack("I", struct.pack("f", resolved_scale))[0]

    num_work_units = B * H_q
    grid_size = query.device().compute_with_storage_grid_size()
    (num_cores, all_cores, core_group_1, core_group_2, units_per_core_1, units_per_core_2) = ttnn.split_work_to_cores(
        grid_size, num_work_units, row_wise=True
    )

    has_mask = attn_mask is not None

    num_q_blocks = (S_q_tiles + B_Q_T - 1) // B_Q_T
    num_o_tiles = B_Q_T * D_t
    num_score_tiles = B_Q_T * B_KV_T

    CB_Q, CB_K, CB_V, CB_MASK = 0, 1, 2, 3
    CB_SCALER_REDUCE = 4
    CB_SCALE_FACTOR = 5
    CB_ALPHA = 8
    CB_O, CB_OUT = 16, 17
    CB_SCORES, CB_SCORES_MASKED = 24, 25
    CB_MAX_NEW, CB_MAX_OLD = 26, 27
    CB_EXP_SCORES, CB_SUM_NEW, CB_SUM_OLD = 28, 29, 30
    CB_O_ACCUM = 31

    cbs = [
        ttnn.CBDescriptor(
            total_size=num_o_tiles * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_Q, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=2 * B_KV_T * D_t * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_K, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=2 * B_KV_T * D_t * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_V, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=2 * num_score_tiles * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_MASK, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=2 * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_SCALER_REDUCE, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=1 * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_SCALE_FACTOR, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=B_Q_T * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_ALPHA, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=num_o_tiles * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_O, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=num_o_tiles * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_OUT, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=num_score_tiles * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_SCORES, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        ttnn.CBDescriptor(
            total_size=num_score_tiles * tile_size,
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
            total_size=num_score_tiles * tile_size,
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
            total_size=num_o_tiles * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_O_ACCUM, data_format=query.dtype, page_size=tile_size)
            ],
        ),
    ]

    # Reader kernel
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
            units_this_core = units_per_core_1 if core_idx < group1_count else units_per_core_2
        rt = [units_this_core, B_Q_T, B_KV_T, D_t, S_q_tiles, S_kv_tiles]
        for i in range(units_this_core):
            bh = work_unit_assigned + i
            rt.append(bh // H_q)
            rt.append(bh % H_q)
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

    # Writer kernel
    writer_ct_args = [num_o_tiles]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_rt_args = ttnn.RuntimeArgs()
    for core_idx, core in enumerate(cores):
        if units_per_core_2 == 0:
            units_this_core = units_per_core_1
        else:
            group1_count = (num_work_units - num_cores * units_per_core_2) // (units_per_core_1 - units_per_core_2)
            units_this_core = units_per_core_1 if core_idx < group1_count else units_per_core_2
        total_tiles = units_this_core * num_q_blocks * num_o_tiles
        writer_rt_args[core.x][core.y] = [output_tensor.buffer_address(), total_tiles]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # Compute kernel
    compute_ct_args = [1 if has_mask else 0, B_Q_T, B_KV_T, D_t, S_q_tiles, S_kv_tiles]
    compute_rt_args = ttnn.RuntimeArgs()
    for core_idx, core in enumerate(cores):
        if units_per_core_2 == 0:
            units_this_core = units_per_core_1
        else:
            group1_count = (num_work_units - num_cores * units_per_core_2) // (units_per_core_1 - units_per_core_2)
            units_this_core = units_per_core_1 if core_idx < group1_count else units_per_core_2
        compute_rt_args[core.x][core.y] = [units_this_core]

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=math_fidelity, fp32_dest_acc_en=fp32_dest_acc_en, math_approx_mode=math_approx_mode
        ),
    )

    return ttnn.ProgramDescriptor(kernels=[reader_kernel, writer_kernel, compute_kernel], semaphores=[], cbs=cbs)
