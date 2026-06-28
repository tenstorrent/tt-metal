# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Program descriptor for scaled_dot_product_attention (Flash Attention)."""
from pathlib import Path
import math, struct, ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE_DIM = 32


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
    B, H_q, H_kv = q_shape[0], q_shape[1], k_shape[1]
    S_q, S_kv, D = q_shape[2], k_shape[2], q_shape[-1]
    D_t = D // TILE_DIM
    S_q_tiles = S_q // TILE_DIM
    S_kv_tiles = S_kv // TILE_DIM
    B_q_t = min(4, S_q_tiles)
    B_kv_t = min(4, S_kv_tiles)
    B_q_t = min(4, S_q_tiles)
    B_kv_t = min(4, S_kv_tiles)
    tile_size = ttnn.tile_size(query.dtype)
    resolved_scale = scale if scale is not None else (1.0 / math.sqrt(D))
    scale_bits = struct.unpack("I", struct.pack("f", resolved_scale))[0]
    num_work_units = B * H_q
    grid_size = query.device().compute_with_storage_grid_size()
    num_cores, all_cores, _, _, u1, u2 = ttnn.split_work_to_cores(grid_size, num_work_units, row_wise=True)
    has_mask = attn_mask is not None
    num_o_tiles = B_q_t * D_t
    num_score_tiles = B_q_t * B_kv_t

    def cb(idx, pages):
        return ttnn.CBDescriptor(
            total_size=pages * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=idx, data_format=query.dtype, page_size=tile_size)
            ],
        )

    cbs = [
        cb(0, B_q_t * D_t),
        cb(1, 2 * B_kv_t * D_t),
        cb(2, 2 * B_kv_t * D_t),
        cb(3, 2 * num_score_tiles),
        cb(4, 2),
        cb(5, 1),
        cb(8, B_q_t),
        cb(16, num_o_tiles),
        cb(17, num_o_tiles),
        cb(24, num_score_tiles),
        cb(25, num_score_tiles),
        cb(26, B_q_t),
        cb(27, B_q_t),
        cb(28, num_score_tiles),
        cb(29, B_q_t),
        cb(30, B_q_t),
        cb(31, num_o_tiles),
    ]

    reader_ct_args = [1 if has_mask else 0, H_q, H_kv]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(query).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(key).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(value).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(query).get_compile_time_args())

    cores = ttnn.grid_to_cores(num_cores, grid_size.x, grid_size.y, row_wise=True)

    def units_for_core(ci):
        if u2 == 0:
            return u1
        g1c = (num_work_units - num_cores * u2) // (u1 - u2)
        return u1 if ci < g1c else u2

    reader_rt_args = ttnn.RuntimeArgs()
    wu = 0
    for ci, core in enumerate(cores):
        uc = units_for_core(ci)
        rt = [uc, B_q_t, B_kv_t, D_t, S_q_tiles, S_kv_tiles]
        for i in range(uc):
            rt.append((wu + i) // H_q)
            rt.append((wu + i) % H_q)
        wu += uc
        rt.extend(
            [
                query.buffer_address(),
                key.buffer_address(),
                scale_bits,
                value.buffer_address(),
                attn_mask.buffer_address() if has_mask else 0,
            ]
        )
        reader_rt_args[core.x][core.y] = rt

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    writer_ct_args = list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_rt_args = ttnn.RuntimeArgs()
    for ci, core in enumerate(cores):
        uc = units_for_core(ci)
        writer_rt_args[core.x][core.y] = [output_tensor.buffer_address(), uc * S_q_tiles * D_t]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    compute_ct_args = [1 if has_mask else 0, B_q_t, B_kv_t, D_t, S_q_tiles, S_kv_tiles]
    compute_rt_args = ttnn.RuntimeArgs()
    for ci, core in enumerate(cores):
        compute_rt_args[core.x][core.y] = [units_for_core(ci)]
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
