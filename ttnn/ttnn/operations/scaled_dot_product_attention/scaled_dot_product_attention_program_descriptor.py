# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Program descriptor for scaled_dot_product_attention (Flash Attention)."""
from pathlib import Path
import math, struct, ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE_DIM = 32
MAX_B_Q_T = 4
MAX_B_KV_T = 4


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
    # Ceiling division for tile counts: non-aligned dims (e.g. D=50, S_q=47)
    # round UP to the next tile. The tile padding (zeros from from_torch or
    # -inf mask from _make_padding_mask) is handled at the op level.
    D_t = (D + TILE_DIM - 1) // TILE_DIM
    S_q_tiles = (S_q + TILE_DIM - 1) // TILE_DIM
    S_kv_tiles = (S_kv + TILE_DIM - 1) // TILE_DIM
    B_q_t = min(MAX_B_Q_T, S_q_tiles)
    B_kv_t = min(MAX_B_KV_T, S_kv_tiles)
    # Ensure B_q_t divides S_q_tiles so every Q-block is full (no partial last block).
    # The kernel uses B_q_t as a compile-time constant for CB sizing and loop bounds,
    # so a partial last block would read/write out-of-bounds tiles. Reduce B_q_t to
    # the largest divisor of S_q_tiles that is <= MAX_B_Q_T.
    while S_q_tiles % B_q_t != 0 and B_q_t > 1:
        B_q_t -= 1
    while S_kv_tiles % B_kv_t != 0 and B_kv_t > 1:
        B_kv_t -= 1
    tile_size = ttnn.tile_size(query.dtype)
    fp32_tile_size = ttnn.tile_size(ttnn.float32)
    bf16_tile_size = ttnn.tile_size(ttnn.bfloat16)
    resolved_scale = scale if scale is not None else (1.0 / math.sqrt(D))
    scale_bits = struct.unpack("I", struct.pack("f", resolved_scale))[0]
    num_work_units = B * H_q
    grid_size = query.device().compute_with_storage_grid_size()
    num_cores, all_cores, _, _, u1, u2 = ttnn.split_work_to_cores(grid_size, num_work_units, row_wise=True)
    has_mask = attn_mask is not None
    mask_is_per_head = has_mask and (attn_mask.shape[1] == H_q)
    num_q_blocks = (S_q_tiles + B_q_t - 1) // B_q_t
    num_kv_blocks = (S_kv_tiles + B_kv_t - 1) // B_kv_t
    num_o_tiles = B_q_t * D_t
    num_score_tiles = B_q_t * B_kv_t

    # Intermediate CB format: fp32 when fp32_dest_acc_en=True (accumulation crosses
    # the CB between phases — running max/sum/output are parked there), Float16_b
    # (bf16) when fp32_dest_acc_en=False.  We use bf16 — NOT the input dtype —
    # because bf8b is a block-float *storage* format whose shared-exponent grid
    # cannot represent the dynamic range of accumulated values (scores, running
    # max/sum, O).  bf16 is the 16-bit DEST-register format; matching it keeps
    # the pack→unpack round-trip lossless at the phase boundary.  See
    # /numeric-formats-metal §4.
    if fp32_dest_acc_en:
        interm_dtype = ttnn.float32
        interm_tile_size = fp32_tile_size
    else:
        interm_dtype = ttnn.bfloat16
        interm_tile_size = bf16_tile_size

    def cb(idx, pages):
        return ttnn.CBDescriptor(
            total_size=pages * tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=idx, data_format=query.dtype, page_size=tile_size)
            ],
        )

    def cb_bf16(idx, pages):
        return ttnn.CBDescriptor(
            total_size=pages * bf16_tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.bfloat16, page_size=bf16_tile_size)
            ],
        )

    def cb_interm(idx, pages):
        return ttnn.CBDescriptor(
            total_size=pages * interm_tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=idx, data_format=interm_dtype, page_size=interm_tile_size)
            ],
        )

    cbs = [
        cb(0, num_o_tiles),  # cb_q (input dtype)
        cb(1, 2 * B_kv_t * D_t),  # cb_k (input dtype)
        cb(2, 2 * B_kv_t * D_t),  # cb_v (input dtype)
        cb(3, 2 * num_score_tiles),  # cb_mask (input dtype)
        # Constants: always bf16. The reader fills scale_factor with bf16 bits,
        # and calculate_and_prepare_reduce_scaler fills scalers in the CB's format.
        # The FPU reads these through srcA/srcB (TF32) — bf16 is lossless for 1.0
        # and sufficient precision for the scale value.
        cb_bf16(6, 1),  # cb_scaler_max (bf16 — constant 1.0)
        cb_bf16(7, 1),  # cb_scaler_sum (bf16 — constant 1.0)
        cb_bf16(5, 1),  # cb_scale_factor (bf16 — reader writes bf16 bits)
        # Intermediates: fp32 when fp32_dest_acc_en=True, input dtype when False.
        cb_interm(8, B_q_t),  # cb_alpha (running state)
        cb_interm(16, num_o_tiles),  # cb_o (running accumulator)
        cb(17, 2 * num_o_tiles),  # cb_out (input dtype, double-buffered)
        cb_interm(24, num_score_tiles),  # cb_scores
        cb_interm(25, num_score_tiles),  # cb_scores_masked
        cb_interm(26, B_q_t),  # cb_max_new (running state)
        cb_interm(27, B_q_t),  # cb_max_old (running state)
        cb_interm(28, num_score_tiles),  # cb_exp_scores
        cb_interm(29, B_q_t),  # cb_sum_new (running state)
        cb_interm(30, B_q_t),  # cb_sum_old (running state)
        cb_interm(31, num_o_tiles),  # cb_o_accum (PV matmul scratch)
    ]

    # Reader CT args: [has_mask, H_q, H_kv, mask_is_per_head, ...Q_acc, ...K_acc, ...V_acc, ...mask_acc]
    reader_ct_args = [1 if has_mask else 0, H_q, H_kv, 1 if mask_is_per_head else 0]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(query).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(key).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(value).get_compile_time_args())
    if has_mask:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(attn_mask).get_compile_time_args())
    else:
        reader_ct_args.extend(ttnn.TensorAccessorArgs().get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    cores = ttnn.grid_to_cores(num_cores, grid_size.x, grid_size.y, row_wise=True)
    wu_assigned = 0
    for ci, core in enumerate(cores):
        if u2 == 0:
            units = u1
        else:
            g1c = (num_work_units - num_cores * u2) // (u1 - u2)
            units = u1 if ci < g1c else u2
        rt = [units, B_q_t, B_kv_t, D_t, S_q_tiles, S_kv_tiles]
        for i in range(units):
            bh = wu_assigned + i
            rt.append(bh // H_q)
            rt.append(bh % H_q)
        wu_assigned += units
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

    # Writer CT args: [B_q_t, D_t, num_q_blocks, ...output_acc]
    writer_ct_args = [B_q_t, D_t, num_q_blocks]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    wu_assigned = 0
    for ci, core in enumerate(cores):
        if u2 == 0:
            units = u1
        else:
            g1c = (num_work_units - num_cores * u2) // (u1 - u2)
            units = u1 if ci < g1c else u2
        rt = [output_tensor.buffer_address(), units, S_q_tiles, H_q]
        for i in range(units):
            bh = wu_assigned + i
            rt.append(bh // H_q)
            rt.append(bh % H_q)
        wu_assigned += units
        writer_rt_args[core.x][core.y] = rt

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # Compute CT args: [B_q_t, B_kv_t, D_t, has_mask, num_q_blocks, num_kv_blocks]
    compute_ct_args = [B_q_t, B_kv_t, D_t, 1 if has_mask else 0, num_q_blocks, num_kv_blocks]
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
