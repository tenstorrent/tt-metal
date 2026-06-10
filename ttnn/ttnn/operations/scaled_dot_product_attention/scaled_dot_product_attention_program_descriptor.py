# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Program descriptor for scaled_dot_product_attention (Flash Attention).

Work unit = one (b, h, q_chunk): c_q tile-rows of Q against all of K/V.
Each core independently owns a contiguous range of flattened
(b*H + h)*Nq + q_chunk indices. Per-block CBs are sized c_q x c_kv tiles,
independent of S_kv (the load-bearing Flash Attention constraint).
"""

import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE = 32

# --- CB indices (semantic names; numeric slot ranges: 0-7 in, 8-15 special,
# 16-23 out, 24-31 intermediate) ---
CB_Q_TILES = 0
CB_KT_TILES = 1
CB_V_TILES = 2
CB_MASK_TILES = 3
CB_SCALER_MAX = 8
CB_SCALER_SUM = 9
CB_CUR_SUM = 10
CB_PREV_MAX = 11
CB_RUNNING_MAX = 12
CB_ALPHA = 13
CB_RUNNING_SUM = 14
CB_INV_SUM = 15
CB_OUT_TILES = 16
CB_SCORES = 24
CB_SCORES_SCALED = 25
CB_PROBS = 26
CB_PV = 27
CB_O_ACC = 28


def _float_bits(value: float) -> int:
    return struct.unpack("I", struct.pack("f", value))[0]


def _flatten_cores(grid_size):
    return [ttnn.CoreCoord(x, y) for y in range(grid_size.y) for x in range(grid_size.x)]


def create_program_descriptor(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    mask: ttnn.Tensor,
    output: ttnn.Tensor,
    *,
    scale: float,
    compute_kernel_config,
) -> ttnn.ProgramDescriptor:
    B, H, S_q, D = list(q.shape)
    _, H_kv, S_kv, _ = list(k.shape)

    Dt = D // TILE
    Sq_t = S_q // TILE
    Skv_t = S_kv // TILE

    # Chunk size: c = clamp(16 / Dt, 1, 4)
    c = max(1, min(4, 16 // Dt))
    c_q = min(c, Sq_t)
    c_kv = min(c, Skv_t)
    Nq = -(-Sq_t // c_q)
    Nkv = -(-Skv_t // c_kv)
    c_q_last = Sq_t - (Nq - 1) * c_q
    c_kv_last = Skv_t - (Nkv - 1) * c_kv

    # P@V subblock width: largest divisor of Dt that is <= 4 (fp32 DEST limit).
    sw = 1
    for cand in (4, 3, 2, 1):
        if Dt % cand == 0:
            sw = cand
            break
    n_sub_w = Dt // sw

    has_mask = mask is not None
    mask_is_per_head = has_mask and list(mask.shape)[1] == H

    # Input/output CB formats follow the input dtype (validate() enforces
    # Q/K/V/mask dtype equality, and output dtype == query dtype). Stat /
    # accumulator intermediates AND cb_probs follow fp32_dest_acc_en: Float32
    # when the fp32 DEST accumulation crosses those CBs (default), Float16_b
    # otherwise — packing a 16-bit DEST into Float32 CBs corrupts values
    # (probe_007: pcc ~0.41 at every fidelity). cb_probs at input dtype was the
    # Refinement 3 long-context precision miss: quantized probs made rowsum l
    # and P@V inconsistent (probe_012). Reduce-scaler tiles stay bfloat16.
    in_fmt = q.dtype
    t_in = ttnn.tile_size(in_fmt)
    t_bf = ttnn.tile_size(ttnn.bfloat16)
    acc_fmt = ttnn.float32 if compute_kernel_config.fp32_dest_acc_en else ttnn.bfloat16
    t_acc = ttnn.tile_size(acc_fmt)

    # --- Work distribution over flattened (b, h, q_chunk) units ---
    total_units = B * H * Nq
    grid_size = q.device().compute_with_storage_grid_size()
    cores = _flatten_cores(grid_size)
    num_cores = min(len(cores), total_units)
    base = total_units // num_cores
    rem = total_units % num_cores

    full_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))]
    )

    # --- Circular buffers ---
    def cb(index, pages, page_size, fmt):
        return ttnn.CBDescriptor(
            total_size=pages * page_size,
            core_ranges=full_grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page_size)],
        )

    cbs = [
        cb(CB_Q_TILES, c_q * Dt, t_in, in_fmt),
        cb(CB_KT_TILES, 2 * c_kv * Dt, t_in, in_fmt),
        cb(CB_V_TILES, 2 * c_kv * Dt, t_in, in_fmt),
        cb(CB_SCALER_MAX, 1, t_bf, ttnn.bfloat16),
        cb(CB_SCALER_SUM, 1, t_bf, ttnn.bfloat16),
        cb(CB_CUR_SUM, c_q, t_acc, acc_fmt),
        cb(CB_PREV_MAX, c_q, t_acc, acc_fmt),
        cb(CB_RUNNING_MAX, 2 * c_q, t_acc, acc_fmt),
        cb(CB_ALPHA, c_q, t_acc, acc_fmt),
        cb(CB_RUNNING_SUM, 2 * c_q, t_acc, acc_fmt),
        cb(CB_INV_SUM, c_q, t_acc, acc_fmt),
        cb(CB_OUT_TILES, 2 * c_q * Dt, t_in, in_fmt),
        cb(CB_SCORES, c_q * c_kv, t_acc, acc_fmt),
        cb(CB_SCORES_SCALED, c_q * c_kv, t_acc, acc_fmt),
        cb(CB_PROBS, c_q * c_kv, t_acc, acc_fmt),
        cb(CB_PV, c_q * Dt, t_acc, acc_fmt),
        cb(CB_O_ACC, 2 * c_q * Dt, t_acc, acc_fmt),
    ]
    if has_mask:
        cbs.append(cb(CB_MASK_TILES, 2 * c_q * c_kv, t_in, in_fmt))

    # --- Reader ---
    reader_ct_args = [
        H,
        Sq_t,
        Skv_t,
        Dt,
        c_q,
        c_kv,
        Nq,
        Nkv,
        c_q_last,
        c_kv_last,
        1 if has_mask else 0,
        1 if mask_is_per_head else 0,
        H_kv,  # GQA/MQA: Q head h -> KV head h / (H // H_kv); == H for MHA
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(q).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(k).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(v).get_compile_time_args())
    reader_ct_args.extend(
        ttnn.TensorAccessorArgs(mask).get_compile_time_args()
        if has_mask
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    # --- Writer ---
    writer_ct_args = [
        H,
        Sq_t,
        Dt,
        c_q,
        Nq,
        c_q_last,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output).get_compile_time_args())

    # --- Compute ---
    compute_ct_args = [
        Dt,
        c_q,
        c_kv,
        Nq,
        Nkv,
        c_q_last,
        c_kv_last,
        sw,
        n_sub_w,
        1 if has_mask else 0,
    ]

    scale_bits = _float_bits(scale)

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    start = 0
    for i, core in enumerate(cores):
        count = (base + (1 if i < rem else 0)) if i < num_cores else 0
        reader_rt[core.x][core.y] = [
            q.buffer_address(),
            k.buffer_address(),
            v.buffer_address(),
            mask.buffer_address() if has_mask else 0,
            start,
            count,
        ]
        writer_rt[core.x][core.y] = [output.buffer_address(), start, count]
        compute_rt[core.x][core.y] = [start, count, scale_bits]
        start += count

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_reader.cpp"),
        core_ranges=full_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_writer.cpp"),
        core_ranges=full_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_compute.cpp"),
        core_ranges=full_grid,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=compute_kernel_config.math_fidelity,
            fp32_dest_acc_en=compute_kernel_config.fp32_dest_acc_en,
            math_approx_mode=compute_kernel_config.math_approx_mode,
            dst_full_sync_en=compute_kernel_config.dst_full_sync_en,
        ),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
