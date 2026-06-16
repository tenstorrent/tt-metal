# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for the FlashAttention scaled_dot_product_attention op.

Work unit = one (b, h, q-chunk) with Bq_t = Bkv_t = 1 tile. Each unit streams
all Skv_t KV-chunks once and folds them into running flash-attention stats.
Units are split contiguously across the compute grid.
"""

import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE_DIM = 32


def _f32_bits(x: float) -> int:
    return struct.unpack("I", struct.pack("f", float(x)))[0]


def create_program_descriptor(
    query: ttnn.Tensor,
    key: ttnn.Tensor,
    value: ttnn.Tensor,
    attn_mask,
    output: ttnn.Tensor,
    *,
    scale: float,
    fp32_dest_acc_en: bool,
    math_fidelity: "ttnn.MathFidelity",
) -> ttnn.ProgramDescriptor:
    q_shape = list(query.shape)
    k_shape = list(key.shape)

    B, H_q, S_q, D = q_shape
    H_kv, S_kv = k_shape[1], k_shape[2]

    Sq_t = S_q // TILE_DIM
    Skv_t = S_kv // TILE_DIM
    d_t = D // TILE_DIM
    group = H_q // H_kv

    has_mask = 1 if attn_mask is not None else 0
    mask_B = int(attn_mask.shape[0]) if attn_mask is not None else 0
    mask_H = int(attn_mask.shape[1]) if attn_mask is not None else 0

    total_units = B * H_q * Sq_t

    tile_size = ttnn.tile_size(ttnn.bfloat16)
    q_page = query.buffer_page_size()
    out_page = output.buffer_page_size()

    scale_bits = _f32_bits(scale)

    # --- Work distribution: contiguous unit range per core ---
    device = query.device()
    grid = device.compute_with_storage_grid_size()  # CoreCoord(x, y)
    num_cores, all_cores, group_1, group_2, units_1, units_2 = ttnn.split_work_to_cores(grid, total_units)

    cores_1 = ttnn.corerange_to_cores(group_1, None, False)
    cores_2 = ttnn.corerange_to_cores(group_2, None, False) if units_2 > 0 else []

    # Ordered (core, num_units) assignment with running start offsets.
    assignments = []
    start = 0
    for c in cores_1:
        assignments.append((c, start, units_1))
        start += units_1
    for c in cores_2:
        assignments.append((c, start, units_2))
        start += units_2

    # --- Circular buffers ---
    CB_Q_IN, CB_K_IN, CB_V_IN, CB_MASK_IN = 0, 1, 2, 3
    CB_SCALER_MAX, CB_SCALER_SUM = 8, 9
    CB_P, CB_O, CB_PV, CB_O_RESC, CB_RECIP_L = 10, 11, 12, 13, 14
    CB_OUT = 16
    CB_Q, CB_SCORES, CB_M_CUR, CB_M, CB_M_NEW, CB_L, CB_L_CUR, CB_CORR = 24, 25, 26, 27, 28, 29, 30, 31

    def cb(index, num_pages, page_size=tile_size, data_format=ttnn.bfloat16):
        return ttnn.CBDescriptor(
            total_size=num_pages * page_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)
            ],
        )

    cbs = [
        cb(CB_Q_IN, 2 * d_t, q_page, query.dtype),
        cb(CB_K_IN, 2 * d_t, q_page, key.dtype),
        cb(CB_V_IN, 2 * d_t, q_page, value.dtype),
        cb(CB_SCALER_MAX, 1),
        cb(CB_SCALER_SUM, 1),
        cb(CB_P, 1),
        cb(CB_O, d_t),
        cb(CB_PV, d_t),
        cb(CB_O_RESC, d_t),
        cb(CB_RECIP_L, 1),
        cb(CB_OUT, 2 * d_t, out_page, output.dtype),
        cb(CB_Q, d_t),
        cb(CB_SCORES, 1),
        cb(CB_M_CUR, 1),
        cb(CB_M, 1),
        cb(CB_M_NEW, 1),
        cb(CB_L, 1),
        cb(CB_L_CUR, 1),
        cb(CB_CORR, 1),
    ]
    if has_mask:
        cbs.append(cb(CB_MASK_IN, 2, tile_size, attn_mask.dtype))

    # --- Reader kernel ---
    reader_ct = [H_q, H_kv, Sq_t, Skv_t, d_t, group, has_mask, mask_H, mask_B]
    reader_ct.extend(ttnn.TensorAccessorArgs(query).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(key).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(value).get_compile_time_args())
    reader_ct.extend(
        ttnn.TensorAccessorArgs(attn_mask).get_compile_time_args()
        if attn_mask is not None
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    q_addr = query.buffer_address()
    k_addr = key.buffer_address()
    v_addr = value.buffer_address()
    mask_addr = attn_mask.buffer_address() if attn_mask is not None else 0
    out_addr = output.buffer_address()

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    for c, start_unit, num_units in assignments:
        reader_rt[c.x][c.y] = [start_unit, num_units, q_addr, k_addr, v_addr, mask_addr]
        writer_rt[c.x][c.y] = [start_unit, num_units, out_addr]
        compute_rt[c.x][c.y] = [num_units]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Writer kernel ---
    writer_ct = [H_q, Sq_t, d_t]
    writer_ct.extend(ttnn.TensorAccessorArgs(output).get_compile_time_args())

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    # --- Compute kernel ---
    compute_ct = [Skv_t, d_t, has_mask, scale_bits]

    compute_config = ttnn.ComputeConfigDescriptor()
    compute_config.math_fidelity = math_fidelity
    compute_config.fp32_dest_acc_en = fp32_dest_acc_en
    compute_config.math_approx_mode = False
    compute_config.dst_full_sync_en = False

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=compute_config,
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
