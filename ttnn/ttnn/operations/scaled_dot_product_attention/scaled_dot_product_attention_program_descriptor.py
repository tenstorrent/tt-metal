# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Program Descriptor for scaled_dot_product_attention (Flash Attention).

Defines circular buffers, kernel descriptors, and runtime/compile-time args
for the Flash Attention algorithm:
  - Reader: streams Q, K, V tile blocks (and optional attn_mask) from DRAM
  - Compute: online softmax with running max/sum, QK^T, P@V matmuls
  - Writer: writes output tiles from L1 CB to DRAM
"""

import struct
from pathlib import Path

import ttnn


KERNEL_DIR = Path(__file__).parent / "kernels"

# Block-size parameters (compile-time)
B_q = 4  # Q block size in tiles along S_q (128 rows)
B_kv = 4  # KV block size in tiles along S_kv (128 rows)


def create_program_descriptor(
    query: ttnn.Tensor,
    key: ttnn.Tensor,
    value: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    attn_mask: ttnn.Tensor | None = None,
    scale: float = 1.0,
    compute_kernel_config: ttnn.ComputeConfigDescriptor | None = None,
) -> ttnn.ProgramDescriptor:
    """Create the ProgramDescriptor for Flash Attention.

    Args:
        query: (B, H, S_q, D) bf16 TILE
        key: (B, H, S_kv, D) bf16 TILE
        value: (B, H, S_kv, D) bf16 TILE
        output_tensor: pre-allocated (B, H, S_q, D) bf16 TILE
        attn_mask: optional (B, 1, S_q, S_kv) or (B, H, S_q, S_kv) bf16 TILE
        scale: scale factor for QK^T
        compute_kernel_config: compute kernel precision config
    """
    # ========== 1. EXTRACT TENSOR METADATA ==========
    B = int(query.shape[0])
    H = int(query.shape[1])
    S_q = int(query.shape[2])
    D = int(query.shape[3])
    S_kv = int(key.shape[2])

    S_q_t = S_q // 32  # tiles along S_q
    S_kv_t = S_kv // 32  # tiles along S_kv
    D_t = D // 32  # tiles along D (head dim)

    dtype = query.dtype
    tile_size = ttnn.tile_size(dtype)  # 2048 for bf16

    has_mask = attn_mask is not None
    # Mask shape: (B, H_m, S_q, S_kv) where H_m is 1 or H
    if has_mask:
        mask_H = int(attn_mask.shape[1])
        mask_per_head = mask_H == H
    else:
        mask_per_head = False

    num_q_blocks = (S_q_t + B_q - 1) // B_q
    num_kv_blocks = (S_kv_t + B_kv - 1) // B_kv

    # ========== 2. CORE GRID AND WORK DISTRIBUTION ==========
    num_work_units = B * H  # one (B,H) pair per core
    max_core = ttnn.CoreCoord(7, 7)
    all_cores_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), max_core)])

    (_, core_grid, core_group_1, core_group_2, units_per_core_g1, units_per_core_g2) = ttnn.split_work_to_cores(
        all_cores_crs, num_work_units, row_wise=True
    )

    # Helper to iterate all active cores in order
    def iter_cores():
        """Yield (x, y, work_unit_idx) for each core in the work split."""
        idx = 0
        for cr in core_group_1.ranges():
            for x in range(cr.start.x, cr.end.x + 1):
                for y in range(cr.start.y, cr.end.y + 1):
                    yield x, y, idx
                    idx += 1
        for cr in core_group_2.ranges():
            for x in range(cr.start.x, cr.end.x + 1):
                for y in range(cr.start.y, cr.end.y + 1):
                    yield x, y, idx
                    idx += 1

    # ========== 3. CIRCULAR BUFFER DESCRIPTORS ==========
    # CB indices
    cb_q_id = 0
    cb_k_id = 1
    cb_v_id = 2
    cb_attn_mask_id = 3
    cb_scale_id = 4
    cb_output_id = 16
    cb_scores_id = 24
    cb_m_id = 25
    cb_l_id = 26
    cb_o_id = 27
    cb_m_new_id = 28
    cb_psum_id = 29
    cb_pv_id = 30
    cb_scaler_id = 31

    # CB page counts
    q_pages = 2 * (B_q * D_t)
    k_pages = 2 * (B_kv * D_t)
    v_pages = 2 * (B_kv * D_t)
    mask_pages = 2 * (B_q * B_kv) if has_mask else 1
    scale_pages = 1
    output_pages = 2 * (B_q * D_t)
    scores_pages = B_q * B_kv
    m_pages = max(2, B_q)
    l_pages = max(2, B_q)
    o_pages = max(2, B_q * D_t)
    m_new_pages = max(2, B_q)
    psum_pages = max(2, B_q)
    pv_pages = max(2, B_q * B_kv)
    scaler_pages = 2

    cbs = []

    def make_cb(cb_id, num_pages):
        return ttnn.CBDescriptor(
            total_size=num_pages * tile_size,
            core_ranges=core_grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=dtype, page_size=tile_size)],
        )

    cbs.append(make_cb(cb_q_id, q_pages))
    cbs.append(make_cb(cb_k_id, k_pages))
    cbs.append(make_cb(cb_v_id, v_pages))
    cbs.append(make_cb(cb_attn_mask_id, mask_pages))
    cbs.append(make_cb(cb_scale_id, scale_pages))
    cbs.append(make_cb(cb_output_id, output_pages))
    cbs.append(make_cb(cb_scores_id, scores_pages))
    cbs.append(make_cb(cb_m_id, m_pages))
    cbs.append(make_cb(cb_l_id, l_pages))
    cbs.append(make_cb(cb_o_id, o_pages))
    cbs.append(make_cb(cb_m_new_id, m_new_pages))
    cbs.append(make_cb(cb_psum_id, psum_pages))
    cbs.append(make_cb(cb_pv_id, pv_pages))
    cbs.append(make_cb(cb_scaler_id, scaler_pages))

    # ========== 4. KERNEL DESCRIPTORS ==========
    if compute_kernel_config is None:
        compute_kernel_config = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            math_approx_mode=False,
        )

    # --- Reader kernel ---
    # CT args: scalar params first, then TensorAccessorArgs for Q, K, V, mask
    reader_ct_args = [
        B,
        H,
        S_q_t,
        S_kv_t,
        D_t,
        num_q_blocks,
        num_kv_blocks,
        B_q,
        B_kv,
        1 if has_mask else 0,
        1 if mask_per_head else 0,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(query).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(key).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(value).get_compile_time_args())
    if has_mask:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(attn_mask).get_compile_time_args())
    else:
        reader_ct_args.extend(ttnn.TensorAccessorArgs().get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    for x, y, bh in iter_cores():
        b_idx = bh // H
        h_idx = bh % H
        reader_rt_args[x][y] = [
            query.buffer_address(),
            key.buffer_address(),
            value.buffer_address(),
            attn_mask.buffer_address() if has_mask else 0,
            b_idx,
            h_idx,
        ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Writer kernel ---
    writer_ct_args = [
        S_q_t,
        D_t,
        H,
        B_q,
        num_q_blocks,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    for x, y, bh in iter_cores():
        b_idx = bh // H
        h_idx = bh % H
        # Output tile layout: (B, H, S_q_t, D_t)
        # start_tile_id = b_idx * H * S_q_t * D_t + h_idx * S_q_t * D_t
        start_tile_id = b_idx * H * S_q_t * D_t + h_idx * S_q_t * D_t
        writer_rt_args[x][y] = [
            output_tensor.buffer_address(),
            b_idx,
            h_idx,
            start_tile_id,
        ]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # --- Compute kernel ---
    scale_bits = struct.unpack("<I", struct.pack("<f", float(scale)))[0]

    compute_ct_args = [
        B_q,
        B_kv,
        D_t,
        S_q_t,
        S_kv_t,
        num_q_blocks,
        num_kv_blocks,
        1 if has_mask else 0,
        scale_bits,
    ]

    compute_rt_args = ttnn.RuntimeArgs()
    for x, y, _ in iter_cores():
        compute_rt_args[x][y] = []

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_compute.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=compute_kernel_config,
    )

    # ========== 5. RETURN PROGRAM DESCRIPTOR ==========
    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
