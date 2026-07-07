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
B_q = 1  # Q block size in tiles along S_q (process 1 tile at a time)
B_kv = 1  # KV block size in tiles along S_kv (process 1 tile at a time)


def create_program_descriptor(
    query: ttnn.Tensor,
    key: ttnn.Tensor,
    value: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    attn_mask: ttnn.Tensor | None = None,
    is_causal: bool = False,
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
        is_causal: if True, generate causal mask on-device
        scale: scale factor for QK^T
        compute_kernel_config: compute kernel precision config
    """
    # ========== 1. EXTRACT TENSOR METADATA ==========
    B = int(query.shape[0])
    H = int(query.shape[1])  # H_q (Q num heads)
    S_q = int(query.shape[2])
    D = int(query.shape[3])
    S_kv = int(key.shape[2])
    H_kv = int(key.shape[1])  # K/V num heads (GQA/MQA: H_kv <= H_q)

    # Use ceildiv for tile counts — non-aligned dimensions (S % 32 != 0 or
    # D % 32 != 0) still occupy a full tile in TILE_LAYOUT (zero-padded by
    # ttnn.from_torch). The kernel must process these partially-filled tiles.
    S_q_t = (S_q + 31) // 32  # tiles along S_q (ceildiv)
    S_kv_t = (S_kv + 31) // 32  # tiles along S_kv (ceildiv)
    D_t = (D + 31) // 32  # tiles along D (head dim, ceildiv)

    dtype = query.dtype
    tile_size = ttnn.tile_size(dtype)

    # ========== 1b. COMPUTE CONFIG DEFAULTS ==========
    # Ensure compute_kernel_config is resolved early.
    if compute_kernel_config is None:
        compute_kernel_config = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            math_approx_mode=False,
        )

    # Intermediate accumulator CBs must be Float32 when fp32_dest_acc_en=True
    # to preserve precision across phase boundaries (online softmax running
    # max/sum/output accumulation). When fp32_dest_acc_en=False, intermediates
    # use Float16_b (bf16) — the best 16-bit format that avoids block-float
    # shared-exponent precision loss in the running max/sum/output accumulators.
    fp32_dest_acc_en = getattr(compute_kernel_config, "fp32_dest_acc_en", True)
    if fp32_dest_acc_en:
        intermediate_format = ttnn.float32
    else:
        intermediate_format = ttnn.bfloat16  # bf16, not block-float
    intermediate_tile_size = ttnn.tile_size(intermediate_format)

    has_mask = attn_mask is not None
    is_causal_mask = is_causal  # causal mask is generated on-device, no caller tensor
    # Mask shape: (B, H_m, S_q, S_kv) where H_m is 1 or H
    if has_mask:
        mask_H = int(attn_mask.shape[1])
        mask_per_head = mask_H == H
    else:
        mask_per_head = False

    # Non-aligned S_kv: the last KV tile contains zero-padded rows (positions
    # S_kv..32*S_kv_t-1). These produce Q·0=0 scores, and exp(0)=1 in softmax,
    # which corrupts the normalization. We must mask them out with -1e9.
    # Non-aligned S_q: padded Q rows produce garbage output, but the writer
    # only writes S_q_t*32 >= S_q tiles — the padded rows are never written
    # back. Non-aligned D: padded D columns are zeros in K/V, so Q·0=0 in the
    # dot product — this doesn't corrupt the result (just reduces magnitude
    # uniformly, which softmax normalization handles). So only S_kv padding
    # needs masking.
    has_kv_padding = S_kv % 32 != 0
    S_kv_padded = S_kv  # actual KV length (for padding mask generation)
    # When padding is needed and no mask is provided, we force a padding-only
    # mask via the on-device generation path (like causal).
    needs_padding_mask = has_kv_padding and not has_mask

    num_q_blocks = (S_q_t + B_q - 1) // B_q
    num_kv_blocks = (S_kv_t + B_kv - 1) // B_kv

    # Adjust block sizes to not exceed actual tile counts
    # This ensures reader/compute/writer all agree on tile counts per block
    actual_B_q = min(B_q, S_q_t)
    actual_B_kv = min(B_kv, S_kv_t)

    # ========== 2. CORE GRID AND WORK DISTRIBUTION ==========
    num_work_units = B * H  # one (B,H) pair per work unit
    # Use 8×7 grid (56 worker cores). The 8th row (y=7) contains dispatch
    # cores that cannot host user kernels.
    max_core = ttnn.CoreCoord(7, 6)
    all_cores_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), max_core)])

    (_, core_grid, core_group_1, core_group_2, units_per_core_g1, units_per_core_g2) = ttnn.split_work_to_cores(
        all_cores_crs, num_work_units, row_wise=True
    )

    # Build per-core work unit lists. When num_work_units > num_cores, some
    # cores get multiple (B,H) pairs — the kernel loops over them.
    # core_to_work_units: { (x,y) : [(b_idx, h_idx, start_tile_id), ...] }
    core_to_work_units = {}
    work_idx = 0
    for cr in core_group_1.ranges():
        for x in range(cr.start.x, cr.end.x + 1):
            for y in range(cr.start.y, cr.end.y + 1):
                for _ in range(units_per_core_g1):
                    b_idx = work_idx // H
                    h_idx = work_idx % H
                    start_tile_id = b_idx * H * S_q_t * D_t + h_idx * S_q_t * D_t
                    core_to_work_units.setdefault((x, y), []).append((b_idx, h_idx, start_tile_id))
                    work_idx += 1
    for cr in core_group_2.ranges():
        for x in range(cr.start.x, cr.end.x + 1):
            for y in range(cr.start.y, cr.end.y + 1):
                for _ in range(units_per_core_g2):
                    b_idx = work_idx // H
                    h_idx = work_idx % H
                    start_tile_id = b_idx * H * S_q_t * D_t + h_idx * S_q_t * D_t
                    core_to_work_units.setdefault((x, y), []).append((b_idx, h_idx, start_tile_id))
                    work_idx += 1

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
    cb_pv_out_id = 23  # PV matmul output (separate from cb_scores)
    cb_scaler_id = 31

    # CB page counts — use actual block sizes
    q_pages = 2 * (actual_B_q * D_t)
    k_pages = 2 * (actual_B_kv * D_t)
    v_pages = 2 * (actual_B_kv * D_t)
    # Mask CB: for custom mask, reader streams from DRAM (double-buffered).
    # For causal mask, reader generates 1 tile per diagonal KV block (B_q*B_kv=1).
    # For padding mask (non-aligned S_kv, no caller mask), reader generates
    # padding mask tiles on-device.
    needs_mask_cb = has_mask or is_causal_mask or needs_padding_mask
    mask_pages = 2 * (actual_B_q * actual_B_kv) if needs_mask_cb else 1
    scale_pages = 1
    output_pages = 2 * (actual_B_q * D_t)
    # cb_scores holds QK^T scores (B_q×B_kv tiles, Phase 1) — used for scale,
    # mask, rowmax, exp, copyP, and rowsum. All these phases push/pop 1 tile
    # at a time (B_q*B_kv tiles total).
    # cb_pv_out holds the PV matmul output (B_q×D_t tiles, Phase 12) — consumed
    # by Phase 13 (O_i += PV). Separated from cb_scores to avoid CB write-pointer
    # alignment issues from mixed 1-tile and multi-tile push patterns.
    scores_pages = max(2, actual_B_q * actual_B_kv)
    pv_out_pages = max(2, actual_B_q * D_t)
    m_pages = max(2, actual_B_q)
    l_pages = max(2, actual_B_q)
    o_pages = max(2, actual_B_q * D_t)
    m_new_pages = max(2, actual_B_q)
    psum_pages = max(2, actual_B_q)
    pv_pages = max(2, actual_B_q * actual_B_kv)
    # cb_scaler: 2 tiles per KV block (MAX + SUM). The reader runs ahead of compute
    # (limited by Q/K/V CB double-buffering to ~2 KV blocks lookahead), so the scaler
    # CB must hold enough for the reader's full lookahead. Use 2 * num_kv_blocks to
    # be safe (the reader pushes all KV blocks' scalers if compute is slow).
    scaler_pages = 2 * num_kv_blocks

    cbs = []

    def make_cb(cb_id, num_pages, cb_dtype, cb_tile_size):
        return ttnn.CBDescriptor(
            total_size=num_pages * cb_tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=cb_dtype, page_size=cb_tile_size)
            ],
        )

    # The reduce scaler CB must use Float16_b or Float32 (prepare_reduce_scaler
    # does not support block-float formats). Always use Float16_b — the scaler
    # is just a constant tile, not an accumulator, so Float32 precision is not
    # needed. Using bf16 keeps the scaler CB small (2KB/tile vs 4KB for Float32)
    # which is critical for long sequences (S=8192 → 512 scaler tiles).
    scaler_format = ttnn.bfloat16
    scaler_tile_size = ttnn.tile_size(scaler_format)

    # Input/output CBs follow the input tensor's dtype
    cbs.append(make_cb(cb_q_id, q_pages, dtype, tile_size))
    cbs.append(make_cb(cb_k_id, k_pages, dtype, tile_size))
    cbs.append(make_cb(cb_v_id, v_pages, dtype, tile_size))
    # Causal mask CB uses intermediate format (generated on-device in the reader
    # kernel via direct L1 writes). For custom mask, it follows the input dtype
    # (streamed from DRAM). Using intermediate format for causal ensures the
    # mask values (-1e38f) match the score accumulation precision.
    # Padding mask (non-aligned S_kv, no caller mask) also uses intermediate
    # format — it's generated on-device like causal.
    mask_cb_format = intermediate_format if (is_causal_mask or needs_padding_mask) else dtype
    mask_cb_tile_size = ttnn.tile_size(mask_cb_format)
    cbs.append(make_cb(cb_attn_mask_id, mask_pages, mask_cb_format, mask_cb_tile_size))
    cbs.append(make_cb(cb_scale_id, scale_pages, dtype, tile_size))
    cbs.append(make_cb(cb_output_id, output_pages, dtype, tile_size))
    # Intermediate accumulator CBs use Float32 when fp32_dest_acc_en=True
    cbs.append(make_cb(cb_scores_id, scores_pages, intermediate_format, intermediate_tile_size))
    cbs.append(make_cb(cb_m_id, m_pages, intermediate_format, intermediate_tile_size))
    cbs.append(make_cb(cb_l_id, l_pages, intermediate_format, intermediate_tile_size))
    cbs.append(make_cb(cb_o_id, o_pages, intermediate_format, intermediate_tile_size))
    cbs.append(make_cb(cb_m_new_id, m_new_pages, intermediate_format, intermediate_tile_size))
    cbs.append(make_cb(cb_psum_id, psum_pages, intermediate_format, intermediate_tile_size))
    cbs.append(make_cb(cb_pv_id, pv_pages, intermediate_format, intermediate_tile_size))
    cbs.append(make_cb(cb_pv_out_id, pv_out_pages, intermediate_format, intermediate_tile_size))
    cbs.append(make_cb(cb_scaler_id, scaler_pages, scaler_format, scaler_tile_size))

    # ========== 4. KERNEL DESCRIPTORS ==========
    # (compute_kernel_config already resolved above)

    # --- Reader kernel ---
    # CT args: scalar params first, then TensorAccessorArgs for Q, K, V, mask
    reader_ct_args = [
        B,
        H,  # [1] H_q (Q num heads)
        S_q_t,
        S_kv_t,
        D_t,
        num_q_blocks,
        num_kv_blocks,
        actual_B_q,
        actual_B_kv,
        1 if has_mask else 0,
        1 if mask_per_head else 0,
        H_kv,  # [11] K/V num heads (GQA/MQA broadcasting)
        1 if is_causal_mask else 0,  # [12] is_causal
        1 if has_kv_padding else 0,  # [13] has_kv_padding
        S_kv_padded,  # [14] actual S_kv (for padding mask generation)
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(query).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(key).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(value).get_compile_time_args())
    if has_mask:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(attn_mask).get_compile_time_args())
    else:
        reader_ct_args.extend(ttnn.TensorAccessorArgs().get_compile_time_args())

    # Reader RT args: flatten all work units for each core into a single list.
    # Format per core: [num_work_units, addr_q, addr_k, addr_v, addr_mask,
    #                   b0, h0, b1, h1, ...]
    reader_rt_args = ttnn.RuntimeArgs()
    for (x, y), work_units in core_to_work_units.items():
        rt = [
            len(work_units),
            query.buffer_address(),
            key.buffer_address(),
            value.buffer_address(),
            attn_mask.buffer_address() if has_mask else 0,
        ]
        for b_idx, h_idx, _ in work_units:
            rt.extend([b_idx, h_idx])
        reader_rt_args[x][y] = rt

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
        actual_B_q,
        num_q_blocks,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # Writer RT args: flatten all work units for each core.
    # Format per core: [num_work_units, output_addr, start_tile_0, start_tile_1, ...]
    writer_rt_args = ttnn.RuntimeArgs()
    for (x, y), work_units in core_to_work_units.items():
        rt = [len(work_units), output_tensor.buffer_address()]
        for _, _, start_tile_id in work_units:
            rt.append(start_tile_id)
        writer_rt_args[x][y] = rt

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
        actual_B_q,
        actual_B_kv,
        D_t,
        S_q_t,
        S_kv_t,
        num_q_blocks,
        num_kv_blocks,
        1 if has_mask else 0,
        scale_bits,
        1 if is_causal_mask else 0,  # [9] is_causal
        1 if has_kv_padding else 0,  # [10] has_kv_padding
    ]

    compute_rt_args = ttnn.RuntimeArgs()
    for (x, y), work_units in core_to_work_units.items():
        compute_rt_args[x][y] = [len(work_units)]

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
