# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for scaled_dot_product_attention (Flash Attention).

Work unit = one Q-block = (batch, query-head, Q-chunk). The flat work-list
``total_q_blocks = B · H_q · q_num_chunks`` is split across the grid with
``ttnn.split_work_to_cores``; each core streams every KV-block for each Q-block
it owns (online softmax). No cross-core communication in Phase-1.

Block-size knobs (single source of truth):
  * ``Q_CHUNK_TILES`` / ``K_CHUNK_TILES`` — per-core Q/KV chunk sizes (tiles).
    The effective chunk is the largest divisor of Sqt / Skvt that is ≤ the knob
    (keeps every chunk full — no partial-tail chunk in Phase-1). CB page counts
    and loop bounds are all derived from these.
  * ``KV_BUFFER_FACTOR`` / ``Q_BUFFER_FACTOR`` — streaming CB depths.
"""

import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# ---- Block-size / buffer-depth knobs (parameters, never inlined downstream) ----
Q_CHUNK_TILES = 4  # Sq_chunk_t upper bound (tiles of 32 query rows)
K_CHUNK_TILES = 4  # Sk_chunk_t upper bound (tiles of 32 key/value rows)
KV_BUFFER_FACTOR = 2  # double-buffer K/V/mask to overlap DRAM read with compute
Q_BUFFER_FACTOR = 1  # Q held resident across the whole KV loop

# ---- CB indices (semantic) ----
CB_Q_IN = 0
CB_K_IN = 1
CB_V_IN = 2
CB_MASK_IN = 3
CB_SCALER_MAX = 4
CB_SCALER_SUM = 5
CB_OUT = 16
# intermediates
CB_QK_SCORES = 24
CB_MAX_A = 25
CB_MAX_B = 26
CB_MAX_NEW = 27
CB_SUM_A = 28
CB_SUM_B = 29
CB_SUM_NEW = 30
CB_EXP_MAX_DIFF = 31
CB_OUT_A = 6
CB_OUT_B = 7
CB_OUT_NEW = 8
CB_SUM_SCALED = 9
CB_OUT_SCALED = 10


def _largest_divisor_leq(n, cap):
    """Largest d in [1, cap] with n % d == 0."""
    for d in range(min(n, cap), 0, -1):
        if n % d == 0:
            return d
    return 1


def _pick_subblock(m_tiles, n_tiles, dst_limit):
    """Pick (sb_h, sb_w) with sb_h*sb_w <= dst_limit, sb_w | n, sb_h | m.
    Returns (sb_h, sb_w, in0_num_subblocks=m/sb_h, in1_num_subblocks=n/sb_w)."""
    best = (1, 1)
    for w in range(min(n_tiles, dst_limit), 0, -1):
        if n_tiles % w != 0:
            continue
        max_h = dst_limit // w
        for h in range(min(m_tiles, max_h), 0, -1):
            if m_tiles % h == 0:
                best = (h, w)
                return best[0], best[1], m_tiles // best[0], n_tiles // best[1]
    h, w = best
    return h, w, m_tiles // h, n_tiles // w


def _f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def create_program_descriptor(query, key, value, attn_mask, output_tensor, *, scale, compute_kernel_config):
    b, h_q, s_q, d = tuple(query.shape)
    _, h_kv, s_kv, _ = tuple(key.shape)

    sqt = s_q // 32
    skvt = s_kv // 32
    dht = d // 32

    sq_chunk_t = _largest_divisor_leq(sqt, Q_CHUNK_TILES)
    sk_chunk_t = _largest_divisor_leq(skvt, K_CHUNK_TILES)
    q_num_chunks = sqt // sq_chunk_t
    k_num_chunks = skvt // sk_chunk_t

    has_mask = attn_mask is not None
    mask_broadcast_head = 1 if (has_mask and tuple(attn_mask.shape)[1] == 1) else 0

    fp32_dest_acc_en = bool(getattr(compute_kernel_config, "fp32_dest_acc_en", True))
    dst_limit = 4 if fp32_dest_acc_en else 8

    # Subblock decomposition for the two matmuls (block held in DEST, num_k_blocks=1).
    qk_sb_h, qk_sb_w, qk_in0_sb, qk_in1_sb = _pick_subblock(sq_chunk_t, sk_chunk_t, dst_limit)
    pv_sb_h, pv_sb_w, pv_in0_sb, pv_in1_sb = _pick_subblock(sq_chunk_t, dht, dst_limit)

    total_q_blocks = b * h_q * q_num_chunks

    grid = query.device().compute_with_storage_grid_size()
    (
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        units_per_core_g1,
        units_per_core_g2,
    ) = ttnn.split_work_to_cores(grid, total_q_blocks)

    # Per-core flat work slice [q_start, q_start+q_count).
    assignment = []
    start = 0
    for group, per_core in ((core_group_1, units_per_core_g1), (core_group_2, units_per_core_g2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment.append((core, start, per_core))
            start += per_core

    # ---- Circular buffers ----
    tile = ttnn.tile_size(ttnn.bfloat16)
    bf16 = ttnn.bfloat16

    def cb(index, num_pages):
        return ttnn.CBDescriptor(
            total_size=num_pages * tile,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=bf16, page_size=tile)],
        )

    q_tiles = sq_chunk_t * dht
    k_tiles = sk_chunk_t * dht
    qk_tiles = sq_chunk_t * sk_chunk_t
    o_tiles = sq_chunk_t * dht

    cbs = [
        cb(CB_Q_IN, q_tiles * Q_BUFFER_FACTOR),
        cb(CB_K_IN, k_tiles * KV_BUFFER_FACTOR),
        cb(CB_V_IN, k_tiles * KV_BUFFER_FACTOR),
        cb(CB_SCALER_MAX, 1),
        cb(CB_SCALER_SUM, 1),
        cb(CB_OUT, o_tiles * 2),
        cb(CB_QK_SCORES, qk_tiles),
        cb(CB_MAX_A, sq_chunk_t),
        cb(CB_MAX_B, sq_chunk_t),
        cb(CB_MAX_NEW, sq_chunk_t),
        cb(CB_SUM_A, sq_chunk_t),
        cb(CB_SUM_B, sq_chunk_t),
        cb(CB_SUM_NEW, sq_chunk_t),
        cb(CB_SUM_SCALED, sq_chunk_t),
        cb(CB_EXP_MAX_DIFF, sq_chunk_t),
        cb(CB_OUT_A, o_tiles),
        cb(CB_OUT_B, o_tiles),
        cb(CB_OUT_NEW, o_tiles),
        cb(CB_OUT_SCALED, o_tiles),
    ]
    if has_mask:
        cbs.append(cb(CB_MASK_IN, qk_tiles * KV_BUFFER_FACTOR))

    # ---- Reader kernel ----
    reader_ct = [
        b,
        h_q,
        h_kv,
        sqt,
        skvt,
        dht,
        sq_chunk_t,
        sk_chunk_t,
        q_num_chunks,
        k_num_chunks,
        1 if has_mask else 0,
        mask_broadcast_head,
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(query).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(key).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(value).get_compile_time_args())
    reader_ct.extend(
        ttnn.TensorAccessorArgs(attn_mask).get_compile_time_args()
        if has_mask
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    q_addr = query.buffer_address()
    k_addr = key.buffer_address()
    v_addr = value.buffer_address()
    m_addr = attn_mask.buffer_address() if has_mask else 0
    o_addr = output_tensor.buffer_address()
    for core, q_start, q_count in assignment:
        reader_rt[core.x][core.y] = [q_addr, k_addr, v_addr, m_addr, q_start, q_count]
        writer_rt[core.x][core.y] = [o_addr, q_start, q_count]
        compute_rt[core.x][core.y] = [q_count]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ---- Compute kernel ----
    compute_ct = [
        sq_chunk_t,
        sk_chunk_t,
        dht,
        k_num_chunks,
        1 if has_mask else 0,
        _f32_bits(scale),
        qk_in0_sb,
        qk_in1_sb,
        qk_sb_h,
        qk_sb_w,
        pv_in0_sb,
        pv_in1_sb,
        pv_sb_h,
        pv_sb_w,
    ]
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=compute_kernel_config,
    )

    # ---- Writer kernel ----
    writer_ct = [b, h_q, sqt, dht, sq_chunk_t, q_num_chunks]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    descriptor = ttnn.ProgramDescriptor(
        kernels=[reader_kernel, compute_kernel, writer_kernel],
        semaphores=[],
        cbs=cbs,
    )

    ordered = [query, key, value] + ([attn_mask] if has_mask else []) + [output_tensor]
    return descriptor, ordered
