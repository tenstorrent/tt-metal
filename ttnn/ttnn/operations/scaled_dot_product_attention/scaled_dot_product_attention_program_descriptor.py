# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for Flash-Attention scaled_dot_product_attention.

Work unit = one Q-block (b, h_q, q_chunk). Each core gets a contiguous run of
Q-blocks; per block it loads Q once, streams every KV-block, runs the online-
softmax recurrence, normalizes, and writes the output chunk. See op_design.md.
"""

import math
import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE_DIM = 32

# CB indices (semantic names in kernels)
CB_Q = 0
CB_K = 1
CB_V = 2
CB_MASK = 3
CB_QS = 4
CB_SCALER_MAX = 8
CB_SCALER_SUM = 9
CB_L_NEW = 10
CB_PV = 11
CB_O_RUN = 12
CB_O_NEW = 13
CB_L_INV = 14
CB_MASKED = 15
CB_OUT = 16
CB_SCORES = 24
CB_PROBS = 25
CB_M_CUR = 26
CB_M_RUN = 27
CB_M_NEW = 28
CB_CORR = 29
CB_L_CUR = 30
CB_L_RUN = 31

MAX_CHUNK = 4  # q_chunk_t / kv_chunk_t cap (~128 rows)


def _f32_bits(f: float) -> int:
    return struct.unpack("I", struct.pack("f", float(f)))[0]


def _enumerate_cores(gx: int, gy: int, n: int):
    """First n cores of a gx*gy grid in row-wise order."""
    cores = []
    for y in range(gy):
        for x in range(gx):
            if len(cores) >= n:
                return cores
            cores.append((x, y))
    return cores


def create_program_descriptor(
    query: ttnn.Tensor,
    key: ttnn.Tensor,
    value: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    attn_mask: ttnn.Tensor = None,
    scale: float,
    compute_kernel_config,
):
    q_shape = list(query.shape)
    k_shape = list(key.shape)
    B, H_q, S_q, D = q_shape
    _, H_kv, S_kv, _ = k_shape

    Sq_t = (S_q + TILE_DIM - 1) // TILE_DIM
    Skv_t = (S_kv + TILE_DIM - 1) // TILE_DIM
    Dt = (D + TILE_DIM - 1) // TILE_DIM

    q_chunk_t = min(Sq_t, MAX_CHUNK)
    kv_chunk_t = min(Skv_t, MAX_CHUNK)
    q_blocks_per_bh = (Sq_t + q_chunk_t - 1) // q_chunk_t
    num_kv_blocks = (Skv_t + kv_chunk_t - 1) // kv_chunk_t
    num_q_blocks = B * H_q * q_blocks_per_bh
    gqa_factor = H_q // H_kv

    has_mask = attn_mask is not None
    mask_mode = 1 if has_mask else 0
    mask_H = int(attn_mask.shape[1]) if has_mask else 1
    scale_bits = _f32_bits(scale)

    # Non-tile-aligned S_kv: the physical last KV tile carries (32 - skv_last_valid)
    # zero-padded key columns. Those padded keys score 0 (Q·0) and would leak
    # exp(0 - row_max) into the softmax row-sum, inflating l and shrinking the
    # output magnitude. Exclude them from the SUM reduce via a partial reduce
    # scaler on the last kv-tile of the last KV block. (Row-MAX is left on the
    # full scaler: an inflated max cancels in the normalized softmax; PV is
    # unaffected because padded V rows are zero.) D (head_dim) non-alignment
    # needs no handling — padded Q/K/V columns are zero, so QKᵀ/PV are exact.
    skv_last_valid = S_kv % TILE_DIM
    skv_non_aligned = 1 if skv_last_valid != 0 else 0

    # --- Work distribution: contiguous Q-block run per core ---
    device = query.device()
    grid = device.compute_with_storage_grid_size()
    gx, gy = grid.x, grid.y
    num_cores = min(num_q_blocks, gx * gy)
    num_cores = max(num_cores, 1)
    used_cores = _enumerate_cores(gx, gy, num_cores)

    base = num_q_blocks // num_cores
    rem = num_q_blocks % num_cores
    per_core = []  # (x, y, start_qb, count)
    cursor = 0
    for i, (x, y) in enumerate(used_cores):
        count = base + (1 if i < rem else 0)
        per_core.append((x, y, cursor, count))
        cursor += count

    core_ranges = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for (x, y) in used_cores]
    )

    # --- CB sizing (pages) ---
    tile_dtype = query.dtype
    tile_bytes = ttnn.tile_size(tile_dtype)
    scaler_bytes = ttnn.tile_size(ttnn.bfloat16)

    qcb = q_chunk_t
    kcb = kv_chunk_t
    block_pages = qcb * kcb
    o_pages = qcb * Dt

    def cb(idx, pages, fmt=tile_dtype, psize=tile_bytes):
        return ttnn.CBDescriptor(
            total_size=pages * psize,
            core_ranges=core_ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=psize)],
        )

    # Softmax-path score intermediates (QKᵀ scores, masked scores, exp probs).
    # For bf8b input these MUST NOT be stored as bf8b: the additive mask writes
    # -inf into cb_masked, and bf8b's shared-per-16-element exponent lets one -inf
    # entry corrupt the valid scores in its block (PCC ~0.9 on custom-mask cells).
    # bf16 has a per-element exponent, so -inf stays local. Q/K/V and the output
    # stay in the caller's dtype; only the internal score path is promoted.
    score_dtype = ttnn.bfloat16 if tile_dtype == ttnn.bfloat8_b else tile_dtype
    score_bytes = ttnn.tile_size(score_dtype)

    def cbs_(idx, pages):
        return cb(idx, pages, fmt=score_dtype, psize=score_bytes)

    # fp32 accumulation path: the running output/sum are updated over up to
    # num_kv_blocks (long context => dozens) online-softmax steps; bf16 storage
    # loses precision there. The matmul/reduce already accumulate in fp32 DEST.
    fp32 = ttnn.float32
    fp32_bytes = ttnn.tile_size(ttnn.float32)

    def cbf(idx, pages):
        return cb(idx, pages, fmt=fp32, psize=fp32_bytes)

    cbs = [
        cb(CB_Q, qcb * Dt),
        cb(CB_QS, qcb * Dt),
        cb(CB_K, 2 * kcb * Dt),
        cb(CB_V, 2 * kcb * Dt),
        cb(CB_MASK, 2 * block_pages if has_mask else 1),
        cb(CB_SCALER_MAX, 1, fmt=ttnn.bfloat16, psize=scaler_bytes),
        # 2 tiles when S_kv is non-aligned: [full scaler, partial scaler].
        cb(CB_SCALER_SUM, 2 if skv_non_aligned else 1, fmt=ttnn.bfloat16, psize=scaler_bytes),
        cbf(CB_L_NEW, 2 * qcb),
        cbf(CB_PV, 2 * o_pages),
        cbf(CB_O_RUN, 2 * o_pages),
        cbf(CB_O_NEW, 2 * o_pages),
        cbf(CB_L_INV, 2 * qcb),
        cbs_(CB_MASKED, 2 * block_pages if has_mask else 1),
        cb(CB_OUT, 2 * o_pages),
        cbs_(CB_SCORES, 2 * block_pages),
        cbs_(CB_PROBS, 2 * block_pages),
        cbf(CB_M_CUR, 2 * qcb),
        cbf(CB_M_RUN, 2 * qcb),
        cbf(CB_M_NEW, 2 * qcb),
        cbf(CB_CORR, 2 * qcb),
        cbf(CB_L_CUR, 2 * qcb),
        cbf(CB_L_RUN, 2 * qcb),
    ]

    # --- Reader ---
    reader_base = [
        B,
        H_q,
        H_kv,
        Sq_t,
        Skv_t,
        Dt,
        q_chunk_t,
        kv_chunk_t,
        q_blocks_per_bh,
        num_kv_blocks,
        gqa_factor,
        mask_mode,
        mask_H,
        skv_non_aligned,
        skv_last_valid,
    ]
    reader_ct = list(reader_base)
    reader_ct.extend(ttnn.TensorAccessorArgs(query).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(key).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(value).get_compile_time_args())
    reader_ct.extend(
        ttnn.TensorAccessorArgs(attn_mask).get_compile_time_args()
        if has_mask
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    reader_rt = ttnn.RuntimeArgs()
    for x, y, start_qb, count in per_core:
        reader_rt[x][y] = [
            query.buffer_address(),
            key.buffer_address(),
            value.buffer_address(),
            attn_mask.buffer_address() if has_mask else 0,
            start_qb,
            count,
        ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_reader.cpp"),
        core_ranges=core_ranges,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Compute ---
    compute_ct = [
        B,
        H_q,
        H_kv,
        Sq_t,
        Skv_t,
        Dt,
        q_chunk_t,
        kv_chunk_t,
        q_blocks_per_bh,
        num_kv_blocks,
        gqa_factor,
        mask_mode,
        scale_bits,
        skv_non_aligned,
    ]
    compute_rt = ttnn.RuntimeArgs()
    for x, y, start_qb, count in per_core:
        compute_rt[x][y] = [start_qb, count]

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_compute.cpp"),
        core_ranges=core_ranges,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=compute_kernel_config,
    )

    # --- Writer ---
    writer_ct = [B, H_q, Sq_t, Dt, q_chunk_t, q_blocks_per_bh]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_rt = ttnn.RuntimeArgs()
    for x, y, start_qb, count in per_core:
        writer_rt[x][y] = [output_tensor.buffer_address(), start_qb, count]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_writer.cpp"),
        core_ranges=core_ranges,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    tensors = [query, key, value]
    if has_mask:
        tensors.append(attn_mask)
    tensors.append(output_tensor)

    program_descriptor = ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
    return tensors, program_descriptor
