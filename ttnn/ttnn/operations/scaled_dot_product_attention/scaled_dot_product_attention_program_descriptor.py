# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
#
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for Flash-Attention scaled_dot_product_attention.

Work distribution: one work unit = (batch b, head h, query tile-row q) → output
tile-row O[b,h,q,:] (Dt tiles). Total B·H·Sq_t units split across the full
compute grid via split_work_to_cores. Each core loops its contiguous unit range;
for each unit it streams all num_kv_chunks KV blocks through the online-softmax
recurrence.

CBs are sized to a single (q_chunk_t=1 × kv_chunk_t) score block + Dt output
accumulator — never Sq_t × Skv_t — which is what makes this Flash Attention.
"""

import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# fp32 + half-sync DEST capacity (must match ComputeConfig fp32_dest_acc_en=True,
# dst_full_sync_en=False). Bounds matmul out-subblock tile count and kv_chunk_t.
DEST_LIMIT = 4
KV_CHUNK_MAX = DEST_LIMIT  # QK out-subblock = q_chunk_t(1) × kv_chunk_t ≤ DEST_LIMIT


def _largest_divisor_leq(n, limit):
    """Largest divisor of n that is ≤ limit (≥ 1)."""
    for d in range(min(n, limit), 0, -1):
        if n % d == 0:
            return d
    return 1


def _f32_to_u32(f):
    """Bit-reinterpret a python float as fp32 → uint32 (for MulUnary scale)."""
    return struct.unpack("<I", struct.pack("<f", float(f)))[0]


def create_program_descriptor(Q, K, V, output_tensor, *, attention_mask=None, scale=None):
    # ---------------- 1. Tensor metadata ----------------
    B, H, S_q, D = (int(x) for x in Q.shape)
    H_kv, S_kv = int(K.shape[1]), int(K.shape[2])
    assert S_q % 32 == 0 and S_kv % 32 == 0 and D % 32 == 0, "Phase 0 is tile-aligned"

    Sq_t = S_q // 32
    Skv_t = S_kv // 32
    Dt = D // 32

    use_mask = 1 if attention_mask is not None else 0
    mask_H = int(attention_mask.shape[1]) if use_mask else 1

    kv_chunk_t = _largest_divisor_leq(Skv_t, KV_CHUNK_MAX)
    num_kv_chunks = Skv_t // kv_chunk_t

    # Matmul sub-block params (out_subblock_h is always 1 since q_chunk_t = 1).
    osw_qk = _largest_divisor_leq(kv_chunk_t, DEST_LIMIT)  # = kv_chunk_t
    in1sb_qk = kv_chunk_t // osw_qk
    osw_pv = _largest_divisor_leq(Dt, DEST_LIMIT)
    in1sb_pv = Dt // osw_pv

    scale_u32 = _f32_to_u32(scale)

    bf16 = ttnn.bfloat16
    fp32 = ttnn.float32
    tile_bf16 = ttnn.tile_size(bf16)
    tile_fp32 = ttnn.tile_size(fp32)

    # ---------------- 2. Work distribution ----------------
    grid = Q.device().compute_with_storage_grid_size()
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    total_units = B * H * Sq_t
    (_, core_grid, core_group_1, core_group_2, units_g1, units_g2) = ttnn.split_work_to_cores(all_cores, total_units)

    # ---------------- 3. Circular buffers ----------------
    CB_Q = 0
    CB_K = 1
    CB_V = 2
    CB_MASK = 3
    CB_SCALER_MAX = 8
    CB_SCALER_SUM = 9
    CB_MBLOCK = 10
    CB_MNEW = 11
    CB_LBLOCK = 12
    CB_OUT = 16
    CB_SCORES = 24
    CB_P = 25
    CB_PV = 26
    CB_OUT_ACCUM = 27
    CB_MAX = 28
    CB_SUM = 29
    CB_ALPHA = 30
    CB_RECIP = 31

    def cb(index, data_format, page_size, num_pages):
        return ttnn.CBDescriptor(
            total_size=page_size * num_pages,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)
            ],
        )

    cbs = [
        cb(CB_Q, bf16, tile_bf16, Dt),  # resident scaled Q
        cb(CB_K, bf16, tile_bf16, 2 * kv_chunk_t * Dt),  # streaming, double-buffered
        cb(CB_V, bf16, tile_bf16, 2 * kv_chunk_t * Dt),
        cb(CB_SCALER_MAX, bf16, tile_bf16, 1),
        cb(CB_SCALER_SUM, bf16, tile_bf16, 1),
        cb(CB_MBLOCK, fp32, tile_fp32, 1),
        cb(CB_MNEW, fp32, tile_fp32, 1),
        cb(CB_LBLOCK, fp32, tile_fp32, 1),
        cb(CB_OUT, bf16, tile_bf16, 2 * Dt),  # streaming output, double-buffered
        cb(CB_SCORES, fp32, tile_fp32, kv_chunk_t),  # one QK block, held across max+exp
        cb(CB_P, bf16, tile_bf16, kv_chunk_t),  # exp-probabilities (≤1.0), bf16 for PV
        cb(CB_PV, fp32, tile_fp32, Dt),
        cb(CB_OUT_ACCUM, fp32, tile_fp32, Dt),  # running O, persists across KV loop
        cb(CB_MAX, fp32, tile_fp32, 1),  # running m
        cb(CB_SUM, fp32, tile_fp32, 1),  # running l
        cb(CB_ALPHA, fp32, tile_fp32, 1),
        cb(CB_RECIP, fp32, tile_fp32, 1),
    ]
    if use_mask:
        cbs.append(cb(CB_MASK, bf16, tile_bf16, 2 * kv_chunk_t))

    # ---------------- 4. Kernels ----------------
    # ---- Reader ----
    reader_ct_args = [
        H,
        H_kv,
        Sq_t,
        Skv_t,
        Dt,
        kv_chunk_t,
        num_kv_chunks,
        mask_H,
        use_mask,
        CB_Q,
        CB_K,
        CB_V,
        CB_MASK,
        CB_SCALER_MAX,
        CB_SCALER_SUM,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(Q).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(K).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(V).get_compile_time_args())
    # Mask accessor is declared unconditionally in the kernel (chained CT offset)
    # but only used under `if constexpr (use_mask)`. When absent, pass Q's accessor
    # args as a structurally-valid placeholder (never read at runtime).
    reader_ct_args.extend(ttnn.TensorAccessorArgs(attention_mask if use_mask else Q).get_compile_time_args())

    # ---- Compute ----
    compute_ct_args = [
        Dt,
        kv_chunk_t,
        num_kv_chunks,
        use_mask,
        osw_qk,
        in1sb_qk,
        osw_pv,
        in1sb_pv,
        CB_Q,
        CB_K,
        CB_V,
        CB_MASK,
        CB_SCALER_MAX,
        CB_SCALER_SUM,
        CB_MBLOCK,
        CB_MNEW,
        CB_LBLOCK,
        CB_OUT,
        CB_SCORES,
        CB_P,
        CB_PV,
        CB_OUT_ACCUM,
        CB_MAX,
        CB_SUM,
        CB_ALPHA,
        CB_RECIP,
    ]

    # ---- Writer ----
    writer_ct_args = [H, Sq_t, Dt, CB_OUT]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # ---- Per-core runtime args ----
    reader_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()

    q_addr = Q.buffer_address()
    k_addr = K.buffer_address()
    v_addr = V.buffer_address()
    mask_addr = attention_mask.buffer_address() if use_mask else 0
    o_addr = output_tensor.buffer_address()

    def assign(core_group, units_per_core, start_ref):
        unit = start_ref
        for core_range in core_group.ranges():
            for x in range(core_range.start.x, core_range.end.x + 1):
                for y in range(core_range.start.y, core_range.end.y + 1):
                    reader_rt[x][y] = [q_addr, k_addr, v_addr, mask_addr, unit, units_per_core]
                    compute_rt[x][y] = [units_per_core, scale_u32]
                    writer_rt[x][y] = [o_addr, unit, units_per_core]
                    unit += units_per_core
        return unit

    next_unit = assign(core_group_1, units_g1, 0)
    if units_g2 > 0:
        assign(core_group_2, units_g2, next_unit)

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    compute_config = ttnn.ComputeConfigDescriptor()
    compute_config.math_fidelity = ttnn.MathFidelity.HiFi2
    compute_config.fp32_dest_acc_en = True
    compute_config.math_approx_mode = False
    compute_config.dst_full_sync_en = False

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_compute.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt,
        config=compute_config,
    )

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
