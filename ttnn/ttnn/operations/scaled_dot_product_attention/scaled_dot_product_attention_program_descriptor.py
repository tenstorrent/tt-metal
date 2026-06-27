# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Program descriptor for scaled_dot_product_attention (Flash Attention).

Stage 1 (qkt_matmul): The reader streams Q-block and K-block tiles from DRAM
into L1 CBs. The compute kernel boots the matmul pipeline and calls
matmul_block (transpose=true) to produce cb_scores = Q @ K^T.

Stage 4 (rowmax): The reader prepares the reduce scaler tile (1.0 for MAX) in
cb_scaler_reduce. The compute kernel calls reduce<MAX, REDUCE_ROW, WaitUpfrontNoPop>
to produce cb_max_new = per-row max of cb_scores_masked.

CB layout (from op_design.md, with corrected sizing — see CB sizing note below):

  cb_q             (0)  — Q-block tiles, B_q_t * D_t pages (16 for D=128)
  cb_k             (1)  — K-block tiles, B_kv_t * D_t pages (16 for D=128)
  cb_scaler_reduce (4)  — reduce scaler tile (1.0 for MAX), 1 page
  cb_scale_factor  (5)  — scale value tile, 1 page
  cb_o             (16) — running output, B_q_t * D_t tiles
  cb_scores        (24) — Q@K^T score matmul out, B_q_t * B_kv_t tiles (16)
  cb_scores_masked (25) — scores after mask/scale, 16 tiles
  cb_max_new       (26) — per-row max of scores, B_q_t tiles
  cb_max_old       (27) — running max m_i, B_q_t tiles
  cb_sum_old       (30) — running sum l_i, B_q_t tiles

CB sizing note: The design specifies cb_q with B_q_t=4 pages, but matmul_block
with k=D_t and num_k=1 requires all M*K = B_q_t*D_t Q-tiles resident in one
call (in0_block_num_tiles = out_subblock_h * in0_block_k * in0_num_subblocks
= 2*4*2 = 16). Similarly cb_k needs B_kv_t*D_t = 16 pages. The design's
in0_sb=1,in1_sb=1 only produces 4 output tiles (a 2×2 subblock); we use
in0_sb=2,in1_sb=2 to produce all 16 score tiles in one call.

The design's in0_policy=NoWaitNoPop is impossible (static_assert forbids it
for in0). We use WaitAndRetainOnLastBlock, which retains Q when num_k_blocks=1.

Tile block sizing (from op_design.md):
  B_q_t = 4  (128 rows per Q-block)
  B_kv_t = 4 (128 cols per KV-block)
  D_t = D // 32 (head-dim tiles)
"""

from pathlib import Path
import math
import struct

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE_DIM = 32

# Block sizes (op_design.md tile block sizing).
B_Q_T = 4  # Q-block tile rows (128 rows)
B_KV_T = 4  # KV-block tile cols (128 cols)


def create_program_descriptor(
    query: ttnn.Tensor,
    key: ttnn.Tensor,
    value: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    attn_mask: ttnn.Tensor | None = None,
    is_causal: bool = False,
    scale: float | None = None,
    math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en: bool = True,
    math_approx_mode: bool = False,
) -> ttnn.ProgramDescriptor:
    """Build the Stage-1 program descriptor.

    Args mirror the op entry point's resolved parameters. Stage 1 streams
    Q and K tiles from DRAM, runs the QK^T matmul, and DPRINTs the score
    block for TDD verification against reference_phase_qkt.
    """
    q_shape = list(query.shape)
    D = q_shape[-1]
    D_t = D // TILE_DIM

    tile_size = ttnn.tile_size(query.dtype)  # bf16 → 2048 bytes

    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # Resolve scale: explicit value or 1/sqrt(D).
    resolved_scale = scale if scale is not None else (1.0 / math.sqrt(D))
    # Pack as fp32 bits for the reader (converts to bf16 in-kernel).
    scale_bits = struct.unpack("I", struct.pack("f", resolved_scale))[0]

    # --- Circular Buffers ---
    CB_Q = 0
    CB_K = 1
    CB_SCALER_REDUCE = 4
    CB_SCALE_FACTOR = 5
    CB_ALPHA = 8
    CB_O = 16
    CB_SCORES = 24
    CB_SCORES_MASKED = 25
    CB_MAX_NEW = 26
    CB_MAX_OLD = 27
    CB_EXP_SCORES = 28
    CB_SUM_NEW = 29
    CB_SUM_OLD = 30

    num_q_tiles = B_Q_T * D_t  # 16 for D=128
    num_k_tiles = B_KV_T * D_t  # 16 for D=128
    num_o_tiles = B_Q_T * D_t
    num_score_tiles = B_Q_T * B_KV_T  # 16

    cbs = [
        # cb_q: one Q-block (B_q_t * D_t tiles). Filled by reader; retained
        # across KV-blocks by compute (WaitAndRetainOnLastBlock).
        ttnn.CBDescriptor(
            total_size=num_q_tiles * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_Q, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        # cb_k: one KV-block of K (B_kv_t * D_t tiles). Streamed by reader;
        # consumed per KV-block by compute (WaitAndPopPerKBlock).
        ttnn.CBDescriptor(
            total_size=num_k_tiles * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_K, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        # cb_scaler_reduce: 2 tiles holding reduce scalers. The reader prepares
        # two scaler tiles — one for MAX (row-0 fill) and one for SUM (col-0
        # fill, since SUM REDUCE_ROW uses the matmul path). Each reduce call
        # waits for its scaler but does NOT pop it; the compute kernel pops
        # the MAX scaler after Stage 4 so the SUM scaler is at the front for
        # Stage 10.
        ttnn.CBDescriptor(
            total_size=2 * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_SCALER_REDUCE, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        # cb_scale_factor: single tile holding the scale value (1/sqrt(D) or
        # explicit). Filled by reader; held as HeldBulk by compute's eltwise mul.
        ttnn.CBDescriptor(
            total_size=1 * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_SCALE_FACTOR, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        # cb_alpha: alpha = exp(m_old - m_new) per Q-block row. B_q_t tiles.
        # Produced by compute (eltwise_chain: BinaryFpu Sub + Exp + PackTile),
        # consumed by compute (rescale O and l via eltwise mul with Col broadcast).
        ttnn.CBDescriptor(
            total_size=B_Q_T * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_ALPHA, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        # cb_o: running output, B_q_t * D_t tiles. Initialized to 0 in Stage 0.
        ttnn.CBDescriptor(
            total_size=num_o_tiles * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_O, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        # cb_scores: Q@K^T score matmul output (16 tiles).
        ttnn.CBDescriptor(
            total_size=num_score_tiles * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_SCORES, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        # cb_scores_masked: scores after mask-add (or passthrough copy when no
        # mask). 16 tiles. Produced by compute (copy/add eltwise), consumed by
        # compute (row-max reduce, subtract, exp).
        ttnn.CBDescriptor(
            total_size=num_score_tiles * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_SCORES_MASKED, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        # cb_max_new: per-row max of scores (B_q_t tiles). Produced by compute
        # (reduce MAX REDUCE_ROW, WaitUpfrontNoPop), consumed by compute
        # (alpha computation, update m_i, subtract max).
        ttnn.CBDescriptor(
            total_size=B_Q_T * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_MAX_NEW, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        # cb_max_old: running max m_i, B_q_t tiles. Initialized to -inf in Stage 0.
        ttnn.CBDescriptor(
            total_size=B_Q_T * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_MAX_OLD, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        # cb_exp_scores: exp(S - m_new), B_q_t * B_kv_t tiles (16). Produced by
        # compute (unary<Exp> eltwise), consumed by compute (row-sum reduce, PV matmul).
        ttnn.CBDescriptor(
            total_size=num_score_tiles * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_EXP_SCORES, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        # cb_sum_old: running sum l_i, B_q_t tiles. Initialized to 0 in Stage 0.
        ttnn.CBDescriptor(
            total_size=B_Q_T * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_SUM_OLD, data_format=query.dtype, page_size=tile_size)
            ],
        ),
        # cb_sum_new: per-row sum of exp scores (B_q_t tiles). Produced by
        # compute (reduce SUM REDUCE_ROW), consumed by compute (update l_i).
        ttnn.CBDescriptor(
            total_size=B_Q_T * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_SUM_NEW, data_format=query.dtype, page_size=tile_size)
            ],
        ),
    ]

    # --- Reader ---
    # CT args: [B_q_t, D_t, B_kv_t, ...TensorAccessorArgs(Q), ...TensorAccessorArgs(K)]
    reader_ct_args = [B_Q_T, D_t, B_KV_T]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(query).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(key).get_compile_time_args())

    # RT args: [q_addr, k_addr, scale_bits (fp32 bits)]
    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [
        query.buffer_address(),
        key.buffer_address(),
        scale_bits,
    ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Writer (stub) ---
    writer_ct_args = []
    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = []

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # --- Compute ---
    # CT args: [B_q_t, B_kv_t, D_t]
    compute_ct_args = [B_Q_T, B_KV_T, D_t]
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_compute.cpp"),
        core_ranges=core_grid,
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
