# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Program descriptor for scaled_dot_product_attention (Flash Attention).

Stage 0 (init): allocates the running-state CBs and launches kernels that
initialize them. The reader fills:
  cb_max_old (27) ← -inf  (B_q_t tiles, running max m_i)
  cb_sum_old (30) ← 0.0   (B_q_t tiles, running sum l_i)
  cb_o       (16) ← 0.0   (B_q_t * D_t tiles, running output O_i)
The compute kernel boots the Tensix pipeline; the writer is a stub.

CB layout (from op_design.md). Only the CBs touched by Stage 0 are
allocated here; later stages add cb_k, cb_v, cb_mask, cb_scaler_reduce,
cb_scale_factor, cb_scores_masked, cb_max_new, cb_exp_scores, cb_sum_new,
cb_o_accum, cb_alpha as the recurrence is built up.

  cb_q          (0)  — Q-block tiles (allocated, filled by later stages)
  cb_o          (16) — running output, B_q_t * D_t tiles
  cb_scores     (24) — score matmul out (referenced by hw_startup, filled later)
  cb_max_old    (27) — running max m_i, B_q_t tiles
  cb_sum_old    (30) — running sum l_i, B_q_t tiles

Tile block sizing (from op_design.md):
  B_q_t = 4  (128 rows per Q-block)
  B_kv_t = 4 (128 cols per KV-block)
  D_t = D // 32 (head-dim tiles)
"""

from pathlib import Path

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
    """Build the Stage-0 program descriptor.

    Args mirror the op entry point's resolved parameters. Stage 0 only uses
    the shapes (to size CBs) and the compute config; attn_mask / is_causal /
    scale are accepted for forward-compatibility and ignored here.
    """
    q_shape = list(query.shape)
    D = q_shape[-1]
    D_t = D // TILE_DIM

    tile_size = ttnn.tile_size(query.dtype)  # bf16 → 2048 bytes

    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # --- Circular Buffers (Stage 0 subset) ---
    CB_Q = 0
    CB_O = 16
    CB_SCORES = 24
    CB_MAX_OLD = 27
    CB_SUM_OLD = 30

    num_o_tiles = B_Q_T * D_t

    cbs = [
        # cb_q: one Q-block (4 tiles). Filled by reader in later stages;
        # allocated now so compute_kernel_hw_startup(cb_q, cb_scores) has a
        # valid CB to reference.
        ttnn.CBDescriptor(
            total_size=B_Q_T * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_Q, data_format=query.dtype, page_size=tile_size)
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
        # cb_scores: score matmul output (16 tiles). Referenced by hw_startup;
        # filled by matmul in Stage 1. Allocated so the compute kernel can boot.
        ttnn.CBDescriptor(
            total_size=B_Q_T * B_KV_T * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_SCORES, data_format=query.dtype, page_size=tile_size)
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
        # cb_sum_old: running sum l_i, B_q_t tiles. Initialized to 0 in Stage 0.
        ttnn.CBDescriptor(
            total_size=B_Q_T * tile_size,
            core_ranges=core_grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_SUM_OLD, data_format=query.dtype, page_size=tile_size)
            ],
        ),
    ]

    # --- Reader ---
    # CT args: [B_q_t, D_t, tile_bytes]
    reader_ct_args = [B_Q_T, D_t, tile_size]
    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = []

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
    # CT args: [B_q_t, B_kv_t, D_t] (used by later stages; Stage 0 only boots).
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
