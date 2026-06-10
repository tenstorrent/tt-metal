# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Program descriptor for fused SDPA.

Multi-core distribution (Refinement 1) — query tile-rows (total_rows =
B * H * Qt) are split across the device's compute_with_storage_grid via
``ttnn.split_work_to_cores``. Each core's reader/writer/compute kernel
gets ``(num_rows, start_row)`` runtime args and processes its slice
independently. No inter-core communication — the kernel is
embarrassingly parallel along the query-tile-row axis.

Numerical configurability (Refinement 1) — the entry point's
``ttnn.ComputeKernelConfig`` flows in through ``compute_kernel_config``:

  - ``math_fidelity`` / ``fp32_dest_acc_en`` / ``math_approx_mode`` /
    ``dst_full_sync_en`` are wired to ``ComputeConfigDescriptor``.

  - Input / output CB formats follow the tensors' dtypes
    (bf16 / fp32 / bf8b).

  - Running-state CBs (cb_cur_max / cb_cur_sum_exp / cb_cur_mm_out) are
    forced to Float32 when ``fp32_dest_acc_en=True`` so the K-loop
    online-softmax update accumulates against an fp32 reload rather
    than truncating to bf16 on every iteration — closes the bulk of
    the S=8192 precision gap from Phase 0.

  - ``UnpackToDestFp32`` is applied selectively (Refinement 6).
    Slots 12 (``cb_cur_sum_exp``) and 21 (``cb_cur_mm_out``) carry the
    tag when ``fp32_dest_acc_en=True``: their per-K-iter SFPU reload in
    ``update_cur_sum_exp_pass2`` / ``matmul_attn_by_v_accumulate`` was
    the dominant S=8192 fp32 precision floor (~10 mantissa bits via
    srcA/srcB TF32 vs. the 24 the FP32 storage holds), and both
    accumulators are read SFPU-only inside the K-loop. The FPU
    ``mul_tiles_bcast_cols`` in the final divide is incompatible with
    ``UnpackToDestFp32``, so the divide pulls from untagged
    ``cb_cur_*_for_divide`` intermediates populated by a single SFPU
    ``copy_tile`` per tile. ``cb_cur_max`` stays untagged — it is
    FPU-read every pass-2 iter (``sub_tiles_bcast_cols`` against
    ``cb_attention_weights``) and tagging would silently corrupt that
    sub. R6 dropped RMS on the named ``Q1x1x8192x64 fp32`` cells from
    0.0272 to 0.000291 (~100× improvement) — see changelog.

L1 budget (Refinement 5) — the R4 two-pass restructure eliminated the
ping-pong correction cascade, leaving four CBs allocated but never
written/read at runtime: ``cb_prev_max``, ``cb_prev_sum_exp``,
``cb_exp_max_diff`` (each 2 × running_state_tile_size), and
``cb_prev_mm_out`` (2 × Dt × running_state_tile_size). At fp32 + D=1024
the unused ``cb_prev_mm_out`` alone was 256 KB, blowing the per-core L1
budget. Refinement 5 reclaims all four — saves ~287 KB at fp32 D=1024
and closes the four ``Q1x1x128x1024 fp32`` golden cells. The kernel's
remaining Dt-scaling CBs (``cb_query``, ``cb_key``, ``cb_value``,
``cb_cur_mm_out``, ``cb_output``) fit at D=1024 with ~187 KB headroom;
D ≥ 2048 would still OOM and would need true D-blocking on the matmul
(filed as a follow-up).

The per-row pipeline is the Flash-Attention-1 online softmax documented
in op_design.md and unchanged in this refinement.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Optional

import ttnn


KERNEL_DIR = Path(__file__).parent / "kernels"
TILE_DIM = 32


# CB indices (semantic names mirror the kernel-side constants).
CB_QUERY = 0
CB_KEY = 1
CB_VALUE = 2
CB_ATTN_MASK = 3

CB_REDUCTION_SCALER = 5  # bf16 1.0 tile, row-axis fill (reduce-LLK)
CB_MATMUL_REDUCE = 6  # bf16 col-0=1.0 tile for matmul-as-reduce
CB_ATTENTION_WEIGHTS = 7  # scores after QK^T (then softmax exps)
# Slots 8, 10 (the old cb_prev_max / cb_exp_max_diff) went unused after
# R4-iter3 collapsed the pass-1 ping-pong cascade. R5 reclaimed slot 11
# (cb_prev_sum_exp) and slot 20 (cb_prev_mm_out) too, but Refinement 6
# now reuses those two for the untagged final-divide intermediate CBs
# (see CB_CUR_*_FOR_DIVIDE below). Slots 8 and 10 remain free.
CB_CUR_MAX = 9
# R6: slot 11 reused for cb_cur_sum_exp_for_divide — an UNTAGGED copy of
# cb_cur_sum_exp consumed by the FPU mul_tiles_bcast_cols in the final
# divide. Only allocated when fp32_dest_acc_en=True (the only path where
# UnpackToDestFp32 tagging makes sense).
CB_CUR_SUM_EXP_FOR_DIVIDE = 11
CB_CUR_SUM_EXP = 12

CB_OUTPUT = 16

# Bigger CBs (sized in Dt) live in the 20s range.
# R6: slot 20 reused for cb_cur_mm_out_for_divide (untagged Dt-tile copy
# of cb_cur_mm_out for the FPU final divide). Only allocated when
# fp32_dest_acc_en=True. At fp32 D=64 (Dt=2) this is 32 KB single-buffered;
# at fp32 D=1024 (Dt=32) it is 128 KB.
CB_CUR_MM_OUT_FOR_DIVIDE = 20
CB_CUR_MM_OUT = 21


def _running_state_cb_format(fp32_dest_acc_en: bool) -> ttnn.DataType:
    """Format for the 6 running-state CBs.

    fp32 if dest-fp32 accumulation is on (preserves the per-iteration
    accumulator); falls back to bf16 otherwise (matches the pre-
    refinement behavior bit-for-bit).
    """
    return ttnn.float32 if fp32_dest_acc_en else ttnn.bfloat16


def _tile_size_for_format(data_format: ttnn.DataType) -> int:
    """Wrapper around ttnn.tile_size for read-side clarity."""
    return ttnn.tile_size(data_format)


def create_program_descriptor(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    attention_mask: Optional[ttnn.Tensor],
    scale_value: float,
    compute_kernel_config: ttnn.WormholeComputeKernelConfig,
) -> ttnn.ProgramDescriptor:
    B, H_q, Sq, D = list(q.shape)
    _, H_kv, Skv, _ = list(k.shape)

    # Refinement 3: ceil-divide instead of floor-divide. TTNN's TILE_LAYOUT
    # pads non-aligned final dims to the next multiple of 32 with zeros,
    # so the kernel always operates on Qt/Kt/Dt full tiles — the logical
    # shape's leftover positions live in the last partial tile.
    Qt = (Sq + TILE_DIM - 1) // TILE_DIM
    Kt = (Skv + TILE_DIM - 1) // TILE_DIM
    Dt = (D + TILE_DIM - 1) // TILE_DIM

    # Refinement 3: tail counts for the last partial tile along each axis.
    # 0 ⇒ aligned (no special handling); >0 ⇒ that many valid positions
    # in the last tile, the rest are TTNN-padded zeros.
    #   - keys_in_last_tile: drives the synthetic alignment-mask path so
    #     softmax doesn't normalize over padded keys (S_kv non-aligned).
    #   - queries_in_last_tile / dims_in_last_tile: informational only; the
    #     existing math is benign on Q/D padding because TTNN auto-zeros
    #     the pad and zero × anything = zero in the QK^T and attn@V matmuls.
    keys_in_last_tile = Skv % TILE_DIM
    needs_alignment_mask = keys_in_last_tile != 0

    has_user_mask = attention_mask is not None
    has_mask = has_user_mask or needs_alignment_mask
    mask_per_head = False
    if has_user_mask:
        Hm = int(attention_mask.shape[1])
        mask_per_head = Hm == H_q

    # Total query tile-rows: B × H_q × Qt. The work distribution and the
    # reader/writer per-row decoding key off H_q (Q's head count); the
    # reader's KV addressing keys off H_kv via h_kv = h_q / (H_q / H_kv).
    total_rows = B * H_q * Qt

    # ---- Compute-config-derived knobs -----------------------------------
    fp32_dest_acc_en = bool(compute_kernel_config.fp32_dest_acc_en)
    math_fidelity = compute_kernel_config.math_fidelity
    math_approx_mode = bool(compute_kernel_config.math_approx_mode)
    dst_full_sync_en = bool(compute_kernel_config.dst_full_sync_en)

    # Running-state CB format and per-tile byte size — Float32 when fp32
    # dest accumulation is on (Refinement 1 precision lift), bf16
    # otherwise.
    running_state_format = _running_state_cb_format(fp32_dest_acc_en)
    running_state_tile_size = _tile_size_for_format(running_state_format)

    # Page sizes — Q/K/V/output use their tensors' page sizes. Mask CB
    # follows q.dtype so the kernel's copy_tile path doesn't have to do
    # a cross-format conversion.
    q_page_size = q.buffer_page_size()
    k_page_size = k.buffer_page_size()
    v_page_size = v.buffer_page_size()
    out_page_size = output_tensor.buffer_page_size()
    bf16_tile = ttnn.tile_size(ttnn.bfloat16)
    # Refinement 3: mask CB exists whenever has_mask is true, including
    # when the alignment mask is synthesized in the absence of a user mask.
    # In the synthetic-only case there is no mask tensor to read a page size
    # from — fall back to the q.dtype tile size so the kernel's
    # copy_tile-and-add path sees a uniformly-formatted mask CB.
    if has_user_mask:
        mask_page_size = attention_mask.buffer_page_size()
        mask_format = attention_mask.dtype
    else:
        mask_page_size = _tile_size_for_format(q.dtype)
        mask_format = q.dtype

    # ---- Core grid (Refinement 1: multi-core split) ----------------------
    grid_size = q.device().compute_with_storage_grid_size()
    (
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        rows_per_core_1,
        rows_per_core_2,
    ) = ttnn.split_work_to_cores(grid_size, total_rows, row_wise=True)

    # ---- Circular buffers ------------------------------------------------
    cbs = [
        # cb_query: Dt tiles held across K-loop (no double buffer — reader
        # pushes once per row, compute pops once per row).
        ttnn.CBDescriptor(
            total_size=Dt * q_page_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_QUERY, data_format=q.dtype, page_size=q_page_size)
            ],
        ),
        # cb_key: Dt tiles per iter, double-buffered.
        ttnn.CBDescriptor(
            total_size=2 * Dt * k_page_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_KEY, data_format=k.dtype, page_size=k_page_size)
            ],
        ),
        # cb_value: Dt tiles per iter, double-buffered.
        ttnn.CBDescriptor(
            total_size=2 * Dt * v_page_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_VALUE, data_format=v.dtype, page_size=v_page_size)
            ],
        ),
        # cb_reduction_scaler: 1 bf16 tile (1.0), row-axis fill. Stays
        # bf16 regardless of input dtype — the reduce_helpers_dataflow
        # path deduces the L1 format from the CB descriptor and the
        # reduce LLK unpacks bf16→TF32 cleanly for any input dtype.
        ttnn.CBDescriptor(
            total_size=bf16_tile,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_REDUCTION_SCALER,
                    data_format=ttnn.bfloat16,
                    page_size=bf16_tile,
                )
            ],
        ),
        # cb_matmul_reduce: 1 bf16 tile (col-0 ones), generated by writer.
        # Stays bf16 — it's a per-column-0 ones tile that feeds a matmul
        # as in1; the bf16→TF32 unpack is lossless.
        ttnn.CBDescriptor(
            total_size=bf16_tile,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_MATMUL_REDUCE,
                    data_format=ttnn.bfloat16,
                    page_size=bf16_tile,
                )
            ],
        ),
        # cb_attention_weights: 1 tile per K-iter, double-buffered.
        # Refinement 4: widened to running_state_format (fp32 when
        # fp32_dest_acc_en=True) — the per-K-iter scores/attn weights
        # were the dominant precision floor for fp32 + S >= 4096 cells.
        # bf16 storage truncated mantissa from fp32 DST on every pack
        # (Q@K^T → cb_attention_weights, then in-place exp rewrite
        # → cb_attention_weights again), and the bf16 unpack on FPU
        # consumers (sub_tiles_bcast_cols, matmul_tiles) carried only
        # 8 mantissa bits. fp32 storage costs 2× page size but preserves
        # the full 10-bit TF32 unpack mantissa to FPU srcA/srcB —
        # 2 extra bits per round-trip × 2 round-trips per K-iter ×
        # Kt iters compounded the cascade error the original R4
        # two-pass design alone could not close.
        ttnn.CBDescriptor(
            total_size=2 * running_state_tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_ATTENTION_WEIGHTS,
                    data_format=running_state_format,
                    page_size=running_state_tile_size,
                )
            ],
        ),
        # ---- Per-iter softmax-state CBs ----------------------------------
        # cb_cur_max — running max across K-iterations. Float32 when
        # fp32_dest_acc_en (precision lift); bf16 otherwise. R4-iter3
        # update_cur_max_inplace overwrites the same slot each iter, so
        # we no longer need a cb_prev_max ping-pong companion (R5 reclaim).
        ttnn.CBDescriptor(
            total_size=2 * running_state_tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_CUR_MAX,
                    data_format=running_state_format,
                    page_size=running_state_tile_size,
                )
            ],
        ),
        # cb_cur_sum_exp — pass-2 running sum-of-exp accumulator
        # (update_cur_sum_exp_pass2 against the fixed global_max).
        # Float32 when fp32_dest_acc_en; bf16 otherwise. R4-iter3 left
        # cb_prev_sum_exp and cb_exp_max_diff allocated but unused (the
        # pre-R4 multiplicative cascade `cur = prev * corr + new_row_sum`
        # is gone); R5 reclaims both — see top-of-file note.
        #
        # Refinement 6: tagged UnpackToDestFp32 when fp32_dest_acc_en=True.
        # Pass-2 readers (update_cur_sum_exp_pass2's copy_tile of the prior
        # accumulator, then recip_tile_inplace at the divide boundary) are
        # all SFPU; the FPU mul_tiles_bcast_cols in the final divide reads
        # cb_cur_sum_exp_for_divide (untagged) instead.
        ttnn.CBDescriptor(
            total_size=2 * running_state_tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_CUR_SUM_EXP,
                    data_format=running_state_format,
                    page_size=running_state_tile_size,
                )
            ],
        ),
        # cb_cur_mm_out — Dt tiles, running attn @ V accumulator.
        # Float32 when fp32_dest_acc_en; bf16 otherwise. This is the
        # dominant precision-lift CB AND the dominant L1 consumer at
        # large head_dim — at fp32 D=1024 (Dt=32) the double-buffered
        # CB is 256 KB. R4-iter3 made matmul_attn_by_v_accumulate
        # in-place against this CB (no `prev` companion needed) — R5
        # reclaims cb_prev_mm_out's 256 KB on the same shapes.
        #
        # Refinement 6: this CB is tagged UnpackToDestFp32 when
        # fp32_dest_acc_en=True (see unpack_to_dest_mode vector below).
        # The tag preserves the full 24-bit FP32 mantissa across the
        # per-K-iter SFPU copy_tile reload in matmul_attn_by_v_accumulate,
        # vs. the 10-bit TF32 cap that an untagged unpack would impose.
        # All readers inside the K-loop are SFPU (copy_tile only), so
        # the tag is safe; the FPU mul_tiles_bcast_cols in the final
        # divide goes through cb_cur_mm_out_for_divide instead.
        ttnn.CBDescriptor(
            total_size=2 * Dt * running_state_tile_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_CUR_MM_OUT,
                    data_format=running_state_format,
                    page_size=running_state_tile_size,
                )
            ],
        ),
        # cb_output: Dt tiles per row, double-buffered.
        ttnn.CBDescriptor(
            total_size=2 * Dt * out_page_size,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_OUTPUT, data_format=output_tensor.dtype, page_size=out_page_size
                )
            ],
        ),
    ]

    # Refinement 6: untagged intermediate CBs that hold a copy of the
    # running-state accumulators for the final FPU divide. Only allocated
    # when fp32_dest_acc_en=True — that's the only path where the
    # running-state CBs get the UnpackToDestFp32 tag (so the FPU
    # mul_tiles_bcast_cols needs an untagged source). When
    # fp32_dest_acc_en=False the tags don't apply, the running-state CBs
    # stay FPU-readable, and the compute kernel takes the direct-divide
    # branch instead — no intermediate needed.
    if fp32_dest_acc_en:
        cbs.append(
            # cb_cur_mm_out_for_divide: single-buffered Dt-tile copy of
            # cb_cur_mm_out. Source for mul_tiles_bcast_cols in the final
            # divide. Written once at end-of-pass-2, consumed immediately,
            # so single-buffer is sufficient. At fp32 D=1024 this is
            # 128 KB — fits inside the ~187 KB headroom R5 left after
            # reclaiming the unused R4-iter3 CBs.
            ttnn.CBDescriptor(
                total_size=Dt * running_state_tile_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_CUR_MM_OUT_FOR_DIVIDE,
                        data_format=running_state_format,
                        page_size=running_state_tile_size,
                    )
                ],
            )
        )
        cbs.append(
            # cb_cur_sum_exp_for_divide: single-buffered 1-tile untagged
            # copy of cb_cur_sum_exp (post-reciprocal). 4 KB at fp32.
            ttnn.CBDescriptor(
                total_size=running_state_tile_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_CUR_SUM_EXP_FOR_DIVIDE,
                        data_format=running_state_format,
                        page_size=running_state_tile_size,
                    )
                ],
            )
        )

    # cb_attn_mask: declared whenever the kernel needs to see a mask CB.
    # Refinement 3: has_mask now covers BOTH the user-supplied mask and
    # the synthetic alignment mask (S_kv non-aligned), so this CB
    # appears whenever either path is active. CB format follows mask
    # tensor's dtype when a user mask is present; otherwise falls back
    # to q.dtype (set in mask_format above) so the kernel's
    # additive-mask path operates in the same precision as Q/K/V.
    if has_mask:
        cbs.append(
            ttnn.CBDescriptor(
                total_size=2 * mask_page_size,
                core_ranges=all_cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=CB_ATTN_MASK,
                        data_format=mask_format,
                        page_size=mask_page_size,
                    )
                ],
            )
        )

    # ---- Scaler bit-pattern for the compute kernel ----------------------
    scale_fp32_bits = struct.unpack("I", struct.pack("f", scale_value))[0]

    # ---- Per-core runtime-arg helpers -----------------------------------
    # split_work_to_cores returns two groups with potentially-different
    # rows-per-core values. Iterate both, assigning each core a
    # contiguous slice [start, start+rows).
    work_groups = [
        (core_group_1, rows_per_core_1),
        (core_group_2, rows_per_core_2),
    ]

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()

    start_row = 0
    q_addr = q.buffer_address()
    k_addr = k.buffer_address()
    v_addr = v.buffer_address()
    # Refinement 3: only resolve a mask buffer address when the USER
    # supplied a mask tensor. has_mask is broader now (also true for
    # synthetic-only alignment masking) — using it here would crash on
    # the attribute access when attention_mask is None.
    mask_addr = attention_mask.buffer_address() if has_user_mask else 0
    out_addr = output_tensor.buffer_address()

    for core_range_set, rows_per_core in work_groups:
        if rows_per_core == 0:
            continue
        for core_range in core_range_set.ranges():
            for x in range(core_range.start.x, core_range.end.x + 1):
                for y in range(core_range.start.y, core_range.end.y + 1):
                    reader_rt_args[x][y] = [
                        q_addr,
                        k_addr,
                        v_addr,
                        mask_addr,
                        rows_per_core,
                        start_row,
                    ]
                    writer_rt_args[x][y] = [
                        out_addr,
                        rows_per_core,
                        start_row,
                    ]
                    compute_rt_args[x][y] = [
                        rows_per_core,
                        start_row,
                    ]
                    start_row += rows_per_core

    # ---- Reader kernel --------------------------------------------------
    # Refinement 2: H_q and H_kv passed as separate CT args. Reader's
    # KV addressing uses h_kv = h_q / (H_q / H_kv); mha/mqa/gqa are
    # handled uniformly by that integer division (mha: group=1 →
    # identity; mqa: group=H_q → h_kv≡0).
    #
    # Refinement 3: HAS_MASK now also fires when the kernel needs to
    # inject a synthetic alignment mask (S_kv non-aligned), even if the
    # user passed no mask. HAS_USER_MASK is the narrower flag — the
    # reader only NoC-reads a mask tile when HAS_USER_MASK; otherwise it
    # zero-fills the mask CB. KEYS_IN_LAST_TILE drives the last-tile
    # -inf overlay path. MASK_ELEM_BYTES tells the overlay helper the
    # mask dtype's byte width (2 for bf16, 4 for fp32) so the right
    # -inf bit pattern lands at each padded position.
    # Mask dtype byte width — drives the reader's -inf overlay (2 for
    # bf16, 4 for fp32). bf8b is excluded for non-aligned cells
    # (EXCLUSIONS) so we don't need a 1-byte path.
    if mask_format == ttnn.float32:
        mask_elem_bytes = 4
    elif mask_format == ttnn.bfloat16:
        mask_elem_bytes = 2
    else:
        # Defensive fallback — Refinement 3 EXCLUSIONS reject bf8b +
        # non-aligned, so we should never hit this branch in a
        # SUPPORTED cell. Pick 2 (bf16) as a sensible default for
        # any tile-aligned bf8b + has_user_mask case where the
        # synthetic-mask path is inactive anyway.
        mask_elem_bytes = 2
    reader_ct_args = [
        B,
        H_q,
        H_kv,
        Qt,
        Kt,
        Dt,
        1 if has_mask else 0,
        1 if (has_user_mask and mask_per_head) else 0,
        1 if has_user_mask else 0,
        keys_in_last_tile,
        mask_elem_bytes,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(q).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(k).get_compile_time_args())
    reader_ct_args.extend(ttnn.TensorAccessorArgs(v).get_compile_time_args())
    if has_user_mask:
        reader_ct_args.extend(ttnn.TensorAccessorArgs(attention_mask).get_compile_time_args())
    else:
        reader_ct_args.extend(ttnn.TensorAccessorArgs().get_compile_time_args())

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ---- Writer kernel --------------------------------------------------
    # Writer's per-row decoding (r → b, h, qt) addresses the OUTPUT tensor,
    # which is (B, H_q, S_q, D) — so H here is H_q (unchanged from
    # Refinement 1; GQA/MQA only affects KV addressing on the reader side).
    writer_ct_args = [Dt, Qt, H_q]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ---- Compute kernel -------------------------------------------------
    # Refinement 6: USE_UNTAGGED_DIVIDE drives the final-divide branch
    # in the compute kernel. When fp32_dest_acc_en=True the running-state
    # CBs (cb_cur_sum_exp, cb_cur_mm_out) carry the UnpackToDestFp32 tag
    # so per-K-iter SFPU reloads preserve the full fp32 mantissa; the
    # final FPU mul_tiles_bcast_cols then reads from the untagged
    # cb_*_for_divide intermediates instead. When fp32_dest_acc_en=False
    # the tags don't apply and the kernel uses the direct-divide branch
    # (the running-state CBs are bf16-formatted and FPU-readable).
    compute_ct_args = [
        Dt,
        Kt,
        scale_fp32_bits,
        1 if has_mask else 0,
        1 if fp32_dest_acc_en else 0,  # USE_UNTAGGED_DIVIDE
    ]

    # Refinement 6: build the per-CB UnpackToDestMode vector. Indexed by
    # buffer_index (CB slot), length = NUM_CIRCULAR_BUFFERS (32 on
    # Wormhole). Only the two running-state CBs that are read by SFPU-
    # only consumers get UnpackToDestFp32; all other CBs stay Default
    # (any FPU reader is fine via the standard srcA/srcB TF32 path).
    # When fp32_dest_acc_en=False the running-state CBs are bf16, the
    # tag wouldn't buy anything anyway, and we leave them Default.
    NUM_CIRCULAR_BUFFERS = 32
    unpack_to_dest_mode = [ttnn.UnpackToDestMode.Default] * NUM_CIRCULAR_BUFFERS
    if fp32_dest_acc_en:
        unpack_to_dest_mode[CB_CUR_SUM_EXP] = ttnn.UnpackToDestMode.UnpackToDestFp32
        unpack_to_dest_mode[CB_CUR_MM_OUT] = ttnn.UnpackToDestMode.UnpackToDestFp32

    compute_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_dest_acc_en,
        math_approx_mode=math_approx_mode,
        dst_full_sync_en=dst_full_sync_en,
    )
    compute_config.unpack_to_dest_mode = unpack_to_dest_mode

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=compute_config,
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
