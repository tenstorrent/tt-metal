// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// sparse_sdpa_msa compute: online softmax over selected pre-tiled K/V blocks. Each token tilizes Q, streams
// selected blocks through QK and PV, combines running max/sum/output, then normalizes the final output.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
// compute_streaming.hpp needs declarations from compute_common.hpp (LightweightMaskContext, reduce helpers,
// DEST_AUTO_LIMIT); include it first. Only compute_streaming primitives are used.
#include "compute_common.hpp"
#include "compute_streaming.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "api/dataflow/circular_buffer.h"  // CircularBuffer: COMPILE_FOR_TRISC-aware CB lifecycle
#include <tt-metalium/constants.hpp>       // tt::constants::TILE_HEIGHT

// Make in-place packer writes to a held CB visible to the next unpacker read.
ALWI void pack_to_unpack_sync() {
    PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));
    UNPACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE)));
    UNPACK((t6_semaphore_get<>(semaphore::PACK_DONE)));
}

ALWI void swap_cb(CircularBuffer& a, CircularBuffer& b) {
    const CircularBuffer t = a;
    a = b;
    b = t;
}

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t DHt = get_compile_time_arg_val(1);
    constexpr uint32_t vDHt = get_compile_time_arg_val(2);
    constexpr uint32_t Skt = get_compile_time_arg_val(3);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(4);

    // CB ids match the factory's compute compile-arg block.
    constexpr uint32_t cb_q_rm = get_compile_time_arg_val(5);
    constexpr uint32_t cb_q_in = get_compile_time_arg_val(6);
    constexpr uint32_t cb_k_in = get_compile_time_arg_val(7);  // K tiled [Skt, DHt] (reader-filled, pre-tiled)
    constexpr uint32_t cb_v_in = get_compile_time_arg_val(8);  // V tiled [Skt, vDHt] (reader-filled, pre-tiled)
    constexpr uint32_t cb_scale = get_compile_time_arg_val(9);
    constexpr uint32_t cb_qk_im = get_compile_time_arg_val(10);
    constexpr uint32_t cb_max_a = get_compile_time_arg_val(11);
    constexpr uint32_t cb_max_b = get_compile_time_arg_val(12);
    constexpr uint32_t cb_sum_a = get_compile_time_arg_val(13);
    constexpr uint32_t cb_sum_b = get_compile_time_arg_val(14);
    constexpr uint32_t cb_out_a = get_compile_time_arg_val(15);
    constexpr uint32_t cb_out_b = get_compile_time_arg_val(16);
    constexpr uint32_t cb_corr = get_compile_time_arg_val(17);
    constexpr uint32_t cb_out_im = get_compile_time_arg_val(18);
    constexpr uint32_t cb_out_rm = get_compile_time_arg_val(19);
    constexpr uint32_t cb_ctrl = get_compile_time_arg_val(20);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(21);
    constexpr uint32_t cb_recip_scratch = get_compile_time_arg_val(22);

    constexpr uint32_t qsb = get_compile_time_arg_val(23);    // query tile-rows per DST group (<= dst_size)
    // Causal masking (token-level diagonal-block mask).
    constexpr bool CAUSAL_MASK_ENABLED = get_compile_time_arg_val(24) != 0;
    constexpr uint32_t cb_neginf = get_compile_time_arg_val(25);  // persistent all -inf tile (future key-tiles)
    constexpr uint32_t cb_vmask = get_compile_time_arg_val(26);   // per-token partial-column boundary tile
    constexpr uint32_t Sqt = H / tt::constants::TILE_HEIGHT;  // total query tile-rows (32 heads each)
    constexpr uint32_t q_groups = Sqt / qsb;                  // DST-bound work runs in this many query-row passes
    constexpr uint32_t KT_stride = Skt;                       // cb_qk_im physical row width
    constexpr uint32_t dst_size = compute_kernel_lib::DEST_AUTO_LIMIT;
    // sub_exp uses the full key width when it fits in DEST, otherwise one key-tile column at a time.
    constexpr uint32_t exp_sbw = (qsb * Skt <= dst_size) ? Skt : 1;

    // CB wrappers for the fixed (non-ping-pong) buffers' lifecycle verbs.
    CircularBuffer q_in_cb(cb_q_in), k_in_cb(cb_k_in), v_in_cb(cb_v_in), qk_cb(cb_qk_im), scale_cb(cb_scale),
        ctrl_cb(cb_ctrl);
    CircularBuffer corr_cb(cb_corr);

    const uint32_t tok_count = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q_in, cb_k_in, cb_qk_im);
    matmul_init(cb_q_in, cb_k_in);  // one-time matmul init; the no_mop matmuls reinit off this

    scale_cb.wait_front(1);  // persistent reduce scaler; the streaming reduce assumes it is ready
    if constexpr (CAUSAL_MASK_ENABLED) {
        CircularBuffer(cb_neginf).wait_front(1);  // persistent -inf mask tile; built once by the writer, never popped
    }

    for (uint32_t tok = 0; tok < tok_count; ++tok) {
        // tilize Q rows -> [Sqt, DHt]
        compute_kernel_lib::tilize<DHt, cb_q_rm, cb_q_in>(/*num_blocks=*/Sqt, /*total_input_pages=*/H);
        // fp8 Q tilize leaves the packer in bfp8 format; restore bf16 before QK writes scores.
        pack_reconfig_data_format(cb_qk_im);

        // Per-token control from the reader: active block count, and (causal) the diagonal block's chunk
        // index + within-block mask boundary.
        ctrl_cb.wait_front(1);
        const uint32_t num_active_chunks = ckernel::read_tile_value(cb_ctrl, /*tile=*/0, /*element_offset=*/0);
        [[maybe_unused]] uint32_t diag_chunk = 0xFFFFFFFFu;
        [[maybe_unused]] uint32_t boundary_tile = 0;
        [[maybe_unused]] uint32_t boundary_col = 0;
        if constexpr (CAUSAL_MASK_ENABLED) {
            diag_chunk = ckernel::read_tile_value(cb_ctrl, /*tile=*/0, /*element_offset=*/1);
            boundary_tile = ckernel::read_tile_value(cb_ctrl, /*tile=*/0, /*element_offset=*/2);
            boundary_col = ckernel::read_tile_value(cb_ctrl, /*tile=*/0, /*element_offset=*/3);
        }
        ctrl_cb.pop_front(1);
        // vmask is pushed by the reader exactly when the boundary key-tile is split (boundary_col > 0).
        [[maybe_unused]] const bool diag_has_vmask =
            CAUSAL_MASK_ENABLED && (diag_chunk != 0xFFFFFFFFu) && (boundary_col > 0);

        // Flash running state, ping-pong. Reset every token; all buffers start empty.
        CircularBuffer max_prev(cb_max_a), max_cur(cb_max_b);
        CircularBuffer sum_prev(cb_sum_a), sum_cur(cb_sum_b);
        CircularBuffer out_prev(cb_out_a), out_cur(cb_out_b);

        for (uint32_t chunk = 0; chunk < num_active_chunks; ++chunk) {
            // K/V are already tiled. Set source formats for QK; scores remain bf16.
            reconfig_data_format(cb_k_in, cb_q_in);

            const bool is_first = (chunk == 0);
            const bool is_last = (chunk == num_active_chunks - 1);

            // The diagonal block needs a token-level causal mask. Front the per-token boundary tile once.
            if constexpr (CAUSAL_MASK_ENABLED) {
                if ((chunk == diag_chunk) && diag_has_vmask) {
                    CircularBuffer(cb_vmask).wait_front(1);
                }
            }

            // cb_qk_im / sum_cur / out_cur span all Sqt rows; the qg loop fills them band-by-band (sub_exp & PV
            // pack at absolute row offsets; reduce/corr reserve+push per band).
            qk_cb.reserve_back(Sqt * KT_stride);
            sum_cur.reserve_back(Sqt);
            out_cur.reserve_back(Sqt * vDHt);
            k_in_cb.wait_front(Skt * DHt);   // K: shared by every query group (QK)
            v_in_cb.wait_front(Skt * vDHt);  // V: shared by every query group (PV)
            q_in_cb.wait_front(Sqt * DHt);

            // DST holds qsb query tile-rows; process Sqt rows in q_groups passes (one pass when qsb==Sqt).
            for (uint32_t qg = 0; qg < q_groups; ++qg) {
                const uint32_t row_base = qg * qsb;  // first query tile-row of this group

                // Set exp to the softmax scale; salad's correction below re-inits it to unit scale.
                exp_packthread_tile_init<true, scale_fp32, InputClamping::None>();

                // Phase 1: Q@K^T -> scores and running row max.
                {
                    // Use pack width 1 because Q tilize changes packer addrmods; wider packs need extra reinit.
                    mm_no_mop_init_short(cb_q_in, cb_k_in, /*transpose=*/true, 1, qsb, DHt);
                    configure_row_pack_width(cb_qk_im, 1);
                    for (uint32_t kt = 0; kt < Skt; ++kt) {
                        blocked_matmul_and_pack<true, /*in1_stride=*/1, /*out_num_cols=*/KT_stride>(
                            cb_q_in,
                            cb_k_in,
                            cb_qk_im,
                            /*in0_index_start=*/row_base * DHt,
                            /*in1_index_start=*/kt * DHt,
                            /*row_subblock_idx=*/qg,
                            /*out_col_offset=*/kt,
                            /*subblock_w=*/1,
                            /*subblock_h=*/qsb,
                            /*inner_dim=*/DHt,
                            /*matmul_stride=*/DHt,
                            /*skip_pack_configure=*/true);
                    }
                    // Publish the band to UNPACK while holding wr_ptr for in-place sub_exp.
                    cb_push_back_hold_wr_ptr(cb_qk_im, qsb * KT_stride);
                }

                // Causal mask on the diagonal block: -inf the future-token columns (a contiguous tail beyond
                // the query position) BEFORE the row-max reduce, so they never inflate the max and map to ~0
                // after sub_exp. All score rows are heads sharing one query position -> a pure column mask.
                if constexpr (CAUSAL_MASK_ENABLED) {
                    if (chunk == diag_chunk) {
                        if (boundary_col > 0) {  // boundary key-tile is split: partial-column mask
                            apply_partial_mask_lightweight(
                                cb_vmask, 0, cb_qk_im, boundary_tile, KT_stride, qsb, row_base);
                        }
                        // Key-tiles entirely after the boundary are fully future -> stamp -inf.
                        const uint32_t full_neginf =
                            (boundary_col > 0) ? (Skt - boundary_tile - 1) : (Skt - boundary_tile);
                        if (full_neginf > 0) {
                            apply_padded_mask_lightweight_runtime<dst_size>(
                                cb_neginf, 0, cb_qk_im, full_neginf, KT_stride, qsb, row_base);
                        }
                        pack_to_unpack_sync();  // masked writes must be visible to the row-max reduce's UNPACK
                    }
                }

                {
                    // Reduce/sub_exp read scores and bf16 scalers.
                    reconfig_data_format(cb_qk_im, cb_scale);
                    // running row-max (MAX-only; eltwise-max against prev on chunk>0)
                    max_cur.reserve_back(qsb);
                    configure_single_tile_pack(max_cur.get_cb_id());
                    reduce_c_row_group<cb_qk_im, cb_scale, KT_stride>(
                        max_cur.get_cb_id(),
                        max_prev.get_cb_id(),
                        /*row_group_index=*/qg,
                        /*do_eltwise_max=*/!is_first,
                        qsb,
                        Skt);
                    max_cur.push_back(qsb);

                    // sub_exp in place: cb_qk_im = exp((cb_qk_im - max)*scale); partial row-sum -> sum_cur (L1-acc).
                    // Walk key-tile columns in DEST-sized steps.
                    for (uint32_t kc = 0; kc < Skt; kc += exp_sbw) {
                        sub_exp_block_bcast_cols<false, scale_fp32>(
                            cb_qk_im,
                            max_cur.get_cb_id(),
                            sum_cur.get_cb_id(),
                            /*cols_in_row=*/KT_stride,
                            /*q_subblock=*/qg,
                            /*global_col_base=*/kc,
                            /*sbh=*/qsb,
                            /*sbw=*/exp_sbw);
                    }
                    pack_to_unpack_sync();  // sub_exp writes must be visible to V-matmul
                }

                // Phase 2: probs@V -> current output band.
                {
                    qk_cb.wait_front((qg + 1) * qsb * KT_stride);
                    // PV reads V as srcA and probabilities as srcB.
                    reconfig_data_format(cb_v_in, cb_qk_im);
                    mm_no_mop_init_short(cb_qk_im, cb_v_in, /*transpose=*/false, 1, qsb, Skt);
                    configure_row_pack_width(out_cur.get_cb_id(), 1);
                    for (uint32_t vd = 0; vd < vDHt; ++vd) {
                        blocked_matmul_and_pack<false, /*in1_stride=*/vDHt, /*out_num_cols=*/vDHt>(
                            cb_qk_im,
                            cb_v_in,
                            out_cur.get_cb_id(),
                            /*in0_index_start=*/row_base * Skt,
                            /*in1_index_start=*/vd,
                            /*row_subblock_idx=*/qg,
                            /*out_col_offset=*/vd,
                            /*subblock_w=*/1,
                            /*subblock_h=*/qsb,
                            /*inner_dim=*/Skt,
                            /*matmul_stride=*/KT_stride,
                            /*skip_pack_configure=*/true);
                    }
                    pack_to_unpack_sync();                // publish held out_cur packs before flash combine
                    reconfig_data_format_srca(cb_qk_im);  // PV left srcA in cb_v_in's format; restore bf16
                }

                // ===== SALAD flash combine (skip on the first chunk) =====
                if (!is_first) {
                    // correction = exp((prev_max - cur_max) * scale)
                    exp_packthread_tile_init<EXP_APPROX_MODE>();
                    corr_cb.reserve_back(qsb);
                    sub_exp_first_col_blocks<false, scale_fp32>(
                        max_prev.get_cb_id(), max_cur.get_cb_id(), cb_corr, /*q_subblock=*/qg, qsb);
                    corr_cb.push_back(qsb);
                    // Restore default packer geometry before the fused flash correction.
                    PACK((
                        llk_pack_init<ckernel::PackMode::Default, false, false, false>(out_cur.get_cb_id(), dst_size)));
                    pack_reconfig_l1_acc(1);
                    // out_prev and corr are consumed per group; sum_prev is indexed by group.
                    salad_correct_fused<qsb, vDHt, dst_size>(
                        out_prev.get_cb_id(),
                        sum_prev.get_cb_id(),
                        cb_corr,
                        out_cur.get_cb_id(),
                        sum_cur.get_cb_id(),
                        /*ob_q_subblock=*/0,
                        /*sum_q_subblock=*/qg,
                        /*write_q_subblock=*/qg);
                    pack_reconfig_l1_acc(0);
                    corr_cb.pop_front(qsb);
                    out_prev.pop_front(qsb * vDHt);  // this band's prev.out consumed into cur
                }
            }  // query groups

            // Release the per-token boundary mask tile once the diagonal chunk's bands are done.
            if constexpr (CAUSAL_MASK_ENABLED) {
                if ((chunk == diag_chunk) && diag_has_vmask) {
                    CircularBuffer(cb_vmask).pop_front(1);
                }
            }

            // prev max/sum consumed into cur (out_prev popped per band above).
            if (!is_first) {
                max_prev.pop_front(Sqt);
                sum_prev.pop_front(Sqt);
            }

            // Publish cur.sum / cur.out (running state for the next chunk, or the final result).
            sum_cur.push_back(Sqt);
            out_cur.push_back(Sqt * vDHt);

            if (is_last) {
                // Finalize: reciprocal row sum, then out *= 1/sum.
                normalize_row_streaming<
                    /*profiling_enabled=*/false,
                    vDHt,
                    dst_size,
                    cb_col_identity,
                    cb_recip_scratch,
                    cb_out_im,
                    scale_fp32>(sum_cur.get_cb_id(), out_cur.get_cb_id(), Sqt);
                max_cur.pop_front(Sqt);  // running max no longer needed
            }

            // Release the held cb_qk_im rows + this chunk's K (QK) and V (PV).
            qk_cb.pop_front(Sqt * KT_stride);
            k_in_cb.pop_front(Skt * DHt);
            v_in_cb.pop_front(Skt * vDHt);

            swap_cb(max_prev, max_cur);  // prev <-> cur for the next chunk
            swap_cb(sum_prev, sum_cur);
            swap_cb(out_prev, out_cur);
        }

        q_in_cb.pop_front(Sqt * DHt);  // Q reused across all chunks; drop it so >1 token/core stays clean

        // cb_out_im was written by normalize_row_streaming; untilize -> row-major out for the writer.
        compute_kernel_lib::untilize<vDHt, cb_out_im, cb_out_rm>(/*num_blocks=*/Sqt);
    }
}
