// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// vsa_sdpa compute (forked from sparse_sdpa_msa_compute): online softmax over chunks of up to m selected KV
// blocks per (head, 64-token query tile). Q arrives pre-tiled (no tilize) and the output leaves as tiles (no
// untilize). Ragged blocks are masked to -inf via count-derived partial-column tiles before the row-max
// reduce; a partial last chunk simply has a narrower runtime width (its CB tail tiles are never read).

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
// compute_streaming.hpp needs declarations from compute_common.hpp (mask helpers, reduce helpers,
// DEST_AUTO_LIMIT); include it first.
#include "compute_common.hpp"
#include "compute_streaming.hpp"
#include "api/dataflow/circular_buffer.h"  // CircularBuffer: COMPILE_FOR_TRISC-aware CB lifecycle
#include <tt-metalium/constants.hpp>       // tt::constants::TILE_HEIGHT

// Make in-place packer writes to a held CB visible to the next unpacker read.
ALWI void vsa_pack_to_unpack_sync() {
    PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));
    UNPACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE)));
    UNPACK((t6_semaphore_get<>(semaphore::PACK_DONE)));
}

ALWI void vsa_swap_cb(CircularBuffer& a, CircularBuffer& b) {
    const CircularBuffer t = a;
    a = b;
    b = t;
}

void kernel_main() {
    constexpr uint32_t DHt = get_compile_time_arg_val(0);
    constexpr uint32_t vDHt = get_compile_time_arg_val(1);
    constexpr uint32_t Skt = get_compile_time_arg_val(2);  // key tile-columns per block (2)
    constexpr uint32_t m = get_compile_time_arg_val(3);    // blocks per chunk
    constexpr uint32_t block_size = get_compile_time_arg_val(4);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(5);

    // CB ids match the factory's compute compile-arg block.
    constexpr uint32_t cb_q_in = get_compile_time_arg_val(6);
    constexpr uint32_t cb_k_in = get_compile_time_arg_val(7);
    constexpr uint32_t cb_v_in = get_compile_time_arg_val(8);
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
    constexpr uint32_t cb_ctrl = get_compile_time_arg_val(19);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(20);
    constexpr uint32_t cb_recip_scratch = get_compile_time_arg_val(21);
    constexpr uint32_t cb_neginf = get_compile_time_arg_val(22);
    constexpr uint32_t cb_vmask = get_compile_time_arg_val(23);
    constexpr uint32_t qsb = get_compile_time_arg_val(24);  // == Sqt: one query band

    constexpr uint32_t Sqt = qsb;               // 64-token query tile = 2 tile-rows; a single DEST band
    constexpr uint32_t KT_stride = m * Skt;     // cb_qk_im physical row width
    constexpr uint32_t keys_per_tile = tt::constants::TILE_WIDTH;
    constexpr uint32_t dst_size = compute_kernel_lib::DEST_AUTO_LIMIT;
    // sub_exp walks key-tile columns in DEST-sized steps (clamped to the runtime chunk width).
    constexpr uint32_t exp_sbw = (qsb * KT_stride <= dst_size) ? KT_stride : (dst_size / qsb);

    CircularBuffer q_in_cb(cb_q_in), k_in_cb(cb_k_in), v_in_cb(cb_v_in), qk_cb(cb_qk_im), scale_cb(cb_scale),
        ctrl_cb(cb_ctrl);
    CircularBuffer corr_cb(cb_corr);

    const uint32_t work_count = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q_in, cb_k_in, cb_qk_im);
    matmul_init(cb_q_in, cb_k_in);  // one-time matmul init; the no_mop matmuls reinit off this

    scale_cb.wait_front(1);                    // persistent reduce scaler (writer-built)
    CircularBuffer(cb_neginf).wait_front(1);   // persistent -inf mask tile (writer-built, never popped)

    for (uint32_t work = 0; work < work_count; ++work) {
        // Flash running state, ping-pong. Reset every row; all buffers start empty.
        CircularBuffer max_prev(cb_max_a), max_cur(cb_max_b);
        CircularBuffer sum_prev(cb_sum_a), sum_cur(cb_sum_b);
        CircularBuffer out_prev(cb_out_a), out_cur(cb_out_b);

        bool is_first = true;
        bool is_last = false;
        while (!is_last) {
            // Per-chunk control from the reader: valid block count, last flag, per-block valid token counts.
            ctrl_cb.wait_front(1);
            const uint32_t n_valid = ckernel::read_tile_value(cb_ctrl, /*tile=*/0, /*element_offset=*/0);
            is_last = ckernel::read_tile_value(cb_ctrl, /*tile=*/0, /*element_offset=*/1) != 0;
            uint32_t counts[m];
            uint32_t n_masks = 0;
            for (uint32_t b = 0; b < n_valid; ++b) {
                counts[b] = ckernel::read_tile_value(cb_ctrl, /*tile=*/0, /*element_offset=*/2 + b);
                if (counts[b] < block_size && (counts[b] % keys_per_tile) != 0) {
                    ++n_masks;
                }
            }
            ctrl_cb.pop_front(1);

            const uint32_t Skt_chunk = n_valid * Skt;  // runtime chunk width in key tiles

            reconfig_data_format(cb_k_in, cb_q_in);

            // cb_qk_im / sum_cur / out_cur span the full physical width; a partial chunk leaves tail columns
            // unwritten and unread. The K/V waits use the fixed batch the reader pushes.
            qk_cb.reserve_back(Sqt * KT_stride);
            sum_cur.reserve_back(Sqt);
            out_cur.reserve_back(Sqt * vDHt);
            k_in_cb.wait_front(m * Skt * DHt);
            v_in_cb.wait_front(m * Skt * vDHt);
            q_in_cb.wait_front(Sqt * DHt);  // Q reused across all chunks of this row

            // Set exp to the softmax scale; salad's correction below re-inits it to unit scale.
            exp_packthread_tile_init<true, scale_fp32, InputClamping::None>();

            // Phase 1: Q@K^T -> scores (columns [0, Skt_chunk)).
            {
                mm_no_mop_init_short(cb_q_in, cb_k_in, /*transpose=*/true, 1, qsb, DHt);
                pack_reconfig_data_format(cb_qk_im);
                configure_row_pack_width(cb_qk_im, 1);
                for (uint32_t kt = 0; kt < Skt_chunk; ++kt) {
                    blocked_matmul_and_pack<true, /*in1_stride=*/1, /*out_num_cols=*/KT_stride>(
                        cb_q_in,
                        cb_k_in,
                        cb_qk_im,
                        /*in0_index_start=*/0,
                        /*in1_index_start=*/kt * DHt,
                        /*row_subblock_idx=*/0,
                        /*out_col_offset=*/kt,
                        /*subblock_w=*/1,
                        /*subblock_h=*/qsb,
                        /*inner_dim=*/DHt,
                        /*matmul_stride=*/DHt,
                        /*skip_pack_configure=*/true);
                }
                // Publish the band to UNPACK while holding wr_ptr for in-place masking and sub_exp.
                cb_push_back_hold_wr_ptr(cb_qk_im, Sqt * KT_stride);
            }

            // Ragged-block masking BEFORE the row-max reduce: pad columns must never inflate the max, and they
            // map to exp(-inf) = 0 after sub_exp. Block b's key tiles sit at columns [b*Skt, (b+1)*Skt); its
            // boundary tile (count % 32 != 0) takes the reader-built partial tile at slot b, and every later
            // tile of the block is stamped fully -inf from the persistent tile. All stamps are L1-accumulated
            // adds onto finite scores.
            if (n_masks > 0) {
                CircularBuffer(cb_vmask).wait_front(m);
            }
            bool masked_any = false;
            for (uint32_t b = 0; b < n_valid; ++b) {
                const uint32_t count = counts[b];
                if (count >= block_size) {
                    continue;
                }
                const uint32_t col0 = b * Skt;
                const uint32_t boundary_tile = count / keys_per_tile;
                const uint32_t boundary_col = count % keys_per_tile;
                uint32_t first_full_masked = boundary_tile;
                if (boundary_col != 0) {
                    apply_partial_mask_lightweight(cb_vmask, b, cb_qk_im, col0 + boundary_tile, KT_stride, qsb, 0);
                    first_full_masked = boundary_tile + 1;
                }
                for (uint32_t kt = first_full_masked; kt < Skt; ++kt) {
                    apply_partial_mask_lightweight(cb_neginf, 0, cb_qk_im, col0 + kt, KT_stride, qsb, 0);
                }
                masked_any = true;
            }
            if (masked_any) {
                vsa_pack_to_unpack_sync();  // masked writes must be visible to the row-max reduce's UNPACK
            }

            {
                // Reduce/sub_exp read scores and bf16 scalers.
                reconfig_data_format(cb_qk_im, cb_scale);
                // running row-max (MAX-only; eltwise-max against prev on chunk > 0)
                max_cur.reserve_back(qsb);
                configure_single_tile_pack(max_cur.get_cb_id());
                reduce_c_row_group<cb_qk_im, cb_scale, KT_stride>(
                    max_cur.get_cb_id(),
                    max_prev.get_cb_id(),
                    /*row_group_index=*/0,
                    /*do_eltwise_max=*/!is_first,
                    qsb,
                    Skt_chunk);
                max_cur.push_back(qsb);

                // sub_exp in place: cb_qk_im = exp((cb_qk_im - max)*scale); partial row-sum -> sum_cur (L1-acc).
                for (uint32_t kc = 0; kc < Skt_chunk; kc += exp_sbw) {
                    const uint32_t sbw = (Skt_chunk - kc < exp_sbw) ? (Skt_chunk - kc) : exp_sbw;
                    sub_exp_block_bcast_cols<false, scale_fp32>(
                        cb_qk_im,
                        max_cur.get_cb_id(),
                        sum_cur.get_cb_id(),
                        /*cols_in_row=*/KT_stride,
                        /*q_subblock=*/0,
                        /*global_col_base=*/kc,
                        /*sbh=*/qsb,
                        /*sbw=*/sbw);
                }
                vsa_pack_to_unpack_sync();  // sub_exp writes must be visible to the V matmul
            }

            // Phase 2: probs@V -> current output.
            {
                qk_cb.wait_front(Sqt * KT_stride);
                reconfig_data_format(cb_v_in, cb_qk_im);
                // kt_dim here is in0's PHYSICAL row width (the unpacker's per-row address stride), not the
                // runtime chunk width -- they differ on a partial chunk and whenever m > 1.
                mm_no_mop_init_short(cb_qk_im, cb_v_in, /*transpose=*/false, 1, qsb, KT_stride);
                configure_row_pack_width(out_cur.get_cb_id(), 1);
                for (uint32_t vd = 0; vd < vDHt; ++vd) {
                    blocked_matmul_and_pack<false, /*in1_stride=*/vDHt, /*out_num_cols=*/vDHt>(
                        cb_qk_im,
                        cb_v_in,
                        out_cur.get_cb_id(),
                        /*in0_index_start=*/0,
                        /*in1_index_start=*/vd,
                        /*row_subblock_idx=*/0,
                        /*out_col_offset=*/vd,
                        /*subblock_w=*/1,
                        /*subblock_h=*/qsb,
                        /*inner_dim=*/Skt_chunk,
                        /*matmul_stride=*/KT_stride,
                        /*skip_pack_configure=*/true);
                }
                vsa_pack_to_unpack_sync();            // publish held out_cur packs before the flash combine
                reconfig_data_format_srca(cb_qk_im);  // PV left srcA in cb_v_in's format; restore bf16
            }

            // ===== SALAD flash combine (skip on the first chunk) =====
            if (!is_first) {
                // correction = exp((prev_max - cur_max) * scale)
                exp_packthread_tile_init<EXP_APPROX_MODE>();
                corr_cb.reserve_back(qsb);
                sub_exp_first_col_blocks<false, scale_fp32>(
                    max_prev.get_cb_id(), max_cur.get_cb_id(), cb_corr, /*q_subblock=*/0, qsb);
                corr_cb.push_back(qsb);
                // Restore default packer geometry before the fused flash correction.
                PACK((llk_pack_init<ckernel::PackMode::Default, false, false, false>(out_cur.get_cb_id(), dst_size)));
                pack_reconfig_l1_acc(1);
                salad_correct_fused<qsb, vDHt, dst_size>(
                    out_prev.get_cb_id(),
                    sum_prev.get_cb_id(),
                    cb_corr,
                    out_cur.get_cb_id(),
                    sum_cur.get_cb_id(),
                    /*ob_q_subblock=*/0,
                    /*sum_q_subblock=*/0,
                    /*write_q_subblock=*/0);
                pack_reconfig_l1_acc(0);
                corr_cb.pop_front(qsb);
                out_prev.pop_front(qsb * vDHt);
                max_prev.pop_front(Sqt);
                sum_prev.pop_front(Sqt);
            }

            // Release the chunk's mask batch once its scores are consumed downstream of the reduce.
            if (n_masks > 0) {
                CircularBuffer(cb_vmask).pop_front(m);
            }

            // Publish cur.sum / cur.out (running state for the next chunk, or the final result).
            sum_cur.push_back(Sqt);
            out_cur.push_back(Sqt * vDHt);

            if (is_last) {
                // Finalize: reciprocal row sum, then out *= 1/sum; tiles land in cb_out_im for the writer.
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
            k_in_cb.pop_front(m * Skt * DHt);
            v_in_cb.pop_front(m * Skt * vDHt);

            vsa_swap_cb(max_prev, max_cur);
            vsa_swap_cb(sum_prev, sum_cur);
            vsa_swap_cb(out_prev, out_cur);
            is_first = false;
        }

        q_in_cb.pop_front(Sqt * DHt);  // Q reused across all chunks; drop it so >1 row/core stays clean
    }
}
