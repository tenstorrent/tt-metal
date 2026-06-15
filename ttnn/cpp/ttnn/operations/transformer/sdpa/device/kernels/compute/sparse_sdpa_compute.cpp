// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// sparse_sdpa compute (compute_streaming primitives, flash/online-softmax over k_chunks).
// Per token: tilize Q. Per chunk: tilize K into the natural [Skt,DHt] layout (both matmuls read it
// directly — no Kᵀ/V copies), then run the streaming Phase-1/Phase-2 inner loop over the Sqt = H/32
// query tile-rows (32 heads each) as one subblock:
//   Phase 1: Q@Kᵀ -> cb_qk_im (held wr_ptr), mask add (row-broadcast), running row-max.
//   Phase 2: sub_exp in place (exp((s-max)*scale)) + partial row-sum (L1-acc into cur.sum),
//            probs@V -> cur.out; then the SALAD flash combine (exp(prev_max-cur_max) correction
//            applied to prev.out and prev.sum, L1-accumulated into cur.out/cur.sum).
// The partial (per-key-column) row-sum is carried across chunks and finalized once on the last
// chunk by normalize_row_streaming (matmul_reduce vs a col-identity ones-vector + recip + scale).
// num_active_chunks==1 (single active chunk) degenerates to a plain single-chunk softmax (no SALAD).

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/bcast.h"  // add_tiles_bcast_rows (mask add)
// compute_streaming.hpp is not self-contained: it relies on declarations from compute_common.hpp
// (LightweightMaskContext, reduce_block_max_row_*, sdpa_reduce_copy_tile_to_dst_init_short, ...) and
// compute_kernel_lib::DEST_AUTO_LIMIT. Include it first. The kernel uses only compute_streaming
// primitives — no compute_common compute primitives.
#include "compute_common.hpp"
#include "compute_streaming.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include <tt-metalium/constants.hpp>           // tt::constants::TILE_HEIGHT
#include <tools/profiler/kernel_profiler.hpp>  // DeviceZoneScopedN

// In-place scores[score_tile] += bcast_rows(mask_cb row 0): broadcast mask row 0 across all 32 query
// rows and add to one score tile. cb_qk_im is visible (pushed via hold_wr_ptr) with wr_ptr held, so we
// re-pack in place at the absolute position. Caller has init'd the bcast-rows op + pack width for scores.
ALWI void add_bcast_row_mask_tile(uint32_t scores_cb, uint32_t mask_cb, uint32_t score_tile) {
    tile_regs_acquire();
    add_tiles_bcast_rows(scores_cb, mask_cb, score_tile, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(0, scores_cb, score_tile);
    tile_regs_release();
}

// PACK->UNPACK handshake: make in-place PACK writes to a held CB visible to the next UNPACK read.
// Needed wherever we write cb_qk_im in place (mask, sub_exp) or out_cur (V matmul) without a
// cb_push_back to order the following reader.
ALWI void pack_to_unpack_sync() {
    PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));
    UNPACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE)));
    UNPACK((t6_semaphore_get<>(semaphore::PACK_DONE)));
}

ALWI void swap_cb(uint32_t& a, uint32_t& b) {
    const uint32_t t = a;
    a = b;
    b = t;
}

void kernel_main() {
    constexpr uint32_t H = get_compile_time_arg_val(0);
    constexpr uint32_t DHt = get_compile_time_arg_val(1);
    constexpr uint32_t vDHt = get_compile_time_arg_val(2);
    constexpr uint32_t Skt = get_compile_time_arg_val(3);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(4);

    // CB ids — passed by the factory (the SparseCB enum is the single source); order must match the
    // factory's compute CB-arg block (compile args 5..24).
    constexpr uint32_t cb_q_rm = get_compile_time_arg_val(5);
    constexpr uint32_t cb_q_in = get_compile_time_arg_val(6);
    constexpr uint32_t cb_k_rm = get_compile_time_arg_val(7);
    constexpr uint32_t cb_k_in = get_compile_time_arg_val(8);
    constexpr uint32_t cb_neginf = get_compile_time_arg_val(9);
    constexpr uint32_t cb_mask_part = get_compile_time_arg_val(10);
    constexpr uint32_t cb_scale = get_compile_time_arg_val(11);
    constexpr uint32_t cb_qk_im = get_compile_time_arg_val(12);
    constexpr uint32_t cb_max_a = get_compile_time_arg_val(13);
    constexpr uint32_t cb_max_b = get_compile_time_arg_val(14);
    constexpr uint32_t cb_sum_a = get_compile_time_arg_val(15);
    constexpr uint32_t cb_sum_b = get_compile_time_arg_val(16);
    constexpr uint32_t cb_out_a = get_compile_time_arg_val(17);
    constexpr uint32_t cb_out_b = get_compile_time_arg_val(18);
    constexpr uint32_t cb_corr = get_compile_time_arg_val(19);
    constexpr uint32_t cb_out_im = get_compile_time_arg_val(20);
    constexpr uint32_t cb_out_rm = get_compile_time_arg_val(21);
    constexpr uint32_t cb_ctrl = get_compile_time_arg_val(22);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(23);
    constexpr uint32_t cb_recip_scratch = get_compile_time_arg_val(24);

    constexpr uint32_t Sqt = H / tt::constants::TILE_HEIGHT;  // query tile-rows (32 heads each); one subblock (<=8)
    constexpr uint32_t KT_stride = Skt;                       // cb_qk_im physical row width
    constexpr uint32_t dst_size = compute_kernel_lib::DEST_AUTO_LIMIT;
    // sub_exp packs sbh*sbw tiles into DST; use the full Skt width only when the whole row group fits, else
    // walk one key-tile column at a time so DST never overflows for Sqt>1.
    constexpr uint32_t exp_sbw = (Sqt * Skt <= dst_size) ? Skt : 1;

    const uint32_t tok_count = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(cb_q_rm, cb_q_in);
    mm_init(cb_q_in, cb_k_in, cb_out_im);  // one-time full matmul init; the no_mop matmuls reinit off this

    cb_wait_front(cb_scale, 1);  // reduce identity scaler is persistent; streaming reduce assumes it's ready

    for (uint32_t tok = 0; tok < tok_count; ++tok) {
        // --- tilize Q rows -> q_tiles [Sqt, DHt] ---
        compute_kernel_lib::tilize<DHt, cb_q_rm, cb_q_in>(/*num_blocks=*/Sqt, /*total_input_pages=*/H);

        // Per-token control from the reader: num_active_chunks (>=1; reader skips all-sentinel chunks) and
        // num_valid_keys (locates the last chunk's boundary mask). read_tile_value mailbox-distributes the
        // UNPACK read so all three compute threads see identical values — they drive the loop bound + mask
        // branches and must match across threads.
        cb_wait_front(cb_ctrl, 1);
        const uint32_t num_active_chunks = ckernel::read_tile_value(cb_ctrl, /*tile=*/0, /*element_offset=*/0);
        const uint32_t num_valid_keys = ckernel::read_tile_value(cb_ctrl, /*tile=*/0, /*element_offset=*/1);
        cb_pop_front(cb_ctrl, 1);

        // Flash running-state CB handles (ping-pong). Reset every token; all buffers start empty.
        uint32_t max_prev = cb_max_a;
        uint32_t max_cur = cb_max_b;
        uint32_t sum_prev = cb_sum_a;
        uint32_t sum_cur = cb_sum_b;
        uint32_t out_prev = cb_out_a;
        uint32_t out_cur = cb_out_b;

        for (uint32_t chunk = 0; chunk < num_active_chunks; ++chunk) {
            // --- tilize K rows -> k_tiles [Skt, DHt] (waits on reader cb_k_rm -> absorbs K-read stall) ---
            {
                DeviceZoneScopedN("cmp_tilizeK");
                compute_kernel_lib::tilize<DHt, cb_k_rm, cb_k_in>(
                    /*num_blocks=*/Skt, /*total_input_pages=*/Skt * tt::constants::TILE_HEIGHT);
            }

            // No grid-reposition: both matmuls read the natural tilized K[Skt,DHt] directly. QK@Kᵀ uses
            // the matmul's within-tile transpose with in1 walking features inside a key-row (stride 1);
            // PV@V walks keys (stride DHt) and only reads the first vDHt feature cols (rope cols skipped).

            const bool is_first = (chunk == 0);
            const bool is_last = (chunk == num_active_chunks - 1);
            // Boundary mask applies only to the last active chunk (only it can hold sentinels).
            const bool has_mask = is_last;

            // Per-chunk exp init + reserve the score / sum buffers (mirrors streaming Phase-1 setup).
            exp_packthread_tile_init<true, scale_fp32, InputClamping::None>();
            cb_reserve_back(cb_qk_im, Sqt * KT_stride);
            cb_reserve_back(sum_cur, Sqt);

            // ===== Phase 1: Q@Kᵀ -> cb_qk_im, mask, running row-max =====
            {
                DeviceZoneScopedN("cmp_QK");
                cb_wait_front(cb_k_in, Skt * DHt);
                cb_wait_front(cb_q_in, Sqt * DHt);
                // scores[q,sk] = sum_d Q[q,d]·K[sk,d]. in1 = natural K[Skt,DHt], transpose=true (within-tile
                // transpose makes K[sk,d] act as Kᵀ[d,sk]); inner walks features in key-row sk (in1_stride=1).
                // One column per matmul (ct/pack_width=1): the wider BH blocked pack is corrupted by the
                // per-chunk tilize (it reprograms the packer addrmods/strides; configure_pack_width can't
                // restore them). matmul ct>1 itself is fine.
                mm_no_mop_init_short(cb_q_in, cb_k_in, /*transpose=*/true, 1, Sqt, DHt);
                configure_row_pack_width(cb_qk_im, 1);
                for (uint32_t kt = 0; kt < Skt; ++kt) {
                    blocked_matmul_and_pack<true, /*in1_stride=*/1, /*out_num_cols=*/KT_stride>(
                        cb_q_in,
                        cb_k_in,
                        cb_qk_im,
                        /*in0_index_start=*/0,
                        /*in1_index_start=*/kt * DHt,
                        /*row_subblock_idx=*/0,
                        /*out_col_offset=*/kt,
                        /*subblock_w=*/1,
                        /*subblock_h=*/Sqt,
                        /*inner_dim=*/DHt,
                        /*matmul_stride=*/DHt,
                        /*trigger_reduce=*/false,
                        /*skip_pack_configure=*/true);
                }
                // Make the matmul output visible to UNPACK while holding wr_ptr (in-place mask/sub_exp).
                cb_push_back_hold_wr_ptr(cb_qk_im, Sqt * KT_stride);
            }

            {
                DeviceZoneScopedN("cmp_softmax");
                if (has_mask) {
                    // Boundary: keys [valid_last, k_chunk) of this last chunk are sentinels -> -inf. Split
                    // into fully-masked key tiles [full_start, Skt) (bcast-add the persistent all-(-inf)
                    // tile) and, if the boundary lands mid-tile, the straddling tile (bcast-add the
                    // reader-built cb_mask_part). valid_last>=1, so tile 0 is never fully masked.
                    const uint32_t k_chunk = Skt * tt::constants::TILE_WIDTH;
                    const uint32_t valid_last = num_valid_keys - (num_active_chunks - 1) * k_chunk;
                    const uint32_t full_start =
                        (valid_last + tt::constants::TILE_WIDTH - 1) / tt::constants::TILE_WIDTH;
                    cb_wait_front(cb_qk_im, Sqt * KT_stride);  // QK output visible before in-place mask add
                    // The mask is the same for every query row, so stamp each masked key tile across all
                    // Sqt rows (cb_qk_im row r, key tile t = r*Skt + t).
                    if (full_start < Skt) {
                        cb_wait_front(cb_neginf, 1);
                        add_bcast_rows_init_short(cb_qk_im, cb_neginf);
                        configure_single_tile_pack(cb_qk_im);
                        for (uint32_t r = 0; r < Sqt; ++r) {
                            for (uint32_t t = full_start; t < Skt; ++t) {
                                add_bcast_row_mask_tile(cb_qk_im, cb_neginf, r * Skt + t);
                            }
                        }
                    }
                    if (valid_last % tt::constants::TILE_WIDTH != 0) {
                        const uint32_t part_t = valid_last / tt::constants::TILE_WIDTH;
                        cb_wait_front(cb_mask_part, 1);
                        add_bcast_rows_init_short(cb_qk_im, cb_mask_part);
                        configure_single_tile_pack(cb_qk_im);
                        for (uint32_t r = 0; r < Sqt; ++r) {
                            add_bcast_row_mask_tile(cb_qk_im, cb_mask_part, r * Skt + part_t);
                        }
                        cb_pop_front(cb_mask_part, 1);
                    }
                    // The in-place mask PACK writes must be visible to the row-max UNPACK below.
                    pack_to_unpack_sync();
                }
                // running row-max (MAX-only; eltwise-max against prev on chunk>0)
                cb_reserve_back(max_cur, Sqt);
                configure_single_tile_pack(max_cur);
                reduce_c_row_group<cb_qk_im, cb_scale, KT_stride>(
                    max_cur, max_prev, /*row_group_index=*/0, /*do_eltwise_max=*/!is_first, Sqt, Skt);
                cb_push_back(max_cur, Sqt);

                // sub_exp in place: cb_qk_im = exp((cb_qk_im - max)*scale); partial row-sum -> sum_cur (L1-acc).
                // Walk key-tile columns in exp_sbw-wide steps (1 step when the row group fits DST); the
                // L1-accumulated partial sum builds up across steps (global_col_base drives the accumulate).
                for (uint32_t kc = 0; kc < Skt; kc += exp_sbw) {
                    sub_exp_block_bcast_cols<false, scale_fp32>(
                        cb_qk_im,
                        max_cur,
                        sum_cur,
                        /*cols_in_row=*/KT_stride,
                        /*q_subblock=*/0,
                        /*global_col_base=*/kc,
                        /*sbh=*/Sqt,
                        /*sbw=*/exp_sbw);
                }

                // sub_exp's in-place PACK writes must be visible to the V-matmul UNPACK.
                pack_to_unpack_sync();
            }

            // ===== Phase 2: probs@V -> cur.out =====
            {
                DeviceZoneScopedN("cmp_PV");
                cb_wait_front(cb_qk_im, Sqt * KT_stride);
                cb_wait_front(cb_k_in, Skt * DHt);
                cb_reserve_back(out_cur, Sqt * vDHt);
                // out[q,vd] = sum_sk probs[q,sk]·K[sk,vd], V = first vDHt feature cols (rope cols skipped).
                // in1 = natural K[Skt,DHt], transpose=false, inner walks keys (in1_stride=DHt). One column
                // per matmul (ct/pack_width=1) for the same blocked-pack reason as QK.
                mm_no_mop_init_short(cb_qk_im, cb_k_in, /*transpose=*/false, 1, Sqt, Skt);
                configure_row_pack_width(out_cur, 1);
                for (uint32_t vd = 0; vd < vDHt; ++vd) {
                    blocked_matmul_and_pack<false, /*in1_stride=*/DHt, /*out_num_cols=*/vDHt>(
                        cb_qk_im,
                        cb_k_in,
                        out_cur,
                        /*in0_index_start=*/0,
                        /*in1_index_start=*/vd,
                        /*row_subblock_idx=*/0,
                        /*out_col_offset=*/vd,
                        /*subblock_w=*/1,
                        /*subblock_h=*/Sqt,
                        /*inner_dim=*/Skt,
                        /*matmul_stride=*/KT_stride,
                        /*trigger_reduce=*/false,
                        /*skip_pack_configure=*/true);
                }
                // V-matmul PACK writes to out_cur must be visible before SALAD/normalize read/accumulate them.
                pack_to_unpack_sync();
            }

            // ===== SALAD flash combine (skip on the first chunk) =====
            if (!is_first) {
                DeviceZoneScopedN("cmp_salad");
                // correction = exp((prev_max - cur_max)*scale) -> cb_corr col 0. Re-init the exp SFPU at unit
                // scale: sub_exp_first_col_blocks applies scale_bf16 itself, but the Phase-1 init baked scale_fp32.
                exp_packthread_tile_init<EXP_APPROX_MODE>();
                cb_reserve_back(cb_corr, Sqt);
                sub_exp_first_col_blocks<false, scale_fp32>(max_prev, max_cur, cb_corr, /*q_subblock=*/0, Sqt);
                cb_push_back(cb_corr, Sqt);
                // cur.out += prev.out*corr ; cur.sum += prev.sum*corr (L1-accumulated). salad_correct_fused
                // packs width=dst_size (blocked); restore Default packer geometry first — the per-chunk
                // tilize left it in Tilize layout and salad's configure_pack_width skips the stride reconfig.
                PACK((llk_pack_init<ckernel::PackMode::Default, false, false, false>(out_cur, dst_size)));
                PACK((llk_pack_reconfig_l1_acc(1)));
                salad_correct_fused<Sqt, vDHt, dst_size>(
                    out_prev,
                    sum_prev,
                    cb_corr,
                    out_cur,
                    sum_cur,
                    /*ob_q_subblock=*/0,
                    /*sum_q_subblock=*/0,
                    /*write_q_subblock=*/0);
                PACK((llk_pack_reconfig_l1_acc(0)));
                cb_pop_front(cb_corr, Sqt);
                cb_pop_front(out_prev, Sqt * vDHt);
                cb_pop_front(max_prev, Sqt);  // old running max/sum consumed into cur
                cb_pop_front(sum_prev, Sqt);
            }

            // Publish cur.sum / cur.out (now the running state for the next chunk, or the final result).
            cb_push_back(sum_cur, Sqt);
            cb_push_back(out_cur, Sqt * vDHt);

            if (is_last) {
                // Finalize: row-sum (matmul vs col-identity) -> recip -> out *= 1/sum -> cb_out_im.
                DeviceZoneScopedN("cmp_normalize");
                normalize_row_streaming<
                    /*profiling_enabled=*/false,
                    vDHt,
                    dst_size,
                    cb_col_identity,
                    cb_recip_scratch,
                    cb_out_im,
                    scale_fp32>(sum_cur, out_cur, Sqt);
                cb_pop_front(max_cur, Sqt);  // running max no longer needed
            }

            // Release the held cb_qk_im row + this chunk's K (consumed by both QK and PV).
            cb_pop_front(cb_qk_im, Sqt * KT_stride);
            cb_pop_front(cb_k_in, Skt * DHt);

            // swap prev <-> cur for the next chunk
            swap_cb(max_prev, max_cur);
            swap_cb(sum_prev, sum_cur);
            swap_cb(out_prev, out_cur);
        }

        cb_pop_front(cb_q_in, Sqt * DHt);  // Q reused across all chunks; drop it now so >1 token/core stays clean

        // cb_out_im was written by normalize_row_streaming; untilize -> row-major out for the writer.
        compute_kernel_lib::untilize<vDHt, cb_out_im, cb_out_rm>(/*num_blocks=*/Sqt);
    }
}
