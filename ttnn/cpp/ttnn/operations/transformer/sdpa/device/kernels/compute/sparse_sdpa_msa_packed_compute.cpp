// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// sparse_sdpa_msa packed-group compute (16 query heads per KV group). A group holds up to G tokens in G/2 Q tile
// rows (two tokens per 32-row tile row). Each union block of the group is loaded once; every tile row that
// holds a token that selected the block runs one online-softmax step on it (exactly the legacy kernel's
// one-tile-row step), with per-token -inf masks:
//   - a token that did not select the block: its 16 rows are fully masked (top/bottom half -inf tile);
//   - a token whose diagonal block this is: key-tiles past its boundary -inf, the boundary key-tile gets the
//     token's half-tile partial-column mask (reader-built).
// A tile row neither of whose tokens selected the block skips it (its running state is untouched, which is
// what a fully masked step would compute: P = 0, correction = 1). Every row's running state lives in its own
// ping-pong CBs, so skipping needs no copy. The lead block (union entry 0) is selected by every token, so all
// rows start on a block with visible keys.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "compute_common.hpp"
#include "compute_streaming.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "api/dataflow/circular_buffer.h"
#include <tt-metalium/constants.hpp>

ALWI void pack_to_unpack_sync() {
    PACK((t6_semaphore_post<p_stall::STALL_PACK>(semaphore::PACK_DONE)));
    UNPACK((t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE)));
    UNPACK((t6_semaphore_get<>(semaphore::PACK_DONE)));
}

void kernel_main() {
    constexpr uint32_t G = get_compile_time_arg_val(0);  // max tokens per group; G/2 tile rows
    constexpr uint32_t DHt = get_compile_time_arg_val(1);
    constexpr uint32_t vDHt = get_compile_time_arg_val(2);
    constexpr uint32_t Skt = get_compile_time_arg_val(3);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(4);
    constexpr uint32_t cb_q_rm = get_compile_time_arg_val(5);
    constexpr uint32_t cb_q_in = get_compile_time_arg_val(6);
    constexpr uint32_t cb_k_in = get_compile_time_arg_val(7);
    constexpr uint32_t cb_v_in = get_compile_time_arg_val(8);
    constexpr uint32_t cb_scale = get_compile_time_arg_val(9);
    constexpr uint32_t cb_qk_im = get_compile_time_arg_val(10);
    constexpr uint32_t cb_corr = get_compile_time_arg_val(11);
    constexpr uint32_t cb_out_im = get_compile_time_arg_val(12);
    constexpr uint32_t cb_out_rm = get_compile_time_arg_val(13);
    constexpr uint32_t cb_ctrl = get_compile_time_arg_val(14);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(15);
    constexpr uint32_t cb_recip_scratch = get_compile_time_arg_val(16);
    constexpr uint32_t cb_neginf = get_compile_time_arg_val(17);    // all -inf
    constexpr uint32_t cb_halfmask = get_compile_time_arg_val(18);  // tile 0: rows 0-15 -inf; tile 1: rows 16-31
    constexpr uint32_t cb_vmask = get_compile_time_arg_val(19);     // G half-tile partial-column tiles per group
    constexpr uint32_t cb_state0 = get_compile_time_arg_val(20);    // per tile row: max a/b, sum a/b, out a/b
    constexpr uint32_t Sqt = G / 2;
    constexpr uint32_t H = Sqt * tt::constants::TILE_HEIGHT;
    constexpr uint32_t KT_stride = Skt;
    constexpr uint32_t dst_size = compute_kernel_lib::DEST_AUTO_LIMIT;
    constexpr uint32_t exp_sbw = (Skt <= dst_size) ? Skt : 1;
    constexpr uint32_t hdr_words = 2 + G;
    constexpr uint32_t kDiagShift = 8;

    CircularBuffer q_in_cb(cb_q_in), k_in_cb(cb_k_in), v_in_cb(cb_v_in), qk_cb(cb_qk_im), scale_cb(cb_scale),
        ctrl_cb(cb_ctrl), corr_cb(cb_corr), vmask_cb(cb_vmask);

    const uint32_t tok_count = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q_in, cb_k_in, cb_qk_im);
    matmul_init(cb_q_in, cb_k_in);

    scale_cb.wait_front(1);
    CircularBuffer(cb_neginf).wait_front(1);
    CircularBuffer(cb_halfmask).wait_front(2);

    uint32_t side[Sqt];  // which half (0: a, 1: b) of each row's ping-pong holds its running state

    uint32_t done = 0;
    while (done < tok_count) {
        // tilize the group's Q rows -> [Sqt, DHt]
        compute_kernel_lib::tilize<DHt, cb_q_rm, cb_q_in>(/*num_blocks=*/Sqt, /*total_input_pages=*/H);
        pack_reconfig_data_format(cb_qk_im);

        ctrl_cb.wait_front(1);
        const uint32_t n_union = ckernel::read_tile_value(cb_ctrl, 0, 0);
        const uint32_t g = ckernel::read_tile_value(cb_ctrl, 0, 1);
        uint32_t geo[G];  // per slot: diagonal boundary key-tile | boundary column << 8
        for (uint32_t s = 0; s < G; ++s) {
            geo[s] = ckernel::read_tile_value(cb_ctrl, 0, 2 + s);
        }
        const uint32_t n_rows = (g + 1) / 2;
        const uint32_t valid_slots = (1u << g) - 1;
        vmask_cb.wait_front(G);
        q_in_cb.wait_front(Sqt * DHt);
        uint32_t started = 0;  // rows that hold running state (all of them after the lead block)

        for (uint32_t u = 0; u < n_union; ++u) {
            const uint32_t word = ckernel::read_tile_value(cb_ctrl, 0, hdr_words + u);
            const uint32_t sel = word & valid_slots;
            const uint32_t diag = (word >> kDiagShift) & valid_slots;

            k_in_cb.wait_front(Skt * DHt);
            v_in_cb.wait_front(Skt * vDHt);

            for (uint32_t r = 0; r < n_rows; ++r) {
                const uint32_t s0 = 2 * r, s1 = 2 * r + 1;
                const uint32_t row_sel = (sel >> s0) & 3u;
                if (row_sel == 0) {
                    continue;  // neither token of this row selected the block: state unchanged
                }
                // Per-half mask modes: 0 visible, 1 hidden, 2 diagonal (an empty bottom slot is visible: zero Q).
                const bool v1 = s1 < g;
                const uint32_t m0 = !(row_sel & 1u) ? 1u : (((diag >> s0) & 1u) ? 2u : 0u);
                const uint32_t m1 = !v1 ? 0u : (!(row_sel & 2u) ? 1u : (((diag >> s1) & 1u) ? 2u : 0u));

                const bool is_first = ((started >> r) & 1u) == 0;
                const uint32_t st = is_first ? 0u : side[r];
                const uint32_t nx = is_first ? 0u : 1u - st;
                const uint32_t base = cb_state0 + r * 6;
                CircularBuffer max_prev(base + 0 + st), max_cur(base + 0 + nx);
                CircularBuffer sum_prev(base + 2 + st), sum_cur(base + 2 + nx);
                CircularBuffer out_prev(base + 4 + st), out_cur(base + 4 + nx);

                reconfig_data_format(cb_k_in, cb_q_in);
                qk_cb.reserve_back(KT_stride);
                sum_cur.reserve_back(1);
                out_cur.reserve_back(vDHt);

                exp_packthread_tile_init<true, scale_fp32, InputClamping::None>();

                // Phase 1: Q(row r)@K^T -> scores [1, Skt].
                mm_no_mop_init_short(cb_q_in, cb_k_in, /*transpose=*/true, 1, 1, DHt);
                configure_row_pack_width(cb_qk_im, 1);
                for (uint32_t kt = 0; kt < Skt; ++kt) {
                    blocked_matmul_and_pack<true, /*in1_stride=*/1, /*out_num_cols=*/KT_stride>(
                        cb_q_in,
                        cb_k_in,
                        cb_qk_im,
                        /*in0_index_start=*/r * DHt,
                        /*in1_index_start=*/kt * DHt,
                        /*row_subblock_idx=*/0,
                        /*out_col_offset=*/kt,
                        /*subblock_w=*/1,
                        /*subblock_h=*/1,
                        /*inner_dim=*/DHt,
                        /*matmul_stride=*/DHt,
                        /*skip_pack_configure=*/true);
                }
                cb_push_back_hold_wr_ptr(cb_qk_im, KT_stride);

                // Masks (L1-accumulated -inf) before the row-max reduce. Stamp kinds: 0 all -inf (both tokens),
                // 1 top-half -inf, 2 bottom-half -inf, 3 top token's partial-column tile, 4 bottom token's.
                if ((m0 | m1) != 0) {
                    const uint32_t bt0 = geo[s0] & 0xFF, bc0 = geo[s0] >> 8;
                    const uint32_t bt1 = v1 ? (geo[s1] & 0xFF) : Skt, bc1 = v1 ? (geo[s1] >> 8) : 0;
                    uint8_t stamps[2 * Skt];
                    uint32_t n = 0;
                    for (uint32_t kt = 0; kt < Skt; ++kt) {
                        const bool full0 = (m0 == 1) || (m0 == 2 && (kt > bt0 || (kt == bt0 && bc0 == 0)));
                        const bool part0 = (m0 == 2) && kt == bt0 && bc0 > 0;
                        const bool full1 = (m1 == 1) || (m1 == 2 && (kt > bt1 || (kt == bt1 && bc1 == 0)));
                        const bool part1 = (m1 == 2) && kt == bt1 && bc1 > 0;
                        if (full0 && full1) {
                            stamps[n++] = (kt << 4) | 0;
                        } else {
                            if (full0) {
                                stamps[n++] = (kt << 4) | 1;
                            }
                            if (full1) {
                                stamps[n++] = (kt << 4) | 2;
                            }
                            if (part0) {
                                stamps[n++] = (kt << 4) | 3;
                            }
                            if (part1) {
                                stamps[n++] = (kt << 4) | 4;
                            }
                        }
                    }
                    if (n > 0) {
                        reconfig_data_format_srca(cb_neginf);
                        pack_reconfig_data_format(cb_qk_im);
                        copy_init(cb_neginf);
                        PACK((llk_pack_reconfig_l1_acc(1)));
                        for (uint32_t b0 = 0; b0 < n; b0 += dst_size) {
                            const uint32_t nb = (n - b0 < dst_size) ? (n - b0) : dst_size;
                            tile_regs_acquire();
                            for (uint32_t i = 0; i < nb; ++i) {
                                const uint32_t kind = stamps[b0 + i] & 0xF;
                                const uint32_t mcb = kind == 0 ? cb_neginf : (kind <= 2 ? cb_halfmask : cb_vmask);
                                const uint32_t midx = kind == 0 ? 0 : (kind <= 2 ? kind - 1 : (kind == 3 ? s0 : s1));
                                copy_tile(mcb, midx, i);
                            }
                            tile_regs_commit();
                            tile_regs_wait();
                            for (uint32_t i = 0; i < nb; ++i) {
                                pack_tile<true>(i, cb_qk_im, stamps[b0 + i] >> 4);
                            }
                            tile_regs_release();
                        }
                        PACK((llk_pack_reconfig_l1_acc(0)));
                        pack_to_unpack_sync();  // masked writes visible to the row-max reduce's UNPACK
                    }
                }

                {
                    reconfig_data_format(cb_qk_im, cb_scale);
                    max_cur.reserve_back(1);
                    configure_single_tile_pack(max_cur.get_cb_id());
                    reduce_c_row_group<cb_qk_im, cb_scale, KT_stride>(
                        max_cur.get_cb_id(),
                        max_prev.get_cb_id(),
                        /*row_group_index=*/0,
                        /*do_eltwise_max=*/!is_first,
                        1,
                        Skt);
                    max_cur.push_back(1);

                    for (uint32_t kc = 0; kc < Skt; kc += exp_sbw) {
                        sub_exp_block_bcast_cols<false, scale_fp32>(
                            cb_qk_im,
                            max_cur.get_cb_id(),
                            sum_cur.get_cb_id(),
                            /*cols_in_row=*/KT_stride,
                            /*q_subblock=*/0,
                            /*global_col_base=*/kc,
                            /*sbh=*/1,
                            /*sbw=*/exp_sbw);
                    }
                    pack_to_unpack_sync();
                }

                // Phase 2: probs@V -> out_cur.
                {
                    qk_cb.wait_front(KT_stride);
                    reconfig_data_format(cb_v_in, cb_qk_im);
                    mm_no_mop_init_short(cb_qk_im, cb_v_in, /*transpose=*/false, 1, 1, Skt);
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
                            /*subblock_h=*/1,
                            /*inner_dim=*/Skt,
                            /*matmul_stride=*/KT_stride,
                            /*skip_pack_configure=*/true);
                    }
                    pack_to_unpack_sync();
                    reconfig_data_format_srca(cb_qk_im);
                }

                if (!is_first) {
                    exp_packthread_tile_init<EXP_APPROX_MODE>();
                    corr_cb.reserve_back(1);
                    sub_exp_first_col_blocks<false, scale_fp32>(
                        max_prev.get_cb_id(), max_cur.get_cb_id(), cb_corr, /*q_subblock=*/0, 1);
                    corr_cb.push_back(1);
                    PACK((
                        llk_pack_init<ckernel::PackMode::Default, false, false, false>(out_cur.get_cb_id(), dst_size)));
                    pack_reconfig_l1_acc(1);
                    salad_correct_fused<1, vDHt, dst_size>(
                        out_prev.get_cb_id(),
                        sum_prev.get_cb_id(),
                        cb_corr,
                        out_cur.get_cb_id(),
                        sum_cur.get_cb_id(),
                        /*ob_q_subblock=*/0,
                        /*sum_q_subblock=*/0,
                        /*write_q_subblock=*/0);
                    pack_reconfig_l1_acc(0);
                    corr_cb.pop_front(1);
                    out_prev.pop_front(vDHt);
                    max_prev.pop_front(1);
                    sum_prev.pop_front(1);
                }

                sum_cur.push_back(1);
                out_cur.push_back(vDHt);
                qk_cb.pop_front(KT_stride);
                side[r] = nx;
                started |= 1u << r;
            }

            k_in_cb.pop_front(Skt * DHt);
            v_in_cb.pop_front(Skt * vDHt);
        }

        // Finalize every row of the group: out *= 1/sum (all rows first, then one untilize, as the legacy kernel).
        for (uint32_t r = 0; r < n_rows; ++r) {
            const uint32_t base = cb_state0 + r * 6;
            const uint32_t st = side[r];
            if (((started >> r) & 1u) == 0) {
                // Unreachable (the lead block is every token's); keep the writer's row count regardless.
                CircularBuffer(cb_out_im).reserve_back(vDHt);
                CircularBuffer(cb_out_im).push_back(vDHt);
                continue;
            }
            normalize_row_streaming<
                /*profiling_enabled=*/false,
                vDHt,
                dst_size,
                cb_col_identity,
                cb_recip_scratch,
                cb_out_im,
                scale_fp32>(base + 2 + st, base + 4 + st, 1);
            CircularBuffer(base + 0 + st).pop_front(1);
        }

        ctrl_cb.pop_front(1);
        vmask_cb.pop_front(G);
        q_in_cb.pop_front(Sqt * DHt);

        compute_kernel_lib::untilize<vDHt, cb_out_im, cb_out_rm>(/*num_blocks=*/n_rows);
        done += g;
    }
}
