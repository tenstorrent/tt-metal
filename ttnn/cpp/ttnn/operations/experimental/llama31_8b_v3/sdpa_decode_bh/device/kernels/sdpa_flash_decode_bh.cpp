// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#define LLK_ZEROFLAG_OUTLINE 1

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)
#define MAX_TREE_REDUCTION_ROUNDS 6

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/rt_args_common_dec.hpp"
#include "ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/sd_api.hpp"

#ifndef SDB_N
#define SDB_N 4
#endif
static_assert(SDB_N % 4 == 0 && SDB_N <= 16);
#if defined(SDB_M) && (defined(SDB_C0) || defined(SDB_C1) || defined(SDB_C4)) && SDB_N != 4 * SDB_M
#error "SDB_M with row trim needs SDB_N = 4 * rows"
#endif

__attribute__((noinline, noclone)) void move_block_ool(uint32_t in_cb, uint32_t out_cb, uint32_t num_tiles) {
    sd_move_block<true>(in_cb, out_cb, num_tiles);
}

#include "ttnn/operations/experimental/llama31_8b_v3/sdpa_decode_bh/device/kernels/sd_steps.hpp"

void kernel_main() {
    constexpr uint32_t St = get_compile_time_arg_val(0);
    constexpr uint32_t DHt = get_compile_time_arg_val(1);
    constexpr uint32_t vDHt = get_compile_time_arg_val(2);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(3);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(4);

    constexpr uint32_t qk_in0_block_w = get_compile_time_arg_val(5);
    constexpr uint32_t qk_subblock_w = get_compile_time_arg_val(6);
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(7);
    constexpr uint32_t qk_in0_num_subblocks = get_compile_time_arg_val(8);
    constexpr uint32_t qk_in1_num_subblocks = get_compile_time_arg_val(9);

    constexpr uint32_t out_in0_block_w = get_compile_time_arg_val(10);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(11);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(12);
    constexpr uint32_t out_in0_num_subblocks = get_compile_time_arg_val(13);
    constexpr uint32_t out_in1_num_subblocks = get_compile_time_arg_val(14);

    constexpr uint32_t num_cores_per_head = get_compile_time_arg_val(15);
    constexpr uint32_t num_heads_per_core = get_compile_time_arg_val(16);

    constexpr bool is_causal = get_compile_time_arg_val(17) == 1;
    constexpr bool use_attention_mask = get_compile_time_arg_val(18) == 1;
    constexpr bool use_attention_sink = get_compile_time_arg_val(19) == 1;
    constexpr uint32_t max_dynamic_chunk_size = get_compile_time_arg_val(20);
    constexpr bool tilize_q = get_compile_time_arg_val(21) == 1;
    constexpr uint32_t q_heads_parallel_factor = get_compile_time_arg_val(22);
    constexpr bool use_half_tile = get_compile_time_arg_val(23);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(24);
    constexpr uint32_t sliding_window_size = get_compile_time_arg_val(25);
    constexpr uint32_t original_block_size = get_compile_time_arg_val(26);
    constexpr bool has_block_padding = original_block_size > 0 && original_block_size < 32;

    constexpr uint32_t spec_multi_pos_T = get_compile_time_arg_val(27);
    constexpr bool spec_multi_pos = spec_multi_pos_T > 0;

    constexpr bool single_local_chunk = !is_causal && sliding_window_size == 0 && Sk_chunk_t > 0 &&
                                        ((St + Sk_chunk_t - 1) / Sk_chunk_t <= num_cores_per_head);
    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_sliding_window_mask_in = tt::CBIndex::c_13;
    constexpr uint32_t cb_block_pad_mask = tt::CBIndex::c_14;
    constexpr uint32_t cb_attention_sink = tt::CBIndex::c_4;
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    constexpr uint32_t cb_m_in = tt::CBIndex::c_6;
    constexpr uint32_t cb_l_in = tt::CBIndex::c_7;
    constexpr uint32_t cb_q_rm = tt::CBIndex::c_10;
    constexpr uint32_t cb_zero_in = tt::CBIndex::c_12;

    constexpr uint32_t cb_cur_pos = tt::CBIndex::c_15;

    constexpr uint32_t cb_qk_im = tt::CBIndex::c_24;
    constexpr uint32_t cb_out_im = tt::CBIndex::c_25;
    constexpr uint32_t cb_out_accumulate_im = tt::CBIndex::c_26;
    constexpr uint32_t cb_max_1 = tt::CBIndex::c_27;
    constexpr uint32_t cb_max_2 = tt::CBIndex::c_28;
    constexpr uint32_t cb_sum_1 = tt::CBIndex::c_29;
    constexpr uint32_t cb_sum_2 = tt::CBIndex::c_30;
    constexpr uint32_t cb_exp_max_diff = tt::CBIndex::c_31;
    constexpr uint32_t cb_prev_sum_2 = tt::CBIndex::c_21;
    constexpr uint32_t cb_exp_max_diff_2 = tt::CBIndex::c_22;
    constexpr uint32_t cb_out_accumulate_im_2 = tt::CBIndex::c_23;

    constexpr uint32_t cb_out_o = tt::CBIndex::c_16;
    constexpr uint32_t cb_out_m = tt::CBIndex::c_17;
    constexpr uint32_t cb_out_l = tt::CBIndex::c_18;
    constexpr uint32_t cb_out_final = tt::CBIndex::c_20;

    uint32_t arg_idx = 0;
    const bool do_reduce = get_arg_val<uint32_t>(arg_idx++) == 1;
    const bool apply_mask_at_last_chunk = do_reduce && is_causal;
    const uint32_t cur_batch = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_num_in_reduce = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cur_pos_arg = get_arg_val<uint32_t>(arg_idx++);

    const bool is_tree_root = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint32_t parent_core_in_group = get_arg_val<uint32_t>(arg_idx++);
    const bool has_parent = parent_core_in_group != UINT32_MAX;

    constexpr uint32_t num_tree_rounds = num_cores_per_head <= 1 ? 0 : 32 - __builtin_clz(num_cores_per_head - 1);
    static_assert(num_tree_rounds <= MAX_TREE_REDUCTION_ROUNDS);
    uint32_t children_per_round[MAX_TREE_REDUCTION_ROUNDS];
    for (uint32_t r = 0; r < num_tree_rounds; ++r) {
        children_per_round[r] = get_arg_val<uint32_t>(arg_idx + r);
    }
#if defined(SDB_C0) || defined(SDB_C1)
    const uint32_t sdk_row0 = get_arg_val<uint32_t>(6 + MAX_TREE_REDUCTION_ROUNDS);
#endif

    if (get_arg_val<uint32_t>(0) == 65) {
        return;
    }

    constexpr uint32_t cur_pos_base = St * 32 - 1;
    uint32_t cur_pos = cur_pos_base;

    uint32_t spec_pos_min = 0;
    if constexpr (is_causal) {
        if (!spec_multi_pos && cur_pos_arg != UINT32_MAX) {
            cur_pos = cur_pos_arg;
        } else {
            CircularBuffer(cb_cur_pos).wait_front(1);
            if constexpr (spec_multi_pos) {
                const uint32_t spec_pos_base = cur_batch * spec_multi_pos_T;
                spec_pos_min = read_tile_value(cb_cur_pos, 0, spec_pos_base);
                cur_pos = read_tile_value(cb_cur_pos, 0, spec_pos_base + spec_multi_pos_T - 1);
            } else {
                cur_pos = read_tile_value(cb_cur_pos, 0, cur_batch / q_heads_parallel_factor);
            }
            CircularBuffer(cb_cur_pos).pop_front(1);
        }
        if (cur_pos == UINT32_MAX) {
            return;
        }
    }

    if constexpr (has_block_padding && is_causal) {
        cur_pos = (cur_pos / original_block_size) * 32 + (cur_pos % original_block_size);
    }

    auto Sk_chunk_t_dynamic = get_dynamic_Sk_chunk_t<Sk_chunk_t, max_dynamic_chunk_size>(cur_pos);
    auto k_chunk_size_dynamic = Sk_chunk_t_dynamic * tt::constants::TILE_HEIGHT;

    auto [PSt, k_num_chunks, k_chunk_start, k_chunk_end, window_start_unaligned, window_start_chunk] =
        get_workload_for_core(
            cur_pos,
            core_num_in_reduce,
            num_cores_per_head,
            k_chunk_size_dynamic,
            sliding_window_size > 0 ? std::optional<uint32_t>(sliding_window_size) : std::nullopt);

    const bool has_local_data = (k_chunk_start != k_chunk_end);

    if (!has_local_data) {
        return;
    }

    uint32_t num_active_children = 0;
    uint32_t active_children_per_round[MAX_TREE_REDUCTION_ROUNDS];
    uint32_t num_active_rounds = 0;

    for (uint32_t r = 0; r < num_tree_rounds; ++r) {
        uint32_t child_id = children_per_round[r];
        if (child_id != UINT32_MAX && child_id < k_num_chunks) {
            active_children_per_round[r] = child_id;
            num_active_children++;
            num_active_rounds = r + 1;
        } else {
            active_children_per_round[r] = UINT32_MAX;
        }
    }

    static_assert(!tilize_q);
    sdo_hw_start(cb_q_in, cb_k_in, cb_qk_im);
    sd_matmul_init(cb_q_in, cb_k_in);
    CircularBuffer(cb_q_in).wait_front(q_chunk_tiles);

    if constexpr (has_block_padding) {
        uint32_t block_pad_mask_tiles = Sq_chunk_t * Sk_chunk_t_dynamic;
        CircularBuffer(cb_block_pad_mask).wait_front(block_pad_mask_tiles);
    }

#ifdef DYNAMIC_CHUNK_SIZE
    const uint32_t qk_subblock_h_dynamic = 1;
    const uint32_t qk_subblock_w_dynamic = Sk_chunk_t_dynamic;

    const uint32_t qk_in0_num_subblocks_dynamic = Sq_chunk_t;
    const uint32_t qk_in1_num_subblocks_dynamic = 1;
    const uint32_t out_in0_block_w_dynamic = Sk_chunk_t_dynamic;
    const uint32_t qk_chunk_tiles_dynamic = Sq_chunk_t * Sk_chunk_t_dynamic;
#else
    constexpr uint32_t qk_subblock_h_dynamic = qk_subblock_h;
    constexpr uint32_t qk_subblock_w_dynamic = qk_subblock_w;
    constexpr uint32_t qk_in0_num_subblocks_dynamic = qk_in0_num_subblocks;
    constexpr uint32_t qk_in1_num_subblocks_dynamic = qk_in1_num_subblocks;
    constexpr uint32_t out_in0_block_w_dynamic = out_in0_block_w;
    constexpr uint32_t qk_chunk_tiles_dynamic = Sq_chunk_t * Sk_chunk_t;
#endif

    constexpr VectorMode vector_mode = use_half_tile ? VectorMode::R : VectorMode::RC;

    uint32_t cb_cur_max = cb_max_1;
    uint32_t cb_prev_max = cb_max_2;
    uint32_t cb_cur_sum = cb_sum_1;
    uint32_t cb_prev_sum = cb_sum_2;

    for (uint32_t cur_head_work = 0; cur_head_work < num_heads_per_core; ++cur_head_work) {
        cb_cur_max = cb_max_1;
        cb_prev_max = cb_max_2;
        cb_cur_sum = cb_sum_1;
        cb_prev_sum = cb_sum_2;

        uint32_t cb_out_mm = cb_out_accumulate_im;

        for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
            sdo_pack_reconfig(cb_qk_im);

            const bool spec_chunk_needs_mask =
                spec_multi_pos && (k_chunk * k_chunk_size_dynamic + k_chunk_size_dynamic - 1) > spec_pos_min;

#ifdef DYNAMIC_CHUNK_SIZE

            bool add_causal_mask_fusion =
                !spec_multi_pos && is_causal && k_chunk == k_chunk_end - 1 && apply_mask_at_last_chunk;

            bool add_sliding_window_mask_fusion =
                sliding_window_size > 0 && k_chunk == window_start_chunk && window_start_unaligned > 0;
            bool add_mask_fusion = add_causal_mask_fusion || use_attention_mask || add_sliding_window_mask_fusion;
#else
            bool add_mask_fusion = false;
            bool add_sliding_window_mask_fusion = false;
#endif

            uint32_t mask_cb_to_use = cb_mask_in;
            if (add_sliding_window_mask_fusion) {
                mask_cb_to_use = cb_sliding_window_mask_in;
            }
            sd_matmul_blocks(
                cb_q_in,
                cb_k_in,
                cb_qk_im,
                Sq_chunk_t,
                Sk_chunk_t_dynamic,
                DHt,
                qk_in0_num_subblocks_dynamic,
                qk_in1_num_subblocks_dynamic,
                qk_in0_block_w,
                qk_subblock_h_dynamic,
                qk_subblock_w_dynamic,
                true,
                add_mask_fusion,
                mask_cb_to_use,
                cb_zero_in);

            if constexpr (has_block_padding) {
                sdo_reconfig(cb_qk_im, cb_block_pad_mask);
                sd_add_block_inplace<false>(cb_qk_im, cb_block_pad_mask, qk_chunk_tiles_dynamic);
            }

            if (!add_mask_fusion) {
                if constexpr (spec_multi_pos) {
                    if (spec_chunk_needs_mask) {
                        sdo_reconfig(cb_qk_im, cb_mask_in);
                        sd_add_block_inplace<true>(cb_qk_im, cb_mask_in, qk_chunk_tiles_dynamic);
                    }
                } else if constexpr (is_causal) {
                    if (k_chunk == k_chunk_end - 1 && apply_mask_at_last_chunk) {
                        sdo_reconfig(cb_qk_im, cb_mask_in);
                        sd_add_block_inplace<false>(cb_qk_im, cb_mask_in, qk_chunk_tiles_dynamic);
                    }
                } else {
                    if constexpr (use_attention_mask) {
                        sdo_reconfig(cb_qk_im, cb_mask_in);
                        sd_add_block_inplace<true>(cb_qk_im, cb_mask_in, qk_chunk_tiles_dynamic);
                    }
                }

                if (sliding_window_size > 0 && k_chunk == window_start_chunk && window_start_unaligned > 0) {
                    sdo_reconfig(cb_qk_im, cb_sliding_window_mask_in);
                    sd_add_block_inplace<false>(cb_qk_im, cb_sliding_window_mask_in, qk_chunk_tiles_dynamic);
                }
            }

            {
                {
                    sdo_reconfig(cb_qk_im, cb_identity_scale_in);
                    sdo_pack_reconfig(cb_cur_max);

                    sd_reduce_c<
                        PoolType::MAX,
                        ReduceDim::REDUCE_ROW,
                        cb_qk_im,
                        cb_identity_scale_in,
                        Sq_chunk_t,
                        vector_mode>(cb_cur_max, cb_prev_max, Sk_chunk_t_dynamic, k_chunk > k_chunk_start);
                }
            }

#ifdef SDB_C0
            {
                sdk_sub_exp_rows4<cb_qk_im, scale_fp32>(cb_cur_max, Sk_chunk_t_dynamic, sdk_row0);
            }
#else
            {
                sd_sub_exp_block_bcast_cols_inplace<cb_qk_im, Sq_chunk_t, scale_fp32, true, false, vector_mode>(
                    cb_cur_max, cb_cur_sum, Sk_chunk_t_dynamic);
            }
#endif
            CircularBuffer(cb_qk_im).wait_front(qk_chunk_tiles_dynamic);

            sdo_reconfig(cb_qk_im, cb_identity_scale_in);
            sdo_pack_reconfig(cb_cur_sum);

            sd_reduce_c<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity_scale_in, Sq_chunk_t, vector_mode>(
                cb_cur_sum, cb_cur_sum, Sk_chunk_t_dynamic, false);

            sdo_pack_reconfig(cb_out_im);
            sd_matmul_blocks(
                cb_qk_im,
                cb_v_in,
                cb_out_mm,
                Sq_chunk_t,
                vDHt,
                Sk_chunk_t_dynamic,
                out_in0_num_subblocks,
                out_in1_num_subblocks,
                out_in0_block_w_dynamic,
                out_subblock_h,
                out_subblock_w,
                false,
                false,
                cb_mask_in,
                cb_zero_in);

            sdo_reconfig_srca(cb_out_im);
            CircularBuffer(cb_qk_im).pop_front(qk_chunk_tiles_dynamic);

            if (k_chunk == k_chunk_start) {
                cb_out_mm = cb_out_im;
            } else if constexpr (!single_local_chunk) {
                sdo_reconfig(cb_prev_max, cb_cur_max);
                sdo_pack_reconfig(cb_exp_max_diff);

                sd_sub_exp_block<scale_fp32>(cb_prev_max, cb_cur_max, cb_exp_max_diff, Sq_chunk_t);
                CircularBuffer(cb_prev_max).pop_front(Sq_chunk_t);

#ifdef SDB_C4
                if constexpr (Sq_chunk_t == 1 && out_chunk_tiles <= 4) {
                    sdk_trs_rescale<out_chunk_tiles>(
                        cb_prev_sum, cb_cur_sum, cb_exp_max_diff, cb_out_accumulate_im, cb_out_im, sdk_row0);
                } else
#endif
                {

                    sd_mul_block_inplace(cb_prev_sum, cb_exp_max_diff, Sq_chunk_t);

                    sd_mul_block_bcast_cols<Sq_chunk_t, vDHt, true, false>(
                        cb_out_accumulate_im, cb_exp_max_diff, cb_out_accumulate_im);

                    sdo_reconfig(cb_cur_sum, cb_prev_sum);
                    sdo_pack_reconfig(cb_cur_sum);
                    sd_add_block_inplace<true>(cb_cur_sum, cb_prev_sum, Sq_chunk_t);

                    sdo_reconfig(cb_out_accumulate_im, cb_out_im);
                    sdo_pack_reconfig(cb_out_accumulate_im);
                    sd_add_block_inplace<true>(cb_out_accumulate_im, cb_out_im, out_chunk_tiles);
                }
            }

#if defined(SDB_C5)
            {
                {
                    const uint32_t swf_max = cb_cur_max;
                    cb_cur_max = cb_prev_max;
                    cb_prev_max = swf_max;
                }
                const uint32_t swf_sum = cb_cur_sum;
                cb_cur_sum = cb_prev_sum;
                cb_prev_sum = swf_sum;
            }
            if constexpr (false)
#elif 0
            if (k_chunk + 1 == k_chunk_end) {
                const uint32_t swf_max = cb_cur_max;
                cb_cur_max = cb_prev_max;
                cb_prev_max = swf_max;
                const uint32_t swf_sum = cb_cur_sum;
                cb_cur_sum = cb_prev_sum;
                cb_prev_sum = swf_sum;
            } else
#endif
            {

                sdo_reconfig(cb_cur_max, cb_cur_max);
                sdo_pack_reconfig(cb_prev_max);

                sd_move_block<true>(cb_cur_max, cb_prev_max, Sq_chunk_t);

                sd_move_block<true>(cb_cur_sum, cb_prev_sum, Sq_chunk_t);
            }
        }

        bool sdo_done = false;
        if (num_active_children > 0) {
            for (uint32_t round = 0; round < num_active_rounds; ++round) {
                uint32_t child_id = active_children_per_round[round];
                if (child_id != UINT32_MAX) {
#ifdef SDB_C1
                    sdk_correction_block_rows4<scale_fp32>(
                        cb_m_in,
                        cb_l_in,
                        cb_cur_max,
                        cb_prev_max,
                        cb_cur_sum,
                        cb_prev_sum,
                        cb_exp_max_diff,
                        cb_exp_max_diff_2,
                        Sq_chunk_t,
                        sdk_row0);
#else
                    sd_correction_block<scale_fp32, vector_mode>(
                        cb_m_in,
                        cb_l_in,
                        cb_cur_max,
                        cb_prev_max,
                        cb_cur_sum,
                        cb_prev_sum,
                        cb_exp_max_diff,
                        cb_exp_max_diff_2,
                        Sq_chunk_t);
#endif

#ifdef SDB_C4
                    if constexpr (Sq_chunk_t == 1 && out_chunk_tiles <= 4) {
                        sd_mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(cb_out_accumulate_im, cb_exp_max_diff);
                        const bool sdo_last = has_parent && round + 1 == num_active_rounds;
                        sdk_trm_child<out_chunk_tiles>(
                            cb_out_o,
                            cb_exp_max_diff_2,
                            cb_out_accumulate_im,
                            sdk_row0,
                            sdo_last ? cb_out_o : cb_out_accumulate_im);
                        sdo_done = sdo_last;
                    } else
#endif
                    {
                        move_block_ool(cb_out_o, cb_out_accumulate_im_2, out_chunk_tiles);

                        sd_mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(cb_out_accumulate_im, cb_exp_max_diff);
                        sd_mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(cb_out_accumulate_im_2, cb_exp_max_diff_2);

                        sd_add_block_inplace<true>(cb_out_accumulate_im, cb_out_accumulate_im_2, out_chunk_tiles);
                    }

                    CircularBuffer(cb_prev_max).pop_front(Sq_chunk_t);
                    CircularBuffer(cb_m_in).pop_front(Sq_chunk_t);
#ifdef SDB_C2
                    {
                        const uint32_t swp_max = cb_cur_max;
                        cb_cur_max = cb_prev_max;
                        cb_prev_max = swp_max;
                        const uint32_t swp_sum = cb_cur_sum;
                        cb_cur_sum = cb_prev_sum;
                        cb_prev_sum = swp_sum;
                    }
#else
                    move_block_ool(cb_cur_max, cb_prev_max, Sq_chunk_t);
                    move_block_ool(cb_cur_sum, cb_prev_sum, Sq_chunk_t);
#endif
                }
            }
        }

        if (is_tree_root) {
            sdo_reconfig(cb_prev_sum, cb_prev_sum);
            sdo_pack_reconfig(cb_prev_sum);

            if constexpr (use_attention_sink) {
                uint32_t max_cb_for_sink = cb_prev_max;

                sd_max_block<vector_mode>(cb_attention_sink, max_cb_for_sink, cb_cur_max, Sq_chunk_t);

                sd_sub_exp_block<scale_fp32>(max_cb_for_sink, cb_cur_max, cb_exp_max_diff, Sq_chunk_t);

                sd_mul_block_inplace(cb_prev_sum, cb_exp_max_diff, Sq_chunk_t);

                sd_sub_exp_block<scale_fp32>(cb_attention_sink, cb_cur_max, cb_exp_max_diff_2, Sq_chunk_t);
                CircularBuffer(cb_cur_max).pop_front(Sq_chunk_t);

                sd_add_block_inplace<true>(cb_prev_sum, cb_exp_max_diff_2, Sq_chunk_t);

                sd_mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(cb_out_accumulate_im, cb_exp_max_diff);
            }

            sdo_reconfig(cb_prev_sum, cb_prev_sum);
            sdo_pack_reconfig(cb_prev_sum);
#ifdef SDB_C1
            sdk_recip_block_rows4(cb_prev_sum, Sq_chunk_t, sdk_row0);
#else
            sd_recip_block_inplace(cb_prev_sum, Sq_chunk_t);
#endif

            sdo_reconfig(cb_out_accumulate_im, cb_prev_sum);
            sdo_pack_reconfig(cb_out_accumulate_im);

#ifdef SDB_C6
            sd_mul_block_bcast_cols<Sq_chunk_t, vDHt, true, false>(cb_out_accumulate_im, cb_prev_sum, cb_out_final);
            CircularBuffer(cb_prev_max).pop_front(Sq_chunk_t);

            continue;
#endif

            sd_mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(cb_out_accumulate_im, cb_prev_sum);
            sdo_pack_reconfig(cb_out_final);

            CircularBuffer(cb_prev_max).pop_front(Sq_chunk_t);

            move_block_ool(cb_out_accumulate_im, cb_out_final, out_chunk_tiles);

        } else if (has_parent) {
            if (!sdo_done) {
                move_block_ool(cb_out_accumulate_im, cb_out_o, out_chunk_tiles);
            }

            move_block_ool(cb_prev_max, cb_out_m, Sq_chunk_t);

            move_block_ool(cb_prev_sum, cb_out_l, Sq_chunk_t);
        }
    }

    CircularBuffer(cb_q_in).pop_front(q_chunk_tiles);
}
