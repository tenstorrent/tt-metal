// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)
#define MAX_TREE_REDUCTION_ROUNDS 6

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/tilize.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/untilize.h"
#include "ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"
#include "api/compute/pack_untilize.h"
#include "api/compute/untilize.h"

constexpr uint32_t MAX_PACK_UNTILIZE_WIDTH = 8;

void kernel_main() {
    // Compile time arguments

    // Input dimensions in tiles
    constexpr uint32_t St = get_compile_time_arg_val(0);
    constexpr uint32_t DHt = get_compile_time_arg_val(1);
    constexpr uint32_t vDHt = get_compile_time_arg_val(2);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(3);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(4);

    // Matmul configs
    constexpr uint32_t qk_in0_block_w = get_compile_time_arg_val(5);
    constexpr uint32_t qk_subblock_w = get_compile_time_arg_val(6);
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(7);
    constexpr uint32_t qk_in0_num_subblocks = get_compile_time_arg_val(8);
    constexpr uint32_t qk_in1_num_subblocks = get_compile_time_arg_val(9);
    constexpr uint32_t qk_num_blocks = get_compile_time_arg_val(10);
    constexpr uint32_t out_in0_block_w = get_compile_time_arg_val(11);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(12);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(13);
    constexpr uint32_t out_in0_num_subblocks = get_compile_time_arg_val(14);
    constexpr uint32_t out_in1_num_subblocks = get_compile_time_arg_val(15);
    constexpr uint32_t out_num_blocks = get_compile_time_arg_val(16);
    constexpr uint32_t num_cores_per_head = get_compile_time_arg_val(19);
    constexpr uint32_t num_heads_per_core = get_compile_time_arg_val(20);

    // Attention-specific parameters
    constexpr bool is_causal = get_compile_time_arg_val(21) == 1;
    constexpr bool use_attention_mask = get_compile_time_arg_val(22) == 1;
    constexpr bool use_attention_sink = get_compile_time_arg_val(23) == 1;
    constexpr uint32_t max_dynamic_chunk_size = get_compile_time_arg_val(24);
    constexpr bool tilize_q = get_compile_time_arg_val(25) == 1;
    constexpr uint32_t q_heads_parallel_factor = get_compile_time_arg_val(26);
    constexpr bool use_half_tile = get_compile_time_arg_val(27);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(28);
    constexpr uint32_t sliding_window_size = get_compile_time_arg_val(29);
    constexpr uint32_t num_tree_reduction_rounds = get_compile_time_arg_val(30);

    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;
    constexpr bool untilize_output = tilize_q;
    constexpr bool use_pack_untilize = out_chunk_tiles <= MAX_PACK_UNTILIZE_WIDTH;

    // CB index definitions
    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_sliding_window_mask_in = tt::CBIndex::c_13;  // Separate buffer for sliding window mask
    constexpr uint32_t cb_attention_sink = tt::CBIndex::c_4;
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    constexpr uint32_t cb_m_in = tt::CBIndex::c_6;
    constexpr uint32_t cb_l_in = tt::CBIndex::c_7;
    constexpr uint32_t cb_q_rm = tt::CBIndex::c_10;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_11;
    constexpr uint32_t cb_zero_in = tt::CBIndex::c_12;

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

    // Runtime arguments
    uint32_t arg_idx = 0;
    const bool do_reduce = get_arg_val<uint32_t>(arg_idx++) == 1;
    const bool apply_mask_at_last_chunk = do_reduce && is_causal;
    const bool do_output = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint32_t cur_head = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cur_batch = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_num_in_reduce = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_num_in_output = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cur_pos_arg = get_arg_val<uint32_t>(arg_idx++);

    // Tree reduction runtime arguments
    const bool is_tree_root = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint32_t parent_core_in_group = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t send_at_round = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_children = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t my_active_rounds = get_arg_val<uint32_t>(arg_idx++);
    const bool has_parent = parent_core_in_group != UINT32_MAX;

    // Read children_per_round array
    uint32_t children_per_round[MAX_TREE_REDUCTION_ROUNDS];
    for (uint32_t r = 0; r < MAX_TREE_REDUCTION_ROUNDS; ++r) {
        children_per_round[r] = get_arg_val<uint32_t>(arg_idx++);
    }

    // Idle core
    // get_arg_val<uint32_t>(0) can go from 0-63 for the core_num; for active cores 65 is out of range so 65 indicates
    // an idle_core
    if (get_arg_val<uint32_t>(0) == 65) {
        return;
    }

    // Get cur_pos
    constexpr uint32_t cur_pos_base = St * 32 - 1;
    uint32_t cur_pos = cur_pos_base;  // default to non-causal, which we do attention on the entire kv cache. In this
                                      // case we set cur_pos to the last position
    if constexpr (is_causal) {
        // using UINT32_MAX as a flag to indicate that cur_pos is not provided as a list
        if (cur_pos_arg != UINT32_MAX) {
            cur_pos = cur_pos_arg;
        } else {
            // Read cur_pos from CB using mailbox-based synchronization (issue #27979)
            constexpr uint32_t cb_index_id = tt::CBIndex::c_8;

            cb_wait_front(cb_index_id, 1);
            cur_pos = read_tile_value(cb_index_id, 0, cur_batch / q_heads_parallel_factor);
            cb_pop_front(cb_index_id, 1);
        }
        if (cur_pos == UINT32_MAX) {
            // cur_pos of -1 indicates that the user should be skipped
            return;
        }
    }

    // Get dynamic chunk size for K in tiles
    auto Sk_chunk_t_dynamic = get_dynamic_Sk_chunk_t<Sk_chunk_t, max_dynamic_chunk_size>(cur_pos);
    auto k_chunk_size_dynamic = Sk_chunk_t_dynamic * tt::constants::TILE_HEIGHT;

    // Get the sequence length assignment
    auto [PSt, k_num_chunks, k_chunk_start, k_chunk_end, window_start_unaligned, window_start_chunk] =
        get_workload_for_core(
            cur_pos,
            cur_batch,
            core_num_in_reduce,
            num_cores_per_head,
            k_chunk_size_dynamic,
            sliding_window_size > 0 ? std::optional<uint32_t>(sliding_window_size) : std::nullopt);

    // Check if this core has local data to process
    const bool has_local_data = (k_chunk_start != k_chunk_end);

    // Cores without data don't participate in tree reduction at all
    // They just exit early - no sending, no receiving
    if (!has_local_data) {
        return;
    }

    // Determine which children actually have data (based on chunk allocation)
    // A child at core_num has data if core_num < k_num_chunks
    uint32_t actual_num_children = 0;
    uint32_t actual_children_per_round[MAX_TREE_REDUCTION_ROUNDS];
    uint32_t actual_my_active_rounds = 0;

    for (uint32_t r = 0; r < MAX_TREE_REDUCTION_ROUNDS; ++r) {
        uint32_t child_id = children_per_round[r];
        if (child_id != UINT32_MAX && child_id < k_num_chunks) {
            // This child has data
            actual_children_per_round[r] = child_id;
            actual_num_children++;
            actual_my_active_rounds = r + 1;
        } else {
            actual_children_per_round[r] = UINT32_MAX;
        }
    }

    // Determine if we have an actual parent (parent must have data too, but root always has data)
    // Actually, we only need to check if WE should send - parent will handle receiving
    // We send if we have a parent AND we have data (which we do if we reach here)
    const bool should_send_to_parent = has_parent;

    // Get number of worker cores to wait for
    uint32_t num_cores_to_wait = num_cores_per_head - 1;
    if (num_cores_per_head > k_num_chunks) {
        num_cores_to_wait = k_num_chunks - 1;
    }

    // We tilize input Q if it is in ROW MAJOR layout
    if constexpr (tilize_q) {
        compute_kernel_hw_startup(cb_q_rm, cb_q_in);
        tilize_init(cb_q_rm, q_chunk_tiles, cb_q_in);
        cb_wait_front(cb_q_rm, q_chunk_tiles);
        cb_reserve_back(cb_q_in, q_chunk_tiles);
        tilize_block(cb_q_rm, q_chunk_tiles, cb_q_in);
        tilize_uninit(cb_q_rm, cb_q_in);
        cb_push_back(cb_q_in, q_chunk_tiles);
        cb_pop_front(cb_q_rm, q_chunk_tiles);
        mm_init_short(cb_q_in, cb_k_in);
    } else {
        mm_init(cb_q_in, cb_k_in, cb_qk_im);
    }
    cb_wait_front(cb_q_in, q_chunk_tiles);

    // Define dynamic matmul configs
#ifdef DYNAMIC_CHUNK_SIZE
    const uint32_t qk_subblock_h_dynamic = 1;
    const uint32_t qk_subblock_w_dynamic = Sk_chunk_t_dynamic;  // Guaranteed < DST
    const uint32_t qk_in0_num_subblocks_dynamic = 1;
    const uint32_t qk_in1_num_subblocks_dynamic = 1;
    const uint32_t out_in0_block_w_dynamic = Sk_chunk_t_dynamic;
    const uint32_t out_num_blocks_dynamic = 1;
    const uint32_t qk_chunk_tiles_dynamic = Sq_chunk_t * Sk_chunk_t_dynamic;
#else
    constexpr uint32_t qk_subblock_h_dynamic = qk_subblock_h;
    constexpr uint32_t qk_subblock_w_dynamic = qk_subblock_w;
    constexpr uint32_t qk_in0_num_subblocks_dynamic = qk_in0_num_subblocks;
    constexpr uint32_t qk_in1_num_subblocks_dynamic = qk_in1_num_subblocks;
    constexpr uint32_t out_in0_block_w_dynamic = out_in0_block_w;
    constexpr uint32_t out_num_blocks_dynamic = out_num_blocks;
    constexpr uint32_t qk_chunk_tiles_dynamic = Sq_chunk_t * Sk_chunk_t;
#endif

    // - VectorMode::RC is equivalent to 32x32 tiles
    // - VectorMode::R is equivalent to 16x32 tiles
    // NOTE: Using VectorMode::RC for 16x32 tiles will be correct accuracy, just slower due to unnecessary math
    constexpr int vector_mode = use_half_tile ? VectorMode::R : VectorMode::RC;

    // We set up Ping Pong intermediate buffers between loops
    uint32_t cb_cur_max = cb_max_1;
    uint32_t cb_prev_max = cb_max_2;
    uint32_t cb_cur_sum = cb_sum_1;
    uint32_t cb_prev_sum = cb_sum_2;

    // Loop through all heads assigned to core
    for (uint32_t cur_head_work = 0; cur_head_work < num_heads_per_core; ++cur_head_work) {
        /******************************************************************************
         *                           FLASH ATTENTION LOOP                             *
         ******************************************************************************/
        /**
         * Compute Parameters (most are compile time but some are dynamic):
         * @tparam St - Total sequence length in tiles
         * @tparam DHt - Head dimension in tiles
         * @tparam Sq_chunk_t - Query chunk size in tiles
         * @tparam Sk_chunk_t - Key chunk size in tiles (dynamic)
         * @tparam qk_in0_block_w - QK matmul block width
         * @tparam qk_subblock_w - QK matmul subblock width (dynamic)
         * @tparam qk_subblock_h - QK matmul subblock height (dynamic)
         * @tparam qk_in0_num_subblocks - QK input0 subblocks (dynamic)
         * @tparam qk_in1_num_subblocks - QK input1 subblocks (dynamic)
         * @tparam qk_num_blocks - QK number of blocks
         * @tparam out_in0_block_w - Output matmul block width (dynamic)
         * @tparam out_subblock_w - Output matmul subblock width
         * @tparam out_subblock_h - Output matmul subblock height
         * @tparam out_in0_num_subblocks - Output input0 subblocks
         * @tparam out_in1_num_subblocks - Output input1 subblocks
         * @tparam out_num_blocks - Output number of blocks (dynamic)
         * @tparam is_causal - Whether to use causal attention (if mask is applied)
         * @tparam use_attention_mask - Whether to use attention mask for non-causal attention
         *
         * Circular Buffer Parameters:
         * @tparam cb_q_in - Query input buffer
         * @tparam cb_k_in - Key input buffer
         * @tparam cb_v_in - Value input buffer
         * @tparam cb_mask_in - Mask input buffer
         * @tparam cb_scale_in - Scale input buffer
         * @tparam cb_identity_scale_in - Identity scale buffer
         * @tparam cb_qk_im - QK intermediate buffer
         * @tparam cb_out_im - Output intermediate buffer
         * @tparam cb_out_accumulate_im - Output accumulate buffer
         * @tparam cb_cur_max - Current max buffer
         * @tparam cb_prev_max - Previous max buffer
         * @tparam cb_cur_sum - Current sum buffer
         * @tparam cb_prev_sum - Previous sum buffer
         * @tparam cb_exp_max_diff - Exp max diff buffer
         * @tparam cb_out_o - Output O buffer
         * @tparam cb_out_m - Output M buffer
         * @tparam cb_out_l - Output L buffer
         *
         * Runtime Parameters:
         * @param k_chunk_start - Start index of key chunk
         * @param k_chunk_end - End index of key chunk
         * @param do_reduce - Whether to perform reduction
         * @param qk_chunk_tiles - Number of QK chunk tiles (dynamic)
         * @param out_chunk_tiles - Number of output chunk tiles
         */
        /* START OF FLASH ATTENTION LOOP */
        uint32_t cb_out_mm = cb_out_accumulate_im;

        // Loop through all K chunks
        for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
            // Reconfig register DF
            reconfig_data_format(cb_q_in, cb_k_in);
            pack_reconfig_data_format(cb_qk_im);

            // OPTIMIZATION: Add the attention mask directly on top of DST if chunk sizes are dynamic
#ifdef DYNAMIC_CHUNK_SIZE
                bool add_causal_mask_fusion = is_causal && k_chunk == k_chunk_end - 1 && apply_mask_at_last_chunk;
                bool add_sliding_window_mask_fusion = k_chunk == window_start_chunk && window_start_unaligned > 0;
                bool add_mask_fusion = add_causal_mask_fusion || use_attention_mask || add_sliding_window_mask_fusion;
#else
                bool add_mask_fusion = false;
                bool add_sliding_window_mask_fusion = false;
#endif

                /* QK = Q_CHUNK @ K_CHUNK */
                // Determine which mask buffer to use for fusion
                uint32_t mask_cb_to_use = cb_mask_in;  // Default to causal mask buffer
                if (add_sliding_window_mask_fusion) {
                    mask_cb_to_use = cb_sliding_window_mask_in;  // Use sliding window mask buffer
                }

                matmul_blocks(
                    cb_q_in,
                    cb_k_in,
                    cb_qk_im,
                    Sq_chunk_t,
                    Sk_chunk_t_dynamic,
                    DHt,
                    qk_num_blocks,
                    qk_in0_num_subblocks_dynamic,
                    qk_in1_num_subblocks_dynamic,
                    qk_in0_block_w,
                    qk_subblock_h_dynamic,
                    qk_subblock_w_dynamic,
                    true,
                    add_mask_fusion,
                    mask_cb_to_use,
                    cb_zero_in);

                /* QK += MASK */
                if (!add_mask_fusion) {
                    if constexpr (is_causal) {
                        // For decode, we only apply mask at the last chunk for causal mode
                        if (k_chunk == k_chunk_end - 1 && apply_mask_at_last_chunk) {
                            reconfig_data_format(cb_qk_im, cb_mask_in);
                            add_block_inplace<false>(cb_qk_im, cb_mask_in, qk_chunk_tiles_dynamic);
                        }
                    } else {
                        if constexpr (use_attention_mask) {
                            reconfig_data_format(cb_qk_im, cb_mask_in);
                            add_block_inplace<true>(cb_qk_im, cb_mask_in, qk_chunk_tiles_dynamic);
                        }
                    }

                    // Apply sliding window mask to the first chunk (only on the core that processes it)
                    if (k_chunk == window_start_chunk && window_start_unaligned > 0) {
                        reconfig_data_format(cb_qk_im, cb_sliding_window_mask_in);
                        add_block_inplace<false>(cb_qk_im, cb_sliding_window_mask_in, qk_chunk_tiles_dynamic);
                    }
                }

                /**
                 * OPTIMIZATION
                 * Typically, scores are multiplied by a scalar here, but an optimization was employed
                 * where the scaling is fused into exp both in exp(x - max) and exp(prev_max - cur_max).
                 * This gives us scaling for free on the performance-critical exp(x - max) computation.
                 */

                reconfig_data_format(cb_qk_im, cb_identity_scale_in);
                pack_reconfig_data_format(cb_cur_max);

                /**
                 * OPTIMIZATION
                 * reduce_c can perform both reduce_max and eltwise max with previous result.
                 * if do_eltwise_max:
                 *  cur_max = eltwise_max(prev_max, max(qk, dim=-1))
                 * else:
                 *  cur_max = max(qk, dim=-1)
                 */
                reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity_scale_in, Sq_chunk_t, vector_mode>(
                    cb_cur_max, cb_prev_max, Sk_chunk_t_dynamic, k_chunk > k_chunk_start);

                /* QK -= cb_cur_max */
                /* QK = exp(QK)*/
                reconfig_data_format(cb_qk_im, cb_cur_max);
                pack_reconfig_data_format(cb_qk_im);

                /**
                 * sub_exp performs `QK = exp((QK - cur_max) * scale)`
                 */
                sub_exp_block_bcast_cols_inplace<cb_qk_im, Sq_chunk_t, scale_fp32, true, false, vector_mode>(
                    cb_cur_max, cb_cur_sum, Sk_chunk_t_dynamic);
                cb_wait_front(cb_qk_im, qk_chunk_tiles_dynamic);

                // Reconfig register DF
                reconfig_data_format(cb_qk_im, cb_identity_scale_in);
                pack_reconfig_data_format(cb_cur_sum);

                /* reduce_c performs CUR_SUM = sum(QK, dim = -1) */
                reduce_c<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity_scale_in, Sq_chunk_t, vector_mode>(
                    cb_cur_sum, cb_cur_sum, Sk_chunk_t_dynamic, false);

                /* OUT_IM = QK @ V_CHUNK */
                reconfig_data_format(cb_qk_im, cb_v_in);  // DEBUG
                pack_reconfig_data_format(cb_out_im);
                matmul_blocks(
                    cb_qk_im,
                    cb_v_in,
                    cb_out_mm,
                    Sq_chunk_t,
                    vDHt,
                    Sk_chunk_t_dynamic,
                    out_num_blocks_dynamic,
                    out_in0_num_subblocks,
                    out_in1_num_subblocks,
                    out_in0_block_w_dynamic,
                    out_subblock_h,
                    out_subblock_w,
                    false /*transpose*/,
                    false,
                    cb_mask_in,
                    cb_zero_in);

                // Reconfig register DF
                reconfig_data_format_srca(cb_out_im);
                cb_pop_front(cb_qk_im, qk_chunk_tiles_dynamic);

                /* OUT_ACC += OUT_IM */
                if (k_chunk == k_chunk_start) {
                    cb_out_mm = cb_out_im;
                } else {
                    // When there is more than 1 chunk, we perform Lazy Softmax
                    // Reconfig register DF
                    reconfig_data_format(cb_prev_max, cb_cur_max);
                    pack_reconfig_data_format(cb_exp_max_diff);

                    /* EXP_MAX_DIFF = exp(PREV_MAX - CUR_MAX) */
                    sub_exp_block<scale_fp32>(cb_prev_max, cb_cur_max, cb_exp_max_diff, Sq_chunk_t);
                    cb_pop_front(cb_prev_max, Sq_chunk_t);

                    /* PREV_SUM *= EXP_MAX_DIFF */
                    mul_block_inplace(cb_prev_sum, cb_exp_max_diff, Sq_chunk_t);

                    /* OUT_ACC *= EXP_MAX_DIFF */
                    reconfig_data_format(cb_out_accumulate_im, cb_exp_max_diff);
                    pack_reconfig_data_format(cb_out_accumulate_im);
                    mul_block_bcast_cols<Sq_chunk_t, vDHt, true, false>(
                        cb_out_accumulate_im, cb_exp_max_diff, cb_out_accumulate_im);

                    /* CUR_SUM += PREV_SUM */
                    reconfig_data_format(cb_cur_sum, cb_prev_sum);
                    pack_reconfig_data_format(cb_cur_sum);
                    add_block_inplace<true>(cb_cur_sum, cb_prev_sum, Sq_chunk_t);

                    /* OUT_ACC += OUT_IM */
                    reconfig_data_format(cb_out_accumulate_im, cb_out_im);
                    pack_reconfig_data_format(cb_out_accumulate_im);
                    add_block_inplace<true>(cb_out_accumulate_im, cb_out_im, out_chunk_tiles);
                }

                // Move intermediate sum and max values to appropriate ping pong buffers
                // Always move to prev buffers during FA loop - we'll handle final output later
                if (k_chunk < k_chunk_end - 1) {
                    // More local chunks to process - move to ping-pong buffers
                    reconfig_data_format(cb_cur_max, cb_cur_max);
                    pack_reconfig_data_format(cb_prev_max);

                    // PREV_MAX <- CUR_MAX
                    move_block<true>(cb_cur_max, cb_prev_max, Sq_chunk_t);

                    // PREV_SUM <- CUR_SUM
                    move_block<true>(cb_cur_sum, cb_prev_sum, Sq_chunk_t);
                }
        }

        // After FA loop completes, prepare buffers for tree reduction or output
        // Results are in: cb_out_accumulate_im (O), cb_cur_max (M), cb_cur_sum (L)
        if (actual_num_children > 0 || should_send_to_parent) {
            // Tree reduction will happen - move cur to prev buffers
            reconfig_data_format(cb_cur_max, cb_cur_max);
            pack_reconfig_data_format(cb_prev_max);
            move_block<true>(cb_cur_max, cb_prev_max, Sq_chunk_t);
            move_block<true>(cb_cur_sum, cb_prev_sum, Sq_chunk_t);
        }
        // If root with no children, keep in cur buffers for finalization

        /* END OF FLASH ATTENTION LOOP */

        /******************************************************************************
         *                      TREE REDUCTION LOGIC                                  *
         ******************************************************************************/
        /**
         * Tree reduction reduces the online softmax results in O(log n) rounds.
         *
         * For each round r (0 to my_active_rounds-1):
         *   - If children_per_round[r] != UINT32_MAX, receive from that child
         *   - Combine received data with local accumulator using softmax correction
         *
         * After all receives:
         *   - If is_tree_root: finalize (1/sum normalization) and output
         *   - Else: output intermediate results for writer to send to parent
         */
        // Tree reduction: receive from children and combine
        // Buffer state entering tree reduction:
        //   - cb_out_accumulate_im: local O (output accumulator)
        //   - cb_prev_max: local M (max of logits)
        //   - cb_prev_sum: local L (sum of exp)
        // Only receive from children that actually have data
        if (actual_num_children > 0) {
            // Iterate through each round and receive from child if one exists AND has data
            for (uint32_t round = 0; round < actual_my_active_rounds; ++round) {
                uint32_t child_id = actual_children_per_round[round];
                if (child_id != UINT32_MAX) {
                    // Writer kernel handles the semaphore wait and data transfer to cb_m_in, cb_l_in, cb_out_o
                    // Data arrives in order: l, m, o

                    // Combine child with existing local/accumulated data
                    // Move child's L to cb_prev_sum_2 for correction
                    move_block<true>(cb_l_in, cb_prev_sum_2, Sq_chunk_t);

                    // Fused Softmax Correction
                    // * Fused Correction is a fused operation that performs the following steps:
                    // * 1. CUR_MAX = max(PREV_MAX, WORKER_MAX)
                    // * 2. EXP_MAX_DIFF_2 = exp((WORKER_MAX - CUR_MAX)*scale)
                    // * 3. PREV_SUM_2 *= EXP_MAX_DIFF_2
                    // * 4. EXP_MAX_DIFF = exp((PREV_MAX - CUR_MAX)*scale)
                    // * 5. PREV_SUM *= EXP_MAX_DIFF
                    // * 6. CUR_SUM = PREV_SUM_2 + PREV_SUM
                    correction_block<scale_fp32, vector_mode>(
                        cb_m_in,        // cb child max
                        cb_prev_sum_2,  // cb child sum
                        cb_cur_max,
                        cb_prev_max,
                        cb_cur_sum,
                        cb_prev_sum,
                        cb_exp_max_diff,
                        cb_exp_max_diff_2,
                        Sq_chunk_t);

                    // OUT_ACC_2 <- CHILD_OUT
                    move_block<true>(cb_out_o, cb_out_accumulate_im_2, out_chunk_tiles);

                    // OUT_ACC *= EXP_MAX_DIFF (scale local accumulator)
                    // OUT_ACC_2 *= EXP_MAX_DIFF_2 (scale child's accumulator)
                    mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(cb_out_accumulate_im, cb_exp_max_diff);
                    mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(cb_out_accumulate_im_2, cb_exp_max_diff_2);

                    // OUT_ACC = OUT_ACC + OUT_ACC_2
                    add_block_inplace<true>(cb_out_accumulate_im, cb_out_accumulate_im_2, out_chunk_tiles);

                    // Update prev buffers for next round
                    // PREV_MAX <- CUR_MAX
                    // PREV_SUM <- CUR_SUM
                    cb_pop_front(cb_prev_max, Sq_chunk_t);
                    cb_pop_front(cb_m_in, Sq_chunk_t);
                    move_block<true>(cb_cur_max, cb_prev_max, Sq_chunk_t);
                    move_block<true>(cb_cur_sum, cb_prev_sum, Sq_chunk_t);
                }
            }
        }

        // Finalize output based on tree role
        if (is_tree_root) {
            // Root node: perform final normalization and output
            // Determine which sum/max buffer to use based on whether we did tree reduction
            // If we had children with data, results are in cb_prev_sum/cb_prev_max after tree reduction
            // If single core (no children with data), results are in cb_cur_sum/cb_cur_max from FA loop

            // Select the correct sum buffer based on whether tree reduction happened
            // If tree reduction happened, sum is in cb_prev_sum; otherwise it's in cb_cur_sum
            uint32_t sum_cb = (actual_num_children > 0) ? cb_prev_sum : cb_cur_sum;

            /* SUM = 1.0 / SUM */
            reconfig_data_format(sum_cb, sum_cb);
            pack_reconfig_data_format(sum_cb);

            // Handle attention sink here
            if constexpr (use_attention_sink) {
                // Use appropriate max buffer based on tree reduction
                uint32_t max_cb_for_sink = (actual_num_children > 0) ? cb_prev_max : cb_cur_max;

                // m_new
                max_block<vector_mode>(cb_attention_sink, max_cb_for_sink, cb_cur_max, Sq_chunk_t);

                // exp(m - m_new)
                sub_exp_block<scale_fp32>(max_cb_for_sink, cb_cur_max, cb_exp_max_diff, Sq_chunk_t);

                // l -> l * exp(m - m_new)
                mul_block_inplace(sum_cb, cb_exp_max_diff, Sq_chunk_t);

                // exp(sink - m_new)
                sub_exp_block<scale_fp32>(cb_attention_sink, cb_cur_max, cb_exp_max_diff_2, Sq_chunk_t);
                cb_pop_front(cb_cur_max, Sq_chunk_t);

                // l -> l + exp(sink - m_new)
                add_block_inplace<true>(sum_cb, cb_exp_max_diff_2, Sq_chunk_t);

                // O -> O * exp(m - m_new)
                mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(cb_out_accumulate_im, cb_exp_max_diff);
            }

            reconfig_data_format(sum_cb, sum_cb);
            pack_reconfig_data_format(sum_cb);
            recip_block_inplace(sum_cb, Sq_chunk_t);

            /* OUT_ACC *= 1/SUM */
            reconfig_data_format(cb_out_accumulate_im, sum_cb);
            pack_reconfig_data_format(cb_out_accumulate_im);

            mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(cb_out_accumulate_im, sum_cb);
            pack_reconfig_data_format(cb_out_final);

            // Untilize output to ROW MAJOR if input Q was also ROW MAJOR
            if constexpr (untilize_output) {
                // Conditionally use pack_untilize or untilize
                if constexpr (use_pack_untilize) {
                    pack_untilize_init<out_chunk_tiles>(cb_out_accumulate_im, cb_out_final);
                } else {
                    untilize_init(cb_out_accumulate_im);
                }
                cb_wait_front(cb_out_accumulate_im, out_chunk_tiles);
                cb_reserve_back(cb_out_final, out_chunk_tiles);
                if constexpr (use_pack_untilize) {
                    pack_untilize_block<out_chunk_tiles>(cb_out_accumulate_im, 1, cb_out_final);
                } else {
                    untilize_block(cb_out_accumulate_im, out_chunk_tiles, cb_out_final);
                }
                if constexpr (use_pack_untilize) {
                    pack_untilize_uninit(cb_out_final);
                } else {
                    untilize_uninit(cb_out_final);
                }
                cb_pop_front(cb_out_accumulate_im, out_chunk_tiles);
                cb_push_back(cb_out_final, out_chunk_tiles);
            } else {
                // Move output to buffer for the writer
                move_block<true>(cb_out_accumulate_im, cb_out_final, out_chunk_tiles);
            }

        } else if (should_send_to_parent) {
            // Non-root node with parent: send intermediate results
            // We have data (checked at function start), so send it
            // After tree reduction (if any), results are in:
            //   - cb_out_accumulate_im: O
            //   - cb_prev_sum: L
            //   - cb_prev_max: M

            // Move O to output CB
            move_block<true>(cb_out_accumulate_im, cb_out_o, out_chunk_tiles);
            // Move M to output CB
            move_block<true>(cb_prev_max, cb_out_m, Sq_chunk_t);
            // Move L to output CB
            move_block<true>(cb_prev_sum, cb_out_l, Sq_chunk_t);
        }

        // Free up prev buffers if we used them
        cb_pop_front(cb_prev_max, Sq_chunk_t);
        cb_pop_front(cb_prev_sum, Sq_chunk_t);
    }

    // Free up cb_q_in after Q chunks
    cb_pop_front(cb_q_in, q_chunk_tiles);
}
