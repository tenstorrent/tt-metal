// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// This kernel reconfigs ~30x; force the Src zero-flag DEFAULT configurator out-of-line (one shared copy
// reached by a call at each reconfig/init site, instead of an inlined fast path at every one) to reclaim
// kernel-config-buffer space. Must be defined before the compute API includes. Perf-neutral (init-time only).
#define LLK_ZEROFLAG_OUTLINE 1

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)
#define MAX_TREE_REDUCTION_ROUNDS 6

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/tilize.h"
#include "api/compute/pack_untilize.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"
#include "api/compute/pack_untilize.h"
constexpr uint32_t MAX_PACK_UNTILIZE_WIDTH = 8;
#include "ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/kernel_lib/untilize_helpers.hpp"

void kernel_main() {
    // Compile time arguments

    // Input dimensions in tiles
    constexpr auto St = get_arg(args::St);
    constexpr auto DHt = get_arg(args::DHt);
    constexpr auto vDHt = get_arg(args::vDHt);
    constexpr auto Sq_chunk_t = get_arg(args::Sq_chunk_t);
    constexpr auto Sk_chunk_t = get_arg(args::Sk_chunk_t);

    // Matmul configs
    constexpr auto qk_in0_block_w = get_arg(args::qk_in0_block_w);
    constexpr auto qk_subblock_w = get_arg(args::qk_subblock_w);
    constexpr auto qk_subblock_h = get_arg(args::qk_subblock_h);
    constexpr auto qk_in0_num_subblocks = get_arg(args::qk_in0_num_subblocks);
    constexpr auto qk_in1_num_subblocks = get_arg(args::qk_in1_num_subblocks);
    constexpr auto qk_num_blocks = get_arg(args::qk_num_blocks);
    constexpr auto out_in0_block_w = get_arg(args::out_in0_block_w);
    constexpr auto out_subblock_w = get_arg(args::out_subblock_w);
    constexpr auto out_subblock_h = get_arg(args::out_subblock_h);
    constexpr auto out_in0_num_subblocks = get_arg(args::out_in0_num_subblocks);
    constexpr auto out_in1_num_subblocks = get_arg(args::out_in1_num_subblocks);
    constexpr auto out_num_blocks = get_arg(args::out_num_blocks);
    constexpr auto num_cores_per_head = get_arg(args::num_cores_per_head);
    constexpr auto num_heads_per_core = get_arg(args::num_heads_per_core);

    // Attention-specific parameters
    constexpr auto max_dynamic_chunk_size = get_arg(args::max_dynamic_chunk_size);
    constexpr auto q_heads_parallel_factor = get_arg(args::q_heads_parallel_factor);
    constexpr bool use_half_tile = get_arg(args::use_half_tile);
    constexpr auto scale_fp32 = get_arg(args::scale_fp32);
    constexpr auto sliding_window_size = get_arg(args::sliding_window_size);
    constexpr auto num_tree_reduction_rounds = get_arg(args::num_tree_reduction_rounds);
    constexpr auto original_block_size = get_arg(args::original_block_size);

#ifdef IS_CAUSAL
    constexpr bool is_causal = true;
#else
    constexpr bool is_causal = false;
#endif
#ifdef USE_ATTENTION_MASK
    constexpr bool use_attention_mask = true;
#else
    constexpr bool use_attention_mask = false;
#endif
    constexpr bool has_block_padding = original_block_size > 0 && original_block_size < 32;

    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;

    // CB index definitions (each exists only where the host bound it — see #ifdef gates).
    constexpr auto dfb_q_in = dfb::q_in;
    constexpr auto dfb_k_in = dfb::k_in;
    constexpr auto dfb_v_in = dfb::v_in;
    constexpr auto dfb_mask_in = dfb::mask_in;
#ifdef SLIDING_WINDOW
    constexpr auto dfb_sliding_window_mask_in = dfb::sliding_window_mask_in;
#endif
#ifdef HAS_BLOCK_PADDING
    constexpr auto dfb_block_pad_mask = dfb::block_pad_mask;
#endif
#ifdef USE_ATTENTION_SINK
    constexpr auto dfb_attention_sink = dfb::attention_sink;
#endif
    constexpr auto dfb_identity_scale_in = dfb::identity_scale_in;
    constexpr auto dfb_m_in = dfb::m_in;
    constexpr auto dfb_l_in = dfb::l_in;
#ifdef TILIZE_Q
    constexpr auto dfb_q_rm = dfb::q_rm;
#endif
    constexpr auto dfb_zero_in = dfb::zero_in;
#ifdef USE_CUR_POS_TENSOR
    // #44366: compute reads cur_pos from c_15 (writer reads from c_8) — see reader_decode_all.cpp.
    constexpr auto dfb_cur_pos = dfb::cur_pos;
#endif

    constexpr auto dfb_qk_im = dfb::qk_im;
    constexpr auto dfb_out_im = dfb::out_im;
    constexpr auto dfb_out_accumulate_im = dfb::out_accumulate_im;
    constexpr auto dfb_max_1 = dfb::max_1;
    constexpr auto dfb_max_2 = dfb::max_2;
    constexpr auto dfb_sum_1 = dfb::sum_1;
    constexpr auto dfb_sum_2 = dfb::sum_2;
    constexpr auto dfb_exp_max_diff = dfb::exp_max_diff;
    constexpr auto dfb_prev_sum_2 = dfb::prev_sum_2;
    constexpr auto dfb_exp_max_diff_2 = dfb::exp_max_diff_2;
    constexpr auto dfb_out_accumulate_im_2 = dfb::out_accumulate_im_2;

    constexpr auto dfb_out_o = dfb::out_o;
    constexpr auto dfb_out_m = dfb::out_m;
    constexpr auto dfb_out_l = dfb::out_l;
    constexpr auto dfb_out_final = dfb::out;

#ifdef TILIZE_Q
    constexpr bool untilize_output = true;
#else
    constexpr bool untilize_output = false;
#endif

    // Runtime arguments
    const bool do_reduce = get_arg(args::do_reduce) == 1;
    const bool apply_mask_at_last_chunk = do_reduce && is_causal;
    const bool do_output = get_arg(args::do_output) == 1;
    const uint32_t cur_head = get_arg(args::cur_head);
    const uint32_t cur_batch = get_arg(args::cur_batch);
    const uint32_t core_num_in_reduce = get_arg(args::core_num_in_reduce);
    const uint32_t core_num_in_output = get_arg(args::core_num_in_output);
    const uint32_t cur_pos_arg = get_arg(args::cur_pos_arg);

    // Tree reduction runtime arguments
    const bool is_tree_root = get_arg(args::is_tree_root) == 1;
    const uint32_t parent_core_in_group = get_arg(args::parent_core_in_group);
    const uint32_t send_at_round = get_arg(args::send_at_round);
    const uint32_t num_children = get_arg(args::num_children);
    const uint32_t my_active_rounds = get_arg(args::my_active_rounds);
    const bool has_parent = parent_core_in_group != UINT32_MAX;

    // Read children_per_round array
    uint32_t children_per_round[MAX_TREE_REDUCTION_ROUNDS] = {
        get_arg(args::children_per_round_0),
        get_arg(args::children_per_round_1),
        get_arg(args::children_per_round_2),
        get_arg(args::children_per_round_3),
        get_arg(args::children_per_round_4),
        get_arg(args::children_per_round_5)};

    // Get cur_pos
    constexpr uint32_t cur_pos_base = St * 32 - 1;
    uint32_t cur_pos = cur_pos_base;  // default to non-causal, which we do attention on the entire kv cache. In this
                                      // case we set cur_pos to the last position
    if constexpr (is_causal) {
        // using UINT32_MAX as a flag to indicate that cur_pos is not provided as a list
        if (cur_pos_arg != UINT32_MAX) {
            cur_pos = cur_pos_arg;
        } else {
#ifdef USE_CUR_POS_TENSOR
            // Read cur_pos from CB using mailbox-based synchronization (issue #27979).
            DataflowBuffer dfb_cur_pos_buf(dfb_cur_pos);
            dfb_cur_pos_buf.wait_front(1);
            cur_pos = dfb_cur_pos_buf.read_tile_value(0, cur_batch / q_heads_parallel_factor);
            dfb_cur_pos_buf.pop_front(1);
#endif
        }
        if (cur_pos == UINT32_MAX) {
            // cur_pos of -1 indicates that the user should be skipped
            return;
        }
    }

    // When block_size < TILE_HEIGHT, convert cur_pos to padded tile space.
    // Only for causal mode; non-causal cur_pos is already in padded space.
    if constexpr (has_block_padding && is_causal) {
        cur_pos = (cur_pos / original_block_size) * 32 + (cur_pos % original_block_size);
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

    // Determine which children actually participate in reduction (based on chunk allocation)
    // A child at core_num is active or has data if core_num < k_num_chunks
    // E.g. k_num_chunks = 2, num_cores_per_head = 4
    // | core 0 | core 1 | core 2 | core 3 |
    //   chunk 0  chunk 1   NA       NA
    // core 0 would have core 1 and core 2 as children, but only core 1 is active to perform reduction with
    uint32_t num_active_children = 0;
    uint32_t active_children_per_round[MAX_TREE_REDUCTION_ROUNDS];
    uint32_t num_active_rounds = 0;

    for (uint32_t r = 0; r < MAX_TREE_REDUCTION_ROUNDS; ++r) {
        uint32_t child_id = children_per_round[r];
        if (child_id != UINT32_MAX && child_id < k_num_chunks) {
            // This child has data
            active_children_per_round[r] = child_id;
            num_active_children++;
            num_active_rounds = r + 1;
        } else {
            active_children_per_round[r] = UINT32_MAX;
        }
    }

    // We tilize input Q if it is in ROW MAJOR layout
#ifdef TILIZE_Q
    compute_kernel_hw_startup(dfb_q_rm, dfb_q_in);
    // Keep InitAndUninit: the helper picks fast- vs regular-tilize at compile time and its
    // teardown must match (fast_tilize_uninit vs tilize_uninit). tilize_q with a FULL-tile Q
    // (use_half_tile==false, e.g. >16 heads) can take the fast-tilize path, so we must not
    // hand-roll the uninit here.
    compute_kernel_lib::tilize<
        q_chunk_tiles,
        dfb_q_rm,
        dfb_q_in,
        compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
    matmul_init(dfb_q_in, dfb_k_in);
    // #49266: The Q tilize runs on SrcA; on galaxy Q is a half-tile (num_faces=2), and
    // tilize_uninit correctly restores SrcA to Q's geometry. But the QK matmul reads operands
    // REVERSED (SrcA <- in1 = dfb_k_in, num_faces=4), and the per-k_chunk reconfig below is
    // IGNORE (format-only). Reprogram SrcA/SrcB tile geometry ONCE here for the matmul
    // operands (is_tile_dim_reconfig_en=true) so K is unpacked with the correct num_faces.
    // One-time, not per-chunk: nothing after this re-establishes Q's geometry on SrcA (K and V
    // are both full tiles), so the per-chunk reconfig can stay IGNORE. Without this, SrcA stays
    // at num_faces=2 and the matmul reads K wrong -> Top-1 0%. (Full-tile Q: this is a no-op
    // re-assert of num_faces=4.)
    reconfig_full_operand(dfb_k_in, dfb_q_in);
#else
    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb_q_in, dfb_k_in, dfb_qk_im);
    matmul_init(dfb_q_in, dfb_k_in);
#endif
    DataflowBuffer(dfb_q_in).wait_front(q_chunk_tiles);

    // Wait for block padding mask (generated once by writer, reused every chunk without popping)
#ifdef HAS_BLOCK_PADDING
    {
        uint32_t block_pad_mask_tiles = Sq_chunk_t * Sk_chunk_t_dynamic;
        DataflowBuffer(dfb_block_pad_mask).wait_front(block_pad_mask_tiles);
    }
#endif

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
    constexpr VectorMode vector_mode = use_half_tile ? VectorMode::R : VectorMode::RC;

    // We set up Ping Pong intermediate buffers between loops
    uint32_t dfb_cur_max = dfb_max_1;
    uint32_t dfb_prev_max = dfb_max_2;
    uint32_t dfb_cur_sum = dfb_sum_1;
    uint32_t dfb_prev_sum = dfb_sum_2;

    // Loop through all heads assigned to core
    for (uint32_t cur_head_work = 0; cur_head_work < num_heads_per_core; ++cur_head_work) {
        // Reset ping-pong buffer assignments at the start of each head iteration
        dfb_cur_max = dfb_max_1;
        dfb_prev_max = dfb_max_2;
        dfb_cur_sum = dfb_sum_1;
        dfb_prev_sum = dfb_sum_2;

        /* START OF FLASH ATTENTION LOOP */
        uint32_t dfb_out_mm = dfb_out_accumulate_im;

        // Loop through all K chunks
        for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
            // Reconfig register DF
            reconfig_data_format(dfb_k_in, dfb_q_in);
            pack_reconfig_data_format(dfb_qk_im);

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
            uint32_t mask_dfb_to_use = dfb_mask_in;  // Default to causal mask buffer
#ifdef SLIDING_WINDOW
            if (add_sliding_window_mask_fusion) {
                mask_dfb_to_use = dfb_sliding_window_mask_in;  // Use sliding window mask buffer
            }
#endif

            matmul_blocks(
                dfb_q_in,
                dfb_k_in,
                dfb_qk_im,
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
                mask_dfb_to_use,
                dfb_zero_in);

            /* QK += MASK */
            // Apply block padding mask for every chunk when block_size < TILE_HEIGHT.
            // Uses <false> to NOT pop the mask CB so it can be reused for subsequent chunks.
            // Applied outside mask_fusion conditional since it's always needed independently.
#ifdef HAS_BLOCK_PADDING
            reconfig_data_format(dfb_qk_im, dfb_block_pad_mask);
            add_block_inplace<false>(dfb_qk_im, dfb_block_pad_mask, qk_chunk_tiles_dynamic);
#endif

            if (!add_mask_fusion) {
                if constexpr (is_causal) {
                    // For decode, we only apply mask at the last chunk for causal mode
                    if (k_chunk == k_chunk_end - 1 && apply_mask_at_last_chunk) {
                        reconfig_data_format(dfb_qk_im, dfb_mask_in);
                        add_block_inplace<false>(dfb_qk_im, dfb_mask_in, qk_chunk_tiles_dynamic);
                    }
                } else {
                    if constexpr (use_attention_mask) {
                        reconfig_data_format(dfb_qk_im, dfb_mask_in);
                        add_block_inplace<true>(dfb_qk_im, dfb_mask_in, qk_chunk_tiles_dynamic);
                    }
                }

                // Apply sliding window mask to the first chunk (only on the core that processes it)
#ifdef SLIDING_WINDOW
                if (k_chunk == window_start_chunk && window_start_unaligned > 0) {
                    reconfig_data_format(dfb_qk_im, dfb_sliding_window_mask_in);
                    add_block_inplace<false>(dfb_qk_im, dfb_sliding_window_mask_in, qk_chunk_tiles_dynamic);
                }
#endif
            }

            /**
             * OPTIMIZATION
             * Typically, scores are multiplied by a scalar here, but an optimization was employed
             * where the scaling is fused into exp both in exp(x - max) and exp(prev_max - cur_max).
             * This gives us scaling for free on the performance-critical exp(x - max) computation.
             */

            reconfig_data_format(dfb_qk_im, dfb_identity_scale_in);
            pack_reconfig_data_format(dfb_cur_max);

            /**
             * OPTIMIZATION
             * reduce_c can perform both reduce_max and eltwise max with previous result.
             * if do_eltwise_max:
             *  cur_max = eltwise_max(prev_max, max(qk, dim=-1))
             * else:
             *  cur_max = max(qk, dim=-1)
             */
            reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, dfb_qk_im, dfb_identity_scale_in, Sq_chunk_t, vector_mode>(
                dfb_cur_max, dfb_prev_max, Sk_chunk_t_dynamic, k_chunk > k_chunk_start);
            /* QK -= dfb_cur_max */
            /* QK = exp(QK)*/
            reconfig_data_format(dfb_qk_im, dfb_cur_max);
            pack_reconfig_data_format(dfb_qk_im);

            /**
             * sub_exp performs `QK = exp((QK - cur_max) * scale)`
             */
            sub_exp_block_bcast_cols_inplace<dfb_qk_im, Sq_chunk_t, scale_fp32, true, false, vector_mode>(
                dfb_cur_max, dfb_cur_sum, Sk_chunk_t_dynamic);
            DataflowBuffer(dfb_qk_im).wait_front(qk_chunk_tiles_dynamic);

            // Reconfig register DF
            reconfig_data_format(dfb_qk_im, dfb_identity_scale_in);
            pack_reconfig_data_format(dfb_cur_sum);

            /* reduce_c performs CUR_SUM = sum(QK, dim = -1) */
            reduce_c<PoolType::SUM, ReduceDim::REDUCE_ROW, dfb_qk_im, dfb_identity_scale_in, Sq_chunk_t, vector_mode>(
                dfb_cur_sum, dfb_cur_sum, Sk_chunk_t_dynamic, false);

            /* OUT_IM = QK @ V_CHUNK */
            reconfig_data_format(dfb_v_in, dfb_qk_im);  // DEBUG
            pack_reconfig_data_format(dfb_out_im);
            matmul_blocks(
                dfb_qk_im,
                dfb_v_in,
                dfb_out_mm,
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
                dfb_mask_in,
                dfb_zero_in);

            // Reconfig register DF
            reconfig_data_format_srca(dfb_out_im);
            DataflowBuffer(dfb_qk_im).pop_front(qk_chunk_tiles_dynamic);

            /* OUT_ACC += OUT_IM */
            if (k_chunk == k_chunk_start) {
                dfb_out_mm = dfb_out_im;
            } else {
                // When there is more than 1 chunk, we perform Lazy Softmax
                // Reconfig register DF
                reconfig_data_format(dfb_prev_max, dfb_cur_max);
                pack_reconfig_data_format(dfb_exp_max_diff);

                /* EXP_MAX_DIFF = exp(PREV_MAX - CUR_MAX) */
                sub_exp_block<scale_fp32>(dfb_prev_max, dfb_cur_max, dfb_exp_max_diff, Sq_chunk_t);
                DataflowBuffer(dfb_prev_max).pop_front(Sq_chunk_t);

                /* PREV_SUM *= EXP_MAX_DIFF */
                mul_block_inplace(dfb_prev_sum, dfb_exp_max_diff, Sq_chunk_t);

                /* OUT_ACC *= EXP_MAX_DIFF */
                reconfig_data_format(dfb_out_accumulate_im, dfb_exp_max_diff);
                pack_reconfig_data_format(dfb_out_accumulate_im);
                mul_block_bcast_cols<Sq_chunk_t, vDHt, true, false>(
                    dfb_out_accumulate_im, dfb_exp_max_diff, dfb_out_accumulate_im);

                /* CUR_SUM += PREV_SUM */
                reconfig_data_format(dfb_cur_sum, dfb_prev_sum);
                pack_reconfig_data_format(dfb_cur_sum);
                add_block_inplace<true>(dfb_cur_sum, dfb_prev_sum, Sq_chunk_t);

                /* OUT_ACC += OUT_IM */
                reconfig_data_format(dfb_out_accumulate_im, dfb_out_im);
                pack_reconfig_data_format(dfb_out_accumulate_im);
                add_block_inplace<true>(dfb_out_accumulate_im, dfb_out_im, out_chunk_tiles);
            }

            // More local chunks to process - move intermediate sum and max values to ping-pong buffers
            reconfig_data_format(dfb_cur_max, dfb_cur_max);
            pack_reconfig_data_format(dfb_prev_max);

            // PREV_MAX <- CUR_MAX
            move_block<true>(dfb_cur_max, dfb_prev_max, Sq_chunk_t);

            // PREV_SUM <- CUR_SUM
            move_block<true>(dfb_cur_sum, dfb_prev_sum, Sq_chunk_t);
        }

        /* END OF FLASH ATTENTION LOOP */

        /******************************************************************************
         *                      TREE REDUCTION LOGIC                                  *
         ******************************************************************************/
        // Tree reduction: receive from children and combine
        // Buffer state entering tree reduction:
        //   - dfb_out_accumulate_im: local O (output accumulator)
        //   - dfb_prev_max: local M (max of logits)
        //   - dfb_prev_sum: local L (sum of exp)
        // Only receive from children that actually have data
        if (num_active_children > 0) {
            // Iterate through each round and receive from child if one exists AND has data
            for (uint32_t round = 0; round < num_active_rounds; ++round) {
                uint32_t child_id = active_children_per_round[round];
                if (child_id != UINT32_MAX) {
                    // Writer kernel handles the semaphore wait and data transfer to dfb_m_in, dfb_l_in, dfb_out_o
                    // Data arrives in order: l, m, o

                    // Combine child with existing local/accumulated data
                    // Move child's L to dfb_prev_sum_2 for correction
                    move_block<true>(dfb_l_in, dfb_prev_sum_2, Sq_chunk_t);
                    // Fused Softmax Correction
                    correction_block<scale_fp32, vector_mode>(
                        dfb_m_in,        // cb child max
                        dfb_prev_sum_2,  // cb child sum
                        dfb_cur_max,
                        dfb_prev_max,
                        dfb_cur_sum,
                        dfb_prev_sum,
                        dfb_exp_max_diff,
                        dfb_exp_max_diff_2,
                        Sq_chunk_t);

                    // OUT_ACC_2 <- CHILD_OUT
                    move_block<true>(dfb_out_o, dfb_out_accumulate_im_2, out_chunk_tiles);

                    // OUT_ACC *= EXP_MAX_DIFF (scale local accumulator)
                    // OUT_ACC_2 *= EXP_MAX_DIFF_2 (scale child's accumulator)
                    mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(dfb_out_accumulate_im, dfb_exp_max_diff);
                    mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(dfb_out_accumulate_im_2, dfb_exp_max_diff_2);

                    // OUT_ACC = OUT_ACC + OUT_ACC_2
                    add_block_inplace<true>(dfb_out_accumulate_im, dfb_out_accumulate_im_2, out_chunk_tiles);

                    // Update prev buffers for next round
                    // PREV_MAX <- CUR_MAX
                    // PREV_SUM <- CUR_SUM
                    DataflowBuffer(dfb_prev_max).pop_front(Sq_chunk_t);
                    DataflowBuffer(dfb_m_in).pop_front(Sq_chunk_t);
                    move_block<true>(dfb_cur_max, dfb_prev_max, Sq_chunk_t);
                    move_block<true>(dfb_cur_sum, dfb_prev_sum, Sq_chunk_t);
                }
            }
        }

        // Finalize output based on tree role
        if (is_tree_root) {
            // Root node: perform final normalization and output
            /* SUM = 1.0 / SUM */
            reconfig_data_format(dfb_prev_sum, dfb_prev_sum);
            pack_reconfig_data_format(dfb_prev_sum);

            // Handle attention sink here
#ifdef USE_ATTENTION_SINK
            {
                // Use appropriate max buffer based on tree reduction
                uint32_t max_dfb_for_sink = dfb_prev_max;

                // m_new
                max_block<vector_mode>(dfb_attention_sink, max_dfb_for_sink, dfb_cur_max, Sq_chunk_t);

                // exp(m - m_new)
                sub_exp_block<scale_fp32>(max_dfb_for_sink, dfb_cur_max, dfb_exp_max_diff, Sq_chunk_t);

                // l -> l * exp(m - m_new)
                mul_block_inplace(dfb_prev_sum, dfb_exp_max_diff, Sq_chunk_t);

                // exp(sink - m_new)
                sub_exp_block<scale_fp32>(dfb_attention_sink, dfb_cur_max, dfb_exp_max_diff_2, Sq_chunk_t);
                DataflowBuffer(dfb_cur_max).pop_front(Sq_chunk_t);

                // l -> l + exp(sink - m_new)
                add_block_inplace<true>(dfb_prev_sum, dfb_exp_max_diff_2, Sq_chunk_t);

                // O -> O * exp(m - m_new)
                mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(dfb_out_accumulate_im, dfb_exp_max_diff);
            }
#endif

            reconfig_data_format(dfb_prev_sum, dfb_prev_sum);
            pack_reconfig_data_format(dfb_prev_sum);
            recip_block_inplace(dfb_prev_sum, Sq_chunk_t);

            /* OUT_ACC *= 1/SUM */
            reconfig_data_format(dfb_out_accumulate_im, dfb_prev_sum);
            pack_reconfig_data_format(dfb_out_accumulate_im);

            // dfb_prev_sum is consumed and popped by mul_block_bcast_cols_inplace
            mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(dfb_out_accumulate_im, dfb_prev_sum);
            pack_reconfig_data_format(dfb_out_final);

            // Pop the max buffer that still has data
            DataflowBuffer(dfb_prev_max).pop_front(Sq_chunk_t);

            // Untilize output to ROW MAJOR if input Q was also ROW MAJOR
#ifdef TILIZE_Q
            // Unified untilize - auto-dispatches based on out_chunk_tiles vs DEST limit
            compute_kernel_lib::untilize<
                out_chunk_tiles,
                dfb_out_accumulate_im,
                dfb_out_final,
                compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
                compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
#else
            // Move output to buffer for the writer
            move_block<true>(dfb_out_accumulate_im, dfb_out_final, out_chunk_tiles);
#endif

        } else if (has_parent) {
            // Non-root node with parent: send intermediate results
            // We have data (checked at function start), so send it
            // After tree reduction (if any), results are in:
            //   - dfb_out_accumulate_im: O
            //   - dfb_prev_sum: L
            //   - dfb_prev_max: M
            // Move O to output CB
            move_block<true>(dfb_out_accumulate_im, dfb_out_o, out_chunk_tiles);
            // Move M to output CB
            move_block<true>(dfb_prev_max, dfb_out_m, Sq_chunk_t);
            // Move L to output CB
            move_block<true>(dfb_prev_sum, dfb_out_l, Sq_chunk_t);
        }
    }

    // Free up dfb_q_in after Q chunks
    DataflowBuffer(dfb_q_in).pop_front(q_chunk_tiles);
}
