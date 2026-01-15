// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "compute_kernel_api.h"
#include <tt-metalium/constants.hpp>
#include "compute_common.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/fused_op_indexer.hpp"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NH = get_compile_time_arg_val(1);
    constexpr uint32_t DHt = get_compile_time_arg_val(2);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(3);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(4);
    constexpr uint32_t local_padded_N = get_compile_time_arg_val(5);
    constexpr uint32_t local_padded_Nt = get_compile_time_arg_val(6);
    constexpr uint32_t padded_Nt = get_compile_time_arg_val(7);
    constexpr uint32_t logical_n = get_compile_time_arg_val(8);
    constexpr uint32_t logical_nt = get_compile_time_arg_val(9);
    constexpr uint32_t Lt = get_compile_time_arg_val(10);
    constexpr uint32_t L = get_compile_time_arg_val(11);
    constexpr uint32_t num_local_q_chunks = get_compile_time_arg_val(12);
    constexpr uint32_t num_joint_q_chunks = get_compile_time_arg_val(13);
    constexpr uint32_t num_local_k_chunks = get_compile_time_arg_val(14);
    constexpr uint32_t num_joint_k_chunks = get_compile_time_arg_val(15);
    constexpr uint32_t num_q_chunks = get_compile_time_arg_val(16);
    constexpr uint32_t ring_size = get_compile_time_arg_val(17);

    constexpr uint32_t qk_in0_block_w = get_compile_time_arg_val(18);
    constexpr uint32_t qk_subblock_w = get_compile_time_arg_val(19);
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(20);
    constexpr uint32_t qk_in0_num_subblocks = get_compile_time_arg_val(21);
    constexpr uint32_t qk_in1_num_subblocks = get_compile_time_arg_val(22);
    constexpr uint32_t qk_num_blocks = get_compile_time_arg_val(23);
    constexpr uint32_t out_in0_block_w = get_compile_time_arg_val(24);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(25);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(26);
    constexpr uint32_t out_in0_num_subblocks = get_compile_time_arg_val(27);
    constexpr uint32_t out_in1_num_subblocks = get_compile_time_arg_val(28);
    constexpr uint32_t out_num_blocks = get_compile_time_arg_val(29);

    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(30);
    uint32_t argidx = 0;
    const uint32_t global_q_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t global_q_end = get_arg_val<uint32_t>(argidx++);

    RingSDPAOpIndexer fused_op_indexer = RingSDPAOpIndexer(argidx);

    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t qk_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * DHt;

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_scale_in = tt::CBIndex::c_4;
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_8;
    constexpr uint32_t cb_lse_in = tt::CBIndex::c_6;
    constexpr uint32_t cb_prev_out = tt::CBIndex::c_7;
    constexpr uint32_t cb_qk_im = tt::CBIndex::c_24;
    constexpr uint32_t cb_out_im_A = tt::CBIndex::c_25;
    constexpr uint32_t cb_out_im_B = tt::CBIndex::c_26;
    constexpr uint32_t cb_max_A = tt::CBIndex::c_27;
    constexpr uint32_t cb_max_B = tt::CBIndex::c_28;
    constexpr uint32_t cb_sum_A = tt::CBIndex::c_29;
    constexpr uint32_t cb_sum_B = tt::CBIndex::c_30;
    constexpr uint32_t cb_exp_max_diff = tt::CBIndex::c_31;

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_lse_out = tt::CBIndex::c_17;

    mm_init(cb_q_in, cb_k_in, cb_qk_im);

    for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
        uint32_t ring_id = fused_op_indexer.get_next_ring_id_and_sync();
        const bool do_joint_kv = ring_id == ring_size - 1;
        const uint32_t num_kv_chunks = do_joint_kv ? num_local_k_chunks + num_joint_k_chunks : num_local_k_chunks;

        // First, find out if this ring iter processes any KV chunks.
        const uint32_t ring_iter_kv_start_tile = ring_id * local_padded_Nt;
        const uint32_t ring_iter_kv_end_tile = ring_iter_kv_start_tile + num_local_k_chunks * Sk_chunk_t;
        const uint32_t global_n_tile_id = logical_n / tt::constants::TILE_HEIGHT;
        const bool ring_iter_processes_KV_chunks = ring_iter_kv_start_tile <= global_n_tile_id;
        const bool ring_iter_does_work = ring_iter_processes_KV_chunks || (do_joint_kv && L != 0);

        uint32_t KV_chunks_processed_in_iter = 0;
        if (!ring_iter_does_work) {
            continue;
        }

        const int32_t global_n_within_ring_iter = logical_n - ring_id * local_padded_N;
        // Note the > and <=. This means there is real length of logical_n within this ring iter.
        const bool global_n_is_within_ring_iter =
            global_n_within_ring_iter > 0 && global_n_within_ring_iter <= (int32_t)local_padded_N;
        const bool global_n_needs_masking = global_n_within_ring_iter % (Sk_chunk_t * tt::constants::TILE_HEIGHT) != 0;
        const bool ring_iter_needs_global_n_mask = global_n_is_within_ring_iter && global_n_needs_masking;
        const uint32_t global_n_mask_chunk_id = global_n_within_ring_iter / (Sk_chunk_t * tt::constants::TILE_HEIGHT);

        // LOCAL N MASK
        const bool local_n_needs_masking = local_padded_Nt % Sk_chunk_t != 0;
        const uint32_t local_n_mask_chunk_id = local_padded_Nt / Sk_chunk_t;

        // JOINT L MASK
        const bool joint_n_needs_masking = L % (Sk_chunk_t * tt::constants::TILE_HEIGHT) != 0;
        const bool ring_iter_needs_joint_n_mask = joint_n_needs_masking && do_joint_kv;
        const uint32_t joint_n_mask_chunk_id = L / (Sk_chunk_t * tt::constants::TILE_HEIGHT);

        // Iterate over KV gathered on the ring
        for (uint32_t global_q_chunk = global_q_start; global_q_chunk < global_q_end; ++global_q_chunk) {
            // Set up ping pong buffers
            uint32_t alias_prev_sum = cb_sum_A;
            uint32_t alias_cur_sum = cb_sum_B;
            uint32_t alias_prev_max = cb_max_A;
            uint32_t alias_cur_max = cb_max_B;
            uint32_t alias_mm2_prev_out = cb_out_im_A;
            uint32_t alias_mm2_cur_out = cb_out_im_B;

            uint32_t processed_k_chunks = 0;

            for (uint32_t k_chunk = 0; k_chunk < num_kv_chunks; ++k_chunk) {
                const bool kv_chunk_is_joint = k_chunk >= num_local_k_chunks;
                // Global index into the padded KV tensor
                const uint32_t kv_global_start_tile = local_padded_Nt * ring_id + k_chunk * Sk_chunk_t;
                const bool kv_chunk_is_beyond_logical_n = !kv_chunk_is_joint && (kv_global_start_tile >= logical_nt);

                if (kv_chunk_is_beyond_logical_n) {
                    // This is a KV chunk on spatial input beyond the logical N, and not joint KV. Skip it.
                    continue;
                }

                KV_chunks_processed_in_iter++;

                bool should_mask = false;
                if (ring_iter_needs_global_n_mask && k_chunk == global_n_mask_chunk_id) {
                    should_mask = true;
                } else if (local_n_needs_masking && k_chunk == local_n_mask_chunk_id) {
                    should_mask = true;
                } else if (ring_iter_needs_joint_n_mask && (k_chunk - num_local_k_chunks) == joint_n_mask_chunk_id) {
                    should_mask = true;
                }

                /* QK = Q_CHUNK @ K_CHUNK */
                pack_reconfig_data_format(cb_qk_im);
                matmul_blocks(
                    cb_q_in,
                    cb_k_in,
                    cb_qk_im,
                    Sq_chunk_t,
                    Sk_chunk_t,
                    DHt,
                    qk_num_blocks,
                    qk_in0_num_subblocks,
                    qk_in1_num_subblocks,
                    qk_in0_block_w,
                    qk_subblock_h,
                    qk_subblock_w,
                    true /*transpose*/);

                /* QK *= SCALE */
                if (should_mask) {
                    /* QK += MASK */
                    reconfig_data_format(cb_qk_im, cb_mask_in);
                    add_block_inplace(cb_qk_im, cb_mask_in, qk_chunk_tiles);
                }

                /**
                 * reduce_c can perform both reduce_max and eltwise max with previous result.
                 * if do_eltwise_max:
                 *  cur_max = eltwise_max(prev_max, max(qk, dim=-1))
                 * else:
                 *  cur_max = max(qk, dim=-1)
                 */
                reconfig_data_format(cb_qk_im, cb_identity_scale_in);
                reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_qk_im, cb_identity_scale_in, Sq_chunk_t, Sk_chunk_t>(
                    alias_cur_max, alias_prev_max, processed_k_chunks > 0);

                /**
                 * sub_exp fuses a few operations.
                 * In-place it performs `QK = exp((QK - cur_max) * scale)`
                 *
                 * It also partially performs reduce_sum on the output using L1 accumulation.
                 * `cur_sum = sum_tiles(exp((QK - cur_max) * scale), dim=-1)`
                 *
                 * Partial reduce_sum is used to push the final row_reduction within a tile
                 * outside of the loop over K chunks.
                 */
                sub_exp_block_bcast_cols_inplace<cb_qk_im, Sq_chunk_t, Sk_chunk_t, scale_fp32>(
                    alias_cur_max, alias_cur_sum);

                /* OUT_IM = QK @ V_CHUNK */
                matmul_blocks(
                    cb_qk_im,
                    cb_v_in,
                    alias_mm2_cur_out,
                    Sq_chunk_t,
                    DHt,
                    Sk_chunk_t,
                    out_num_blocks,
                    out_in0_num_subblocks,
                    out_in1_num_subblocks,
                    out_in0_block_w,
                    out_subblock_h,
                    out_subblock_w,
                    false /*transpose*/);

                cb_pop_front(cb_qk_im, qk_chunk_tiles);
                reconfig_data_format(alias_prev_max, alias_cur_max);

                /* OUT_ACC += OUT_IM */
                if (processed_k_chunks > 0) {
                    /* cb_exp_max_diff = torch.exp(cb_prev_max - cb_cur_max) */
                    sub_exp_block<scale_fp32>(alias_prev_max, alias_cur_max, cb_exp_max_diff, Sq_chunk_t);
                    cb_pop_front(alias_prev_max, Sq_chunk_t);
                    /* cb_prev_sum *= cb_exp_max_diff */
                    mul_tiles_bcast_cols_inplace(alias_prev_sum, cb_exp_max_diff, Sq_chunk_t);

                    /* cb_cur_sum += cb_prev_sum */
                    add_block_inplace(alias_cur_sum, alias_prev_sum, Sq_chunk_t);

                    /* cb_out_accumulate_im *= cb_exp_max_diff */

                    mul_block_bcast_cols<Sq_chunk_t, DHt>(alias_mm2_prev_out, cb_exp_max_diff, alias_mm2_cur_out, true);
                }

                // Swap ping-pong buffers
                std::swap(alias_prev_sum, alias_cur_sum);
                std::swap(alias_mm2_prev_out, alias_mm2_cur_out);
                std::swap(alias_prev_max, alias_cur_max);
                processed_k_chunks++;
            }

            // Calculate current LSE
            // Use alias_cur_max as intermediate buffer.
            matmul_reduce<Sq_chunk_t>(cb_col_identity, alias_prev_sum);

            log_block(alias_prev_sum, alias_cur_max, Sq_chunk_t);

            // Scale prev_max by scale_fp32
            mul_block_bcast_scalar_inplace<cb_scale_in, Sq_chunk_t>(alias_prev_max);
            add_block_inplace(alias_prev_max, alias_cur_max, Sq_chunk_t);

            /* cb_cur_sum = 1.0 / cb_cur_sum */
            recip_block_inplace(alias_prev_sum, Sq_chunk_t);
            /* cb_out_accumulate_im *= cb_cur_sum */
            mul_block_bcast_cols_inplace<Sq_chunk_t, DHt>(alias_mm2_prev_out, alias_prev_sum);
            if (ring_iter > 0) {
                // Update output according to previous and current LSE
                /**
                 * sig = torch.sigmoid(cur_lse - prev_lse)
                 * out = prev_out - sig * (prev_out - cur_out)
                 * lse = prev_lse - torch.logsigmoid(prev_lse - cur_lse)
                 */
                cb_wait_front(cb_lse_in, Sq_chunk_t);
                cb_wait_front(cb_prev_out, out_chunk_tiles);

                uint32_t alias_cur_lse = alias_prev_max;      // full
                uint32_t alias_sig = alias_cur_max;           // empty
                uint32_t alias_cur_out = alias_mm2_prev_out;  // full
                uint32_t alias_sub = alias_mm2_cur_out;       // empty

                // alias_sig = sigmoid(alias_cur_lse - cb_lse_in)
                sigmoid_sub(alias_cur_lse, cb_lse_in, alias_sig, Sq_chunk_t);

                // alias_sub = cb_prev_out - alias_cur_out
                reconfig_data_format(cb_prev_out, alias_cur_out);
                sub_block(cb_prev_out, alias_cur_out, alias_sub, out_chunk_tiles);
                // alias_sub *= alias_sig
                reconfig_data_format(alias_sub, alias_sig);
                mul_block_bcast_cols_inplace<Sq_chunk_t, DHt>(alias_sub, alias_sig);
                // cb_out = cb_prev_out - alias_sub
                reconfig_data_format(cb_prev_out, alias_sub);
                pack_reconfig_data_format(cb_out);
                sub_block(cb_prev_out, alias_sub, cb_out, out_chunk_tiles);
                cb_pop_front(cb_prev_out, out_chunk_tiles);
                cb_pop_front(alias_cur_out, out_chunk_tiles);
                cb_pop_front(alias_sub, out_chunk_tiles);

                // alias_sig = sigmoid(cb_lse_in - alias_cur_lse)
                // alias_cur_lse = log(alias_sig)
                // cb_lse_out = cb_lse_in - alias_cur_lse
                pack_reconfig_data_format(alias_sig);
                reconfig_data_format(cb_lse_in, alias_cur_lse);
                logsigmoid_sub(cb_lse_in, alias_cur_lse, alias_sig, Sq_chunk_t);
                sub_block(cb_lse_in, alias_sig, cb_lse_out, Sq_chunk_t);
                cb_pop_front(alias_sig, Sq_chunk_t);
                cb_pop_front(alias_cur_lse, Sq_chunk_t);
                cb_pop_front(cb_lse_in, Sq_chunk_t);

            } else {
                pack_reconfig_data_format(cb_out);
                copy_block(alias_mm2_prev_out, cb_out, out_chunk_tiles);

                pack_reconfig_data_format(cb_lse_out);
                copy_block(alias_prev_max, cb_lse_out, Sq_chunk_t);
            }

            cb_pop_front(cb_q_in, q_chunk_tiles);
        }
        if (KV_chunks_processed_in_iter % 2 == 0) {
            cb_wait_front(cb_k_in, k_chunk_tiles);
            cb_wait_front(cb_v_in, k_chunk_tiles);
            cb_pop_front(cb_k_in, k_chunk_tiles);
            cb_pop_front(cb_v_in, k_chunk_tiles);
        }
    }
}
}  // namespace NAMESPACE
