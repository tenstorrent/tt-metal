// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "compute_kernel_api.h"
#include "compute_common.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/fused_op_indexer.hpp"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NH = get_compile_time_arg_val(1);
    constexpr uint32_t Skt = get_compile_time_arg_val(2);
    constexpr uint32_t DHt = get_compile_time_arg_val(3);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(4);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(5);

    constexpr uint32_t qk_in0_block_w = get_compile_time_arg_val(6);
    constexpr uint32_t qk_subblock_w = get_compile_time_arg_val(7);
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(8);
    constexpr uint32_t qk_in0_num_subblocks = get_compile_time_arg_val(9);
    constexpr uint32_t qk_in1_num_subblocks = get_compile_time_arg_val(10);
    constexpr uint32_t qk_num_blocks = get_compile_time_arg_val(11);
    constexpr uint32_t out_in0_block_w = get_compile_time_arg_val(12);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(13);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(14);
    constexpr uint32_t out_in0_num_subblocks = get_compile_time_arg_val(15);
    constexpr uint32_t out_in1_num_subblocks = get_compile_time_arg_val(16);
    constexpr uint32_t out_num_blocks = get_compile_time_arg_val(17);

    constexpr bool use_joint_mask = get_compile_time_arg_val(18) == 1;
    constexpr uint32_t mask_chunk_0 = get_compile_time_arg_val(19);
    constexpr uint32_t mask_chunk_1 = get_compile_time_arg_val(20);
    constexpr uint32_t ring_size = get_compile_time_arg_val(21);
    constexpr uint32_t N_k_num_chunks_local = get_compile_time_arg_val(22);
    constexpr uint32_t L_k_num_chunks = get_compile_time_arg_val(23);
    constexpr uint32_t global_logical_NK_chunks = get_compile_time_arg_val(24);
    constexpr uint32_t global_padded_NK_chunks = get_compile_time_arg_val(25);
    constexpr uint32_t q_num_chunks = get_compile_time_arg_val(26);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(27);
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

    // Only one iteration of the ring will contain the masked portion of the spatial input.
    constexpr uint32_t N_mask_ring_id = mask_chunk_0 / N_k_num_chunks_local;
    // The last iteration will concatenate L, which contains the masked portion of the joint tensor.
    constexpr uint32_t L_mask_ring_id = ring_size - 1;

    mm_init(cb_q_in, cb_k_in, cb_qk_im);

    for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
        uint32_t ring_id = fused_op_indexer.get_next_ring_id_and_sync();
        const uint32_t iter_k_num_chunks =
            ring_id == ring_size - 1 ? (N_k_num_chunks_local + L_k_num_chunks) : N_k_num_chunks_local;
        const uint32_t iter_k_chunk_start = ring_id * N_k_num_chunks_local;
        const uint32_t iter_k_chunk_end = iter_k_chunk_start + iter_k_num_chunks;

        sdpa_ring<cb_qk_im, cb_identity_scale_in, cb_scale_in, Sq_chunk_t, Sk_chunk_t, DHt, use_joint_mask, scale_fp32>(
            Skt,
            qk_in0_block_w,
            qk_subblock_w,
            qk_subblock_h,
            qk_in0_num_subblocks,
            qk_in1_num_subblocks,
            qk_num_blocks,
            out_in0_block_w,
            out_subblock_w,
            out_subblock_h,
            out_in0_num_subblocks,
            out_in1_num_subblocks,
            out_num_blocks,
            global_q_start,
            global_q_end,
            q_num_chunks,
            iter_k_chunk_start,
            iter_k_chunk_end,
            q_chunk_tiles,
            k_chunk_tiles,
            qk_chunk_tiles,
            out_chunk_tiles,
            mask_chunk_0,
            mask_chunk_1,
            ring_iter,
            ring_id,
            N_mask_ring_id,
            L_mask_ring_id,
            global_logical_NK_chunks,
            global_padded_NK_chunks,
            cb_q_in,
            cb_k_in,
            cb_v_in,
            cb_mask_in,
            cb_col_identity,
            cb_out_im_A,
            cb_out_im_B,
            cb_max_A,
            cb_max_B,
            cb_sum_A,
            cb_sum_B,
            cb_exp_max_diff,
            cb_lse_in,
            cb_lse_out,
            cb_prev_out,
            cb_out);
    }
}
}  // namespace NAMESPACE
