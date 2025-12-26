// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "compute_kernel_api.h"
#include "compute_common.hpp"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NQH = get_compile_time_arg_val(1);
    constexpr uint32_t NKH = get_compile_time_arg_val(2);
    constexpr uint32_t Skt = get_compile_time_arg_val(3);
    constexpr uint32_t DHt = get_compile_time_arg_val(4);
    constexpr uint32_t vDHt = get_compile_time_arg_val(5);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(6);
    constexpr uint32_t q_num_chunks = get_compile_time_arg_val(7);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(8);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(9);

    constexpr uint32_t qk_in0_block_w = get_compile_time_arg_val(10);
    constexpr uint32_t qk_subblock_w = get_compile_time_arg_val(11);
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(12);
    constexpr uint32_t qk_in0_num_subblocks = get_compile_time_arg_val(13);
    constexpr uint32_t qk_in1_num_subblocks = get_compile_time_arg_val(14);
    constexpr uint32_t qk_num_blocks = get_compile_time_arg_val(15);
    constexpr uint32_t out_in0_block_w = get_compile_time_arg_val(16);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(17);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(18);
    constexpr uint32_t out_in0_num_subblocks = get_compile_time_arg_val(19);
    constexpr uint32_t out_in1_num_subblocks = get_compile_time_arg_val(20);
    constexpr uint32_t out_num_blocks = get_compile_time_arg_val(21);

    constexpr uint32_t num_cores = get_compile_time_arg_val(22);

    constexpr bool is_causal = get_compile_time_arg_val(23) == 1;
    constexpr bool use_provided_mask = get_compile_time_arg_val(24) == 1;
    constexpr bool use_padded_mask = get_compile_time_arg_val(25) == 1;
    constexpr bool is_chunked = get_compile_time_arg_val(26) == 1;
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(27);
    constexpr uint32_t sliding_window_size = get_compile_time_arg_val(28);
    constexpr bool use_attention_sink = get_compile_time_arg_val(29) == 1;

    const uint32_t core_id = get_arg_val<uint32_t>(0);
    const uint32_t local_batch_start = get_arg_val<uint32_t>(1);
    const uint32_t local_batch_end = get_arg_val<uint32_t>(2);
    const uint32_t local_nh_start = get_arg_val<uint32_t>(3);
    const uint32_t local_nh_end = get_arg_val<uint32_t>(4);
    const uint32_t local_q_start = get_arg_val<uint32_t>(5);
    const uint32_t local_q_end = get_arg_val<uint32_t>(6);
    // const uint32_t chunked_q_chunk_offset = get_arg_val<uint32_t>(7);
    const uint32_t num_phases = get_arg_val<uint32_t>(7);
    const uint32_t chunked_q_chunk_offset_phase_1 = get_arg_val<uint32_t>(8);
    uint32_t chunked_q_chunk_offset_phase_2 = 0;
    if (num_phases == 2) {
        chunked_q_chunk_offset_phase_2 = get_arg_val<uint32_t>(9);
    }

    const uint32_t q_chunks_per_core = local_q_end - local_q_start;

    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t qk_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_attention_sink = tt::CBIndex::c_4;
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_7;

    constexpr uint32_t cb_qk_im = tt::CBIndex::c_24;
    constexpr uint32_t cb_out_im_A = tt::CBIndex::c_25;
    constexpr uint32_t cb_out_im_B = tt::CBIndex::c_26;
    constexpr uint32_t cb_max_A = tt::CBIndex::c_27;
    constexpr uint32_t cb_max_B = tt::CBIndex::c_28;
    constexpr uint32_t cb_sum_A = tt::CBIndex::c_29;
    constexpr uint32_t cb_sum_B = tt::CBIndex::c_30;
    constexpr uint32_t cb_exp_max_diff = tt::CBIndex::c_31;

    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    uint32_t chunked_q_chunk_offset = 0;
    mm_init(cb_q_in, cb_k_in, cb_out);

    for (uint32_t phase = 0; phase < num_phases; ++phase) {
        if (phase == 0) {
            chunked_q_chunk_offset = chunked_q_chunk_offset_phase_1;
        } else {
            chunked_q_chunk_offset = chunked_q_chunk_offset_phase_2;
        }

        for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
            for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
                sdpa_standard<
                    cb_qk_im,
                    cb_identity_scale_in,
                    cb_attention_sink,
                    Sq_chunk_t,
                    Sk_chunk_t,
                    DHt,
                    vDHt,
                    use_attention_sink,
                    is_causal,
                    use_provided_mask,
                    use_padded_mask,
                    is_chunked,
                    scale_fp32,
                    sliding_window_size>(
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
                    0,                  // iter_q_start
                    q_chunks_per_core,  // iter_q_end
                    q_num_chunks,
                    local_q_start,
                    chunked_q_chunk_offset,
                    k_num_chunks,
                    q_chunk_tiles,
                    k_chunk_tiles,
                    qk_chunk_tiles,
                    out_chunk_tiles,
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
                    cb_out);
            }
        }
    }
}
}  // namespace NAMESPACE
