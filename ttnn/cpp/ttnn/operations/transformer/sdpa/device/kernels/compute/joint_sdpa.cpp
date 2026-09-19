// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "compute_common.hpp"
#include "compute_streaming.hpp"

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NH = get_compile_time_arg_val(1);
    constexpr uint32_t Skt = get_compile_time_arg_val(2);
    constexpr uint32_t DHt = get_compile_time_arg_val(3);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(4);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(5);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(6);

    constexpr uint32_t qk_in0_block_w = get_compile_time_arg_val(7);
    constexpr uint32_t qk_subblock_w = get_compile_time_arg_val(8);
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(9);
    constexpr uint32_t qk_in0_num_subblocks = get_compile_time_arg_val(10);
    constexpr uint32_t qk_in1_num_subblocks = get_compile_time_arg_val(11);
    constexpr uint32_t qk_num_blocks = get_compile_time_arg_val(12);
    constexpr uint32_t out_in0_block_w = get_compile_time_arg_val(13);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(14);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(15);
    constexpr uint32_t out_in0_num_subblocks = get_compile_time_arg_val(16);
    constexpr uint32_t out_in1_num_subblocks = get_compile_time_arg_val(17);
    constexpr uint32_t out_num_blocks = get_compile_time_arg_val(18);

    constexpr bool use_joint_mask = get_compile_time_arg_val(19) == 1;
    constexpr uint32_t mask_chunk_0 = get_compile_time_arg_val(20);
    constexpr uint32_t mask_chunk_1 = get_compile_time_arg_val(21);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(22);
    constexpr bool use_streaming_compute = get_compile_time_arg_val(23) == 1;
    constexpr uint32_t valid_Skt = get_compile_time_arg_val(24);
    constexpr uint32_t k_partial_col = get_compile_time_arg_val(25);
    constexpr uint32_t n_partial_col = get_compile_time_arg_val(26);
    constexpr uint32_t mid_padded_tiles = get_compile_time_arg_val(27);

    uint32_t argidx = 0;
    const uint32_t local_batch_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_batch_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_nh_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_nh_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_q_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_q_end = get_arg_val<uint32_t>(argidx++);

    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t qk_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * DHt;

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_7;
    constexpr uint32_t cb_recip_scratch = tt::CBIndex::c_6;

    constexpr uint32_t cb_qk_im = tt::CBIndex::c_24;
    constexpr uint32_t cb_out_im_A = tt::CBIndex::c_25;
    constexpr uint32_t cb_out_im_B = tt::CBIndex::c_26;
    constexpr uint32_t cb_max_A = tt::CBIndex::c_27;
    constexpr uint32_t cb_max_B = tt::CBIndex::c_28;
    constexpr uint32_t cb_sum_A = tt::CBIndex::c_29;
    constexpr uint32_t cb_sum_B = tt::CBIndex::c_30;
    constexpr uint32_t cb_exp_max_diff = tt::CBIndex::c_31;

    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q_in, cb_k_in, use_streaming_compute ? cb_out : cb_qk_im);
    if constexpr (use_streaming_compute) {
        CircularBuffer(cb_identity_scale_in).wait_front(1);
        LightweightMaskContext lw_mask;
        lw_mask.neginf_tile_idx = 0;
        constexpr uint32_t n_partial_tiles = n_partial_col > 0 ? 1u : 0u;
        if constexpr (mask_chunk_0 != static_cast<uint32_t>(-1)) {
            // The spatial segment ends inside chunk mask_chunk_0: its padded tiles are narrowed away and the
            // partial tile, when there is one, is stamped from palette tile 1.
            lw_mask.mid_mask_chunk = mask_chunk_0;
            lw_mask.mid_padded_tiles = mid_padded_tiles;
            lw_mask.mid_partial_col = n_partial_col;
            lw_mask.mid_partial_tile_idx = 1;
        }
        if constexpr (k_partial_col > 0) {
            // The joint tail ends inside the last K chunk: whole padded tiles are narrowed away, the partial
            // tile is stamped from the palette tile after the spatial one.
            lw_mask.global_n_partial_col = k_partial_col;
            lw_mask.global_n_partial_tile_idx = 1 + n_partial_tiles;
            constexpr uint32_t last_chunk_first_tile = ((valid_Skt - 1) / Sk_chunk_t) * Sk_chunk_t;
            lw_mask.global_n_padded_tiles = Sk_chunk_t - (valid_Skt - last_chunk_first_tile);
        }
        if constexpr (n_partial_tiles + (k_partial_col > 0 ? 1u : 0u) > 0) {
            CircularBuffer(cb_mask_in).wait_front(1 + n_partial_tiles + (k_partial_col > 0 ? 1u : 0u));
        }
        // The reader and writer walk nb, nq, q_chunk in this order; each (nb, nq) is one q chunk range.
        for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
            for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
                sdpa_standard_v2<
                    Sq_chunk_t,
                    Sk_chunk_t,
                    valid_Skt,
                    DHt,
                    DHt,
                    scale_fp32,
                    qk_subblock_h,
                    qk_subblock_w,
                    out_subblock_h,
                    out_subblock_w,
                    use_joint_mask,
                    cb_q_in,
                    cb_k_in,
                    cb_v_in,
                    cb_qk_im,
                    cb_identity_scale_in,
                    cb_exp_max_diff,
                    cb_col_identity,
                    cb_recip_scratch,
                    cb_out,
                    cb_mask_in>(
                    local_q_end - local_q_start,
                    k_num_chunks,
                    cb_out_im_A,
                    cb_out_im_B,
                    cb_max_A,
                    cb_max_B,
                    cb_sum_A,
                    cb_sum_B,
                    local_q_start,
                    0,
                    lw_mask,
                    0,
                    false);
            }
        }
        return;
    }
    matmul_init(cb_q_in, cb_k_in);

    for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
        for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
            sdpa_joint<cb_qk_im, cb_identity_scale_in, Sq_chunk_t, Sk_chunk_t, DHt, use_joint_mask, scale_fp32>(
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
                local_q_start,
                local_q_end,
                k_num_chunks,
                q_chunk_tiles,
                k_chunk_tiles,
                qk_chunk_tiles,
                out_chunk_tiles,
                mask_chunk_0,
                mask_chunk_1,
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
