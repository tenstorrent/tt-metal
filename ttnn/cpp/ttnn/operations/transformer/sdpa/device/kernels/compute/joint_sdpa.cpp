// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "compute_common.hpp"

// Metal 2.0 joint SDPA compute kernel. Owned exclusively by JointSDPAProgramFactory, so ported in
// place: named CTAs/RTAs (args::) and DFB handles (dfb::) replace the legacy positional CTAs and
// CBIndex constants. On Gen1 dfb::name is the framework-allocated physical CB id (constexpr), so it
// flows unchanged into the compute_common.hpp template helpers. Behavior is identical.
void kernel_main() {
    constexpr uint32_t B = get_arg(args::B);
    constexpr uint32_t NH = get_arg(args::NH);
    constexpr uint32_t Skt = get_arg(args::Skt);
    constexpr uint32_t DHt = get_arg(args::DHt);
    constexpr uint32_t Sq_chunk_t = get_arg(args::Sq_chunk_t);
    constexpr uint32_t Sk_chunk_t = get_arg(args::Sk_chunk_t);
    constexpr uint32_t k_num_chunks = get_arg(args::k_num_chunks);

    constexpr uint32_t qk_in0_block_w = get_arg(args::qk_in0_block_w);
    constexpr uint32_t qk_subblock_w = get_arg(args::qk_subblock_w);
    constexpr uint32_t qk_subblock_h = get_arg(args::qk_subblock_h);
    constexpr uint32_t qk_in0_num_subblocks = get_arg(args::qk_in0_num_subblocks);
    constexpr uint32_t qk_in1_num_subblocks = get_arg(args::qk_in1_num_subblocks);
    constexpr uint32_t qk_num_blocks = get_arg(args::qk_num_blocks);
    constexpr uint32_t out_in0_block_w = get_arg(args::out_in0_block_w);
    constexpr uint32_t out_subblock_w = get_arg(args::out_subblock_w);
    constexpr uint32_t out_subblock_h = get_arg(args::out_subblock_h);
    constexpr uint32_t out_in0_num_subblocks = get_arg(args::out_in0_num_subblocks);
    constexpr uint32_t out_in1_num_subblocks = get_arg(args::out_in1_num_subblocks);
    constexpr uint32_t out_num_blocks = get_arg(args::out_num_blocks);

    constexpr bool use_joint_mask = get_arg(args::use_joint_mask) == 1;
    constexpr uint32_t mask_chunk_0 = get_arg(args::mask_chunk_0);
    constexpr uint32_t mask_chunk_1 = get_arg(args::mask_chunk_1);
    constexpr uint32_t scale_fp32 = get_arg(args::scale_fp32);

    const uint32_t local_batch_start = get_arg(args::local_batch_start);
    const uint32_t local_batch_end = get_arg(args::local_batch_end);
    const uint32_t local_nh_start = get_arg(args::local_nh_start);
    const uint32_t local_nh_end = get_arg(args::local_nh_end);
    const uint32_t local_q_start = get_arg(args::local_q_start);
    const uint32_t local_q_end = get_arg(args::local_q_end);

    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t qk_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * DHt;

    constexpr uint32_t cb_q_in = dfb::q_in;
    constexpr uint32_t cb_k_in = dfb::k_in;
    constexpr uint32_t cb_v_in = dfb::v_in;
    constexpr uint32_t cb_mask_in = dfb::mask;
    constexpr uint32_t cb_identity_scale_in = dfb::scale;
    constexpr uint32_t cb_col_identity = dfb::col_identity;

    constexpr uint32_t cb_qk_im = dfb::qk_im;
    constexpr uint32_t cb_out_im_A = dfb::out_im_a;
    constexpr uint32_t cb_out_im_B = dfb::out_im_b;
    constexpr uint32_t cb_max_A = dfb::max_a;
    constexpr uint32_t cb_max_B = dfb::max_b;
    constexpr uint32_t cb_sum_A = dfb::sum_a;
    constexpr uint32_t cb_sum_B = dfb::sum_b;
    constexpr uint32_t cb_exp_max_diff = dfb::exp_max_diff;

    constexpr uint32_t cb_out = dfb::out;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q_in, cb_k_in, cb_qk_im);
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
