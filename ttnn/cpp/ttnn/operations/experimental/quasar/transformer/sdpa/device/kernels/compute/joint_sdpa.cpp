// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "experimental/kernel_args.h"
#include "compute_common.hpp"

void kernel_main() {
    constexpr auto Skt = get_arg(args::Skt);
    constexpr auto DHt = get_arg(args::DHt);
    constexpr auto Sq_chunk_t = get_arg(args::Sq_chunk_t);
    constexpr auto Sk_chunk_t = get_arg(args::Sk_chunk_t);
    constexpr auto k_num_chunks = get_arg(args::k_num_chunks);

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

    constexpr auto mask_chunk_0 = get_arg(args::mask_chunk_0);
    constexpr auto mask_chunk_1 = get_arg(args::mask_chunk_1);
    constexpr auto scale_fp32 = get_arg(args::scale_fp32);

    const auto local_batch_start = get_arg(args::local_batch_start);
    const auto local_batch_end = get_arg(args::local_batch_end);
    const auto local_nh_start = get_arg(args::local_nh_start);
    const auto local_nh_end = get_arg(args::local_nh_end);
    const auto local_q_start = get_arg(args::local_q_start);
    const auto local_q_end = get_arg(args::local_q_end);

    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t qk_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * DHt;

    // Mask consumption is gated by use_joint_mask. When the mask is disabled, dfb_mask_in is
    // never bound, so alias its handle to a bound DFB (q_in) as a placeholder — sdpa_joint's
    // discarded (use_joint_mask == false) branch never touches it.
#ifdef USE_JOINT_MASK
    constexpr bool use_joint_mask = true;
    constexpr uint32_t dfb_mask_in = dfb::mask_in;
#else
    constexpr bool use_joint_mask = false;
    constexpr uint32_t dfb_mask_in = dfb::q_in;  // placeholder; not read on the no-mask path
#endif

    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::q_in, dfb::k_in, dfb::qk_im);
    matmul_init(dfb::q_in, dfb::k_in);

    for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
        for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
            sdpa_joint<dfb::qk_im, dfb::identity_scale_in, Sq_chunk_t, Sk_chunk_t, DHt, use_joint_mask, scale_fp32>(
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
                dfb::q_in,
                dfb::k_in,
                dfb::v_in,
                dfb_mask_in,
                dfb::col_identity,
                dfb::out_im_A,
                dfb::out_im_B,
                dfb::max_A,
                dfb::max_B,
                dfb::sum_A,
                dfb::sum_B,
                dfb::exp_max_diff,
                dfb::out);
        }
    }
}
