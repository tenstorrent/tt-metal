// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/compute_kernel_api.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#ifdef TILIZE_IN
#include "api/compute/tilize.h"
#endif
#ifdef UNTILIZE_OUT
#include "api/compute/pack_untilize.h"
#endif
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/normalization/kernel_util/compute/numeric.h"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "ttnn/operations/normalization/kernel_util/generic/bit.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/circular_buffer.h"

#include "experimental/kernel_args.h"

#include "layernorm_compute_utils.h"

namespace generic = norm::kernel_util::generic;
namespace kutil = norm::kernel_util;
namespace numeric = kutil::compute::numeric;
namespace policies = kutil::compute::policies;

ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }

void kernel_main() {
    auto NCHt = get_arg(args::NCHt);
    constexpr auto Wt = get_arg(args::Wt);
    constexpr auto block_size = get_arg(args::block_size);
    constexpr auto do_gamma = get_arg(args::do_gamma);
    constexpr auto do_beta = get_arg(args::do_beta);
    constexpr bool FLOAT32_DTYPE = get_arg(args::FLOAT32_DTYPE) == 1;
    constexpr bool FLOAT32_REDUCTION = get_arg(args::FLOAT32_REDUCTION) == 1;
    constexpr bool LEGACY_RSQRT = get_arg(args::LEGACY_RSQRT) == 1;
    constexpr auto W = get_arg(args::W);
    constexpr auto tile_width = get_arg(args::tile_width);

    // DFB handles — `dfb::name` is constexpr-convertible to uint32_t (the legacy CB index),
    // so the same expressions read cleanly through both LLK signatures and constexpr aliases.
    // CB declarations are gated to match the host's conditional DFB bindings; see
    // METAL2_PORT_PLAN.md for the symbol → host-condition mapping.
    constexpr uint32_t cb_eps = dfb::cb_eps;
    constexpr uint32_t cb_in = dfb::cb_in;
    constexpr uint32_t cb_out = dfb::cb_out;
    constexpr uint32_t cb_ex2 = dfb::cb_ex2;
    constexpr uint32_t cb_ex2pe = dfb::cb_ex2pe;
    DataflowBuffer cb_eps_obj(cb_eps);
    DataflowBuffer cb_in_obj(cb_in);
    DataflowBuffer cb_out_obj(cb_out);
    DataflowBuffer cb_ex2_obj(cb_ex2);
    DataflowBuffer cb_ex2pe_obj(cb_ex2pe);

#ifndef RMSNORM
    // cb_scaler is host-bound when !use_welford; this kernel is the non-welford path.
    constexpr uint32_t cb_scaler = dfb::cb_scaler;
    constexpr uint32_t cb_ex = dfb::cb_ex;
    DataflowBuffer cb_ex_obj(cb_ex);
#endif

    // cb_xmm2 always exists for layernorm.cpp (non-welford path).
    constexpr uint32_t cb_xmm2 = dfb::cb_xmm2;
    DataflowBuffer cb_xmm2_obj(cb_xmm2);

#if defined RMSNORM and not defined FUSE_PRE_ADD
    constexpr uint32_t cb_xmm = cb_in;  // aliased to cb_in by kernel design
#else
    constexpr uint32_t cb_xmm = dfb::cb_xmm;
#endif
    DataflowBuffer cb_xmm_obj(cb_xmm);

#ifdef FUSE_PRE_ADD
    constexpr uint32_t cb_inb = dfb::cb_inb;
    DataflowBuffer cb_inb_obj(cb_inb);
#endif

#ifdef FUSE_GAMMA
    constexpr uint32_t cb_gamma = dfb::cb_gamma;
    DataflowBuffer cb_gamma_obj(cb_gamma);
#endif
#ifdef FUSE_BETA
    constexpr uint32_t cb_beta = dfb::cb_beta;
    DataflowBuffer cb_beta_obj(cb_beta);
#endif

#if defined FUSE_GAMMA || defined FUSE_BETA
    constexpr uint32_t cb_fusion = dfb::cb_fusion;
    DataflowBuffer cb_fusion_obj(cb_fusion);
#endif

#ifdef TILIZE_IN
    constexpr uint32_t cb_in_rm = dfb::cb_in_rm;
#endif

    constexpr int onetile = 1;
    constexpr int dst0 = 0;
    constexpr int dst1 = 1;
    constexpr auto scaler0 = 0;

#ifdef FUSE_PRE_ADD
#ifdef RMSNORM
    constexpr uint32_t cb_x = cb_xmm;
#else
    constexpr uint32_t cb_x = dfb::cb_x;
#endif
#else
    constexpr uint32_t cb_x = cb_in;
#endif
    DataflowBuffer cb_x_obj(cb_x);

#ifdef TILIZE_IN
    binary_op_init_common(cb_in_rm, cb_in_rm, cb_in);
#elif defined(FUSE_PRE_ADD)
    binary_op_init_common(cb_in, cb_inb, cb_x);
#elif defined(RMSNORM)
    binary_op_init_common(cb_xmm, cb_xmm, cb_xmm2);
#else
    binary_op_init_common(cb_x, cb_scaler, cb_ex);
#endif

    cb_eps_obj.wait_front(1);  // comes from the reader

    // cb_im_or_out: cb_fusion when gamma/beta fuse; otherwise cb_out. Gated by #ifdef
    // because cb_fusion only exists when host bound it (FUSE_GAMMA||FUSE_BETA).
#if defined FUSE_GAMMA || defined FUSE_BETA
    constexpr int cb_im_or_out = cb_fusion;
#else
    constexpr int cb_im_or_out = cb_out;
#endif
    DataflowBuffer cb_im_or_out_obj(cb_im_or_out);

    // Intermediate buffers need to be reserved/pushed/popped
    // in full blocks
    const auto total_buffer_size = generic::blocks(Wt, block_size).total_with_remainder();

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
#ifdef TILIZE_IN
        tilize_all_blocks_to_cb<block_size>(cb_in_rm, cb_in, Wt);
        // Re-init binary ops after tilize hardware reconfiguration.
#ifdef FUSE_PRE_ADD
        binary_op_init_common(cb_in, cb_inb, cb_x);
#elif defined(RMSNORM)
        binary_op_init_common(cb_xmm, cb_xmm, cb_xmm2);
#else
        binary_op_init_common(cb_x, cb_scaler, cb_ex);
#endif
#endif
/*
 * X + Y
 */
#ifdef FUSE_PRE_ADD
        reconfig_data_format(cb_in, cb_inb);
        pack_reconfig_data_format(cb_x);
        add_tiles_init(cb_in, cb_inb);
        for (auto block : generic::blocks(Wt, block_size)) {
            ACQ();
            // In/inb come from the reader and need to be
            // synced on full block size. Keep cb_x aligned
            // to full block size as well so pre-add/no-pre-add
            // can be handled the same way.
            cb_in_obj.wait_front(block.full_block_size());
            cb_inb_obj.wait_front(block.full_block_size());
            cb_x_obj.reserve_back(block.full_block_size());
            for (auto i : block.local()) {
                add_tiles(cb_in, cb_inb, i, i, i);
                pack_tile(i, cb_x);
            }
            REL();
            cb_x_obj.push_back(block.full_block_size());  // push the sum into the same buffer
            cb_in_obj.pop_front(block.full_block_size());
            cb_inb_obj.pop_front(block.full_block_size());
        }
#ifndef RMSNORM
        reconfig_data_format(cb_in, cb_x, cb_inb, cb_scaler);
#else
        reconfig_data_format(cb_in, cb_x, cb_inb, cb_x);
#endif
        // by the end of this loop we should end up with Wt tiles in cb_x
#else
#ifdef RMSNORM
        reconfig_data_format(cb_in, cb_in);
        pack_reconfig_data_format(cb_xmm2);
#endif
#endif

#ifndef RMSNORM
        // E[x]
        numeric::row_wise_mean<PoolType::SUM, ReduceDim::REDUCE_ROW, FLOAT32_REDUCTION, policies::FullBlockWithoutPopPolicy>(
            cb_x, cb_scaler, cb_ex, W, Wt, block_size, tile_width);

        // x - E[x]
        reconfig_data_format(cb_x, cb_ex);
        cb_xmm_obj.reserve_back(total_buffer_size);
        sub_bcast_cols_init_short(cb_x, cb_ex);
        for (auto block : generic::blocks(Wt, block_size)) {
            ACQ();
            for (auto i : block.local()) {
                sub_tiles_bcast_cols(cb_x, cb_ex, i, 0, i);
                pack_tile(i, cb_xmm);
            }
            cb_xmm_obj.push_back(block.full_block_size());
            cb_x_obj.pop_front(block.full_block_size());
            REL();
        }
        cb_ex_obj.pop_front(1);

#ifndef FUSE_PRE_ADD
        reconfig_data_format_srca(cb_x, cb_xmm);
#endif
#endif

        /* (x - E[x])^2
         * compute temp = xmm*xmm = (x-E[x])^2
         */
        mul_tiles_init(cb_xmm, cb_xmm);
        for (auto block : generic::blocks(Wt, block_size)) {
#ifndef RMSNORM
            cb_xmm_obj.wait_front(block.start() + block.size());
#else
            cb_xmm_obj.wait_front(block.start() + block.full_block_size());
#endif
            cb_xmm2_obj.reserve_back(block.full_block_size());
            ACQ();
            for (auto i : block.local()) {
                const auto global_i = block.to_global(i);
                mul_tiles(cb_xmm, cb_xmm, global_i, global_i, i);
                pack_tile(i, cb_xmm2);
            }
            cb_xmm2_obj.push_back(block.full_block_size());
            REL();
        }
#if defined RMSNORM and not defined FUSED_PRE_ADD
        reconfig_data_format(cb_xmm, cb_xmm2, cb_xmm, cb_scaler);
#endif

        // Var[x]
        numeric::row_wise_mean<PoolType::SUM, ReduceDim::REDUCE_ROW, FLOAT32_REDUCTION, policies::FullBlockWithPopPolicy>(
            cb_xmm2, cb_scaler, cb_ex2, W, Wt, block_size, tile_width);

        // Var[x] + eps
        cb_ex2_obj.wait_front(1);
        reconfig_data_format(cb_ex2, cb_eps);
        ACQ();
        add_tiles_init(cb_ex2, cb_eps);
        add_tiles(cb_ex2, cb_eps, 0, 0, dst0);

        cb_ex2pe_obj.reserve_back(1);  // 1
        rsqrt_tile_init<LEGACY_RSQRT>();
        rsqrt_tile<LEGACY_RSQRT>(dst0);
        pack_reconfig_data_format(cb_ex2pe);
        pack_tile(dst0, cb_ex2pe);
        cb_ex2pe_obj.push_back(1);
        REL();
        cb_ex2_obj.pop_front(1);

        // (x-E[x]) / sqrt(Var[x] + eps) * gamma + beta
        cb_ex2pe_obj.wait_front(1);
        for (auto block : generic::blocks(Wt, block_size)) {
            reconfig_data_format(cb_xmm, cb_ex2pe);
            if constexpr (do_gamma == 0 && do_beta == 0) {
                pack_reconfig_data_format(cb_out);
            } else {
                pack_reconfig_data_format(cb_fusion);
            }
            cb_im_or_out_obj.reserve_back(block.full_block_size());
#if defined RMSNORM and not defined FUSE_PRE_ADD
            reconfig_data_format_srca(cb_fusion, cb_xmm);
#endif
            ACQ();
            mul_bcast_cols_init_short(cb_xmm, cb_ex2pe);
            for (auto i : block.local()) {
                mul_tiles_bcast_cols(cb_xmm, cb_ex2pe, block.to_global(i), 0, i);  // tile *= 1/(sum(exp(x)))
#ifdef SFPU_OP_INIT_ACTIVATION
                // Activation must be applied last. If do_gamma != 0 or do_beta != 0 then
                // activation will be applied after the gamma/beta multiplication/addition.
                // Otherwise, we can apply the activation here.
                if constexpr (!(do_gamma == 1 || do_beta == 1)) {
                    SFPU_OP_INIT_ACTIVATION
                    SFPU_OP_FUNC_ACTIVATION
                }
#endif
                pack_tile(i, cb_im_or_out);  // pack either to intermediate (cb_fusion or out0)
            }
            cb_im_or_out_obj.push_back(
                block.full_block_size());  // if no gamma/beta are provided, this will be passed on to the writer
            REL();

            if constexpr (!(do_gamma == 0 && do_beta == 0)) {
#if defined RMSNORM and not defined FUSE_PRE_ADD
                reconfig_data_format_srca(cb_xmm, cb_fusion);
#endif
            }

            if constexpr (do_gamma) {
                if constexpr (do_beta == 0) {
                    pack_reconfig_data_format(cb_out);
                }
                reconfig_data_format_srcb(cb_ex2pe, cb_gamma);
                ACQ();
                uint32_t cb_outg = do_beta ? cb_fusion : cb_out;
                DataflowBuffer cb_outg_obj(cb_outg);
                mul_bcast_rows_init_short(cb_fusion, cb_gamma);
                cb_outg_obj.reserve_back(block.full_block_size());
                cb_gamma_obj.wait_front(
                    block.start() + block.full_block_size());  // we don't pop, TODO: only wait on first ht
                cb_fusion_obj.wait_front(block.full_block_size());
                for (auto i : block.local()) {
                    mul_tiles_bcast_rows(cb_fusion, cb_gamma, i, block.to_global(i), i);  // tile *= 1/(sum(exp(x)))
#ifdef SFPU_OP_INIT_ACTIVATION
                    // Activation must be applied last. If do_beta != 0 then
                    // activation will be applied after the beta addition.
                    // Otherwise, we can apply the activation here.
                    if constexpr (!(do_beta == 1)) {
                        SFPU_OP_INIT_ACTIVATION
                        SFPU_OP_FUNC_ACTIVATION
                    }
#endif
                    pack_tile(i, cb_outg);  // pack either to intermediate (cb_fusion or out0)
                }
                cb_fusion_obj.pop_front(block.full_block_size());
                // we don't pop gamma
                cb_outg_obj.push_back(block.full_block_size());
                // We don't pop gamma since it's 1,1,1,Wt and we reuse it for all NCHt
                REL();
            }
            if constexpr (do_beta) {
                pack_reconfig_data_format(cb_out);
                if constexpr (do_gamma) {
                    reconfig_data_format_srcb(cb_gamma, cb_beta);
                } else {
                    reconfig_data_format_srcb(cb_ex2pe, cb_beta);
                }
                ACQ();
                add_bcast_rows_init_short(cb_fusion, cb_beta);
                cb_out_obj.reserve_back(block.full_block_size());
                cb_beta_obj.wait_front(
                    block.start() + block.full_block_size());  // TODO: optimization - only wait on first ht
                cb_fusion_obj.wait_front(block.full_block_size());
                for (auto i : block.local()) {
                    add_tiles_bcast_rows(cb_fusion, cb_beta, i, block.to_global(i), i);  // tile *= 1/(sum(exp(x)))
#ifdef SFPU_OP_INIT_ACTIVATION
                    SFPU_OP_INIT_ACTIVATION
                    SFPU_OP_FUNC_ACTIVATION
#endif
                    pack_tile(i, cb_out);  // pack either to intermediate (cb_fusion or out0)
                }
                cb_fusion_obj.pop_front(block.full_block_size());
                // We don't pop beta since it's 1,1,1,Wt and we reuse it for all NCHt
                cb_out_obj.push_back(block.full_block_size());
                REL();
            }
        }
        cb_ex2pe_obj.pop_front(1);
        cb_xmm_obj.pop_front(total_buffer_size);

#ifdef UNTILIZE_OUT
        constexpr uint32_t cb_out_rm = dfb::cb_out_rm;
        untilize_all_blocks_from_cb<block_size>(cb_out, cb_out_rm, Wt);
#endif
    }  // NCHt loop
}
