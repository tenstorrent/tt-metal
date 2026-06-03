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
#include "api/debug/dprint.h"

#include "experimental/kernel_args.h"

#include "layernorm_compute_utils.h"

// Restrict debug prints to math thread only to avoid multi-TRISC print backpressure
// perturbing synchronization while debugging hangs.
#undef DPRINT
#define DPRINT DPRINT_MATH

namespace generic = norm::kernel_util::generic;
namespace kutil = norm::kernel_util;
namespace numeric = kutil::compute::numeric;
namespace policies = kutil::compute::policies;

ALWI void ACQ() {
    tile_regs_acquire();
    tile_regs_wait();
}
ALWI void REL() {
    WAYPOINT("RL1");
    tile_regs_commit();
    WAYPOINT("RL2");
    tile_regs_release();
    WAYPOINT("RL3");
    WAYPOINT("RL4");
}

void kernel_main() {
    WAYPOINT("LC0");
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
    WAYPOINT("LC1");

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
        WAYPOINT("LC2");
        DPRINT("[layernorm_compute] LOOP_BEGIN ncht ncht={} NCHt={}\n", ncht, NCHt);
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
            DPRINT(
                "[layernorm_compute] LOOP_BEGIN pre_add_block ncht={} start={} size={} full={}\n",
                ncht,
                block.start(),
                block.size(),
                block.full_block_size());
            ACQ();
            // In/inb come from the reader and need to be
            // synced on full block size. Keep cb_x aligned
            // to full block size as well so pre-add/no-pre-add
            // can be handled the same way.
            cb_in_obj.wait_front(block.full_block_size());
            cb_inb_obj.wait_front(block.full_block_size());
            cb_x_obj.reserve_back(block.full_block_size());
            for (auto i : block.local()) {
                DPRINT("[layernorm_compute] LOOP_BEGIN pre_add_i ncht={} start={} i={}\n", ncht, block.start(), i);
                add_tiles(cb_in, cb_inb, i, i, i);
                pack_tile(i, cb_x);
                DPRINT("[layernorm_compute] LOOP_END pre_add_i ncht={} start={} i={}\n", ncht, block.start(), i);
            }
            REL();
            cb_x_obj.push_back(block.full_block_size());  // push the sum into the same buffer
            cb_in_obj.pop_front(block.full_block_size());
            cb_inb_obj.pop_front(block.full_block_size());
            DPRINT(
                "[layernorm_compute] LOOP_END pre_add_block ncht={} start={} size={} full={}\n",
                ncht,
                block.start(),
                block.size(),
                block.full_block_size());
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
        WAYPOINT("EX0");  // before E[x] reduce
        DPRINT("[probe] BUILD_TAG_v2 before Ex reduce ncht={}\n", ncht);
        numeric::row_wise_mean<PoolType::SUM, ReduceDim::REDUCE_ROW, FLOAT32_REDUCTION, policies::FullBlockWithoutPopPolicy>(
            cb_x, cb_scaler, cb_ex, W, Wt, block_size, tile_width);
        WAYPOINT("EX1");  // E[x] reduce returned
        DPRINT("[probe] after Ex reduce\n");

        // x - E[x]
        WAYPOINT("XR0");  // before reconfig_data_format(cb_x, cb_ex)
        reconfig_data_format(cb_x, cb_ex);
        WAYPOINT("XR1");  // before cb_ex.wait_front(1)
        cb_ex_obj.wait_front(1);
        WAYPOINT("XR2");  // before cb_xmm.reserve_back
        cb_xmm_obj.reserve_back(total_buffer_size);
        WAYPOINT("XR3");  // before sub_bcast_cols_init_short
        DPRINT("[probe] before sub_bcast_cols_init_short\n");
        sub_bcast_cols_init_short(cb_x, cb_ex);
        DPRINT("[probe] after sub_bcast_cols_init_short\n");
        WAYPOINT("XR4");  // sub_bcast_cols_init_short returned
        for (auto block : generic::blocks(Wt, block_size)) {
            WAYPOINT("S00");
            cb_x_obj.wait_front(block.start() + block.full_block_size());
            WAYPOINT("S01");
            DPRINT(
                "[layernorm_compute] LOOP_BEGIN sub_block ncht={} start={} size={} full={}\n",
                ncht,
                block.start(),
                block.size(),
                block.full_block_size());
            WAYPOINT("S0A");
            // NOTE: do NOT use ACQ() here. This block uses a split (manual) dest handshake:
            // tile_regs_acquire/commit on MATH and an explicit tile_regs_wait/release on PACK
            // (see tile_regs_commit/tile_regs_wait below). ACQ() already issues tile_regs_wait(),
            // which would give the PACK thread two waits against a single MATH commit and deadlock.
            tile_regs_acquire();
            WAYPOINT("S0B");
            WAYPOINT("S02");
            for (uint32_t i = 0; i < block.full_block_size(); ++i) {
                const uint32_t src_i = (i < block.size()) ? i : 0;
                DPRINT(
                    "[layernorm_compute] LOOP_BEGIN sub_i ncht={} start={} i={} src_i={}\n",
                    ncht,
                    block.start(),
                    i,
                    src_i);
                WAYPOINT("S03");
                sub_tiles_bcast_cols(cb_x, cb_ex, src_i, 0, i);
                WAYPOINT("S04");
                DPRINT(
                    "[layernorm_compute] LOOP_END sub_i ncht={} start={} i={} src_i={}\n",
                    ncht,
                    block.start(),
                    i,
                    src_i);
            }
            tile_regs_commit();
            tile_regs_wait();
            WAYPOINT("S05");
            pack_reconfig_data_format(cb_xmm);
            for (uint32_t i = 0; i < block.full_block_size(); ++i) {
                pack_tile(i, cb_xmm);
            }
            // Release tile regs before any potentially blocking CB operations.
            // Quasar appears more sensitive to holding tile regs across CB waits.
            WAYPOINT("S54");
            tile_regs_release();
            WAYPOINT("S55");
            WAYPOINT("S51");
            cb_xmm_obj.push_back(block.full_block_size());
            WAYPOINT("S52");
            WAYPOINT("S56");
            cb_x_obj.pop_front(block.full_block_size());
            WAYPOINT("S53");
            WAYPOINT("S57");
            WAYPOINT("S06");
            DPRINT(
                "[layernorm_compute] LOOP_END sub_block ncht={} start={} size={} full={}\n",
                ncht,
                block.start(),
                block.size(),
                block.full_block_size());
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
            DPRINT(
                "[layernorm_compute] LOOP_BEGIN varmul_block ncht={} start={} size={} full={}\n",
                ncht,
                block.start(),
                block.size(),
                block.full_block_size());
#ifndef RMSNORM
            cb_xmm_obj.wait_front(block.start() + block.size());
#else
            cb_xmm_obj.wait_front(block.start() + block.full_block_size());
#endif
            cb_xmm2_obj.reserve_back(block.full_block_size());
            ACQ();
            for (auto i : block.local()) {
                DPRINT("[layernorm_compute] LOOP_BEGIN varmul_i ncht={} start={} i={}\n", ncht, block.start(), i);
                const auto global_i = block.to_global(i);
                mul_tiles(cb_xmm, cb_xmm, global_i, global_i, i);
                pack_tile(i, cb_xmm2);
                DPRINT("[layernorm_compute] LOOP_END varmul_i ncht={} start={} i={}\n", ncht, block.start(), i);
            }
            cb_xmm2_obj.push_back(block.full_block_size());
            REL();
            DPRINT(
                "[layernorm_compute] LOOP_END varmul_block ncht={} start={} size={} full={}\n",
                ncht,
                block.start(),
                block.size(),
                block.full_block_size());
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
        // rsqrt_tile_init<LEGACY_RSQRT>();
        // rsqrt_tile<LEGACY_RSQRT>(dst0);
        pack_reconfig_data_format(cb_ex2pe);
        pack_tile(dst0, cb_ex2pe);
        cb_ex2pe_obj.push_back(1);
        REL();
        cb_ex2_obj.pop_front(1);

        // (x-E[x]) / sqrt(Var[x] + eps) * gamma + beta
        cb_ex2pe_obj.wait_front(1);
        for (auto block : generic::blocks(Wt, block_size)) {
            DPRINT(
                "[layernorm_compute] LOOP_BEGIN norm_block ncht={} start={} size={} full={}\n",
                ncht,
                block.start(),
                block.size(),
                block.full_block_size());
            reconfig_data_format(cb_xmm, cb_ex2pe);
#if defined FUSE_GAMMA || defined FUSE_BETA
            pack_reconfig_data_format(cb_fusion);
#else
            pack_reconfig_data_format(cb_out);
#endif
            WAYPOINT("CO0");
            cb_im_or_out_obj.reserve_back(block.full_block_size());
#if defined RMSNORM and not defined FUSE_PRE_ADD && (defined FUSE_GAMMA || defined FUSE_BETA)
            reconfig_data_format_srca(cb_fusion, cb_xmm);
#endif
            ACQ();
            mul_bcast_cols_init_short(cb_xmm, cb_ex2pe);
            for (auto i : block.local()) {
                DPRINT("[layernorm_compute] LOOP_BEGIN norm_i ncht={} start={} i={}\n", ncht, block.start(), i);
                mul_tiles_bcast_cols(cb_xmm, cb_ex2pe, block.to_global(i), 0, i);
#ifdef SFPU_OP_INIT_ACTIVATION
                if constexpr (!(do_gamma == 1 || do_beta == 1)) {
                    SFPU_OP_INIT_ACTIVATION
                    SFPU_OP_FUNC_ACTIVATION
                }
#endif
                pack_tile(i, cb_im_or_out);
                DPRINT("[layernorm_compute] LOOP_END norm_i ncht={} start={} i={}\n", ncht, block.start(), i);
            }
            cb_im_or_out_obj.push_back(block.full_block_size());
            WAYPOINT("CO1");
            REL();

#if defined FUSE_GAMMA || defined FUSE_BETA
#if defined RMSNORM and not defined FUSE_PRE_ADD
            reconfig_data_format_srca(cb_xmm, cb_fusion);
#endif
#endif

#ifdef FUSE_GAMMA
            {
#ifndef FUSE_BETA
                pack_reconfig_data_format(cb_out);
#endif
                reconfig_data_format_srcb(cb_ex2pe, cb_gamma);
                ACQ();
#ifdef FUSE_BETA
                uint32_t cb_outg = cb_fusion;
#else
                uint32_t cb_outg = cb_out;
#endif
                DataflowBuffer cb_outg_obj(cb_outg);
                mul_bcast_rows_init_short(cb_fusion, cb_gamma);
                cb_outg_obj.reserve_back(block.full_block_size());
                cb_gamma_obj.wait_front(block.start() + block.full_block_size());
                cb_fusion_obj.wait_front(block.full_block_size());
                for (auto i : block.local()) {
                    DPRINT("[layernorm_compute] LOOP_BEGIN gamma_i ncht={} start={} i={}\n", ncht, block.start(), i);
                    mul_tiles_bcast_rows(cb_fusion, cb_gamma, i, block.to_global(i), i);
#ifdef SFPU_OP_INIT_ACTIVATION
                    if constexpr (!(do_beta == 1)) {
                        SFPU_OP_INIT_ACTIVATION
                        SFPU_OP_FUNC_ACTIVATION
                    }
#endif
                    pack_tile(i, cb_outg);
                    DPRINT("[layernorm_compute] LOOP_END gamma_i ncht={} start={} i={}\n", ncht, block.start(), i);
                }
                cb_fusion_obj.pop_front(block.full_block_size());
                cb_outg_obj.push_back(block.full_block_size());
                REL();
            }
#endif
#ifdef FUSE_BETA
            {
                pack_reconfig_data_format(cb_out);
#ifdef FUSE_GAMMA
                reconfig_data_format_srcb(cb_gamma, cb_beta);
#else
                reconfig_data_format_srcb(cb_ex2pe, cb_beta);
#endif
                ACQ();
                add_bcast_rows_init_short(cb_fusion, cb_beta);
                cb_out_obj.reserve_back(block.full_block_size());
                cb_beta_obj.wait_front(block.start() + block.full_block_size());
                cb_fusion_obj.wait_front(block.full_block_size());
                for (auto i : block.local()) {
                    DPRINT("[layernorm_compute] LOOP_BEGIN beta_i ncht={} start={} i={}\n", ncht, block.start(), i);
                    add_tiles_bcast_rows(cb_fusion, cb_beta, i, block.to_global(i), i);
#ifdef SFPU_OP_INIT_ACTIVATION
                    SFPU_OP_INIT_ACTIVATION
                    SFPU_OP_FUNC_ACTIVATION
#endif
                    pack_tile(i, cb_out);
                    DPRINT("[layernorm_compute] LOOP_END beta_i ncht={} start={} i={}\n", ncht, block.start(), i);
                }
                cb_fusion_obj.pop_front(block.full_block_size());
                cb_out_obj.push_back(block.full_block_size());
                REL();
            }
#endif
            DPRINT(
                "[layernorm_compute] LOOP_END norm_block ncht={} start={} size={} full={}\n",
                ncht,
                block.start(),
                block.size(),
                block.full_block_size());
        }
        cb_ex2pe_obj.pop_front(1);
        cb_xmm_obj.pop_front(total_buffer_size);

#ifdef UNTILIZE_OUT
        constexpr uint32_t cb_out_rm = dfb::cb_out_rm;
        untilize_all_blocks_from_cb<block_size>(cb_out, cb_out_rm, Wt);
#endif
        DPRINT("[layernorm_compute] LOOP_END ncht ncht={} NCHt={}\n", ncht, NCHt);
    }  // NCHt loop
}
