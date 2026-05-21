// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/compute_kernel_api.h"
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/operations/normalization/kernel_util/compute/numeric.h"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "api/dataflow/circular_buffer.h"
#include "experimental/kernel_args.h"

#include "layernorm_compute_utils.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace kutil = norm::kernel_util;
namespace numeric = kutil::compute::numeric;
namespace policies = kutil::compute::policies;
namespace generic = kutil::generic;

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

    constexpr uint32_t onetile = 1;

    constexpr uint32_t cb_scaler = dfb::cb_scaler;
    constexpr uint32_t cb_eps = dfb::cb_eps;
    constexpr uint32_t cb_in = dfb::cb_in;
    constexpr uint32_t cb_inb = dfb::cb_inb;
    constexpr uint32_t cb_out = dfb::cb_out;
    constexpr uint32_t cb_gamma = dfb::cb_gamma;
    constexpr uint32_t cb_beta = dfb::cb_beta;
    constexpr uint32_t cb_xmm = dfb::cb_xmm;
    constexpr uint32_t cb_ex = dfb::cb_ex;
    constexpr uint32_t cb_ex2 = dfb::cb_ex2;
    constexpr uint32_t cb_xmm2 = dfb::cb_xmm2;
    constexpr uint32_t cb_ex2pe = dfb::cb_ex2pe;
    uint32_t cb_fusion = dfb::cb_fusion;
    constexpr auto scaler0 = 0;
    constexpr uint32_t cb_accumulate = dfb::cb_accumulate;

#ifdef TILIZE_IN
    constexpr uint32_t cb_in_rm = dfb::cb_in_rm;
#endif

#ifdef FUSE_PRE_ADD
#ifdef RMSNORM
    constexpr uint32_t cb_x = cb_xmm;
#else
    constexpr uint32_t cb_x = dfb::cb_x;
#endif
#else
    constexpr uint32_t cb_x = cb_in;
#endif

    DataflowBuffer cb_eps_obj(cb_eps);
    DataflowBuffer cb_in_obj(cb_in);
    DataflowBuffer cb_inb_obj(cb_inb);
    DataflowBuffer cb_out_obj(cb_out);
    DataflowBuffer cb_gamma_obj(cb_gamma);
    DataflowBuffer cb_beta_obj(cb_beta);
    DataflowBuffer cb_xmm_obj(cb_xmm);
    DataflowBuffer cb_ex_obj(cb_ex);
    DataflowBuffer cb_ex2_obj(cb_ex2);
    DataflowBuffer cb_xmm2_obj(cb_xmm2);
    DataflowBuffer cb_ex2pe_obj(cb_ex2pe);
    DataflowBuffer cb_accumulate_obj(cb_accumulate);

#ifdef FUSE_PRE_ADD
    binary_op_init_common(cb_in, cb_inb, cb_x);
#else
    // Always call binary_op_init_common regardless of TILIZE_IN.
    // This initializes llk_pack_dest_init, which sets up the MATH-PACK DST semaphore
    // in the "available for MATH" state.  Without it, the first tilize_block call's
    // internal llk_math_wait_for_dest_available() spins forever (deadlock).
    binary_op_init_common(cb_in, cb_scaler, cb_ex);
#endif
    cb_eps_obj.wait_front(1);  // comes from the reader

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        constexpr int onetile = 1;
        constexpr int dst0 = 0;
#ifndef RMSNORM
        // Start of
        //  E[x]
        //  aka   ∑(x)
        //      --------
        //         n
#ifdef FUSE_PRE_ADD
        numeric::row_wise_mean_with_pre_add<
            PoolType::SUM,
            ReduceDim::REDUCE_ROW,
            FLOAT32_REDUCTION,
            policies::FullBlockWithPopPolicy>(cb_in, cb_inb, cb_scaler, cb_ex, W, Wt, block_size, tile_width);
#else
        numeric::row_wise_mean<PoolType::SUM, ReduceDim::REDUCE_ROW, FLOAT32_REDUCTION, policies::FullBlockWithPopPolicy>(
            cb_in, cb_scaler, cb_ex, W, Wt, block_size, tile_width);
#endif
#endif  // !RMS ifdef end
        // Start of
        // Var Calculation
        // Var(X) = ∑(x-E[x])^2
        //         -----------
        //              n
        const bool last_tile_is_partial = W % tile_width > 0;
        for (auto block : generic::blocks(Wt, block_size)) {
#ifdef TILIZE_IN
            tilize_row_major_block(cb_in_rm, cb_in, block_size, block);
            binary_op_init_common(cb_in, cb_scaler, cb_ex);
#endif
            cb_in_obj.wait_front(block.full_block_size());
            tile_regs_acquire();
#ifdef RMSNORM
            reconfig_data_format_srca(cb_in);
            copy_tile_init(cb_in);
            for (auto i : block.local()) {
                copy_tile(cb_in, i, i);
            }
#else
            // x-E[x]
            reconfig_data_format(cb_in, cb_ex);
            sub_bcast_cols_init_short(cb_in, cb_ex);
            for (auto i : block.local()) {
                sub_tiles_bcast_cols(cb_in, cb_ex, i, 0, i);
            }
#endif
            cb_in_obj.pop_front(block.full_block_size());
#ifdef FUSE_PRE_ADD
            cb_inb_obj.wait_front(block.full_block_size());
            reconfig_data_format_srca(cb_in, cb_inb);
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_inb);
            for (auto i : block.local()) {
                binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                    cb_inb, i, i);
            }
            cb_inb_obj.pop_front(block.full_block_size());
#endif
            // (x-E[x])^2. Pack to CB
            square_tile_init();
            for (auto i : block.local()) {
                square_tile(i);
            }
            tile_regs_commit();
            tile_regs_wait();
            cb_xmm2_obj.reserve_back(block.full_block_size());
            pack_reconfig_data_format(cb_xmm2);
            for (auto i : block.local()) {
                pack_tile(i, cb_xmm2);
            }
            tile_regs_release();
            cb_xmm2_obj.push_back(block.full_block_size());

            tile_regs_acquire();
            if (!block.is_first()) {
                cb_accumulate_obj.wait_front(onetile);
                reconfig_data_format_srca(cb_accumulate);
                copy_tile_init(cb_accumulate);
                copy_tile(cb_accumulate, 0, dst0);
                cb_accumulate_obj.pop_front(onetile);
            }
            cb_xmm2_obj.wait_front(block.full_block_size());

            // Accumulate (x-E[x])^2
            reconfig_data_format(cb_xmm2, cb_scaler);
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW, FLOAT32_REDUCTION>(cb_xmm2, cb_scaler, cb_accumulate);
            for (auto i : block.local()) {
                const auto scaler_tile_idx = block.to_global(i) == Wt - 1 && last_tile_is_partial ? 1 : 0;
                reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW, FLOAT32_REDUCTION>(
                    cb_xmm2, cb_scaler, i, scaler_tile_idx, dst0);
            }

            cb_pop_front(cb_xmm2, block.full_block_size());

            const auto final_iter = block.last() == Wt;
            const auto pack_cb = final_iter ? cb_ex2 : cb_accumulate;
            if (final_iter) {
                // Divide by W
                binop_with_scalar_tile_init();
                mul_unary_tile(dst0, generic::bit_cast<uint32_t>(1.0f / W));
            }

            reduce_uninit<FLOAT32_REDUCTION>();
            tile_regs_commit();
            tile_regs_wait();

            DataflowBuffer(pack_cb).reserve_back(onetile);
            pack_reconfig_data_format(pack_cb);
            pack_tile(dst0, pack_cb);
            tile_regs_release();
            DataflowBuffer(pack_cb).push_back(onetile);
        }

        // End of
        // Var Calculation
        // Var(X) = ∑(x-E[x])^2
        //         -----------

        // Start of
        // Calculation
        //                     1
        //  cb_ex2pe =   -------------
        //               √(Var(X) + ε)
        cb_ex2_obj.wait_front(onetile);
        reconfig_data_format(cb_ex2, cb_eps);
        tile_regs_acquire();

        add_tiles_init(cb_ex2, cb_eps);
        add_tiles(cb_ex2, cb_eps, 0, 0, dst0);

        rsqrt_tile_init<LEGACY_RSQRT>();
        rsqrt_tile<LEGACY_RSQRT>(dst0);

        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_ex2pe);
        pack_tile(dst0, cb_ex2pe);
        tile_regs_release();
        cb_ex2pe_obj.push_back(onetile);
        cb_ex2_obj.pop_front(onetile);

        // broadcasts the tile since cb_ex2pe is a column vector that contains the important data
        cb_ex2pe_obj.wait_front(onetile);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_ex2pe);

        unary_bcast_init<BroadcastType::COL>(cb_ex2pe, cb_ex2pe);
        unary_bcast<BroadcastType::COL>(cb_ex2pe, 0, dst0);
        cb_ex2pe_obj.pop_front(onetile);

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_ex2pe);
        tile_regs_release();
        cb_ex2pe_obj.push_back(onetile);
        cb_ex2pe_obj.wait_front(onetile);

        // End of
        // Calculation
        //                     1
        //  cb_ex2pe =   -------------
        //               √(Var(X) + ε)

        // Start of
        // Final Val Calc
        //    x-E[X]
        //(---------------*𝛄)+ß
        //  √(Var(X)+ε)
        for (auto block : generic::blocks(Wt, block_size)) {
#ifdef TILIZE_IN
            // Tilize one block from cb_in_rm → cb_in per loop iteration (Pass 2).
            // Reader supplies this second pass of data after the variance data.
            tilize_row_major_block(cb_in_rm, cb_in, block_size, block);

            binary_op_init_common(cb_in, cb_scaler, cb_ex);
#endif
            tile_regs_acquire();
            cb_in_obj.wait_front(block.full_block_size());
#ifdef RMSNORM
            reconfig_data_format_srca(cb_in);
            copy_tile_init(cb_in);
            for (auto i : block.local()) {
                copy_tile(cb_in, i, i);
            }
#else
            cb_ex_obj.wait_front(1);
            reconfig_data_format(cb_in, cb_ex);
            sub_bcast_cols_init_short(cb_in, cb_ex);
            // x-E[x]
            for (auto i : block.local()) {
                sub_tiles_bcast_cols(cb_in, cb_ex, i, 0, i);
            }
#endif
            cb_in_obj.pop_front(block.full_block_size());
#ifdef FUSE_PRE_ADD
            cb_inb_obj.wait_front(block.full_block_size());
            reconfig_data_format_srca(cb_inb);
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_inb);
            for (auto i : block.local()) {
                binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                    cb_inb, i, i);
            }
            cb_inb_obj.pop_front(block.full_block_size());
#endif
            tile_regs_commit();
            tile_regs_wait();
            // Note: We shouldn't have to pack to
            // intermediate CB. We should be able to
            // do a binary dest with reuse (as we used
            // to). However, tt-llk #868 is preventing
            // that from working at the moment.
            cb_xmm_obj.reserve_back(block.full_block_size());
            pack_reconfig_data_format(cb_xmm);
            for (auto i : block.local()) {
                pack_tile(i, cb_xmm);
            }
            cb_xmm_obj.push_back(block.full_block_size());
            tile_regs_release();

            cb_xmm_obj.wait_front(block.full_block_size());
            reconfig_data_format(cb_xmm, cb_ex2pe);
            tile_regs_acquire();

            mul_tiles_init(cb_xmm, cb_ex2pe);
            for (auto i : block.local()) {
                mul_tiles(cb_xmm, cb_ex2pe, i, 0, i);
#ifdef SFPU_OP_INIT_ACTIVATION
                // Activation must be applied last. If do_gamma != 0 or do_beta != 0 then
                // activation will be applied after the gamma/beta multiplication/addition.
                // Otherwise, we can apply the activation here.
                if constexpr (!(do_gamma == 1 || do_beta == 1)) {
                    SFPU_OP_INIT_ACTIVATION
                    SFPU_OP_FUNC_ACTIVATION
                }
#endif
            }
            tile_regs_commit();
            tile_regs_wait();

            if constexpr (!(do_gamma == 1 or do_beta == 1)) {
                cb_fusion = cb_out;
            }
            DataflowBuffer(cb_fusion).reserve_back(block.full_block_size());
            pack_reconfig_data_format(cb_fusion);
            for (auto i : block.local()) {
                pack_tile(i, cb_fusion);
            }
            tile_regs_release();
            DataflowBuffer(cb_fusion).push_back(block.full_block_size());
            cb_xmm_obj.pop_front(block.full_block_size());

            if constexpr (do_gamma == 1) {
                tile_regs_acquire();
                tile_regs_wait();
                reconfig_data_format(cb_fusion, cb_gamma);
                if constexpr (!do_beta) {
                    pack_reconfig_data_format(cb_out);
                }
                cb_gamma_obj.wait_front(block.full_block_size());
                DataflowBuffer(cb_fusion).wait_front(block.full_block_size());
                mul_bcast_rows_init_short(cb_fusion, cb_gamma);
                for (auto i : block.local()) {
                    mul_tiles_bcast_rows(cb_fusion, cb_gamma, i, i, i);
#ifdef SFPU_OP_INIT_ACTIVATION
                    // Activation must be applied last. If do_beta != 0 then
                    // activation will be applied after the beta addition.
                    // Otherwise, we can apply the activation here.
                    if constexpr (!(do_beta == 1)) {
                        SFPU_OP_INIT_ACTIVATION
                        SFPU_OP_FUNC_ACTIVATION
                    }
#endif
                }
                tile_regs_commit();
                cb_gamma_obj.pop_front(block.full_block_size());
                DataflowBuffer(cb_fusion).pop_front(block.full_block_size());
                if constexpr (!do_beta) {
                    cb_out_obj.reserve_back(block.full_block_size());
                    for (auto i : block.local()) {
                        pack_tile(i, cb_out);
                    }
                    cb_out_obj.push_back(block.full_block_size());
                } else {
                    DataflowBuffer(cb_fusion).reserve_back(block.full_block_size());
                    for (auto i : block.local()) {
                        pack_tile(i, cb_fusion);
                    }
                    DataflowBuffer(cb_fusion).push_back(block.full_block_size());
                }

                tile_regs_release();
            }
            if constexpr (do_beta == 1) {
                tile_regs_acquire();
                tile_regs_wait();
                reconfig_data_format(cb_fusion, cb_beta);
                pack_reconfig_data_format(cb_out);
                cb_beta_obj.wait_front(block.full_block_size());
                DataflowBuffer(cb_fusion).wait_front(block.full_block_size());
                add_bcast_rows_init_short(cb_fusion, cb_beta);
                for (auto i : block.local()) {
                    add_tiles_bcast_rows(cb_fusion, cb_beta, i, i, i);
#ifdef SFPU_OP_INIT_ACTIVATION
                    SFPU_OP_INIT_ACTIVATION
                    SFPU_OP_FUNC_ACTIVATION
#endif
                }
                tile_regs_commit();
                cb_beta_obj.pop_front(block.full_block_size());
                DataflowBuffer(cb_fusion).pop_front(block.full_block_size());
                cb_out_obj.reserve_back(block.full_block_size());
                for (auto i : block.local()) {
                    pack_tile(i, cb_out);
                }
                tile_regs_release();
                cb_out_obj.push_back(block.full_block_size());
            }

#ifdef UNTILIZE_OUT
            constexpr uint32_t cb_out_rm = dfb::cb_out_rm;
            untilize_row_major_block<decltype(block), block_size>(cb_out, cb_out_rm, block);
#endif
        }  // block loop
        // End of
        // Final Val Calc
        //    x-E[X]
        //(---------------*𝛄)+ß
        //  √(Var(X)+ε)
#ifndef RMSNORM
        cb_ex_obj.pop_front(onetile);
#endif
        cb_ex2pe_obj.pop_front(onetile);
    }  // NCHt loop
}
