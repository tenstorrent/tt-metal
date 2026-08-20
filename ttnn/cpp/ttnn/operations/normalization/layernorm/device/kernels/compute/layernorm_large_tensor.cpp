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
#include "experimental/kernel_args.h"
#include "ttnn/operations/normalization/kernel_util/compute/numeric.h"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "api/dataflow/dataflow_buffer.h"

#include "layernorm_compute_utils.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"  // Square
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"

namespace ckl = compute_kernel_lib;

namespace kutil = norm::kernel_util;
namespace numeric = kutil::compute::numeric;
namespace policies = kutil::compute::policies;
namespace generic = kutil::generic;

struct FusedActivation : ckl::UnaryOp<FusedActivation, ckl::Dst::D0> {
    static ALWI void init() {
#ifdef SFPU_OP_INIT_ACTIVATION
        SFPU_OP_INIT_ACTIVATION
#endif
    }

    static ALWI void exec_impl([[maybe_unused]] uint32_t i) {
#ifdef SFPU_OP_INIT_ACTIVATION
        SFPU_OP_FUNC_ACTIVATION
#endif
    }
};

#ifdef SFPU_OP_INIT_ACTIVATION
constexpr bool fused_activation_enabled = true;
#else
constexpr bool fused_activation_enabled = false;
#endif

void kernel_main() {
    uint32_t NCHt = get_arg(args::NCHt);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t block_size = get_arg(args::block_size);
    constexpr uint32_t do_gamma = get_arg(args::do_gamma);
    constexpr uint32_t do_beta = get_arg(args::do_beta);
    constexpr bool activate_after_normalize = fused_activation_enabled && !do_gamma && !do_beta;
    constexpr bool activate_after_gamma = fused_activation_enabled && !do_beta;
    // Fused activation runs after the last enabled affine stage: after beta, otherwise after
    // gamma, otherwise immediately after normalization.
    constexpr bool FLOAT32_DTYPE = get_arg(args::fp32_dest_acc_en) == 1;
    constexpr bool FLOAT32_REDUCTION = get_arg(args::float32_reduction) == 1;
    constexpr bool LEGACY_RSQRT = get_arg(args::legacy_rsqrt) == 1;
    constexpr uint32_t W = get_arg(args::W);
    constexpr uint32_t tile_width = get_arg(args::tile_width);

    constexpr uint32_t onetile = 1;

    constexpr auto dfb_scaler_id = dfb::scaler;
    constexpr auto dfb_eps_id = dfb::eps;
    constexpr auto dfb_in_id = dfb::in;
#ifdef FUSE_PRE_ADD
    constexpr auto dfb_inb_id = dfb::inb;
#else
    constexpr auto dfb_inb_id = dfb_in_id;
#endif
    constexpr auto dfb_out_id = dfb::out;
#ifdef FUSE_GAMMA
    constexpr auto dfb_gamma_id = dfb::gamma;
#else
    constexpr auto dfb_gamma_id = dfb_out_id;
#endif
#ifdef FUSE_BETA
    constexpr auto dfb_beta_id = dfb::beta;
#else
    constexpr auto dfb_beta_id = dfb_out_id;
#endif
    constexpr uint32_t dfb_xmm_id = dfb::xmm;
#ifndef RMSNORM
    constexpr auto dfb_ex_id = dfb::ex;
#endif
    constexpr auto dfb_ex2_id = dfb::ex2;
    constexpr auto dfb_xmm2_id = dfb::xmm2;
    constexpr auto dfb_ex2pe_id = dfb::ex2pe;
#if defined(FUSE_GAMMA) || defined(FUSE_BETA)
    constexpr auto dfb_fusion_id = dfb::fusion;
#else
    constexpr auto dfb_fusion_id = dfb_out_id;
#endif
#if defined(FUSE_GAMMA) || defined(FUSE_BETA)
    constexpr auto dfb_im_or_out_id = dfb_fusion_id;
#else
    constexpr auto dfb_im_or_out_id = dfb_out_id;
#endif
    constexpr auto scaler0 = 0;
    constexpr auto dfb_accumulate_id = dfb::accumulate;

#ifdef TILIZE_IN
    constexpr auto dfb_in_rm_id = dfb::in_rm;
#endif

#ifdef RMSNORM
    constexpr bool is_rmsnorm = true;
#else
    constexpr bool is_rmsnorm = false;
#endif
#ifdef FUSE_PRE_ADD
    constexpr bool do_fuse_pre_add = true;
#else
    constexpr bool do_fuse_pre_add = false;
#endif

#ifdef FUSE_PRE_ADD
#ifdef RMSNORM
    constexpr uint32_t dfb_x_id = dfb_xmm_id;
#else
    constexpr uint32_t dfb_x_id = dfb::x;
#endif
#else
    constexpr uint32_t dfb_x_id = dfb_in_id;
#endif

    DataflowBuffer dfb_eps_obj(dfb_eps_id);
    DataflowBuffer dfb_scaler_obj(dfb_scaler_id);
    DataflowBuffer dfb_in_obj(dfb_in_id);
#ifdef TILIZE_IN
    DataflowBuffer dfb_in_rm_obj(dfb_in_rm_id);
#endif
#ifdef FUSE_PRE_ADD
    DataflowBuffer dfb_inb_obj(dfb_inb_id);
#endif
    DataflowBuffer dfb_out_obj(dfb_out_id);
#ifndef RMSNORM
    DataflowBuffer dfb_ex_obj(dfb_ex_id);
#endif
    DataflowBuffer dfb_xmm2_obj(dfb_xmm2_id);
    DataflowBuffer dfb_ex2pe_obj(dfb_ex2pe_id);
    DataflowBuffer dfb_accumulate_obj(dfb_accumulate_id);

#ifdef FUSE_PRE_ADD
    compute_kernel_hw_startup(dfb_in_id, dfb_inb_id, dfb_x_id);
#else
    // Always call compute_kernel_hw_startup regardless of TILIZE_IN.
    // This initializes llk_pack_dest_init, which sets up the MATH-PACK DST semaphore
    // in the "available for MATH" state.  Without it, the first tilize_block call's
    // internal llk_math_wait_for_dest_available() spins forever (deadlock).
#ifdef RMSNORM
    compute_kernel_hw_startup(dfb_in_id, dfb_scaler_id, dfb_xmm2_id);
#else
    compute_kernel_hw_startup(dfb_in_id, dfb_scaler_id, dfb_ex_id);
#endif
#endif
    dfb_eps_obj.wait_front(1);  // comes from the reader

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
            policies::FullBlockWithPopPolicy>(
            dfb_in_obj, dfb_inb_obj, dfb_scaler_obj, dfb_ex_obj, W, Wt, block_size, tile_width);
#else
        numeric::
            row_wise_mean<PoolType::SUM, ReduceDim::REDUCE_ROW, FLOAT32_REDUCTION, policies::FullBlockWithPopPolicy>(
                dfb_in_obj, dfb_scaler_obj, dfb_ex_obj, W, Wt, block_size, tile_width);
#endif
#endif  // !RMS ifdef end
        // Start of
        // Var Calculation
        // Var(X) = ∑(x-E[x])^2
        //         -----------
        //              n
        const bool last_tile_is_partial = W % tile_width > 0;
        for (auto block : generic::blocks(Wt, block_size)) {
            const auto block_shape = ckl::IterationShape::tiles(block.size())
                                         .block_size(block.full_block_size(), ckl::BlockTailSync::FullBlock);
#ifdef TILIZE_IN
            tilize_row_major_block(dfb_in_rm_obj, dfb_in_obj, block_size, block);
            // TODO(#52395): replace this unsafe mid-kernel startup with a targeted DST re-arm.
#ifdef RMSNORM
            compute_kernel_hw_startup(dfb_in_id, dfb_scaler_id, dfb_xmm2_id);
#else
            compute_kernel_hw_startup(dfb_in_id, dfb_scaler_id, dfb_ex_id);
#endif
#endif
            ckl::eltwise_chain(
                block_shape,
                ckl::Optional<
                    is_rmsnorm,  // RMSNORM: copy x (no mean subtraction)
                    ckl::CopyTile<
                        ckl::input(
                            dfb_in_id,
                            ckl::WaitPolicy::PerBlockSize,
                            ckl::PopPolicy::PerBlockSize,
                            ckl::InputTileMapping::Block),
                        ckl::Dst::D0>>{},
                ckl::Optional<
                    !is_rmsnorm,  // LayerNorm: x - E[x] (reads dfb_ex_id; stripped under RMSNORM)
                    ckl::BinaryFpu<
                        ckl::BinaryFpuOp::Sub,
                        ckl::input(
                            dfb_in_id,
                            ckl::WaitPolicy::PerBlockSize,
                            ckl::PopPolicy::PerBlockSize,
                            ckl::InputTileMapping::Block),
                        ckl::input(dfb_ex_id, ckl::BroadcastDim::Col, ckl::WaitPolicy::None, ckl::PopPolicy::None)>>{},
                ckl::Optional<
                    do_fuse_pre_add,  // FUSE_PRE_ADD: + b (DEST-reuse), else stripped
                    ckl::DestReuseBinary<
                        ckl::BinaryFpuOp::Add,
                        ckl::input(
                            dfb_inb_id,
                            ckl::WaitPolicy::PerBlockSize,
                            ckl::PopPolicy::PerBlockSize,
                            ckl::InputTileMapping::Block),
                        ckl::DestReuseType::DEST_TO_SRCB>>{},
                // (x-E[x])^2. Pack to CB
                ckl::Square<ckl::Dst::D0>{},
                ckl::PackTile<ckl::output(
                    dfb_xmm2_id, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>{});

            tile_regs_acquire();
            if (!block.is_first()) {
                dfb_accumulate_obj.wait_front(onetile);
                reconfig_data_format_srca(dfb_accumulate_id);
                copy_tile_init(dfb_accumulate_id);
                copy_tile(dfb_accumulate_id, 0, dst0);
                dfb_accumulate_obj.pop_front(onetile);
            }
            dfb_xmm2_obj.wait_front(block.full_block_size());

            // Accumulate (x-E[x])^2
            reconfig_data_format(dfb_scaler_id, dfb_xmm2_id);
            reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb_xmm2_id, dfb_scaler_id, dfb_accumulate_id);
            for (auto i : block.local()) {
                const auto scaler_tile_idx = block.to_global(i) == Wt - 1 && last_tile_is_partial ? 1 : 0;
                reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb_xmm2_id, dfb_scaler_id, i, scaler_tile_idx, dst0);
            }

            dfb_xmm2_obj.pop_front(block.full_block_size());

            const auto final_iter = block.last() == Wt;
            const auto pack_dfb_id = final_iter ? dfb_ex2_id : dfb_accumulate_id;
            DataflowBuffer pack_dfb_obj(pack_dfb_id);
            if (final_iter) {
                // Divide by W
                binop_with_scalar_tile_init();
                mul_unary_tile(dst0, generic::bit_cast<uint32_t>(1.0f / W));
            }

            reduce_uninit();
            tile_regs_commit();
            tile_regs_wait();

            pack_dfb_obj.reserve_back(onetile);
            pack_reconfig_data_format(pack_dfb_id);
            pack_tile(dst0, pack_dfb_id);
            tile_regs_release();
            pack_dfb_obj.push_back(onetile);
        }

        // End of
        // Var Calculation
        // Var(X) = ∑(x-E[x])^2
        //         -----------

        // Start of
        // Calculation
        //                     1
        //               √(Var(X) + ε)
        ckl::eltwise_chain(
            ckl::IterationShape::tiles(onetile),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Add,
                ckl::input(dfb_ex2_id),
                ckl::input(dfb_eps_id, ckl::WaitPolicy::None, ckl::PopPolicy::None)>{},
            ckl::Rsqrt<ckl::Approx::Exact, LEGACY_RSQRT ? ckl::Legacy::On : ckl::Legacy::Off, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(dfb_ex2pe_id, ckl::ReservePolicy::None, ckl::PushPolicy::AtEnd)>{});

        // broadcasts the tile since dfb_ex2pe_id is a column vector that contains the important data
        ckl::unary_bcast<ckl::BroadcastDim::Col, ckl::input(dfb_ex2pe_id), ckl::output(dfb_ex2pe_id)>(
            ckl::IterationShape::tiles(onetile));
        dfb_ex2pe_obj.wait_front(onetile);

        // End of
        // Calculation
        //                     1
        //               √(Var(X) + ε)

        // Start of
        // Final Val Calc
        //    x-E[X]
        //(---------------*𝛄)+ß
        //  √(Var(X)+ε)
        for (auto block : generic::blocks(Wt, block_size)) {
            const auto block_shape = ckl::IterationShape::tiles(block.size())
                                         .block_size(block.full_block_size(), ckl::BlockTailSync::FullBlock);
#ifdef TILIZE_IN
            // Tilize one block from dfb_in_rm_id → dfb_in_id per loop iteration (Pass 2).
            // Reader supplies this second pass of data after the variance data.
            tilize_row_major_block(dfb_in_rm_obj, dfb_in_obj, block_size, block);

            // TODO(#52395): replace this unsafe mid-kernel startup with a targeted DST re-arm.
#ifdef RMSNORM
            compute_kernel_hw_startup(dfb_in_id, dfb_scaler_id, dfb_xmm2_id);
#else
            compute_kernel_hw_startup(dfb_in_id, dfb_scaler_id, dfb_ex_id);
#endif
#endif
#ifndef RMSNORM
            dfb_ex_obj.wait_front(1);
#endif
            ckl::eltwise_chain(
                block_shape,
                ckl::Optional<
                    is_rmsnorm,  // RMSNORM: copy x (no mean subtraction)
                    ckl::CopyTile<
                        ckl::input(
                            dfb_in_id,
                            ckl::WaitPolicy::PerBlockSize,
                            ckl::PopPolicy::PerBlockSize,
                            ckl::InputTileMapping::Block),
                        ckl::Dst::D0>>{},
                ckl::Optional<
                    !is_rmsnorm,  // LayerNorm: x - E[x] (reads dfb_ex_id; stripped under RMSNORM)
                    ckl::BinaryFpu<
                        ckl::BinaryFpuOp::Sub,
                        ckl::input(
                            dfb_in_id,
                            ckl::WaitPolicy::PerBlockSize,
                            ckl::PopPolicy::PerBlockSize,
                            ckl::InputTileMapping::Block),
                        ckl::input(dfb_ex_id, ckl::BroadcastDim::Col, ckl::WaitPolicy::None, ckl::PopPolicy::None)>>{},
                ckl::Optional<
                    do_fuse_pre_add,  // FUSE_PRE_ADD: + b (DEST-reuse), else stripped
                    ckl::DestReuseBinary<
                        ckl::BinaryFpuOp::Add,
                        ckl::input(
                            dfb_inb_id,
                            ckl::WaitPolicy::PerBlockSize,
                            ckl::PopPolicy::PerBlockSize,
                            ckl::InputTileMapping::Block),
                        ckl::DestReuseType::DEST_TO_SRCB>>{},
                // Note: We shouldn't have to pack to
                // intermediate CB. We should be able to
                // do a binary dest with reuse (as we used
                // to). However, tt-llk #868 is preventing
                // that from working at the moment.
                ckl::PackTile<ckl::output(
                    dfb_xmm_id, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>{});

            ckl::eltwise_chain(
                block_shape,
                ckl::BinaryFpu<
                    ckl::BinaryFpuOp::Mul,
                    ckl::input(
                        dfb_xmm_id,
                        ckl::WaitPolicy::PerBlockSize,
                        ckl::PopPolicy::PerBlockSize,
                        ckl::InputTileMapping::Block),
                    ckl::input(dfb_ex2pe_id, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None)>{},
                ckl::Optional<activate_after_normalize, FusedActivation>{},
                ckl::PackTile<ckl::output(
                    dfb_im_or_out_id, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>{});

            if constexpr (do_gamma == 1) {
                constexpr auto dfb_gamma_out_id = do_beta ? dfb_fusion_id : dfb_out_id;
                ckl::eltwise_chain(
                    block_shape,
                    ckl::BinaryFpu<
                        ckl::BinaryFpuOp::Mul,
                        ckl::input(
                            dfb_fusion_id,
                            ckl::WaitPolicy::PerBlockSize,
                            ckl::PopPolicy::PerBlockSize,
                            ckl::InputTileMapping::Block),
                        ckl::input(
                            dfb_gamma_id,
                            ckl::BroadcastDim::Row,
                            ckl::WaitPolicy::PerBlockSize,
                            ckl::PopPolicy::PerBlockSize,
                            ckl::InputTileMapping::Block)>{},
                    ckl::Optional<activate_after_gamma, FusedActivation>{},
                    ckl::PackTile<ckl::output(
                        dfb_gamma_out_id, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>{});
            }
            if constexpr (do_beta == 1) {
                ckl::eltwise_chain(
                    block_shape,
                    ckl::BinaryFpu<
                        ckl::BinaryFpuOp::Add,
                        ckl::input(
                            dfb_fusion_id,
                            ckl::WaitPolicy::PerBlockSize,
                            ckl::PopPolicy::PerBlockSize,
                            ckl::InputTileMapping::Block),
                        ckl::input(
                            dfb_beta_id,
                            ckl::BroadcastDim::Row,
                            ckl::WaitPolicy::PerBlockSize,
                            ckl::PopPolicy::PerBlockSize,
                            ckl::InputTileMapping::Block)>{},
                    ckl::Optional<fused_activation_enabled, FusedActivation>{},
                    ckl::PackTile<ckl::output(
                        dfb_out_id, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>{});
            }

#ifdef UNTILIZE_OUT
            constexpr auto dfb_out_rm_id = dfb::out_rm;
            DataflowBuffer dfb_out_rm_obj(dfb_out_rm_id);
            untilize_row_major_block<decltype(block), block_size>(dfb_out_obj, dfb_out_rm_obj, block);
#endif
        }  // block loop
        // End of
        // Final Val Calc
        //    x-E[X]
        //(---------------*𝛄)+ß
        //  √(Var(X)+ε)
#ifndef RMSNORM
        dfb_ex_obj.pop_front(onetile);
#endif
        dfb_ex2pe_obj.pop_front(onetile);
    }  // NCHt loop
}
