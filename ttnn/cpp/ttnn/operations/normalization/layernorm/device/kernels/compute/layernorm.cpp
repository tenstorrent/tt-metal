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
#include "experimental/kernel_args.h"
#include "ttnn/operations/normalization/kernel_util/compute/numeric.h"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "ttnn/operations/normalization/kernel_util/generic/bit.h"
#include "ttnn/operations/normalization/layernorm/device/kernels/layernorm_scaler_tiles.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/dataflow_buffer.h"

#include "layernorm_compute_utils.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"  // square
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"

namespace ckl = compute_kernel_lib;

namespace generic = norm::kernel_util::generic;
namespace kutil = norm::kernel_util;
namespace numeric = kutil::compute::numeric;
namespace policies = kutil::compute::policies;

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
    constexpr bool FLOAT32_DTYPE = get_arg(args::fp32_dest_acc_en) == 1;
    constexpr bool FLOAT32_REDUCTION = get_arg(args::float32_reduction) == 1;
    constexpr bool LEGACY_RSQRT = get_arg(args::legacy_rsqrt) == 1;
    constexpr uint32_t W = get_arg(args::W);
    constexpr uint32_t tile_width = get_arg(args::tile_width);

    constexpr auto dfb_scaler_id = dfb::scaler;
    constexpr auto dfb_eps_id = dfb::eps;
    constexpr auto dfb_in_id = dfb::in;
#ifdef FUSE_PRE_ADD
    constexpr auto dfb_inb_id = dfb::inb;
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
#if defined RMSNORM and not defined FUSE_PRE_ADD
    constexpr uint32_t dfb_xmm_id = dfb_in_id;  // x minus mean
#else
    constexpr uint32_t dfb_xmm_id = dfb::xmm;
#endif
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
    DataflowBuffer dfb_eps_obj(dfb_eps_id);
    DataflowBuffer dfb_in_obj(dfb_in_id);
#if defined RMSNORM and not defined FUSE_PRE_ADD
    DataflowBuffer& dfb_xmm_obj = dfb_in_obj;
#else
    DataflowBuffer dfb_xmm_obj(dfb_xmm_id);
#endif
    DataflowBuffer dfb_out_obj(dfb_out_id);
    DataflowBuffer dfb_ex2pe_obj(dfb_ex2pe_id);
    DataflowBuffer dfb_scaler_obj(dfb_scaler_id);

#ifdef TILIZE_IN
    constexpr auto dfb_in_rm_id = dfb::in_rm;
#endif

    constexpr int onetile = 1;
    constexpr int dst0 = 0;
    constexpr int dst1 = 1;
    constexpr auto scaler0 = 0;

#ifdef FUSE_PRE_ADD
#ifdef RMSNORM
    constexpr uint32_t dfb_x_id = dfb_xmm_id;
#else
    constexpr uint32_t dfb_x_id = dfb::x;
#endif
#else
    constexpr uint32_t dfb_x_id = dfb_in_id;
#endif

#ifdef TILIZE_IN
    compute_kernel_hw_startup(dfb_in_rm_id, dfb_in_rm_id, dfb_in_id);
#elif defined(FUSE_PRE_ADD)
    compute_kernel_hw_startup(dfb_in_id, dfb_inb_id, dfb_x_id);
#elif defined(RMSNORM)
    compute_kernel_hw_startup(dfb_xmm_id, dfb_xmm_id, dfb_xmm2_id);
#else
    compute_kernel_hw_startup(dfb_x_id, dfb_scaler_id, dfb_ex_id);
#endif

    dfb_eps_obj.wait_front(1);  // comes from the reader

#if defined(FUSE_GAMMA) || defined(FUSE_BETA)
    constexpr int dfb_im_or_out_id = dfb_fusion_id;
#else
    constexpr int dfb_im_or_out_id = dfb_out_id;
#endif

    // Intermediate buffers need to be reserved/pushed/popped
    // in full blocks
    const auto total_buffer_size = generic::blocks(Wt, block_size).total_with_remainder();
    // Math follows the valid width; Chunked lifecycles still exchange a fixed-size tail block.
    constexpr auto row_shape = ckl::IterationShape::tiles(Wt).block_size(block_size, ckl::BlockTailSync::FullBlock);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
#ifdef TILIZE_IN
        DataflowBuffer dfb_in_rm_obj(dfb_in_rm_id);
        tilize_all_blocks_to_cb<block_size>(dfb_in_rm_obj, dfb_in_obj, Wt);
        // Re-init binary ops after tilize/untilize reconfiguration. compute_kernel_hw_startup is call-once;
        // TODO(#52395): replace this mid-kernel re-init with a targeted DST re-arm.
#ifdef FUSE_PRE_ADD
        compute_kernel_hw_startup(dfb_in_id, dfb_inb_id, dfb_x_id);
#elif defined(RMSNORM)
        compute_kernel_hw_startup(dfb_xmm_id, dfb_xmm_id, dfb_xmm2_id);
#else
        compute_kernel_hw_startup(dfb_x_id, dfb_scaler_id, dfb_ex_id);
#endif
#endif
        // X + Y
#ifdef FUSE_PRE_ADD
        // The reader streams block-sized chunks, so waiting for the whole row would deadlock.
        // In/inb come from the reader and need to be
        // synced on full block size. Keep dfb_x_id aligned
        // to full block size as well so pre-add/no-pre-add
        // can be handled the same way.
        ckl::add<
            ckl::input(
                dfb_in_id, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::InputTileMapping::Block),
            ckl::input(
                dfb_inb_id, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::InputTileMapping::Block),
            ckl::output(dfb_x_id, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>(row_shape);
        // by the end of this loop we should end up with Wt tiles in dfb_x_id
#ifndef RMSNORM
        reconfig_data_format(dfb_in_id, dfb_x_id, dfb_inb_id, dfb_scaler_id);
#else
        reconfig_data_format(dfb_in_id, dfb_x_id, dfb_inb_id, dfb_x_id);
#endif
#else
#ifdef RMSNORM
        reconfig_data_format(dfb_in_id, dfb_in_id);
        pack_reconfig_data_format(dfb_xmm2_id);
#endif
#endif

#ifndef RMSNORM
        // E[x]
        DataflowBuffer dfb_x_obj(dfb_x_id), dfb_ex_obj(dfb_ex_id);
        numeric::
            row_wise_mean<PoolType::SUM, ReduceDim::REDUCE_ROW, FLOAT32_REDUCTION, policies::FullBlockWithoutPopPolicy>(
                dfb_x_obj, dfb_scaler_obj, dfb_ex_obj, W, Wt, block_size, tile_width);

        // x - E[x]; the mean stays resident for the whole row.
        ckl::sub<
            ckl::input(
                dfb_x_id, ckl::WaitPolicy::PerBlockSize, ckl::PopPolicy::PerBlockSize, ckl::InputTileMapping::Block),
            ckl::input(dfb_ex_id, ckl::BroadcastDim::Col, ckl::WaitPolicy::None, ckl::PopPolicy::None),
            ckl::output(
                dfb_xmm_id,
                ckl::ReservePolicy::PerBlockSize,
                ckl::PushPolicy::PerBlockSize,
                ckl::DataFormatReconfig::Disabled)>(row_shape);
        dfb_ex_obj.pop_front(1);

#ifndef FUSE_PRE_ADD
        reconfig_data_format_srca(dfb_x_id, dfb_xmm_id);
#endif
#endif

        // Preserve dfb_xmm_id for the normalization pass; the variance path consumes only its square.
        // compute temp = xmm*xmm = (x-E[x])^2
        ckl::square<
            ckl::input(
                dfb_xmm_id,
                ckl::WaitPolicy::Cumulative,
                ckl::PopPolicy::None,
                ckl::InputTileMapping::Block,
                ckl::DataFormatReconfig::Disabled),
            ckl::output(
                dfb_xmm2_id,
                ckl::ReservePolicy::PerBlockSize,
                ckl::PushPolicy::PerBlockSize,
                ckl::DataFormatReconfig::Disabled)>(row_shape);
#if defined RMSNORM and not defined FUSE_PRE_ADD
        reconfig_data_format(dfb_xmm_id, dfb_xmm2_id, dfb_xmm_id, dfb_scaler_id);
#endif

        // Var[x]
        DataflowBuffer dfb_xmm2_obj(dfb_xmm2_id), dfb_ex2_obj(dfb_ex2_id);
        numeric::
            row_wise_mean<PoolType::SUM, ReduceDim::REDUCE_ROW, FLOAT32_REDUCTION, policies::FullBlockWithPopPolicy>(
                dfb_xmm2_obj, dfb_scaler_obj, dfb_ex2_obj, W, Wt, block_size, tile_width);

        // 1/sqrt(Var[x] + eps)
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Add,
                ckl::input(dfb_ex2_id),
                ckl::input(dfb_eps_id, ckl::WaitPolicy::None, ckl::PopPolicy::None)>{},
            ckl::Rsqrt<ckl::Approx::Exact, LEGACY_RSQRT ? ckl::Legacy::On : ckl::Legacy::Off, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(dfb_ex2pe_id)>{});

        // Gamma and beta each contain one row and remain resident across all NCHt rows; tile
        // offsets select the current width block. TODO: wait on gamma/beta only on the first NCHt row.
        // (x-E[x]) / sqrt(Var[x] + eps) * gamma + beta
        for (auto block : generic::blocks(Wt, block_size)) {
            const auto block_shape = ckl::IterationShape::tiles(block.size())
                                         .block_size(block.full_block_size(), ckl::BlockTailSync::FullBlock);
            ckl::eltwise_chain(
                block_shape,
                ckl::BinaryFpu<
                    ckl::BinaryFpuOp::Mul,
                    ckl::input(
                        dfb_xmm_id,
                        ckl::WaitPolicy::Upfront,
                        ckl::PopPolicy::None,
                        ckl::InputTileMapping::Block,
                        ckl::DataFormatReconfig::Enabled,
                        ckl::TileAddressing::Offset),
                    ckl::input(dfb_ex2pe_id, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None)>{
                    block.start(), 0u},
                // Activation must be applied last. If do_gamma != 0 or do_beta != 0 then
                // activation will be applied after the gamma/beta multiplication/addition.
                // Otherwise, we can apply the activation here.
                ckl::Optional<activate_after_normalize, FusedActivation>{},
                // pack either to intermediate (dfb_fusion or out0)
                // if no gamma/beta are provided, this will be passed on to the writer
                ckl::PackTile<ckl::output(
                    dfb_im_or_out_id, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>{});

#if defined RMSNORM and not defined FUSE_PRE_ADD
            if constexpr (do_gamma || do_beta) {
                reconfig_data_format_srca(dfb_xmm_id, dfb_fusion_id);
            }
#endif
            if constexpr (do_gamma) {
                constexpr uint32_t dfb_outg_id = do_beta ? dfb_fusion_id : dfb_out_id;
                if constexpr (!do_beta) {
                    pack_reconfig_data_format(dfb_out_id);
                }
                reconfig_data_format_srcb(dfb_ex2pe_id, dfb_gamma_id);
                ckl::eltwise_chain(
                    block_shape,
                    ckl::BinaryFpu<
                        ckl::BinaryFpuOp::Mul,
                        ckl::input(
                            dfb_fusion_id,
                            ckl::WaitPolicy::PerBlockSize,
                            ckl::PopPolicy::PerBlockSize,
                            ckl::InputTileMapping::Block,
                            ckl::DataFormatReconfig::Disabled),
                        ckl::input(
                            dfb_gamma_id,
                            ckl::BroadcastDim::Row,
                            ckl::WaitPolicy::Upfront,
                            ckl::PopPolicy::None,
                            ckl::InputTileMapping::Block,
                            ckl::DataFormatReconfig::Disabled,
                            ckl::TileAddressing::Offset)>{0u, block.start()},
                    // Activation must be applied last. If do_beta != 0 then
                    // activation will be applied after the beta addition.
                    // Otherwise, we can apply the activation here.
                    ckl::Optional<activate_after_gamma, FusedActivation>{},
                    // pack either to intermediate (dfb_fusion or out0)
                    ckl::PackTile<ckl::output(
                        dfb_outg_id,
                        ckl::ReservePolicy::PerBlockSize,
                        ckl::PushPolicy::PerBlockSize,
                        ckl::DataFormatReconfig::Disabled)>{});
            }
            if constexpr (do_beta) {
                pack_reconfig_data_format(dfb_out_id);
                if constexpr (do_gamma) {
                    reconfig_data_format_srcb(dfb_gamma_id, dfb_beta_id);
                } else {
                    reconfig_data_format_srcb(dfb_ex2pe_id, dfb_beta_id);
                }
                ckl::eltwise_chain(
                    block_shape,
                    ckl::BinaryFpu<
                        ckl::BinaryFpuOp::Add,
                        ckl::input(
                            dfb_fusion_id,
                            ckl::WaitPolicy::PerBlockSize,
                            ckl::PopPolicy::PerBlockSize,
                            ckl::InputTileMapping::Block,
                            ckl::DataFormatReconfig::Disabled),
                        ckl::input(
                            dfb_beta_id,
                            ckl::BroadcastDim::Row,
                            ckl::WaitPolicy::Upfront,
                            ckl::PopPolicy::None,
                            ckl::InputTileMapping::Block,
                            ckl::DataFormatReconfig::Disabled,
                            ckl::TileAddressing::Offset)>{0u, block.start()},
                    ckl::Optional<fused_activation_enabled, FusedActivation>{},
                    ckl::PackTile<ckl::output(
                        dfb_out_id,
                        ckl::ReservePolicy::PerBlockSize,
                        ckl::PushPolicy::PerBlockSize,
                        ckl::DataFormatReconfig::Disabled)>{});
            }
        }
        dfb_ex2pe_obj.pop_front(1);
        dfb_xmm_obj.pop_front(total_buffer_size);

#ifdef UNTILIZE_OUT
        constexpr auto dfb_out_rm_id = dfb::out_rm;
        DataflowBuffer dfb_out_rm_obj(dfb_out_rm_id);
        untilize_all_blocks_from_cb<block_size>(dfb_out_obj, dfb_out_rm_obj, Wt);
#endif
    }  // NCHt loop
    // The reduce scaler is generated once by the reader and reused (waited inside row_wise_mean)
    // across every NCHt iteration but never popped. Pop the producer's tile count once here to
    // balance the CB. The reader pushes a second scaler tile only when the last column tile is
    // partial (W not a multiple of tile_width), matching row_wise_mean's wait count.
    //
    // The reader generates the scalers using tt::constants::TILE_WIDTH; this kernel must use the
    // same width for the count to match, so derive both from the shared helper. (tile_width is the
    // tensor's tile width, which equals TILE_WIDTH for every supported layernorm config — see the
    // partial-column handling in row_wise_mean above.)
    static_assert(
        tile_width == tt::constants::TILE_WIDTH,
        "layernorm reader generates reduce scalers using TILE_WIDTH; compute must use the same tile "
        "width or cb_scaler push/pop counts diverge (issue #48487)");
    constexpr uint32_t num_scaler_tiles = norm::layernorm::reduce_scaler_tile_count(W, tile_width);
    dfb_scaler_obj.pop_front(num_scaler_tiles);
}
