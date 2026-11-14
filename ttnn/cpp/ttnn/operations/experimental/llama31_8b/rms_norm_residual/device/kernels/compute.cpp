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
#include "api/tile_constants.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/normalization/kernel_util/compute/numeric.h"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "ttnn/operations/normalization/kernel_util/generic/bit.h"
#include "ttnn/operations/normalization/layernorm/device/kernels/layernorm_scaler_tiles.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/dataflow_buffer.h"

#include "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_compute_utils.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"

namespace ckl = compute_kernel_lib;

namespace generic = norm::kernel_util::generic;
namespace kutil = norm::kernel_util;
namespace numeric = kutil::compute::numeric;
namespace policies = kutil::compute::policies;

struct FusedActivation : ckl::UnaryOp<FusedActivation, ckl::Dst::D0> {
    static ALWI void init() {}

    static ALWI void exec_impl([[maybe_unused]] uint32_t i) {}
};

constexpr bool fused_activation_enabled = false;

void kernel_main() {
    const uint32_t NCHt = get_arg(args::NCHt);
    constexpr auto Wt = get_arg(args::Wt);
    constexpr auto block_size = get_arg(args::block_size);
    constexpr auto do_gamma = get_arg(args::do_gamma);
    constexpr auto do_beta = get_arg(args::do_beta);
    constexpr bool activate_after_normalize = fused_activation_enabled && do_gamma == 0 && do_beta == 0;
    constexpr bool activate_after_gamma = fused_activation_enabled && do_beta == 0;
    constexpr bool FLOAT32_DTYPE = get_arg(args::fp32_dest_acc_en) == 1;
    constexpr bool FLOAT32_REDUCTION = get_arg(args::float32_reduction) == 1;
    constexpr auto W = get_arg(args::W);
    constexpr auto tile_width = get_arg(args::tile_width);

    constexpr auto dfb_scaler_id = dfb::scaler;
    constexpr auto dfb_eps_id = dfb::eps;
    constexpr auto dfb_in_id = dfb::in;
    constexpr auto dfb_inb_id = dfb::inb;
    constexpr auto dfb_out_id = dfb::out;
    constexpr auto dfb_gamma_id = dfb::gamma;
    constexpr auto dfb_xmm_id = dfb::xmm;
    constexpr auto dfb_ex2_id = dfb::ex2;
    constexpr auto dfb_xmm2_id = dfb::xmm2;
    constexpr auto dfb_ex2pe_id = dfb::ex2pe;
    constexpr auto dfb_fusion_id = dfb::fusion;
    DataflowBuffer dfb_eps(dfb_eps_id);
    DataflowBuffer dfb_in(dfb_in_id);
    DataflowBuffer dfb_xmm(dfb_xmm_id);
    DataflowBuffer dfb_out(dfb_out_id);
    DataflowBuffer dfb_ex2pe(dfb_ex2pe_id);
    DataflowBuffer dfb_scaler(dfb_scaler_id);
    DataflowBuffer dfb_xmm2(dfb_xmm2_id);
    DataflowBuffer dfb_ex2(dfb_ex2_id);
    DataflowBuffer dfb_inb(dfb_inb_id);
    DataflowBuffer dfb_gamma(dfb_gamma_id);
    DataflowBuffer dfb_fusion(dfb_fusion_id);

    constexpr int onetile = 1;
    constexpr int dst0 = 0;
    constexpr int dst1 = 1;
    constexpr auto scaler0 = 0;

    constexpr auto dfb_x_id = dfb_xmm_id;

    auto& dfb_x = dfb_xmm;

    compute_kernel_hw_startup(dfb_in_id, dfb_inb_id, dfb_x_id);

    dfb_eps.wait_front(1);

    constexpr auto dfb_im_or_out_id = dfb_fusion_id;
    auto& dfb_im_or_out = dfb_fusion;

    const auto total_buffer_size = generic::blocks(Wt, block_size).total_with_remainder();
    constexpr auto row_shape = ckl::IterationShape::tiles(Wt).block_size(block_size, ckl::BlockTailSync::FullBlock);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        {
            DataflowBuffer dfb_hout(dfb::hout);
            reconfig_data_format(dfb_in_id, dfb_inb_id);
            pack_reconfig_data_format(dfb_x_id);
            add_init(dfb_in_id, dfb_inb_id);
            for (uint32_t b = 0; b < Wt; b += block_size) {
                const uint32_t n = (b + block_size <= Wt) ? block_size : (Wt - b);
                dfb_in.wait_front(block_size);
                dfb_inb.wait_front(block_size);
                dfb_x.reserve_back(block_size);
                dfb_hout.reserve_back(block_size);
                tile_regs_acquire();
                for (uint32_t i = 0; i < n; ++i) {
                    add_tiles(dfb_in_id, dfb_inb_id, i, i, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < n; ++i) {
                    pack_tile(i, dfb_x_id);
                }
                for (uint32_t i = 0; i < n; ++i) {
                    pack_tile(i, dfb::hout);
                }
                tile_regs_release();
                dfb_x.push_back(block_size);
                dfb_hout.push_back(block_size);
                dfb_in.pop_front(block_size);
                dfb_inb.pop_front(block_size);
            }
            pack_reconfig_data_format(dfb_xmm2_id);
        }
        reconfig_data_format(dfb_in_id, dfb_x_id, dfb_inb_id, dfb_x_id);

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
                ckl::DataFormatReconfig::Disabled)>(row_shape, dfb_xmm, dfb_xmm2);
        reconfig_data_format(dfb_xmm_id, dfb_xmm2_id, dfb_xmm_id, dfb_scaler_id);

        numeric::
            row_wise_mean<PoolType::SUM, ReduceDim::REDUCE_ROW, FLOAT32_REDUCTION, policies::FullBlockWithPopPolicy>(
                dfb_xmm2, dfb_scaler, dfb_ex2, W, Wt, block_size, tile_width);

        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Add,
                ckl::input(dfb_ex2_id),
                ckl::input(dfb_eps_id, ckl::WaitPolicy::None, ckl::PopPolicy::None)>{dfb_ex2, dfb_eps},
            ckl::Rsqrt<ckl::Approx::Exact, ckl::Dst::D0>{},
            ckl::PackTile<ckl::output(dfb_ex2pe_id)>{dfb_ex2pe});

        constexpr uint32_t norm_tiles = block_size * 2;
        for (auto block : generic::blocks(Wt, norm_tiles)) {
            const auto block_shape =
                ckl::IterationShape::tiles(block.size()).block_size(block_size, ckl::BlockTailSync::FullBlock);
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
                    dfb_xmm, dfb_ex2pe, block.start(), 0u},
                ckl::Optional<activate_after_normalize, FusedActivation>{},
                ckl::PackTile<ckl::output(
                    dfb_im_or_out_id, ckl::ReservePolicy::PerBlockSize, ckl::PushPolicy::PerBlockSize)>{dfb_im_or_out});

            reconfig_data_format_srca(dfb_xmm_id, dfb_fusion_id);
            constexpr auto dfb_outg_id = do_beta ? dfb_fusion_id : dfb_out_id;
            auto& dfb_outg = do_beta ? dfb_fusion : dfb_out;
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
                        ckl::TileAddressing::Offset)>{dfb_fusion, dfb_gamma, 0u, block.start()},
                ckl::Optional<activate_after_gamma, FusedActivation>{},
                ckl::PackTile<ckl::output(
                    dfb_outg_id,
                    ckl::ReservePolicy::PerBlockSize,
                    ckl::PushPolicy::PerBlockSize,
                    ckl::DataFormatReconfig::Disabled)>{dfb_outg});
        }
        dfb_ex2pe.pop_front(1);
        dfb_xmm.pop_front(static_cast<uint16_t>(total_buffer_size));
    }
    static_assert(
        tile_width == tt::constants::TILE_WIDTH,
        "layernorm reader generates reduce scalers using TILE_WIDTH; compute must use the same tile "
        "width or dfb_scaler push/pop counts diverge (issue #48487)");
    constexpr uint32_t num_scaler_tiles = norm::layernorm::reduce_scaler_tile_count(W, tile_width);
    dfb_scaler.pop_front(num_scaler_tiles);

    dfb_eps.pop_front(1);

    if (NCHt > 0) {
        dfb_gamma.pop_front(total_buffer_size);
    }
}
