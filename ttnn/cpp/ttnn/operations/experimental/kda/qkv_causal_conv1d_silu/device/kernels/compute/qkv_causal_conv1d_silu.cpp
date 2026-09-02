// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/tilize.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/activations.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

namespace ckl = compute_kernel_lib;

template <uint32_t block_ct, uint32_t num_blocks>
TT_KERNEL void compute(uint32_t wi_count) {
    // Kimi-K3 uses a fixed four-tap causal convolution, with three preceding rows supplied by history.
    constexpr uint32_t tap_count = 4;
    compute_kernel_hw_startup(dfb::act_rm, dfb::act_tile, dfb::output);
    DataflowBuffer activation(dfb::act_tile);
    DataflowBuffer weights(dfb::weights);

    if constexpr (num_blocks == 1) {
        weights.wait_front(tap_count * block_ct);
    }
    for (uint32_t item = 0; item < wi_count; ++item) {
        if constexpr (num_blocks > 1) {
            weights.wait_front(tap_count * block_ct);
        }
        for (uint32_t tap = 0; tap < tap_count; ++tap) {
            compute_kernel_lib::tilize<block_ct, dfb::act_rm, dfb::act_tile>(1);
            activation.wait_front(block_ct);

            if (tap == 0) {
                // First tap: activation * weight -> partial.
                ckl::eltwise_chain(
                    ckl::IterationShape::tiles(block_ct),
                    ckl::BinaryFpu<
                        ckl::BinaryFpuOp::Mul,
                        ckl::input(
                            dfb::act_tile, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::InputTileMapping::Block),
                        ckl::input(
                            dfb::weights,
                            ckl::BroadcastDim::Row,
                            ckl::WaitPolicy::None,
                            ckl::PopPolicy::None,
                            ckl::InputTileMapping::Block,
                            ckl::DataFormatReconfig::Enabled,
                            ckl::TileAddressing::Offset)>{0, 0},
                    ckl::PackTile<ckl::output(dfb::partial, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile)>{});
            } else if (tap + 1 == tap_count) {
                // Final tap: activation * weight + partial, then SiLU -> output.
                ckl::eltwise_chain(
                    ckl::IterationShape::tiles(block_ct),
                    ckl::BinaryFpu<
                        ckl::BinaryFpuOp::Mul,
                        ckl::input(
                            dfb::act_tile, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::InputTileMapping::Block),
                        ckl::input(
                            dfb::weights,
                            ckl::BroadcastDim::Row,
                            ckl::WaitPolicy::None,
                            ckl::PopPolicy::None,
                            ckl::InputTileMapping::Block,
                            ckl::DataFormatReconfig::Enabled,
                            ckl::TileAddressing::Offset)>{0, tap * block_ct},
                    ckl::DestReuseBinary<
                        ckl::BinaryFpuOp::Add,
                        ckl::input(
                            dfb::partial,
                            ckl::WaitPolicy::Upfront,
                            ckl::PopPolicy::PerTile,
                            ckl::InputTileMapping::Scalar),
                        ckl::DestReuseType::DEST_TO_SRCB>{},
                    ckl::Silu<>{},
                    ckl::PackTile<ckl::output(dfb::output, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile)>{});
            } else {
                // Middle taps: activation * weight + partial -> partial.
                ckl::eltwise_chain(
                    ckl::IterationShape::tiles(block_ct),
                    ckl::BinaryFpu<
                        ckl::BinaryFpuOp::Mul,
                        ckl::input(
                            dfb::act_tile, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::InputTileMapping::Block),
                        ckl::input(
                            dfb::weights,
                            ckl::BroadcastDim::Row,
                            ckl::WaitPolicy::None,
                            ckl::PopPolicy::None,
                            ckl::InputTileMapping::Block,
                            ckl::DataFormatReconfig::Enabled,
                            ckl::TileAddressing::Offset)>{0, tap * block_ct},
                    ckl::DestReuseBinary<
                        ckl::BinaryFpuOp::Add,
                        ckl::input(
                            dfb::partial,
                            ckl::WaitPolicy::Upfront,
                            ckl::PopPolicy::PerTile,
                            ckl::InputTileMapping::Scalar),
                        ckl::DestReuseType::DEST_TO_SRCB>{},
                    ckl::PackTile<ckl::output(dfb::partial, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile)>{});
            }
            activation.pop_front(block_ct);
        }
        if constexpr (num_blocks > 1) {
            weights.pop_front(tap_count * block_ct);
        }
    }
}
