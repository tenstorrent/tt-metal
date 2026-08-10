// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"     // BinaryFpu, DestReuseBinary, PackTile, eltwise_chain
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"     // Rsqrt
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"  // OptionalChainElement

#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

namespace ckl = compute_kernel_lib;

template <
    bool WeightHas,
    bool BiasHas,
    uint32_t dfb_bcast_id,
    uint32_t dfb_other_id,
    uint32_t dfb_batch_var_id,
    uint32_t dfb_eps_id,
    uint32_t dfb_den_id,
    uint32_t dfb_weight_id,
    uint32_t dfb_bias_id,
    uint32_t dfb_output_0_id>
ALWI void batchnorm_bcast_tiles(uint32_t freq, uint32_t tile_start) {
    ckl::eltwise_chain(
        ckl::EltwiseShape::single(),
        ckl::BinaryFpu<
            ckl::input(dfb_batch_var_id, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd),
            ckl::input(dfb_eps_id, ckl::WaitPolicy::None, ckl::PopPolicy::None),
            ckl::BinaryFpuOp::Add,
            ckl::BroadcastDim::None>{},
        ckl::Rsqrt<>{},
        ckl::PackTile<ckl::output(dfb_den_id)>{});

    const uint32_t inner_count = freq - tile_start;

    constexpr auto sub_op = ckl::BinaryFpu<
        ckl::input(dfb_other_id),
        ckl::input(dfb_bcast_id, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd),
        ckl::BinaryFpuOp::Sub,
        ckl::BroadcastDim::None>{};
    constexpr auto mul_den = ckl::DestReuseBinary<
        input(dfb_den_id, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd),
        ckl::BinaryFpuOp::Mul,
        ckl::DestReuseType::DEST_TO_SRCA>{};
    constexpr auto mul_weight = ckl::OptionalChainElement<
        WeightHas,
        ckl::DestReuseBinary<
            ckl::input(dfb_weight_id, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd),
            ckl::BinaryFpuOp::Mul,
            ckl::DestReuseType::DEST_TO_SRCA>>{};
    constexpr auto add_bias = ckl::OptionalChainElement<
        BiasHas,
        ckl::DestReuseBinary<
            ckl::input(dfb_bias_id, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd),
            ckl::BinaryFpuOp::Add,
            ckl::DestReuseType::DEST_TO_SRCA>>{};
    constexpr auto pack_out = ckl::PackTile<ckl::output(dfb_output_0_id)>{};

    ckl::eltwise_chain(ckl::EltwiseShape::tiles(inner_count), sub_op, mul_den, mul_weight, add_bias, pack_out);
}

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t tile_freq = get_arg(args::tile_freq);
    uint32_t tile_start = get_arg(args::tile_start);
    constexpr bool weight_has_value = get_arg(args::weight_has_value) == 1;
    constexpr bool bias_has_value = get_arg(args::bias_has_value) == 1;

    if (num_tiles == 0) {
        return;
    }

    constexpr auto dfb_input_id = dfb::input;
    constexpr auto dfb_batch_mean_id = dfb::batch_mean;
    constexpr auto dfb_output_0_id = dfb::out;
    constexpr auto dfb_batch_var_id = dfb::batch_var;
    constexpr auto dfb_eps_id = dfb::eps;
    constexpr auto dfb_den_id = dfb::den;
    constexpr auto dfb_weight_id = dfb::weight;
    constexpr auto dfb_bias_id = dfb::bias;

    compute_kernel_hw_startup(dfb_input_id, dfb_batch_mean_id, dfb_output_0_id);

    const uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    const uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    DataflowBuffer(dfb_eps_id).wait_front(1);

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        batchnorm_bcast_tiles<
            weight_has_value,
            bias_has_value,
            dfb_batch_mean_id,
            dfb_input_id,
            dfb_batch_var_id,
            dfb_eps_id,
            dfb_den_id,
            dfb_weight_id,
            dfb_bias_id,
            dfb_output_0_id>(tile_freq, tile_start);
    }
    if (remaining_iterations > 0) {
        batchnorm_bcast_tiles<
            weight_has_value,
            bias_has_value,
            dfb_batch_mean_id,
            dfb_input_id,
            dfb_batch_var_id,
            dfb_eps_id,
            dfb_den_id,
            dfb_weight_id,
            dfb_bias_id,
            dfb_output_0_id>(remaining_iterations, tile_start);
    }

    DataflowBuffer(dfb_eps_id).pop_front(1);
}
