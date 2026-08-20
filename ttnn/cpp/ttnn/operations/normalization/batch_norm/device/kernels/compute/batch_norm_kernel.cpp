// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"      // BinaryFpu, DestReuseBinary, PackTile, eltwise_chain
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"     // Rsqrt
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"  // Optional

#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t tile_freq = get_arg(args::tile_freq);
    uint32_t tile_start = get_arg(args::tile_start);
    constexpr bool weight_has_value = get_arg(args::weight_has_value) == 1;
    constexpr bool bias_has_value = get_arg(args::bias_has_value) == 1;

    if (num_tiles == 0) {
        return;
    }

    compute_kernel_hw_startup(dfb::input, dfb::batch_mean, dfb::out);

    const uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    const uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    DataflowBuffer dfb_eps_obj(dfb::eps);  // one tile of eps, filled by the reader
    dfb_eps_obj.wait_front(1);

    // out = ((input - batch_mean) / sqrt(batch_var + eps)) * optional(weight) + optional(bias).
    const auto batchnorm_bcast_tiles = [](uint32_t freq, uint32_t tile_start) __attribute__((always_inline)) {
        // 1/(sqrt(batch_var + eps))
        ckl::eltwise_chain(
            ckl::IterationShape::one_tile(),
            ckl::BinaryFpu<
                ckl::BinaryFpuOp::Add,
                ckl::input(dfb::batch_var, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd),
                ckl::input(dfb::eps, ckl::WaitPolicy::None, ckl::PopPolicy::None)>{},
            ckl::Rsqrt<>{},
            ckl::PackTile<ckl::output(dfb::den)>{});

        const uint32_t inner_count = freq - tile_start;

        // The batch mean is the broadcast operand of the subtraction; the input tiles are the other one.
        constexpr auto sub_op = ckl::BinaryFpu<
            ckl::BinaryFpuOp::Sub,
            ckl::input(dfb::input),
            // batch_mean, broadcast against the input
            ckl::input(dfb::batch_mean, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd)>{};
        // (input - batch_mean)/(sqrt(batch_var + eps)) = result
        constexpr auto mul_den = ckl::DestReuseBinary<
            ckl::BinaryFpuOp::Mul,
            ckl::input(dfb::den, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd),
            ckl::DestReuseType::DEST_TO_SRCA>{};
        // result = result * weight
        constexpr auto mul_weight = ckl::Optional<
            weight_has_value,
            ckl::DestReuseBinary<
                ckl::BinaryFpuOp::Mul,
                // weight tensor
                ckl::input(dfb::weight, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd),
                ckl::DestReuseType::DEST_TO_SRCA>>{};
        // result = result + bias
        constexpr auto add_bias = ckl::Optional<
            bias_has_value,
            ckl::DestReuseBinary<
                ckl::BinaryFpuOp::Add,
                ckl::input(dfb::bias, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd),
                ckl::DestReuseType::DEST_TO_SRCA>>{};
        constexpr auto pack_out = ckl::PackTile<ckl::output(dfb::out)>{};

        ckl::eltwise_chain(ckl::IterationShape::tiles(inner_count), sub_op, mul_den, mul_weight, add_bias, pack_out);
    };

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        batchnorm_bcast_tiles(tile_freq, tile_start);
    }
    if (remaining_iterations > 0) {
        batchnorm_bcast_tiles(remaining_iterations, tile_start);
    }

    dfb_eps_obj.pop_front(1);
}
