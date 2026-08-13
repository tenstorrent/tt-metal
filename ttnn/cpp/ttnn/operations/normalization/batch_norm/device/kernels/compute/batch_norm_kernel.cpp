// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"     // BinaryFpu, DestReuseBinary, PackTile, eltwise_chain
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"     // Rsqrt
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"  // Optional

#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

namespace ckl = compute_kernel_lib;

// out = ((input - batch_mean) / sqrt(batch_var + eps)) * optional(weight) + optional(bias).
template <bool WeightHas, bool BiasHas>
ALWI void batchnorm_bcast_tiles(uint32_t freq, uint32_t tile_start) {
    ckl::eltwise_chain(
        ckl::IterationShape::one_tile(),
        ckl::BinaryFpu<
            ckl::BinaryFpuOp::Add,
            ckl::input(dfb::batch_var, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd),
            ckl::input(dfb::eps, ckl::WaitPolicy::None, ckl::PopPolicy::None)>{},
        ckl::Rsqrt<>{},
        ckl::PackTile<ckl::output(dfb::den)>{});

    const uint32_t inner_count = freq - tile_start;

    constexpr auto sub_op = ckl::BinaryFpu<
        ckl::BinaryFpuOp::Sub,
        ckl::input(dfb::input),
        ckl::input(dfb::batch_mean, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd)>{};
    constexpr auto mul_den = ckl::DestReuseBinary<
        ckl::input(dfb::den, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd),
        ckl::BinaryFpuOp::Mul,
        ckl::DestReuseType::DEST_TO_SRCA>{};
    constexpr auto mul_weight = ckl::Optional<
        WeightHas,
        ckl::DestReuseBinary<
            ckl::input(dfb::weight, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd),
            ckl::BinaryFpuOp::Mul,
            ckl::DestReuseType::DEST_TO_SRCA>>{};
    constexpr auto add_bias = ckl::Optional<
        BiasHas,
        ckl::DestReuseBinary<
            ckl::input(dfb::bias, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd),
            ckl::BinaryFpuOp::Add,
            ckl::DestReuseType::DEST_TO_SRCA>>{};
    constexpr auto pack_out = ckl::PackTile<ckl::output(dfb::out)>{};

    ckl::eltwise_chain(ckl::IterationShape::tiles(inner_count), sub_op, mul_den, mul_weight, add_bias, pack_out);
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

    compute_kernel_hw_startup(dfb::input, dfb::batch_mean, dfb::out);

    const uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    const uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    DataflowBuffer(dfb::eps).wait_front(1);

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        batchnorm_bcast_tiles<weight_has_value, bias_has_value>(tile_freq, tile_start);
    }
    if (remaining_iterations > 0) {
        batchnorm_bcast_tiles<weight_has_value, bias_has_value>(remaining_iterations, tile_start);
    }

    DataflowBuffer(dfb::eps).pop_front(1);
}
