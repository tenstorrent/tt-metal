// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"     // BinaryFpu, DestReuseBinary, PackTile, eltwise_chain
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"      // Rsqrt
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"  // OptionalChainElement

#include "api/dataflow/circular_buffer.h"

namespace ckl = compute_kernel_lib;

template <
    bool WeightHas,
    bool BiasHas,
    uint32_t cb_bcast,
    uint32_t cb_other,
    uint32_t cb_batch_var,
    uint32_t cb_eps,
    uint32_t cb_den,
    uint32_t cb_weight,
    uint32_t cb_bias,
    uint32_t cb_output_0>
ALWI void batchnorm_bcast_tiles(uint32_t freq, uint32_t tile_start) {
    ckl::eltwise_chain(
        ckl::EltwiseShape::single(),
        ckl::BinaryFpu<
            ckl::input(cb_batch_var, ckl::InputLifecycle::Bulk),
            ckl::input(cb_eps, ckl::InputLifecycle::CallerManaged),
            ckl::BinaryFpuOp::Add,
            ckl::BroadcastDim::None>{},
        ckl::Rsqrt<>{},
        ckl::PackTile<ckl::output(cb_den)>{});

    const uint32_t inner_count = freq - tile_start;

    constexpr auto sub_op = ckl::BinaryFpu<
        ckl::input(cb_other),
        ckl::input(cb_bcast, ckl::InputLifecycle::Bulk),
        ckl::BinaryFpuOp::Sub,
        ckl::BroadcastDim::None>{};
    constexpr auto mul_den = ckl::DestReuseBinary<
        input(cb_den, ckl::InputLifecycle::Bulk),
        ckl::BinaryFpuOp::Mul,
        ckl::DestReuseType::DEST_TO_SRCA>{};
    constexpr auto mul_weight = ckl::OptionalChainElement<
        WeightHas,
        ckl::DestReuseBinary<
            ckl::input(cb_weight, ckl::InputLifecycle::Bulk),
            ckl::BinaryFpuOp::Mul,
            ckl::DestReuseType::DEST_TO_SRCA>>{};
    constexpr auto add_bias = ckl::OptionalChainElement<
        BiasHas,
        ckl::DestReuseBinary<
            ckl::input(cb_bias, ckl::InputLifecycle::Bulk),
            ckl::BinaryFpuOp::Add,
            ckl::DestReuseType::DEST_TO_SRCA>>{};
    constexpr auto pack_out = ckl::PackTile<ckl::output(cb_output_0)>{};

    ckl::eltwise_chain(ckl::EltwiseShape::tiles(inner_count), sub_op, mul_den, mul_weight, add_bias, pack_out);
}

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);
    constexpr bool weight_has_value = get_compile_time_arg_val(0) == 1;
    constexpr bool bias_has_value = get_compile_time_arg_val(1) == 1;

    if (num_tiles == 0) {
        return;
    }

    constexpr auto cb_input = get_compile_time_arg_val(2);       // input
    constexpr auto cb_batch_mean = get_compile_time_arg_val(3);  // batch_mean
    constexpr auto cb_output_0 = get_compile_time_arg_val(4);
    constexpr auto cb_batch_var = get_compile_time_arg_val(5);
    constexpr auto cb_eps = get_compile_time_arg_val(6);
    constexpr auto cb_den = get_compile_time_arg_val(7);
    constexpr auto cb_weight = get_compile_time_arg_val(8);
    constexpr auto cb_bias = get_compile_time_arg_val(10);

    binary_op_init_common(cb_input, cb_batch_mean, cb_output_0);

    const uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    const uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    cb_wait_front(cb_eps, 1);

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        batchnorm_bcast_tiles<
            weight_has_value,
            bias_has_value,
            cb_batch_mean,
            cb_input,
            cb_batch_var,
            cb_eps,
            cb_den,
            cb_weight,
            cb_bias,
            cb_output_0>(tile_freq, tile_start);
    }
    if (remaining_iterations > 0) {
        batchnorm_bcast_tiles<
            weight_has_value,
            bias_has_value,
            cb_batch_mean,
            cb_input,
            cb_batch_var,
            cb_eps,
            cb_den,
            cb_weight,
            cb_bias,
            cb_output_0>(remaining_iterations, tile_start);
    }

    cb_pop_front(cb_eps, 1);
}
