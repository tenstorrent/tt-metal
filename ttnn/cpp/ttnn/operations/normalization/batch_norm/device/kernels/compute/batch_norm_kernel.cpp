// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "experimental/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_math.hpp"

// CB ids are promoted to template parameters because DestReuseMul<cb_den, ...>
// requires cb_den to be a compile-time constant. Runtime params (freq,
// tile_start, weight_has, bias_has) stay runtime.
template <
    uint32_t cb_bcast,
    uint32_t cb_other,
    uint32_t cb_batch_var,
    uint32_t cb_eps,
    uint32_t cb_den,
    uint32_t cb_weight,
    uint32_t cb_bias,
    uint32_t cb_tmp_1,
    uint32_t cb_output_0>
ALWI void batchnorm_bcast_tiles(uint32_t freq, uint32_t tile_start, uint32_t weight_has, uint32_t bias_has) {
    constexpr uint32_t onetile = 1;
    const uint32_t weight_has_value = weight_has;
    const uint32_t bias_has_value = bias_has;
    const auto cb_affine_or_out = (weight_has_value || bias_has_value) ? cb_tmp_1 : cb_output_0;
    const auto cb_scaled_output = (bias_has_value) ? cb_tmp_1 : cb_output_0;

    experimental::CircularBuffer cb_bcast_obj(cb_bcast);
    experimental::CircularBuffer cb_den_obj(cb_den);
    experimental::CircularBuffer cb_weight_obj(cb_weight);
    experimental::CircularBuffer cb_bias_obj(cb_bias);

    // Stage A — cb_den = rsqrt(cb_batch_var + cb_eps)
    // cb_batch_var streams (wait/pop one tile); cb_eps is persistent (kernel_main owns wait/pop).
    compute_kernel_lib::add<
        compute_kernel_lib::BroadcastDim::NONE,
        compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
        compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
        cb_batch_var,
        cb_eps,
        cb_den,
        compute_kernel_lib::BinaryInputBlockShape::single(),
        compute_kernel_lib::sfpu_chain(compute_kernel_lib::Rsqrt<>{}));

    // cb_bcast / cb_den / cb_weight / cb_bias are persistent: wait once before loop, pop once after.
    cb_bcast_obj.wait_front(onetile);
    cb_den_obj.wait_front(onetile);
    if (weight_has_value) {
        cb_weight_obj.wait_front(onetile);
    }
    if (bias_has_value) {
        cb_bias_obj.wait_front(onetile);
    }

    for (uint32_t j = tile_start; j < freq; ++j) {
        // Stage B — cb_affine_or_out = (cb_other - cb_bcast) * cb_den
        // Fuses sub + dest-reuse mul in one DEST window via DestReuseMul chain PostOp.
        // cb_other streams; cb_bcast and cb_den are persistent (NoWaitNoPop).
        compute_kernel_lib::sub<
            compute_kernel_lib::BroadcastDim::NONE,
            compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
            cb_other,
            cb_bcast,
            cb_affine_or_out,
            compute_kernel_lib::BinaryInputBlockShape::single(),
            compute_kernel_lib::sfpu_chain(
                compute_kernel_lib::
                    DestReuseMul<cb_den, compute_kernel_lib::Dst::D0, compute_kernel_lib::LoadPolicy::NoWaitNoPop>{}));

        // Stage C — cb_scaled_output = cb_affine_or_out * cb_weight
        if (weight_has_value) {
            compute_kernel_lib::mul<
                compute_kernel_lib::BroadcastDim::NONE,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
                cb_affine_or_out, cb_weight, cb_scaled_output, compute_kernel_lib::BinaryInputBlockShape::single());
        }

        // Stage D — cb_output_0 = cb_tmp_1 + cb_bias
        if (bias_has_value) {
            compute_kernel_lib::add<
                compute_kernel_lib::BroadcastDim::NONE,
                compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
                compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
                cb_tmp_1, cb_bias, cb_output_0, compute_kernel_lib::BinaryInputBlockShape::single());
        }
    }

    cb_bcast_obj.pop_front(onetile);
    cb_den_obj.pop_front(onetile);
    if (weight_has_value) {
        cb_weight_obj.pop_front(onetile);
    }
    if (bias_has_value) {
        cb_bias_obj.pop_front(onetile);
    }
}

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);
    constexpr uint32_t weight_has_value = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t bias_has_value = get_compile_time_arg_val(1) == 1;

    if (num_tiles == 0) {
        return;
    }

    constexpr auto cb_input = get_compile_time_arg_val(2);       // input
    constexpr auto cb_batch_mean = get_compile_time_arg_val(3);  // batch_mean
    constexpr auto cb_output_0 =
        get_compile_time_arg_val(4);  // output -- > [(input - batch_mean)/(sqrt(batch_var + eps))] * weight
    constexpr auto cb_batch_var = get_compile_time_arg_val(5);  // batch_var
    constexpr auto cb_eps = get_compile_time_arg_val(6);        // eps
    constexpr auto cb_den = get_compile_time_arg_val(7);        // 1/(sqrt(batch_var + eps))
    constexpr auto cb_weight = get_compile_time_arg_val(8);     // weight tensor
    constexpr auto cb_tmp_1 = get_compile_time_arg_val(9);      // (input - batch_mean)/(sqrt(batch_var + eps))
    constexpr auto cb_bias = get_compile_time_arg_val(10);      // bias tensor

    constexpr auto cb_bcast = cb_batch_mean;
    constexpr auto cb_other = cb_input;

    binary_op_init_common(cb_other, cb_bcast, cb_output_0);

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    constexpr uint32_t onetile = 1;
    experimental::CircularBuffer cb_eps_obj(cb_eps);
    cb_eps_obj.wait_front(onetile);

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        batchnorm_bcast_tiles<
            cb_bcast,
            cb_other,
            cb_batch_var,
            cb_eps,
            cb_den,
            cb_weight,
            cb_bias,
            cb_tmp_1,
            cb_output_0>(tile_freq, tile_start, weight_has_value, bias_has_value);
    }
    if (remaining_iterations > 0) {
        batchnorm_bcast_tiles<
            cb_bcast,
            cb_other,
            cb_batch_var,
            cb_eps,
            cb_den,
            cb_weight,
            cb_bias,
            cb_tmp_1,
            cb_output_0>(remaining_iterations, tile_start, weight_has_value, bias_has_value);
    }

    cb_eps_obj.pop_front(onetile);
}
