// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"  // Rsqrt
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"  // Typecast
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu_basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp"  // OptionalChainElement
#include "api/dataflow/circular_buffer.h"

template <
    bool WeightHas,
    bool BiasHas,
    bool NeedsTypecast,
    uint32_t TcInFmt,
    uint32_t TcOutFmt,
    uint32_t cb_bcast,
    uint32_t cb_other,
    uint32_t cb_batch_var,
    uint32_t cb_eps,
    uint32_t cb_den,
    uint32_t cb_weight,
    uint32_t cb_bias,
    uint32_t cb_output_0,
    uint32_t cb_output_final>
ALWI void batchnorm_bcast_tiles(uint32_t freq, uint32_t tile_start) {
    using namespace compute_kernel_lib;

    eltwise_chain(
        EltwiseShape::single(),
        CopyTile<input(cb_batch_var, InputLifecycle::Bulk), Dst::D0>{},
        CopyTile<input(cb_eps, InputLifecycle::CallerManaged), Dst::D1>{},
        AddBinary<Dst::D0, Dst::D1, Dst::D0>{},
        Rsqrt<>{},
        PackTile<output(cb_den)>{});

    const uint32_t inner_count = freq - tile_start;

    constexpr uint32_t cb_final_out = NeedsTypecast ? cb_output_final : cb_output_0;

    eltwise_chain(
        EltwiseShape::tiles(inner_count),
        CopyTile<input(cb_other)>{},
        CopyTile<input(cb_bcast, InputLifecycle::Bulk), Dst::D1>{},
        SubBinary<Dst::D0, Dst::D1, Dst::D0>{},
        CopyTile<input(cb_den, InputLifecycle::Bulk), Dst::D1>{},
        MulBinary<Dst::D0, Dst::D1, Dst::D0>{},
        OptionalChainElement<WeightHas, CopyTile<input(cb_weight, InputLifecycle::Bulk), Dst::D1>>{},
        OptionalChainElement<WeightHas, MulBinary<Dst::D0, Dst::D1, Dst::D0>>{},
        OptionalChainElement<BiasHas, CopyTile<input(cb_bias, InputLifecycle::Bulk), Dst::D1>>{},
        OptionalChainElement<BiasHas, AddBinary<Dst::D0, Dst::D1, Dst::D0>>{},
        OptionalChainElement<NeedsTypecast, Typecast<TcInFmt, TcOutFmt, Dst::D0>>{},
        PackTile<output(cb_final_out)>{});
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
    constexpr auto cb_output_final = get_compile_time_arg_val(11);
    constexpr bool needs_output_typecast = get_compile_time_arg_val(12) == 1;
    constexpr uint32_t tc_in_fmt = get_compile_time_arg_val(13);
    constexpr uint32_t tc_out_fmt = get_compile_time_arg_val(14);

    compute_kernel_hw_startup(cb_input, cb_batch_mean, cb_output_0);

    cb_wait_front(cb_eps, 1);

    const uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    const uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        batchnorm_bcast_tiles<
            weight_has_value,
            bias_has_value,
            needs_output_typecast,
            tc_in_fmt,
            tc_out_fmt,
            cb_batch_mean,
            cb_input,
            cb_batch_var,
            cb_eps,
            cb_den,
            cb_weight,
            cb_bias,
            cb_output_0,
            cb_output_final>(tile_freq, tile_start);
    }
    if (remaining_iterations > 0) {
        batchnorm_bcast_tiles<
            weight_has_value,
            bias_has_value,
            needs_output_typecast,
            tc_in_fmt,
            tc_out_fmt,
            cb_batch_mean,
            cb_input,
            cb_batch_var,
            cb_eps,
            cb_den,
            cb_weight,
            cb_bias,
            cb_output_0,
            cb_output_final>(remaining_iterations, tile_start);
    }

    cb_pop_front(cb_eps, 1);
}
