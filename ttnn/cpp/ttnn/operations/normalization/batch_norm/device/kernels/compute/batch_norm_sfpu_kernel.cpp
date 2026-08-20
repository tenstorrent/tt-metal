// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"  // Rsqrt
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"  // Typecast
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"  // Optional
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t tile_freq = get_arg(args::tile_freq);
    uint32_t tile_start = get_arg(args::tile_start);
    constexpr bool weight_has_value = get_arg(args::weight_has_value) == 1;
    constexpr bool bias_has_value = get_arg(args::bias_has_value) == 1;
    constexpr bool needs_output_typecast = get_arg(args::needs_output_typecast) == 1;

    if (num_tiles == 0) {
        return;
    }

    constexpr uint32_t tc_in_fmt = get_arg(args::tc_in_fmt);
    constexpr uint32_t tc_out_fmt = get_arg(args::tc_out_fmt);
    // The batch mean is the broadcast operand of the subtraction; the input tiles are the other one.
    compute_kernel_hw_startup(dfb::input, dfb::batch_mean, dfb::out);

    DataflowBuffer dfb_eps_obj(dfb::eps);  // one tile of eps, filled by the reader
    dfb_eps_obj.wait_front(1);

    const uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    const uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    // out = ((input - batch_mean) / sqrt(batch_var + eps)) * optional(weight) + optional(bias).
    // batchnorm_bcast_tiles: For each output tile in [tile_start, freq), computes batch-norm on tiles from dfb::input
    // (input) broadcast against dfb::batch_mean (batch mean). First builds 1/sqrt(batch_var + eps) in dfb::den, then
    // per tile: (input - mean) * den, optional multiply by weight, optional add bias. When needs_output_typecast,
    // SFPU typecasts to the writer-facing dfb::writer_out.
    const auto batchnorm_bcast_tiles = [](uint32_t freq, uint32_t tile_start) __attribute__((always_inline)) {
        using namespace compute_kernel_lib;

        // 1/(sqrt(batch_var + eps)) = dfb::den
        eltwise_chain(
            IterationShape::one_tile(),
            CopyTile<input(dfb::batch_var, WaitPolicy::Upfront, PopPolicy::AtEnd), Dst::D0>{},
            CopyTile<input(dfb::eps, WaitPolicy::None, PopPolicy::None), Dst::D1>{},
            AddBinary<Dst::D0, Dst::D1, Dst::D0>{},
            Rsqrt<>{},
            PackTile<output(dfb::den)>{});

        const uint32_t inner_count = freq - tile_start;

        // The output binding must be selected by the preprocessor: dfb::writer_out is not generated for
        // non-typecast builds, so even an unselected if-constexpr or ternary branch would fail to compile.
        // Keep this condition in sync with needs_output_typecast above; the writer drains out otherwise.
#ifdef NEEDS_OUTPUT_TYPECAST
        constexpr auto output_final = output(dfb::writer_out);
#else
        constexpr auto output_final = output(dfb::out);
#endif

        eltwise_chain(
            IterationShape::tiles(inner_count),
            // (input - batch_mean) * den
            CopyTile<input(dfb::input)>{},
            // batch_mean, broadcast against the input
            CopyTile<input(dfb::batch_mean, WaitPolicy::Upfront, PopPolicy::AtEnd), Dst::D1>{},
            SubBinary<Dst::D0, Dst::D1, Dst::D0>{},
            // 1/(sqrt(batch_var + eps))
            CopyTile<input(dfb::den, WaitPolicy::Upfront, PopPolicy::AtEnd), Dst::D1>{},
            // (input - batch_mean)/(sqrt(batch_var + eps)) = result
            MulBinary<Dst::D0, Dst::D1, Dst::D0>{},
            // weight tensor
            Optional<weight_has_value, CopyTile<input(dfb::weight, WaitPolicy::Upfront, PopPolicy::AtEnd), Dst::D1>>{},
            // result = result * weight
            Optional<weight_has_value, MulBinary<Dst::D0, Dst::D1, Dst::D0>>{},
            // result = result + bias
            Optional<bias_has_value, CopyTile<input(dfb::bias, WaitPolicy::Upfront, PopPolicy::AtEnd), Dst::D1>>{},
            Optional<bias_has_value, AddBinary<Dst::D0, Dst::D1, Dst::D0>>{},
            Optional<needs_output_typecast, Typecast<tc_in_fmt, tc_out_fmt, Dst::D0>>{},
            PackTile<output_final>{});
    };

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        batchnorm_bcast_tiles(tile_freq, tile_start);
    }
    if (remaining_iterations > 0) {
        batchnorm_bcast_tiles(remaining_iterations, tile_start);
    }

    dfb_eps_obj.pop_front(1);
}
