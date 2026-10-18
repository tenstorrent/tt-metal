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
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t tile_freq = get_arg(args::tile_freq);
    uint32_t tile_start = get_arg(args::tile_start);
    constexpr bool weight_has_value = get_arg(args::weight_has_value) == 1;
    constexpr bool bias_has_value = get_arg(args::bias_has_value) == 1;
#ifdef NEEDS_OUTPUT_TYPECAST
    constexpr bool needs_output_typecast = true;
#else
    constexpr bool needs_output_typecast = false;
#endif

    if (num_tiles == 0) {
        return;
    }

    constexpr uint32_t tc_in_fmt = get_arg(args::tc_in_fmt);
    constexpr uint32_t tc_out_fmt = get_arg(args::tc_out_fmt);
    // The batch mean is the broadcast operand of the subtraction; the input tiles are the other one.
    compute_kernel_hw_startup(dfb::input, dfb::batch_mean, dfb::out);

    DataflowBuffer dfb_eps_obj(dfb::eps);  // one tile of eps, filled by the reader
    dfb_eps_obj.wait_front(1);
    DataflowBuffer dfb_batch_var_obj(dfb::batch_var);
    DataflowBuffer dfb_den_obj(dfb::den);
    DataflowBuffer dfb_input_obj(dfb::input);
    DataflowBuffer dfb_batch_mean_obj(dfb::batch_mean);
    DataflowBuffer dfb_weight_obj(dfb::weight);
    DataflowBuffer dfb_bias_obj(dfb::bias);
#ifdef NEEDS_OUTPUT_TYPECAST
    DataflowBuffer dfb_output_final_obj(dfb::writer_out);
#else
    DataflowBuffer dfb_output_final_obj(dfb::out);
#endif

    const uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    const uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    // out = ((input - batch_mean) / sqrt(batch_var + eps)) * optional(weight) + optional(bias).
    const auto batchnorm_bcast_tiles = [&](uint32_t freq, uint32_t tile_start) __attribute__((always_inline)) {
        using namespace compute_kernel_lib;

        eltwise_chain(
            IterationShape::one_tile(),
            CopyTile<input(dfb::batch_var, WaitPolicy::Upfront, PopPolicy::AtEnd), Dst::D0>{dfb_batch_var_obj},
            CopyTile<input(dfb::eps, WaitPolicy::None, PopPolicy::None), Dst::D1>{dfb_eps_obj},
            AddBinary<Dst::D0, Dst::D1, Dst::D0>{},
            Rsqrt<>{},
            PackTile<output(dfb::den)>{dfb_den_obj});

        const uint32_t inner_count = freq - tile_start;

// The writer-facing output DFB is only bound when the accumulation format is wider than the output
// dtype; on the other path the writer drains the compute output directly, so the same kernel-side
// handle has to name a different DFB. The alias is gated at the preprocessor stage because
// dfb::writer_out simply does not exist on the untypecast build.
#ifdef NEEDS_OUTPUT_TYPECAST
        constexpr auto output_final = output(dfb::writer_out);
#else
        constexpr auto output_final = output(dfb::out);
#endif

        eltwise_chain(
            IterationShape::tiles(inner_count),
            CopyTile<input(dfb::input)>{dfb_input_obj},
            CopyTile<input(dfb::batch_mean, WaitPolicy::Upfront, PopPolicy::AtEnd), Dst::D1>{dfb_batch_mean_obj},
            SubBinary<Dst::D0, Dst::D1, Dst::D0>{},
            CopyTile<input(dfb::den, WaitPolicy::Upfront, PopPolicy::AtEnd), Dst::D1>{dfb_den_obj},
            MulBinary<Dst::D0, Dst::D1, Dst::D0>{},
            Optional<weight_has_value, CopyTile<input(dfb::weight, WaitPolicy::Upfront, PopPolicy::AtEnd), Dst::D1>>{
                dfb_weight_obj},
            Optional<weight_has_value, MulBinary<Dst::D0, Dst::D1, Dst::D0>>{},
            Optional<bias_has_value, CopyTile<input(dfb::bias, WaitPolicy::Upfront, PopPolicy::AtEnd), Dst::D1>>{
                dfb_bias_obj},
            Optional<bias_has_value, AddBinary<Dst::D0, Dst::D1, Dst::D0>>{},
            Optional<needs_output_typecast, Typecast<tc_in_fmt, tc_out_fmt, Dst::D0>>{},
            PackTile<output_final>{dfb_output_final_obj});
    };

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        batchnorm_bcast_tiles(tile_freq, tile_start);
    }
    if (remaining_iterations > 0) {
        batchnorm_bcast_tiles(remaining_iterations, tile_start);
    }

    dfb_eps_obj.pop_front(1);
}
