// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"  // Rsqrt
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"  // Typecast
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"  // Optional
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

// The writer-facing output DFB is only bound when the accumulation format is wider than the output
// dtype; on the other path the writer drains the compute output directly, so the same kernel-side
// handle has to name a different DFB. The alias is gated at the preprocessor stage because
// dfb::writer_out simply does not exist on the untypecast build.
#ifdef NEEDS_OUTPUT_TYPECAST
constexpr bool needs_output_typecast = true;
constexpr auto dfb_output_final_binding = dfb::writer_out;
#else
constexpr bool needs_output_typecast = false;
constexpr auto dfb_output_final_binding = dfb::out;
#endif

template <bool WeightHas, bool BiasHas, bool NeedsTypecast, uint32_t TcInFmt, uint32_t TcOutFmt>
ALWI void batchnorm_bcast_tiles(uint32_t freq, uint32_t tile_start) {
    using namespace compute_kernel_lib;

    eltwise_chain(
        IterationShape::one_tile(),
        CopyTile<input(dfb::batch_var, WaitPolicy::Upfront, PopPolicy::AtEnd), Dst::D0>{},
        CopyTile<input(dfb::eps, WaitPolicy::None, PopPolicy::None), Dst::D1>{},
        AddBinary<Dst::D0, Dst::D1, Dst::D0>{},
        Rsqrt<>{},
        PackTile<output(dfb::den)>{});

    const uint32_t inner_count = freq - tile_start;

    eltwise_chain(
        IterationShape::tiles(inner_count),
        CopyTile<input(dfb::input)>{},
        CopyTile<input(dfb::batch_mean, WaitPolicy::Upfront, PopPolicy::AtEnd), Dst::D1>{},
        SubBinary<Dst::D0, Dst::D1, Dst::D0>{},
        CopyTile<input(dfb::den, WaitPolicy::Upfront, PopPolicy::AtEnd), Dst::D1>{},
        MulBinary<Dst::D0, Dst::D1, Dst::D0>{},
        Optional<WeightHas, CopyTile<input(dfb::weight, WaitPolicy::Upfront, PopPolicy::AtEnd), Dst::D1>>{},
        Optional<WeightHas, MulBinary<Dst::D0, Dst::D1, Dst::D0>>{},
        Optional<BiasHas, CopyTile<input(dfb::bias, WaitPolicy::Upfront, PopPolicy::AtEnd), Dst::D1>>{},
        Optional<BiasHas, AddBinary<Dst::D0, Dst::D1, Dst::D0>>{},
        Optional<NeedsTypecast, Typecast<TcInFmt, TcOutFmt, Dst::D0>>{},
        PackTile<output(dfb_output_final_binding)>{});
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

    constexpr uint32_t tc_in_fmt = get_arg(args::tc_in_fmt);
    constexpr uint32_t tc_out_fmt = get_arg(args::tc_out_fmt);

    compute_kernel_hw_startup(dfb::input, dfb::batch_mean, dfb::out);

    DataflowBuffer(dfb::eps).wait_front(1);

    const uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    const uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        batchnorm_bcast_tiles<weight_has_value, bias_has_value, needs_output_typecast, tc_in_fmt, tc_out_fmt>(
            tile_freq, tile_start);
    }
    if (remaining_iterations > 0) {
        batchnorm_bcast_tiles<weight_has_value, bias_has_value, needs_output_typecast, tc_in_fmt, tc_out_fmt>(
            remaining_iterations, tile_start);
    }

    DataflowBuffer(dfb::eps).pop_front(1);
}
