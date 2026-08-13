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

// out = ((input - batch_mean) / sqrt(batch_var + eps)) * optional(weight) + optional(bias).
#ifdef NEEDS_OUTPUT_TYPECAST
constexpr bool needs_output_typecast = true;
#else
constexpr bool needs_output_typecast = false;
#endif

template <
    bool WeightHas,
    bool BiasHas,
    bool NeedsTypecast,
    uint32_t TcInFmt,
    uint32_t TcOutFmt,
    uint32_t InputDfb,
    uint32_t BatchMeanDfb,
    uint32_t BatchVarDfb,
    uint32_t EpsDfb,
    uint32_t DenDfb,
    uint32_t WeightDfb,
    uint32_t BiasDfb,
    uint32_t OutputFinalDfb>
ALWI void batchnorm_bcast_tiles(uint32_t freq, uint32_t tile_start) {
    using namespace compute_kernel_lib;

    eltwise_chain(
        IterationShape::one_tile(),
        CopyTile<input(BatchVarDfb, WaitPolicy::Upfront, PopPolicy::AtEnd), Dst::D0>{},
        CopyTile<input(EpsDfb, WaitPolicy::None, PopPolicy::None), Dst::D1>{},
        AddBinary<Dst::D0, Dst::D1, Dst::D0>{},
        Rsqrt<>{},
        PackTile<output(DenDfb)>{});

    const uint32_t inner_count = freq - tile_start;

    eltwise_chain(
        IterationShape::tiles(inner_count),
        CopyTile<input(InputDfb)>{},
        CopyTile<input(BatchMeanDfb, WaitPolicy::Upfront, PopPolicy::AtEnd), Dst::D1>{},
        SubBinary<Dst::D0, Dst::D1, Dst::D0>{},
        CopyTile<input(DenDfb, WaitPolicy::Upfront, PopPolicy::AtEnd), Dst::D1>{},
        MulBinary<Dst::D0, Dst::D1, Dst::D0>{},
        Optional<WeightHas, CopyTile<input(WeightDfb, WaitPolicy::Upfront, PopPolicy::AtEnd), Dst::D1>>{},
        Optional<WeightHas, MulBinary<Dst::D0, Dst::D1, Dst::D0>>{},
        Optional<BiasHas, CopyTile<input(BiasDfb, WaitPolicy::Upfront, PopPolicy::AtEnd), Dst::D1>>{},
        Optional<BiasHas, AddBinary<Dst::D0, Dst::D1, Dst::D0>>{},
        Optional<NeedsTypecast, Typecast<TcInFmt, TcOutFmt, Dst::D0>>{},
        PackTile<output(OutputFinalDfb)>{});
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

    // writer_out is only bound when the output needs a typecast; on the other path the writer
    // drains out directly. Keep that selection inside kernel_main because the generated dfb::
    // names are unavailable when fusion compiles the declarations above this function.
    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
#ifdef NEEDS_OUTPUT_TYPECAST
        batchnorm_bcast_tiles<
            weight_has_value,
            bias_has_value,
            needs_output_typecast,
            tc_in_fmt,
            tc_out_fmt,
            dfb::input,
            dfb::batch_mean,
            dfb::batch_var,
            dfb::eps,
            dfb::den,
            dfb::weight,
            dfb::bias,
            dfb::writer_out>(tile_freq, tile_start);
#else
        batchnorm_bcast_tiles<
            weight_has_value,
            bias_has_value,
            needs_output_typecast,
            tc_in_fmt,
            tc_out_fmt,
            dfb::input,
            dfb::batch_mean,
            dfb::batch_var,
            dfb::eps,
            dfb::den,
            dfb::weight,
            dfb::bias,
            dfb::out>(tile_freq, tile_start);
#endif
    }
    if (remaining_iterations > 0) {
#ifdef NEEDS_OUTPUT_TYPECAST
        batchnorm_bcast_tiles<
            weight_has_value,
            bias_has_value,
            needs_output_typecast,
            tc_in_fmt,
            tc_out_fmt,
            dfb::input,
            dfb::batch_mean,
            dfb::batch_var,
            dfb::eps,
            dfb::den,
            dfb::weight,
            dfb::bias,
            dfb::writer_out>(remaining_iterations, tile_start);
#else
        batchnorm_bcast_tiles<
            weight_has_value,
            bias_has_value,
            needs_output_typecast,
            tc_in_fmt,
            tc_out_fmt,
            dfb::input,
            dfb::batch_mean,
            dfb::batch_var,
            dfb::eps,
            dfb::den,
            dfb::weight,
            dfb::bias,
            dfb::out>(remaining_iterations, tile_start);
#endif
    }

    DataflowBuffer(dfb::eps).pop_front(1);
}
