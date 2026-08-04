// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/math.hpp"  // Rsqrt
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"  // Typecast
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/binary/sfpu/basic.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/optional.hpp"  // OptionalChainElement
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

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
        CopyTile<input(cb_batch_var, WaitPolicy::Upfront, PopPolicy::AtEnd), Dst::D0>{},
        CopyTile<input(cb_eps, WaitPolicy::None, PopPolicy::None), Dst::D1>{},
        AddBinary<Dst::D0, Dst::D1, Dst::D0>{},
        Rsqrt<>{},
        PackTile<output(cb_den)>{});

    const uint32_t inner_count = freq - tile_start;

    constexpr uint32_t cb_final_out = NeedsTypecast ? cb_output_final : cb_output_0;

    eltwise_chain(
        EltwiseShape::tiles(inner_count),
        CopyTile<input(cb_other)>{},
        CopyTile<input(cb_bcast, WaitPolicy::Upfront, PopPolicy::AtEnd), Dst::D1>{},
        SubBinary<Dst::D0, Dst::D1, Dst::D0>{},
        CopyTile<input(cb_den, WaitPolicy::Upfront, PopPolicy::AtEnd), Dst::D1>{},
        MulBinary<Dst::D0, Dst::D1, Dst::D0>{},
        OptionalChainElement<WeightHas, CopyTile<input(cb_weight, WaitPolicy::Upfront, PopPolicy::AtEnd), Dst::D1>>{},
        OptionalChainElement<WeightHas, MulBinary<Dst::D0, Dst::D1, Dst::D0>>{},
        OptionalChainElement<BiasHas, CopyTile<input(cb_bias, WaitPolicy::Upfront, PopPolicy::AtEnd), Dst::D1>>{},
        OptionalChainElement<BiasHas, AddBinary<Dst::D0, Dst::D1, Dst::D0>>{},
        OptionalChainElement<NeedsTypecast, Typecast<TcInFmt, TcOutFmt, Dst::D0>>{},
        PackTile<output(cb_final_out)>{});
}

// The writer-facing output DFB is only bound when the accumulation format is wider than the output
// dtype; on the other path the writer drains the compute output directly, so the same kernel-side
// handle has to name a different DFB. The alias is gated at the preprocessor stage because
// dfb::writer_out simply does not exist on the untypecast build.
#ifdef NEEDS_OUTPUT_TYPECAST
constexpr bool needs_output_typecast = true;
constexpr auto dfb_output_final = dfb::writer_out;
#else
constexpr bool needs_output_typecast = false;
constexpr auto dfb_output_final = dfb::out;
#endif

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t tile_freq = get_arg(args::tile_freq);
    uint32_t tile_start = get_arg(args::tile_start);
    constexpr bool weight_has_value = get_arg(args::weight_has_value) == 1;
    constexpr bool bias_has_value = get_arg(args::bias_has_value) == 1;

    if (num_tiles == 0) {
        return;
    }

    constexpr auto cb_input = dfb::input;
    constexpr auto cb_batch_mean = dfb::batch_mean;
    constexpr auto cb_output_0 = dfb::out;
    constexpr auto cb_batch_var = dfb::batch_var;
    constexpr auto cb_eps = dfb::eps;
    constexpr auto cb_den = dfb::den;
    constexpr auto cb_weight = dfb::weight;
    constexpr auto cb_bias = dfb::bias;
    constexpr auto cb_output_final = dfb_output_final;
    constexpr uint32_t tc_in_fmt = get_arg(args::tc_in_fmt);
    constexpr uint32_t tc_out_fmt = get_arg(args::tc_out_fmt);

    compute_kernel_hw_startup(cb_input, cb_batch_mean, cb_output_0);

    DataflowBuffer(cb_eps).wait_front(1);

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

    DataflowBuffer(cb_eps).pop_front(1);
}
