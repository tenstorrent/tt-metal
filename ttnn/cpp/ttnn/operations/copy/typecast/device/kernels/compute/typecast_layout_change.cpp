// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Typecast fused with a layout change. Each block is one tile-row slice of block_width_tiles tiles.
//   TILIZE_INPUT:  dfb::in (row-major) -> tilize -> dfb::mid -> typecast -> dfb::out (tiles)
//   otherwise:     dfb::in (tiles) -> typecast -> dfb::mid -> untilize -> dfb::out (row-major)
// SAME_DTYPE skips the typecast: there is no dfb::mid and the layout op writes dfb::out directly.

#include "api/compute/compute_kernel_hw_startup.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t num_blocks = get_arg(args::num_blocks);
    constexpr uint32_t block_width_tiles = get_arg(args::block_width_tiles);

    DataflowBuffer in(dfb::in);
    DataflowBuffer out(dfb::out);

#ifdef TILIZE_INPUT
    using TilizeMode = ckl::tilize_config::InitUninitMode;
    using TilizeWait = ckl::tilize_config::WaitMode;
    using TilizeReconfig = ckl::tilize_config::ReconfigureRegisterDatatypeMode;
    constexpr auto lossless = ckl::tilize_config::Fp32Mode::Lossless;
#else
    using UntilizeMode = ckl::untilize_config::InitUninitMode;
    using UntilizeWait = ckl::untilize_config::WaitMode;
    using UntilizeReconfig = ckl::untilize_config::ReconfigureRegisterDatatypeMode;
#endif

#ifdef SAME_DTYPE
    compute_kernel_hw_startup(dfb::in, dfb::out);
#ifdef TILIZE_INPUT
    ckl::tilize<
        block_width_tiles,
        dfb::in,
        dfb::out,
        TilizeMode::InitAndUninit,
        TilizeWait::WaitBlock,
        TilizeReconfig::NoReconfigure,
        lossless>(in, out, num_blocks);
#else
    ckl::untilize<
        block_width_tiles,
        dfb::in,
        dfb::out,
        UntilizeMode::InitAndUninit,
        UntilizeWait::WaitBlock,
        UntilizeReconfig::NoReconfigure>(in, out, num_blocks);
#endif
#else
    // Passed explicitly rather than read off the buffers since an Int8 tensor is carried in a UInt8 buffer.
    constexpr uint32_t in_data_format = get_arg(args::in_data_format);
    constexpr uint32_t out_data_format = get_arg(args::out_data_format);
    DataflowBuffer mid(dfb::mid);

#ifdef TILIZE_INPUT
    constexpr uint32_t cast_in_dfb = dfb::mid;
    constexpr uint32_t cast_out_dfb = dfb::out;
#else
    constexpr uint32_t cast_in_dfb = dfb::in;
    constexpr uint32_t cast_out_dfb = dfb::mid;
#endif
    constexpr auto cast_input =
        ckl::input(cast_in_dfb, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Enabled);
    constexpr auto cast_output = ckl::output(
        cast_out_dfb, ckl::ReservePolicy::PerOuter, ckl::PushPolicy::PerOuter, ckl::DataFormatReconfig::Enabled);
    using Cast = ckl::Typecast<in_data_format, out_data_format>;
    constexpr auto block = ckl::IterationShape::grid(1, block_width_tiles);

#ifdef TILIZE_INPUT
    compute_kernel_hw_startup(dfb::in, dfb::mid);
    for (uint32_t b = 0; b < num_blocks; ++b) {
        ckl::tilize<
            block_width_tiles,
            dfb::in,
            dfb::mid,
            TilizeMode::InitAndUninit,
            TilizeWait::WaitBlock,
            TilizeReconfig::UnpackAndPackReconfigure,
            lossless>(in, mid, 1);
        ckl::unary<Cast, cast_input, cast_output>(block, mid, out);
    }
#else
    compute_kernel_hw_startup(dfb::in, dfb::mid);
    for (uint32_t b = 0; b < num_blocks; ++b) {
        ckl::unary<Cast, cast_input, cast_output>(block, in, mid);
        ckl::untilize<
            block_width_tiles,
            dfb::mid,
            dfb::out,
            UntilizeMode::InitAndUninit,
            UntilizeWait::WaitBlock,
            UntilizeReconfig::UnpackAndPackReconfigure>(mid, out, 1);
    }
#endif
#endif
}
