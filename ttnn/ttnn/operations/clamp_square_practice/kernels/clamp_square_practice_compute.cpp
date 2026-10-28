// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// clamp_square_practice compute — clamp_square_block.
// ONE eltwise_chain call spans this core's whole assignment: Copy -> Clamp -> Square -> Pack,
// blocked by block_tiles DEST lanes with FullBlock CB synchronization (nominal block_tiles
// wait/pop/reserve/push even on the ragged last block; math/pack touch only valid tiles).
// Element inits and format reconfig are therefore paid once per core.
//
// CT args: [cb_input_tiles, cb_output_tiles, block_tiles]
// RT args: [core_tiles, min_bits (fp32), max_bits (fp32)]

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(2);
    static_assert(
        block_tiles >= 1 && block_tiles <= compute_kernel_lib::DEST_AUTO_LIMIT, "FullBlock block must fit DEST");

    const uint32_t core_tiles = get_arg_val<uint32_t>(0);
    const uint32_t min_bits = get_arg_val<uint32_t>(1);
    const uint32_t max_bits = get_arg_val<uint32_t>(2);

    DataflowBuffer input_tiles(cb_input_tiles);
    DataflowBuffer output_tiles(cb_output_tiles);

    compute_kernel_hw_startup(cb_input_tiles, cb_output_tiles);

    using namespace compute_kernel_lib;
    eltwise_chain(
        IterationShape::tiles(core_tiles).block_size(block_tiles, BlockTailSync::FullBlock),
        CopyTile<
            input(cb_input_tiles, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize, InputTileMapping::Block),
            Dst::D0>{input_tiles},
        Clamp<Dst::D0>{min_bits, max_bits},
        Square<Dst::D0>{},
        PackTile<output(cb_output_tiles, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)>{output_tiles});
}
