// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_typecast compute — convert_block.
//
// The kernel owns the block handshake (nominal block_tiles on every wait/reserve/pop/push, the
// ragged last block included) and hands the chain caller-managed (None) policies with Block
// mapping, so the chain reads and packs tiles [0, block_extent) of the open slots by index and
// groups them into DEST-sized batches. The pair's formats come from the bound CBs; bf16->f32 and
// bf16->bf8b are pure unpack/pack conversions, bf16->i32 truncates on the SFPU.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

void kernel_main() {
    using namespace compute_kernel_lib;

    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(1);

    const uint32_t block_tiles = get_common_arg_val<uint32_t>(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(0);
    const uint32_t last_block_extent = get_arg_val<uint32_t>(1);

    constexpr InputSpec block_input = input(cb_input_tiles, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block);
    constexpr OutputSpec block_output = output(cb_output_tiles, ReservePolicy::None, PushPolicy::None);

    compute_kernel_hw_startup(cb_input_tiles, cb_output_tiles);
    DataflowBuffer input_tiles(cb_input_tiles);
    DataflowBuffer output_tiles(cb_output_tiles);

    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t block_extent = (block_idx + 1 == num_blocks) ? last_block_extent : block_tiles;
        {
            DeviceZoneScopedN("convert_block.wait");
            input_tiles.wait_front(block_tiles);
        }
        {
            DeviceZoneScopedN("convert_block.reserve");
            output_tiles.reserve_back(block_tiles);
        }
        {
            DeviceZoneScopedN("convert_block.chain");
            typecast<block_input, block_output>(
                IterationShape::tiles(block_extent).block_size(DEST_AUTO_LIMIT), input_tiles, output_tiles);
        }
        input_tiles.pop_front(block_tiles);
        output_tiles.push_back(block_tiles);
    }
}
