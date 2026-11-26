// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_typecast compute — convert_block.
//
// One typecast chain per block: wait the whole block upfront, convert it in DEST groups of
// DEST_AUTO_LIMIT tiles, pack straight into cb_output_tiles, publish the block at the end.
// The LLK in/out formats are derived from the CB formats the host bound, so one source serves
// every (input, output) dtype pair the program descriptor configures.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

void kernel_main() {
    constexpr uint32_t block_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(2);

    const uint32_t core_tiles = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_input_tiles, cb_output_tiles);

    using namespace compute_kernel_lib;
    DataflowBuffer input_tiles(cb_input_tiles);
    DataflowBuffer output_tiles(cb_output_tiles);

    // Block mapping is mandatory with an upfront window: the default Scalar mapping would make
    // every DEST lane read tile 0 of the block.
    constexpr auto input_spec = input(cb_input_tiles, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block);
    constexpr auto output_spec = output(cb_output_tiles, ReservePolicy::Upfront, PushPolicy::AtEnd);

    const uint32_t num_blocks = (core_tiles + block_tiles - 1) / block_tiles;
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t block_extent =
            (block_idx + 1 == num_blocks) ? core_tiles - block_idx * block_tiles : block_tiles;
#if defined(PROFILE_KERNEL)
        {
            // Split the upstream starvation out of TC-CV-BLOCK; the chain's own wait then returns at once.
            DeviceZoneScopedN("TC-CV-WAIT");
            input_tiles.wait_front(block_extent);
        }
#endif
        {
            DeviceZoneScopedN("TC-CV-BLOCK");
            typecast<input_spec, output_spec>(
                IterationShape::tiles(block_extent).block_size(DEST_AUTO_LIMIT), input_tiles, output_tiles);
        }
    }
}
