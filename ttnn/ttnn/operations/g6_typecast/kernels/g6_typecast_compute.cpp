// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_typecast compute — convert_block.
//
// One compute_kernel_lib::typecast call walks the core's whole page range (core_pages tiles), so
// the typecast init and data-format reconfig run once per core; the chain waits/pops
// cb_input_tiles and reserves/pushes cb_output_tiles once per block_tiles group (the tail block
// synchronizes its valid tiles only, BlockTailSync::ValidTiles — matching the reader/writer).
// The LLK formats come from the CB formats: Float16_b -> Float32 (packer), -> Bfp8_b (packer),
// -> Int32 (SFPU fp32_to_int32). The host sets fp32_dest_acc_en for the 32-bit destinations.
//
// CT: [cb_input_tiles, cb_output_tiles, block_tiles]   per-core RT: [core_pages]

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(2);

    const uint32_t core_pages = get_arg_val<uint32_t>(0);

    DataflowBuffer input_tiles(cb_input_tiles);
    DataflowBuffer output_tiles(cb_output_tiles);

    compute_kernel_hw_startup(cb_input_tiles, cb_output_tiles);

    DeviceZoneScopedN("convert");
    using namespace compute_kernel_lib;
    typecast<
        input(cb_input_tiles, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize, InputTileMapping::Block),
        output(cb_output_tiles, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)>(
        IterationShape::tiles(core_pages).block_size(block_tiles), input_tiles, output_tiles);
}
