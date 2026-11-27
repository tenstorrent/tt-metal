// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_embedding compute: tilize_block (the design's block op), one staged block per call.
//
// cb_row_staging and cb_out_tiles are UInt16 CBs holding bf16 bits. UInt16 keeps the tilize path
// bit-opaque (the unpacker's zero flag stays off, no float rewrite of subnormals, -0.0 or NaN); with
// Float16_b CBs the tilize changes 509 of the 65536 bf16 patterns. The formats never change, so the
// helper runs with NoReconfigure and its LLK init/uninit is paid once per kernel.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

namespace {

using compute_kernel_lib::tilize_config::InitUninitMode;
using compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode;
using compute_kernel_lib::tilize_config::WaitMode;

template <uint32_t block_width_tiles, uint32_t cb_row_staging, uint32_t cb_out_tiles, InitUninitMode mode>
ALWI void tilize_staged_block(DataflowBuffer& rows, DataflowBuffer& tiles, uint32_t tile_rows) {
    DeviceZoneScopedN("TILIZE");
    compute_kernel_lib::tilize<
        block_width_tiles,
        cb_row_staging,
        cb_out_tiles,
        mode,
        WaitMode::NoWait,
        ReconfigureRegisterDatatypeMode::NoReconfigure>(rows, tiles, tile_rows);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_row_staging = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t block_height_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t block_pages = block_height_tiles * block_width_tiles;

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_row_staging, cb_out_tiles);
    DataflowBuffer rows(cb_row_staging);
    DataflowBuffer tiles(cb_out_tiles);

    for (uint32_t block = 0; block < num_blocks; ++block) {
        {
            // The whole staged block lands with one reader push; WaitMode::NoWait leaves this wait to
            // the caller so it is timed apart from the tilize.
            DeviceZoneScopedN("STAGE-WAIT");
            rows.wait_front(block_pages);
        }
        const bool first = block == 0;
        const bool last = block + 1 == num_blocks;
        if (first && last) {
            tilize_staged_block<block_width_tiles, cb_row_staging, cb_out_tiles, InitUninitMode::InitAndUninit>(
                rows, tiles, block_height_tiles);
        } else if (first) {
            tilize_staged_block<block_width_tiles, cb_row_staging, cb_out_tiles, InitUninitMode::InitOnly>(
                rows, tiles, block_height_tiles);
        } else if (last) {
            tilize_staged_block<block_width_tiles, cb_row_staging, cb_out_tiles, InitUninitMode::UninitOnly>(
                rows, tiles, block_height_tiles);
        } else {
            tilize_staged_block<block_width_tiles, cb_row_staging, cb_out_tiles, InitUninitMode::Neither>(
                rows, tiles, block_height_tiles);
        }
    }
}
