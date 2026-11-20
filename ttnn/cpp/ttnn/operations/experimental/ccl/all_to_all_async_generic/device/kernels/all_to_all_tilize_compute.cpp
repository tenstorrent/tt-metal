// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "all_to_all_tilize_common.hpp"

// Row-major input: tilizes each tile row the reader fetched (32 input rows, tiles_per_row tiles wide) into the packet
// buffer the writer sends from, one page per tile.
//
// After each destination range it pads the buffer (range_pad_tiles, all_to_all_tilize_common.hpp).
//
// Runtime args: the number of ranges this core converts, then each range's tile rows.
void kernel_main() {
    constexpr uint32_t rows_cb = get_compile_time_arg_val(0);
    constexpr uint32_t packet_cb = get_compile_time_arg_val(1);
    constexpr uint32_t tiles_per_row = get_compile_time_arg_val(2);
    constexpr uint32_t range_align_tiles = get_compile_time_arg_val(3);
    const uint32_t num_ranges = get_arg_val<uint32_t>(0);
    if (num_ranges == 0) {
        return;
    }
    compute_kernel_hw_startup(rows_cb, packet_cb);
    for (uint32_t range = 0; range < num_ranges; ++range) {
        const uint32_t tile_rows = get_arg_val<uint32_t>(1 + range);
        compute_kernel_lib::tilize<
            tiles_per_row,
            rows_cb,
            packet_cb,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure,
            compute_kernel_lib::tilize_config::Fp32Mode::Fast>(tile_rows);
        const uint32_t pad_tiles = range_pad_tiles(tile_rows * tiles_per_row, range_align_tiles);
        if (pad_tiles > 0) {
            cb_reserve_back(packet_cb, pad_tiles);
            cb_push_back(packet_cb, pad_tiles);
        }
    }
}
