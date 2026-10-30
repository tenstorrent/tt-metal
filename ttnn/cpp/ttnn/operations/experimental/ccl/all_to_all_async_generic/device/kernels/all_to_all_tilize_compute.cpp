// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

// Row-major input: tilizes each tile row the reader fetched (32 input rows, tiles_per_row tiles wide) into the packet
// buffer the writer sends from. Runtime arg 0: tile rows this core converts (0 on cores with nothing to send).
void kernel_main() {
    constexpr uint32_t cb_rows = get_compile_time_arg_val(0);
    constexpr uint32_t cb_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t tiles_per_row = get_compile_time_arg_val(2);
    const uint32_t tile_rows = get_arg_val<uint32_t>(0);
    if (tile_rows == 0) {
        return;
    }
    compute_kernel_hw_startup(cb_rows, cb_tiles);
    compute_kernel_lib::tilize<
        tiles_per_row,
        cb_rows,
        cb_tiles,
        compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure,
        compute_kernel_lib::tilize_config::Fp32Mode::Fast>(tile_rows);
}
