// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "api/debug/dprint.h"
#include "api/debug/dprint_tile.h"
#include "api/debug/dprint_tensix.h"

// #include "api/debug/dprint.h"

void print_tile(
    uint8_t cb,
    int tile,
    uint8_t max_h,
    uint8_t max_w,
    const char* info,
    bool endl_rows = true,
    bool print_untilized = false) {
    DPRINT << "++ Tile " << static_cast<int>(cb) << ':' << tile << ' ' << info << ENDL();
    for (uint8_t r = 0; r < max_h; r++) {
        const auto sr = SliceRange{.h0 = r, .h1 = static_cast<uint8_t>(r + 1), .hs = 1, .w0 = 0, .w1 = max_w, .ws = 1};
        DPRINT << static_cast<int>(r) << ": " << TileSlice<64>(cb, tile, sr, endl_rows, print_untilized) << ENDL();
    }
}

void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
    UNPACK(DPRINT << "blk/c=" << uint32_t(per_core_block_cnt) << " tle/blk=" << per_core_block_tile_cnt << ENDL());

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);

    // Use lossless tilize for fp32 inputs to preserve exact values (fast tilize truncates fp32 → tf32)
    constexpr auto fp32_mode = compute_kernel_lib::is_fp32_input_format<tt::CBIndex::c_0>()
                                   ? compute_kernel_lib::tilize_config::Fp32Mode::Lossless
                                   : compute_kernel_lib::tilize_config::Fp32Mode::Fast;
    // I used to do these in the loop, but it's moved to the helper?
    // UNPACK(print_tile(tt::CBIndex::c_0, 0, 32, 32, "in"); print_tile(tt::CBIndex::c_0, 1, 32, 32, "in"););
    // PACK(print_tile(tt::CBIndex::c_16, 0, 32, 32, "out"); print_tile(tt::CBIndex::c_16, 1, 32, 32, "out"));

    compute_kernel_lib::tilize<
        per_core_block_tile_cnt,
        tt::CBIndex::c_0,
        tt::CBIndex::c_16,
        compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure,
        fp32_mode>(per_core_block_cnt);
}
