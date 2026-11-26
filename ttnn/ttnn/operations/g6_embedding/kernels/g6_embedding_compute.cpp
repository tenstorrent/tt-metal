// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_embedding compute: tilize_block, one tile row x col_block_tiles per unit, UInt16 -> UInt16.
//
// Both CBs are UInt16 with 16-bit DEST, so the tilize is a pure bit permutation of the bf16 values
// (a Float16_b-typed tilize flushes subnormals, drops -0.0's sign and rewrites NaN payloads).
// The wait sits outside the helper (NoWait) so the profiler sees wait and tilize as separate
// zones; init/uninit are still paid once per kernel through the InitOnly/Neither/UninitOnly modes.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_rows = get_compile_time_arg_val(0);
    constexpr uint32_t cb_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t col_block_tiles = get_compile_time_arg_val(2);
    const uint32_t num_units = get_arg_val<uint32_t>(0);

    using namespace compute_kernel_lib::tilize_config;
    constexpr auto kWait = WaitMode::NoWait;
    // compute_kernel_hw_startup configures exactly these two CBs; nothing else runs in between.
    constexpr auto kReconfig = ReconfigureRegisterDatatypeMode::NoReconfigure;

    compute_kernel_hw_startup(cb_rows, cb_tiles);
    DataflowBuffer rows(cb_rows);
    DataflowBuffer tiles(cb_tiles);

    for (uint32_t unit = 0; unit < num_units; ++unit) {
        {
            DeviceZoneScopedN("tilize-wait");
            rows.wait_front(col_block_tiles);
        }
        DeviceZoneScopedN("tilize");
        if (unit == 0) {
            if (num_units == 1) {
                compute_kernel_lib::
                    tilize<col_block_tiles, cb_rows, cb_tiles, InitUninitMode::InitAndUninit, kWait, kReconfig>(
                        rows, tiles, 1);
            } else {
                compute_kernel_lib::
                    tilize<col_block_tiles, cb_rows, cb_tiles, InitUninitMode::InitOnly, kWait, kReconfig>(
                        rows, tiles, 1);
            }
        } else if (unit + 1 == num_units) {
            compute_kernel_lib::
                tilize<col_block_tiles, cb_rows, cb_tiles, InitUninitMode::UninitOnly, kWait, kReconfig>(
                    rows, tiles, 1);
        } else {
            compute_kernel_lib::tilize<col_block_tiles, cb_rows, cb_tiles, InitUninitMode::Neither, kWait, kReconfig>(
                rows, tiles, 1);
        }
    }
}
