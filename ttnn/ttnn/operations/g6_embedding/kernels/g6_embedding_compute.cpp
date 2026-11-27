// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_embedding compute: tilize one unit (1 tile row x col_block_tiles) per step.
//
// cb_rows and cb_tiles are typed UInt16 with 16-bit DEST: the bf16 bits pass through the standard
// tilize path untouched (a Float16_b tilize flushes subnormals, drops -0.0 and rewrites NaN).
// One helper call per unit, init on the first and uninit on the last, so the wait for a unit's rows
// sits in its own profiler zone apart from the tilize itself.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

namespace {

using compute_kernel_lib::tilize_config::InitUninitMode;
using compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode;
using compute_kernel_lib::tilize_config::WaitMode;

constexpr uint32_t col_block_tiles = get_compile_time_arg_val(0);
constexpr uint32_t cb_rows = get_compile_time_arg_val(1);
constexpr uint32_t cb_tiles = get_compile_time_arg_val(2);

template <InitUninitMode mode>
void tilize_unit(DataflowBuffer& rows, DataflowBuffer& tiles) {
    {
        DeviceZoneScopedN("tilize-wait");
        rows.wait_front(col_block_tiles);
    }
    DeviceZoneScopedN("tilize");
    compute_kernel_lib::tilize<
        col_block_tiles,
        cb_rows,
        cb_tiles,
        mode,
        WaitMode::NoWait,
        ReconfigureRegisterDatatypeMode::NoReconfigure>(rows, tiles, 1);
}

}  // namespace

void kernel_main() {
    const uint32_t num_units = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_rows, cb_tiles);
    DataflowBuffer rows(cb_rows);
    DataflowBuffer tiles(cb_tiles);

    if (num_units == 1) {
        tilize_unit<InitUninitMode::InitAndUninit>(rows, tiles);
        return;
    }
    tilize_unit<InitUninitMode::InitOnly>(rows, tiles);
    for (uint32_t unit = 1; unit + 1 < num_units; ++unit) {
        tilize_unit<InitUninitMode::Neither>(rows, tiles);
    }
    tilize_unit<InitUninitMode::UninitOnly>(rows, tiles);
}
