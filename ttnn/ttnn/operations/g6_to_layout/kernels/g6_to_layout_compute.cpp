// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_to_layout compute — block operation `untilize_block`.
//
// One untilize helper call over all of this core's segments (each 1 x segment_width tiles), so
// init/uninit run once per core. Both CBs are declared UInt16: with an integer operand MOVA2D keeps
// zero-exponent datums, so subnormals and -0.0 of the bf16 input survive bit-exactly.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_tiles_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_rows_id = get_compile_time_arg_val(1);
    constexpr uint32_t segment_width = get_compile_time_arg_val(2);

    const uint32_t num_segments = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_tiles_id, cb_rows_id);
    DataflowBuffer cb_tiles(cb_tiles_id);
    DataflowBuffer cb_rows(cb_rows_id);

    DeviceZoneScopedN("untilize_block");
    using namespace compute_kernel_lib::untilize_config;
    compute_kernel_lib::untilize<
        segment_width,
        cb_tiles_id,
        cb_rows_id,
        InitUninitMode::InitAndUninit,
        WaitMode::WaitBlock,
        ReconfigureRegisterDatatypeMode::NoReconfigure>(cb_tiles, cb_rows, num_segments);
}
