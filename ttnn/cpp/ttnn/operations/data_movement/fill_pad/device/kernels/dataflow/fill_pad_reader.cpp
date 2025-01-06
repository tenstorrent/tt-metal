// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    // get input tensor DRAM and find starting points for pad iteration
    const std::uint32_t input_dram_buffer_src_addr = get_arg_val<uint32_t>(0);
    const std::uint32_t beginning_row = get_arg_val<uint32_t>(1);
    const std::uint32_t beginning_col = get_arg_val<uint32_t>(2);

    // hardware constraints
    constexpr uint32_t face_size = 16;
    constexpr uint32_t tile_height = 32;

    const std::uint32_t fill_value = get_compile_time_arg_val(1);
}
