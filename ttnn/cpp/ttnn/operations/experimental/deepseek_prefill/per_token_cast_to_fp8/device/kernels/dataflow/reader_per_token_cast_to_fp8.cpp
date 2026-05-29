// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// Reader for per_token_cast_to_fp8.
// For each tile-row (32 sticks of H elements) assigned to this core, reads the 32 sticks
// into cb_in_rm. Each stick is a full row of the row-major input tensor.

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t start_tile_row = get_arg_val<uint32_t>(1);
    uint32_t num_tile_rows = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_in_rm = get_compile_time_arg_val(0);
    constexpr uint32_t stick_size_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr auto src_args = TensorAccessorArgs<2>();

    const auto src = TensorAccessor(src_args, src_addr);

    for (uint32_t row = 0; row < num_tile_rows; ++row) {
        cb_reserve_back(cb_in_rm, TILE_HEIGHT);
        uint32_t l1_addr = get_write_ptr(cb_in_rm);
        for (uint32_t s = 0; s < TILE_HEIGHT; ++s) {
            uint32_t stick_id = (start_tile_row + row) * TILE_HEIGHT + s;
            noc_async_read(src.get_noc_addr(stick_id), l1_addr, stick_size_bytes);
            l1_addr += stick_size_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_in_rm, TILE_HEIGHT);
    }
}
