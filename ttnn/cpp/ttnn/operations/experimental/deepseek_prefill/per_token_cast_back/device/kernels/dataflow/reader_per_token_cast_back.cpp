// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// Reader for per_token_cast_back.
// Reads 32 fp8 sticks (each H bytes) per tile-row into cb_in_fp8.
// v0 ignores the scale tensor (passed for forward-compat with v1).

void kernel_main() {
    uint32_t src_e4m3_addr = get_arg_val<uint32_t>(0);
    /*uint32_t src_scale_addr =*/get_arg_val<uint32_t>(1);  // unused in v0
    uint32_t start_tile_row = get_arg_val<uint32_t>(2);
    uint32_t num_tile_rows = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_in_fp8 = get_compile_time_arg_val(0);
    constexpr uint32_t e4m3_stick_size_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr auto e4m3_args = TensorAccessorArgs<2>();

    const auto e4m3_src = TensorAccessor(e4m3_args, src_e4m3_addr);

    for (uint32_t row = 0; row < num_tile_rows; ++row) {
        cb_reserve_back(cb_in_fp8, TILE_HEIGHT);
        uint32_t l1_addr = get_write_ptr(cb_in_fp8);
        for (uint32_t s = 0; s < TILE_HEIGHT; ++s) {
            uint32_t stick_id = (start_tile_row + row) * TILE_HEIGHT + s;
            noc_async_read(e4m3_src.get_noc_addr(stick_id), l1_addr, e4m3_stick_size_bytes);
            l1_addr += e4m3_stick_size_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_in_fp8, TILE_HEIGHT);
    }
}
