// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Writer kernel for fused DeltaNet decode.
//
// Writes per-head results back to DRAM:
//   - Updated state S: Dk_tiles * Dv_tiles tiles
//   - Output vector: Dv_tiles tiles

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_state_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output    = get_compile_time_arg_val(1);
    constexpr uint32_t Dk_tiles     = get_compile_time_arg_val(2);
    constexpr uint32_t Dv_tiles     = get_compile_time_arg_val(3);
    constexpr auto accessor_args    = TensorAccessorArgs<4>();

    const uint32_t state_out_addr       = get_arg_val<uint32_t>(0);
    const uint32_t output_addr          = get_arg_val<uint32_t>(1);
    const uint32_t state_out_start_tile = get_arg_val<uint32_t>(2);
    const uint32_t output_start_tile    = get_arg_val<uint32_t>(3);

    constexpr uint32_t state_tiles = Dk_tiles * Dv_tiles;
    const uint32_t tile_bytes = get_tile_size(cb_state_out);

    const auto state_out_acc = TensorAccessor(accessor_args, state_out_addr, tile_bytes);
    const auto output_acc    = TensorAccessor(accessor_args, output_addr, tile_bytes);

    // Write updated state back to DRAM
    {
        cb_wait_front(cb_state_out, state_tiles);
        uint32_t l1_addr = get_read_ptr(cb_state_out);
        for (uint32_t t = 0; t < state_tiles; t++) {
            noc_async_write_tile(state_out_start_tile + t, state_out_acc, l1_addr);
            l1_addr += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_state_out, state_tiles);
    }

    // Write output vector
    {
        cb_wait_front(cb_output, Dv_tiles);
        uint32_t l1_addr = get_read_ptr(cb_output);
        for (uint32_t t = 0; t < Dv_tiles; t++) {
            noc_async_write_tile(output_start_tile + t, output_acc, l1_addr);
            l1_addr += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_output, Dv_tiles);
    }
}
