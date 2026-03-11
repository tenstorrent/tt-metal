// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// RMS Norm - Writer Kernel
// RM path: waits for Wt tile-pages in c_17, extracts 32 sticks, writes to DRAM
// TILE path: waits for tiles in c_16 one at a time, writes to DRAM

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

constexpr uint32_t cb_out = 16;
constexpr uint32_t cb_untilized = 17;

// Compile-time args
constexpr uint32_t stick_or_tile_size = get_compile_time_arg_val(0);
constexpr uint32_t input_is_rm = get_compile_time_arg_val(1);
constexpr uint32_t Wt = get_compile_time_arg_val(2);
constexpr uint32_t has_gamma = get_compile_time_arg_val(3);
constexpr auto output_accessor_args = TensorAccessorArgs<4>();

void kernel_main() {
    // Runtime args
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t start_id = get_arg_val<uint32_t>(1);
    uint32_t num_rows = get_arg_val<uint32_t>(2);

    if (num_rows == 0) {
        return;
    }

    const auto output_accessor = TensorAccessor(output_accessor_args, dst_addr, stick_or_tile_size);

    if constexpr (input_is_rm) {
        // RM path: read from c_17 (untilized), write 32 sticks per tile-row
        constexpr uint32_t TILE_H = 32;

        for (uint32_t row = 0; row < num_rows; ++row) {
            // Wait for all Wt tile-pages in c_17
            cb_wait_front(cb_untilized, Wt);
            uint32_t l1_read_addr = get_read_ptr(cb_untilized);

            // Write 32 sticks
            for (uint32_t s = 0; s < TILE_H; ++s) {
                uint32_t stick_id = start_id + row * TILE_H + s;
                uint64_t noc_addr = output_accessor.get_noc_addr(stick_id);
                noc_async_write(l1_read_addr, noc_addr, stick_or_tile_size);
                l1_read_addr += stick_or_tile_size;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_untilized, Wt);
        }
    } else {
        // TILE path: read from c_16, write one tile at a time
        for (uint32_t row = 0; row < num_rows; ++row) {
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                cb_wait_front(cb_out, 1);
                uint32_t l1_read_addr = get_read_ptr(cb_out);
                uint32_t tile_id = start_id + row * Wt + wt;
                uint64_t noc_addr = output_accessor.get_noc_addr(tile_id);
                noc_async_write(l1_read_addr, noc_addr, stick_or_tile_size);
                noc_async_write_barrier();
                cb_pop_front(cb_out, 1);
            }
        }
    }
}
