// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Layer Norm RM - Writer Kernel
// Waits for Wt tile-sized pages in c_17 per tile-row.
// Extracts 32 RM sticks and writes to DRAM via TensorAccessor.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

constexpr uint32_t c_17 = 17;  // Untilized RM output

// Compile-time args
constexpr uint32_t stick_size = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);
constexpr auto output_tensor_args = TensorAccessorArgs<2>();

void kernel_main() {
    // Runtime args
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tile_rows = get_arg_val<uint32_t>(1);
    uint32_t start_stick_id = get_arg_val<uint32_t>(2);

    const auto output_accessor = TensorAccessor(output_tensor_args, dst_addr, stick_size);

    constexpr uint32_t STICKS_PER_TILE_ROW = 32;
    uint32_t stick_id = start_stick_id;

    for (uint32_t tr = 0; tr < num_tile_rows; ++tr) {
        // Wait for Wt tile-sized pages of untilized RM data
        cb_wait_front(c_17, Wt);
        uint32_t l1_read_addr = get_read_ptr(c_17);

        // Extract 32 sticks and write to DRAM
        for (uint32_t s = 0; s < STICKS_PER_TILE_ROW; ++s) {
            uint64_t noc_addr = output_accessor.get_noc_addr(stick_id + s);
            noc_async_write(l1_read_addr, noc_addr, stick_size);
            l1_read_addr += stick_size;
        }
        noc_async_write_barrier();
        cb_pop_front(c_17, Wt);

        stick_id += STICKS_PER_TILE_ROW;
    }
}
