// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// LayerNorm - Reader Kernel
// Reads 32 RM sticks per tile-row into cb_in (c_0).

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

constexpr uint32_t cb_in = tt::CBIndex::c_0;
constexpr uint32_t TILE_HEIGHT = 32;

// Compile-time args
constexpr uint32_t stick_size = get_compile_time_arg_val(0);
constexpr uint32_t has_gamma = get_compile_time_arg_val(1);
constexpr uint32_t has_beta = get_compile_time_arg_val(2);
constexpr auto input_accessor_args = TensorAccessorArgs<3>();

void kernel_main() {
    // Runtime args
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_sticks = get_arg_val<uint32_t>(1);
    uint32_t Wt = get_arg_val<uint32_t>(2);
    uint32_t start_stick_id = get_arg_val<uint32_t>(3);

    // Input tensor accessor
    const auto input_accessor = TensorAccessor(input_accessor_args, src_addr, stick_size);

    // Number of tile-rows
    uint32_t num_tile_rows = num_sticks / TILE_HEIGHT;

    // Main loop: read 32 RM sticks per tile-row into cb_in
    uint32_t stick_id = start_stick_id;
    for (uint32_t tile_row = 0; tile_row < num_tile_rows; ++tile_row) {
        cb_reserve_back(cb_in, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_in);

        for (uint32_t s = 0; s < TILE_HEIGHT; ++s) {
            uint64_t noc_addr = input_accessor.get_noc_addr(stick_id);
            noc_async_read(noc_addr, l1_write_addr, stick_size);
            l1_write_addr += stick_size;
            stick_id++;
        }
        noc_async_read_barrier();

        cb_push_back(cb_in, Wt);
    }
}
