// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Reader kernel for reduce_avg_w_rm operation
// Reads row-major input sticks from DRAM and generates scaler tile for averaging

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    // Compile-time args
    constexpr uint32_t packed_scaler_value = get_compile_time_arg_val(0);
    constexpr uint32_t stick_size = get_compile_time_arg_val(1);
    constexpr auto src_tensor_args = TensorAccessorArgs<2>();  // TensorAccessor args start at index 2

    // Runtime args
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_sticks = get_arg_val<uint32_t>(1);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(2);

    // CB indices (from kernel design document)
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;

    // Setup TensorAccessor for input
    const auto s = TensorAccessor(src_tensor_args, src_addr, stick_size);

    // Phase 1: Generate scaler tile (1/W) once at start
    // generate_reduce_scaler handles cb_reserve_back and cb_push_back internally
    generate_reduce_scaler(cb_scaler, packed_scaler_value);

    // Phase 2: Read input sticks - TILE_HEIGHT (32) sticks per block
    constexpr uint32_t TILE_HEIGHT = 32;
    uint32_t num_blocks = num_sticks / TILE_HEIGHT;
    uint32_t stick_id = start_stick_id;

    for (uint32_t block = 0; block < num_blocks; block++) {
        // Reserve space for one block of sticks (32 sticks)
        // CB page_size is 32 * stick_size, but we push 1 page per block
        cb_reserve_back(cb_in, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_in);

        // Read 32 sticks (one tile row) from DRAM to L1
        for (uint32_t h = 0; h < TILE_HEIGHT; h++) {
            uint64_t noc_addr = s.get_noc_addr(stick_id);
            noc_async_read(noc_addr, l1_write_addr, stick_size);
            l1_write_addr += stick_size;
            stick_id++;
        }

        // Wait for all reads to complete
        noc_async_read_barrier();

        // Signal that one page (containing 32 sticks) is ready
        cb_push_back(cb_in, 1);
    }
}
