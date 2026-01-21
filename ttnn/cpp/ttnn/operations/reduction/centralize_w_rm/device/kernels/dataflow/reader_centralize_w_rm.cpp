// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Compile-time args
    constexpr uint32_t input_stick_size = get_compile_time_arg_val(0);
    constexpr uint32_t packed_scaler_value = get_compile_time_arg_val(1);
    constexpr uint32_t Ht = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);
    constexpr auto tensor_args = TensorAccessorArgs<4>();

    // Runtime args
    const uint32_t src_addr = get_arg_val<uint32_t>(0);

    // CB IDs
    constexpr uint32_t cb_in_rm = tt::CBIndex::c_0;  // Input RM sticks

    // Create tensor accessor for input
    const auto accessor = TensorAccessor(tensor_args, src_addr, input_stick_size);

    // STUB: Read and push 1 tile at a time (compute consumes 1 at a time)
    // Real implementation (Stage 7): Will push Wt at a time for tilize helper
    constexpr uint32_t TILE_HEIGHT = 32;
    uint32_t stick_id = 0;

    const uint32_t num_tiles = Ht * Wt;
    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        cb_reserve_back(cb_in_rm, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_in_rm);

        // Read 32 sticks for one tile (stub: just copy garbage data)
        for (uint32_t s = 0; s < TILE_HEIGHT; ++s) {
            uint64_t noc_addr = accessor.get_noc_addr(stick_id);
            noc_async_read(noc_addr, l1_write_addr, input_stick_size);
            l1_write_addr += input_stick_size;
            stick_id++;
        }
        noc_async_read_barrier();

        cb_push_back(cb_in_rm, 1);
    }
}
