// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Softmax - Writer Kernel
// Writes output tiles from c_16 to DRAM via TensorAccessor (NOC1).

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_output = tt::CBIndex::c_16;

void kernel_main() {
    // Compile-time args
    constexpr uint32_t num_output_tiles = get_compile_time_arg_val(0);
    constexpr auto tensor_accessor_args = TensorAccessorArgs<1>();

    // Runtime args
    const uint32_t output_addr = get_arg_val<uint32_t>(0);

    // Setup tensor accessor
    const uint32_t page_size = get_tile_size(cb_output);
    const auto output_accessor = TensorAccessor(tensor_accessor_args, output_addr, page_size);

    for (uint32_t tile_id = 0; tile_id < num_output_tiles; ++tile_id) {
        // Wait for compute to produce a tile
        cb_wait_front(cb_output, 1);

        // Write tile to DRAM
        uint32_t l1_read_addr = get_read_ptr(cb_output);
        uint64_t noc_addr = output_accessor.get_noc_addr(tile_id);
        noc_async_write(l1_read_addr, noc_addr, page_size);
        noc_async_write_barrier();

        // Release tile from CB
        cb_pop_front(cb_output, 1);
    }
}
