// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// RMS Norm - Writer Kernel
// Writes output data to DRAM

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

constexpr uint32_t cb_out = 16;

// Compile-time args
constexpr uint32_t stick_size = get_compile_time_arg_val(0);
constexpr auto output_accessor_args = TensorAccessorArgs<1>();

void kernel_main() {
    // Runtime args
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);
    const uint32_t start_row_id = get_arg_val<uint32_t>(2);
    const uint32_t Wt = get_arg_val<uint32_t>(3);

    if (num_rows == 0) {
        return;
    }

    const auto output_accessor = TensorAccessor(output_accessor_args, dst_addr, stick_size);

    for (uint32_t row = 0; row < num_rows; ++row) {
        uint32_t row_id = start_row_id + row;

#if IS_INPUT_RM
        // RM path: cb_out has Wt tile-sized pages containing 32 RM sticks
        // Extract 32 sticks and write each to DRAM
        cb_wait_front(cb_out, Wt);
        uint32_t l1_read_addr = get_read_ptr(cb_out);
        uint32_t base_stick = row_id * 32;
        for (uint32_t s = 0; s < 32; ++s) {
            uint64_t noc_addr = output_accessor.get_noc_addr(base_stick + s);
            noc_async_write(l1_read_addr, noc_addr, stick_size);
            l1_read_addr += stick_size;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, Wt);
#else
        // TILE path: read Wt tiles from cb_out one at a time
        uint32_t base_tile = row_id * Wt;
        for (uint32_t t = 0; t < Wt; ++t) {
            cb_wait_front(cb_out, 1);
            uint32_t l1_read_addr = get_read_ptr(cb_out);
            uint64_t noc_addr = output_accessor.get_noc_addr(base_tile + t);
            noc_async_write(l1_read_addr, noc_addr, stick_size);
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
#endif
    }
}
