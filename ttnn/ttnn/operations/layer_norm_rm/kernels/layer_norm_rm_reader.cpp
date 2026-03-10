// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Reader Kernel
// Runs on RISCV_0 (BRISC), reads RM sticks from DRAM via NOC0.
//
// Per block (32 sticks = 1 tile-row = Wt tile-pages):
//   cb_reserve_back(c_0, Wt) -> read 32 sticks -> cb_push_back(c_0, Wt)

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    // Compile-time args
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr auto input_tensor_args = TensorAccessorArgs<1>();

    // Runtime args
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(1);
    const uint32_t num_sticks = get_arg_val<uint32_t>(2);

    // Derived constants
    constexpr uint32_t cb_input_rm = 0;             // c_0
    constexpr uint32_t cb_scaler = 8;               // c_8: reduce scaler (1/W)
    constexpr uint32_t Wt = stick_size / (32 * 2);  // tiles per row = W / 32, element_size=2 for bf16
    constexpr uint32_t W = stick_size / 2;          // width in elements

    // Set up TensorAccessor for input
    const auto input_accessor = TensorAccessor(input_tensor_args, src_addr, stick_size);

    // Fill scaler CB with 1/W for reduce_mean (program lifetime, filled once)
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler>(1.0f / W);

    // Main loop: read blocks of 32 sticks
    uint32_t stick_id = start_stick_id;
    for (uint32_t s = 0; s < num_sticks; s += 32) {
        // Reserve Wt pages in c_0 (tile-sized pages that hold 32 RM sticks)
        cb_reserve_back(cb_input_rm, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_input_rm);

        // Read 32 sticks into the reserved space
        for (uint32_t i = 0; i < 32; i++) {
            uint64_t noc_addr = input_accessor.get_noc_addr(stick_id);
            noc_async_read(noc_addr, l1_write_addr, stick_size);
            l1_write_addr += stick_size;
            stick_id++;
        }
        noc_async_read_barrier();

        // Push Wt pages to compute
        cb_push_back(cb_input_rm, Wt);
    }
}
