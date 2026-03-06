// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Reader Kernel
// Runs on RISCV_0 (BRISC), reads RM sticks from DRAM to L1 CBs via NOC0
//
// Per tile-row: reads 32 RM sticks into cb_in (Wt tile-sized pages).
// On first tile-row (when needed): generates reduce scaler, epsilon, gamma, beta.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

constexpr uint32_t cb_in = 0;
constexpr uint32_t cb_gamma = 1;
constexpr uint32_t cb_beta = 2;
constexpr uint32_t cb_reduce_scaler = 8;
constexpr uint32_t cb_eps = 9;

void kernel_main() {
    // Compile-time args
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t has_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t has_beta = get_compile_time_arg_val(3);
    constexpr auto input_accessor_args = TensorAccessorArgs<4>();

    // Runtime args
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t N = get_arg_val<uint32_t>(1);
    uint32_t start_stick_id = get_arg_val<uint32_t>(2);
    // Args 3-6 used in later stages (scaler, eps, gamma_addr, beta_addr)

    const auto input_accessor = TensorAccessor(input_accessor_args, src_addr, stick_size);

    uint32_t stick_id = start_stick_id;

    for (uint32_t row = 0; row < N; row++) {
        // Reserve Wt tile-sized pages in cb_in
        cb_reserve_back(cb_in, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_in);

        // Read 32 RM sticks (one tile-row)
        for (uint32_t s = 0; s < 32; s++) {
            uint64_t noc_addr = input_accessor.get_noc_addr(stick_id);
            noc_async_read(noc_addr, l1_write_addr, stick_size);
            l1_write_addr += stick_size;
            stick_id++;
        }
        noc_async_read_barrier();

        cb_push_back(cb_in, Wt);
    }
}
