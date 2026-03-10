// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Reader Kernel
// Reads RM sticks from DRAM into c_0 using TensorAccessor.
// Fills c_8 (1/W scaler via prepare_reduce_scaler) and c_9 (epsilon) at program start.
// Optionally reads gamma/beta sticks into c_2/c_3.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include <tt-metalium/constants.hpp>

// CB indices
constexpr uint32_t cb_input_rm = 0;  // c_0: RM sticks for tilize
constexpr uint32_t cb_scaler = 8;    // c_8: Reduce scaler (1/W)
constexpr uint32_t cb_eps = 9;       // c_9: Epsilon tile
constexpr uint32_t cb_gamma_rm = 2;  // c_2: Gamma RM stick (optional)
constexpr uint32_t cb_beta_rm = 3;   // c_3: Beta RM stick (optional)

// Compile-time args
constexpr uint32_t stick_size = get_compile_time_arg_val(0);
constexpr uint32_t has_gamma = get_compile_time_arg_val(1);
constexpr uint32_t has_beta = get_compile_time_arg_val(2);
constexpr auto input_tensor_args = TensorAccessorArgs<3>();

void kernel_main() {
    // Runtime args
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_sticks = get_arg_val<uint32_t>(1);
    const uint32_t Wt = get_arg_val<uint32_t>(2);
    const uint32_t scaler_value = get_arg_val<uint32_t>(3);
    const uint32_t eps_value = get_arg_val<uint32_t>(4);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(5);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(6);
    const uint32_t beta_addr = get_arg_val<uint32_t>(7);

    // Reinterpret uint32 bit patterns as float
    union {
        uint32_t u;
        float f;
    } scaler_conv;
    scaler_conv.u = scaler_value;
    float scaler_f = scaler_conv.f;

    union {
        uint32_t u;
        float f;
    } eps_conv;
    eps_conv.u = eps_value;
    float eps_f = eps_conv.f;

    // Setup input TensorAccessor
    const auto input_accessor = TensorAccessor(input_tensor_args, src_addr, stick_size);

    // ===== 1. Fill c_8 with 1/W scaler (for reduce) =====
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler>(scaler_f);

    // ===== 2. Fill c_9 with epsilon value =====
    dataflow_kernel_lib::prepare_reduce_scaler<cb_eps>(eps_f);

    // ===== 3. Main loop: read RM sticks into c_0 =====
    // Process tile-rows: each tile-row = 32 sticks, Wt tile-pages
    uint32_t stick_id = start_stick_id;

    for (uint32_t s = 0; s < num_sticks; s += 32) {
        // Reserve Wt pages in c_0
        cb_reserve_back(cb_input_rm, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_input_rm);

        // Read 32 RM sticks into the reserved space
        for (uint32_t i = 0; i < 32; ++i) {
            uint64_t noc_addr = input_accessor.get_noc_addr(stick_id);
            noc_async_read(noc_addr, l1_write_addr, stick_size);
            l1_write_addr += stick_size;
            stick_id++;
        }
        noc_async_read_barrier();

        // Push Wt pages (32 sticks = Wt tiles worth of RM data)
        cb_push_back(cb_input_rm, Wt);
    }
}
