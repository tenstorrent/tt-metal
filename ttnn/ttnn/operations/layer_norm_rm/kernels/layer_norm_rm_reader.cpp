// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Reader Kernel
// Runs on RISCV_0 (BRISC), reads RM sticks from DRAM via NOC0.
//
// Before main loop:
//   - Fill c_8 with reduce scaler (1/W)
//   - Fill c_9 with epsilon constant
//   - If gamma: read gamma stick, replicate 32x into c_0, push for compute to tilize into c_5
//   - If beta: read beta stick, replicate 32x into c_0, push for compute to tilize into c_6
//
// Per block (32 sticks = 1 tile-row = Wt tile-pages):
//   cb_reserve_back(c_0, Wt) -> read 32 sticks -> cb_push_back(c_0, Wt)

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

// Helper: read a single stick from interleaved DRAM and replicate 32x into cb_input_rm
inline void read_and_replicate_param(uint32_t param_addr, uint32_t stick_size, uint32_t cb_id, uint32_t Wt) {
    cb_reserve_back(cb_id, Wt);
    uint32_t l1_write_addr = get_write_ptr(cb_id);

    // Read page 0 of the interleaved buffer using InterleavedAddrGenFast
    const InterleavedAddrGenFast<true> param_addrgen = {
        .bank_base_address = param_addr,
        .page_size = stick_size,
        .data_format = DataFormat::Float16_b,
    };
    noc_async_read_tile(0, param_addrgen, l1_write_addr);
    noc_async_read_barrier();

    // Replicate the first stick 31 more times via L1 memcpy
    volatile uint16_t* src = reinterpret_cast<volatile uint16_t*>(l1_write_addr);
    uint32_t num_elements = stick_size / sizeof(uint16_t);
    for (uint32_t row = 1; row < 32; row++) {
        volatile uint16_t* dst = reinterpret_cast<volatile uint16_t*>(l1_write_addr + row * stick_size);
        for (uint32_t e = 0; e < num_elements; e++) {
            dst[e] = src[e];
        }
    }

    cb_push_back(cb_id, Wt);
}

void kernel_main() {
    // Compile-time args
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr auto input_tensor_args = TensorAccessorArgs<1>();

    // Runtime args
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(1);
    const uint32_t num_sticks = get_arg_val<uint32_t>(2);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(3);
    const uint32_t beta_addr = get_arg_val<uint32_t>(4);
    const uint32_t eps_bits = get_arg_val<uint32_t>(5);

    // Derived constants
    constexpr uint32_t cb_input_rm = 0;             // c_0
    constexpr uint32_t cb_scaler = 8;               // c_8: reduce scaler (1/W)
    constexpr uint32_t cb_eps = 9;                  // c_9: epsilon constant
    constexpr uint32_t Wt = stick_size / (32 * 2);  // tiles per row = W / 32, element_size=2 for bf16
    constexpr uint32_t W = stick_size / 2;          // width in elements

    // Reinterpret epsilon bits back to float
    union {
        uint32_t u;
        float f;
    } eps_union;
    eps_union.u = eps_bits;
    float epsilon = eps_union.f;

    // Set up TensorAccessor for input
    const auto input_accessor = TensorAccessor(input_tensor_args, src_addr, stick_size);

    // Fill scaler CB with 1/W for reduce_mean (program lifetime, filled once)
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler>(1.0f / W);

    // Fill epsilon CB (program lifetime, filled once)
    dataflow_kernel_lib::prepare_reduce_scaler<cb_eps>(epsilon);

    // --- Optional: read and replicate gamma/beta for tilize by compute ---
    // Gamma and beta are (1,1,1,W) RM tensors in DRAM. Read the single stick,
    // replicate 32 times into c_0, push for compute to tilize into c_5/c_6.
    if (gamma_addr != 0) {
        read_and_replicate_param(gamma_addr, stick_size, cb_input_rm, Wt);
    }

    if (beta_addr != 0) {
        read_and_replicate_param(beta_addr, stick_size, cb_input_rm, Wt);
    }

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
