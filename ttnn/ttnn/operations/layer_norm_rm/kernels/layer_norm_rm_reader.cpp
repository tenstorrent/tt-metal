// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Reader Kernel
//
// Reads RM input sticks from DRAM using TensorAccessor.
// Generates scaler tile (1/W) into cb_scaler and epsilon tile into cb_eps.
// For affine transform: reads gamma/beta tile-rows into cb_gamma/cb_beta.
//
// Compile-time args:
//   [0]  stick_size       - W * element_size bytes
//   [1+] TensorAccessorArgs(input)
//
// Runtime args:
//   [0] src_addr           - Input buffer base address
//   [1] num_blocks         - Number of tile-rows for this core
//   [2] start_stick_id     - First stick (row) index
//   [3] gamma_addr         - Gamma buffer base address (0 if no gamma)
//   [4] beta_addr          - Beta buffer base address (0 if no beta)
//   [5] eps_value          - Epsilon as bit-cast uint32
//   [6] mean_scaler_value  - 1/W as bit-cast uint32

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

constexpr uint32_t cb_in_rm = 0;
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_eps = 3;
constexpr uint32_t cb_gamma = 4;
constexpr uint32_t cb_beta = 5;

void kernel_main() {
    // Compile-time args
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr auto input_tensor_args = TensorAccessorArgs<1>();

    // Runtime args
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(1);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(2);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(3);
    const uint32_t beta_addr = get_arg_val<uint32_t>(4);
    const uint32_t eps_bits = get_arg_val<uint32_t>(5);
    const uint32_t scaler_bits = get_arg_val<uint32_t>(6);

    // Derived constants
    constexpr uint32_t tile_size_bytes = get_tile_size(cb_in_rm);  // bfloat16 tile = 2048 bytes
    constexpr uint32_t Wt = stick_size / (32 * 2);  // tiles per row: stick_size / (TILE_W * sizeof(bf16))

    // Create TensorAccessor for input (RM tensor: page = 1 stick = stick_size bytes)
    const auto input_accessor = TensorAccessor(input_tensor_args, src_addr, stick_size);

    // Early exit for idle cores
    if (num_blocks == 0) {
        return;
    }

    // === Prepare scaler tile (1/W) into cb_scaler ===
    // Reinterpret scaler_bits as float
    union {
        uint32_t u;
        float f;
    } scaler_union;
    scaler_union.u = scaler_bits;
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler>(scaler_union.f);

    // === Prepare epsilon tile into cb_eps ===
    union {
        uint32_t u;
        float f;
    } eps_union;
    eps_union.u = eps_bits;
    dataflow_kernel_lib::prepare_reduce_scaler<cb_eps>(eps_union.f);

    // === Main loop: read RM sticks per tile-row ===
    uint32_t stick_id = start_stick_id;

    for (uint32_t block = 0; block < num_blocks; ++block) {
        // Each tile-row = 32 sticks, producing Wt tile-sized pages in cb_in_rm
        cb_reserve_back(cb_in_rm, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_in_rm);

        for (uint32_t stick = 0; stick < 32; ++stick) {
            uint64_t noc_addr = input_accessor.get_noc_addr(stick_id);
            noc_async_read(noc_addr, l1_write_addr, stick_size);
            l1_write_addr += stick_size;
            stick_id++;
        }
        noc_async_read_barrier();
        cb_push_back(cb_in_rm, Wt);
    }
}
