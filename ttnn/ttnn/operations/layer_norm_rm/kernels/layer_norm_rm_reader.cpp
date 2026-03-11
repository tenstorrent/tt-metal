// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Reader Kernel
// Reads RM sticks from DRAM, fills scaler/eps CBs, optional gamma/beta reads.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/dataflow/fill_tile_utils.hpp"

void kernel_main() {
    // ========== Compile-time args ==========
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr auto input_accessor_args = TensorAccessorArgs<1>();
    // has_gamma, has_beta follow after input_accessor_args
    // (indices depend on TensorAccessorArgs size, handled by program descriptor)

    // ========== CB indices ==========
    constexpr uint32_t cb_rm_input = 0;  // RM sticks from DRAM
    constexpr uint32_t cb_scaler = 8;    // Reduce scaler (1/W)
    constexpr uint32_t cb_eps = 9;       // Epsilon constant

    // ========== Runtime args ==========
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_blocks = get_arg_val<uint32_t>(1);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(2);
    const uint32_t Wt = get_arg_val<uint32_t>(3);
    const uint32_t W = get_arg_val<uint32_t>(4);
    // gamma_addr and beta_addr at indices 5, 6 (used in later stages)

    // ========== Setup TensorAccessor for input ==========
    const auto input_accessor = TensorAccessor(input_accessor_args, src_addr, stick_size);

    // ========== Fill scaler CB (c_8) with 1/W for reduce_row ==========
    const float scaler_value = 1.0f / static_cast<float>(W);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler>(scaler_value);

    // ========== Main loop: read 32 RM sticks per block ==========
    uint32_t stick_id = start_stick_id;

    for (uint32_t block = 0; block < num_blocks; ++block) {
        // Reserve Wt pages (tile-sized) in cb_rm_input
        cb_reserve_back(cb_rm_input, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_rm_input);

        // Read 32 sticks (one tile-row)
        for (uint32_t stick = 0; stick < 32; ++stick) {
            uint64_t noc_addr = input_accessor.get_noc_addr(stick_id);
            noc_async_read(noc_addr, l1_write_addr, stick_size);
            l1_write_addr += stick_size;
            stick_id++;
        }
        noc_async_read_barrier();

        // Push Wt pages (32 sticks = Wt tiles worth of data)
        cb_push_back(cb_rm_input, Wt);
    }
}
