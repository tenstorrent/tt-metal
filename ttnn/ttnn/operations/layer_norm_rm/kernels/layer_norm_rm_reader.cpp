// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm — Reader Kernel
//
// Reads RM input sticks from DRAM and packs them into tile-sized CB pages
// for the tilize operation in the compute kernel.
//
// Compile-time args:
//   [0]  stick_size          — W * sizeof(bfloat16) bytes per RM stick
//   [1+] TensorAccessorArgs(input) — interleaved input accessor
//
// Runtime args:
//   [0] src_addr       — input buffer address
//   [1] start_stick_id — first RM stick for this core
//   [2] num_sticks     — nblocks * 32 sticks to read
//   [3] gamma_addr     — gamma buffer address (0 if absent)
//   [4] beta_addr      — beta buffer address (0 if absent)

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    // ========== Compile-time args ==========
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr auto input_tensor_args = TensorAccessorArgs<1>();

    // ========== Runtime args ==========
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(1);
    const uint32_t num_sticks = get_arg_val<uint32_t>(2);
    // gamma_addr and beta_addr are runtime args 3 and 4 (used in Stage 4)

    // ========== Constants ==========
    constexpr uint32_t cb_in_rm = 0;
    constexpr uint32_t cb_scaler = 9;

    // Wt = tiles per row = stick_size / (32 * sizeof(bfloat16)) = stick_size / 64
    constexpr uint32_t Wt = stick_size / 64;

    // ========== Setup TensorAccessor ==========
    // For ROW_MAJOR tensor, each page = 1 stick = stick_size bytes
    const auto input_accessor = TensorAccessor(input_tensor_args, src_addr, stick_size);

    // ========== Fill reduce scaler CB (1/W) — program lifetime ==========
    // W = stick_size / sizeof(bfloat16) = stick_size / 2
    constexpr uint32_t W = stick_size / 2;
    constexpr float scaler_val = 1.0f / static_cast<float>(W);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler>(scaler_val);

    // ========== Read RM sticks ==========
    // Process in blocks of 32 sticks (= 1 tile-row)
    const uint32_t nblocks = num_sticks / 32;

    uint32_t stick_id = start_stick_id;

    for (uint32_t block = 0; block < nblocks; ++block) {
        // Reserve Wt tile-sized pages in cb_in_rm
        cb_reserve_back(cb_in_rm, Wt);
        uint32_t l1_write_addr = get_write_ptr(cb_in_rm);

        // Read 32 sticks into the reserved space
        // 32 sticks * stick_size = Wt * tile_size (memory equivalence)
        for (uint32_t s = 0; s < 32; ++s) {
            noc_async_read_page(stick_id, input_accessor, l1_write_addr);
            l1_write_addr += stick_size;
            stick_id++;
        }
        noc_async_read_barrier();

        // Push Wt pages (what compute expects)
        cb_push_back(cb_in_rm, Wt);
    }
}
