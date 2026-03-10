// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Group Norm - Reader Kernel
// Reads RM sticks from DRAM into cb_input_rm (CB 0) for tilize.
// Packs 32 sticks (one tile-row) into Ct tile-sized pages per push.
// Also reads gamma/beta tiles and fills eps/scaler CBs (later stages).

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // ========== COMPILE-TIME ARGS ==========
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr auto input_accessor_args = TensorAccessorArgs<1>();
    constexpr auto gamma_accessor_args = TensorAccessorArgs<input_accessor_args.next_compile_time_args_offset()>();
    constexpr auto beta_accessor_args = TensorAccessorArgs<gamma_accessor_args.next_compile_time_args_offset()>();

    // ========== RUNTIME ARGS ==========
    uint32_t arg_idx = 0;
    const uint32_t input_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t beta_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_sticks = get_arg_val<uint32_t>(arg_idx++);  // N * HW
    const uint32_t Ct = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t block_width_size = get_arg_val<uint32_t>(arg_idx++);  // Ct * 64
    const uint32_t gamma_num_tiles = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t beta_num_tiles = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t packed_eps = get_arg_val<uint32_t>(arg_idx++);

    // ========== CB INDICES ==========
    constexpr uint32_t cb_input_rm = 0;
    constexpr uint32_t cb_gamma = 2;
    constexpr uint32_t cb_beta = 3;
    constexpr uint32_t cb_eps = 4;
    constexpr uint32_t cb_scaler = 5;

    // ========== SETUP TENSOR ACCESSORS ==========
    auto input_accessor = TensorAccessor(input_accessor_args, input_addr, stick_size);

    // ========== MAIN LOOP: READ RM STICKS ==========
    // Process 32 sticks at a time (one tile-row), pushing Ct pages per block.
    // Each tile-row has 32 sticks * stick_size = Ct * tile_page_size bytes.
    uint32_t stick_id = 0;
    const uint32_t num_tile_rows = num_sticks / 32;

    for (uint32_t tr = 0; tr < num_tile_rows; ++tr) {
        cb_reserve_back(cb_input_rm, Ct);
        uint32_t l1_addr = get_write_ptr(cb_input_rm);

        for (uint32_t s = 0; s < 32; ++s) {
            uint64_t noc_addr = input_accessor.get_noc_addr(stick_id);
            noc_async_read(noc_addr, l1_addr, stick_size);
            l1_addr += stick_size;
            stick_id++;
        }

        noc_async_read_barrier();
        cb_push_back(cb_input_rm, Ct);
    }
}
