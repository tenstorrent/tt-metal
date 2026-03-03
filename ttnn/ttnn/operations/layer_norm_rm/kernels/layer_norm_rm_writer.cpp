// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm - Writer Kernel
// Runs on RISCV_1 (NCRISC), writes RM sticks to DRAM via NOC1.
//
// Compile-time args:
//   [0]    stick_size        - bytes per output RM stick (W * sizeof(bfloat16))
//   [1..]  TensorAccessorArgs(output)
//
// Runtime args:
//   [0] dst_addr        - output buffer DRAM address
//   [1] num_rows        - total tile-rows to process
//   [2] Wt              - tiles per row (W / 32)
//   [3] start_stick_id  - first output stick id for this core
//
// CB usage:
//   c_16 (cb_out_rm) - consume Wt pages per tile-row (32 RM sticks from compute)

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t TILE_HEIGHT = 32;

// CB indices
constexpr uint32_t cb_out_rm = 16;

void kernel_main() {
    // ---- Compile-time args ----
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);
    constexpr auto output_args = TensorAccessorArgs<1>();

    // ---- Runtime args ----
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);
    const uint32_t Wt = get_arg_val<uint32_t>(2);
    const uint32_t start_stick_id = get_arg_val<uint32_t>(3);

    // ---- TensorAccessor ----
    const auto output_accessor = TensorAccessor(output_args, dst_addr, stick_size);

    // ---- Drain output CB ----
    // For each tile-row block: wait for Wt tiles in cb_out_rm,
    // read 32 sticks from L1 and write them to DRAM.
    uint32_t stick_id = start_stick_id;
    for (uint32_t row = 0; row < num_rows; ++row) {
        cb_wait_front(cb_out_rm, Wt);
        uint32_t l1_read_addr = get_read_ptr(cb_out_rm);

        for (uint32_t s = 0; s < TILE_HEIGHT; ++s) {
            uint64_t dst_noc_addr = get_noc_addr(stick_id, output_accessor);
            noc_async_write(l1_read_addr, dst_noc_addr, stick_size);
            l1_read_addr += stick_size;
            stick_id++;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out_rm, Wt);
    }
}
