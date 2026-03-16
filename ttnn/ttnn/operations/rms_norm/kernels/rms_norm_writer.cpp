// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// RMS Norm - Writer Kernel
// Runs on RISCV_1 (NCRISC), writes data to DRAM via NOC1
//
// TILE output: wait for tiles from cb_out, write to DRAM via TensorAccessor
// RM output: wait for sticks from cb_out_rm, write sticks to DRAM

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"

// CB indices
constexpr uint32_t cb_out = 16;
constexpr uint32_t cb_out_rm = 17;

// Compile-time args
constexpr uint32_t is_rm_output = get_compile_time_arg_val(0);
constexpr uint32_t output_stick_size = get_compile_time_arg_val(1);

// TensorAccessor args for output start at CT index 2
constexpr auto output_ta_args = TensorAccessorArgs<2>();

void kernel_main() {
    // Runtime args
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_rows = get_arg_val<uint32_t>(1);
    uint32_t Wt = get_arg_val<uint32_t>(2);
    uint32_t num_pages = get_arg_val<uint32_t>(3);

    if constexpr (is_rm_output) {
        // RM output: read sticks from cb_out_rm, write to DRAM
        const auto output_accessor = TensorAccessor(output_ta_args, dst_addr, output_stick_size);

        uint32_t stick_id = 0;
        for (uint32_t row = 0; row < num_rows; ++row) {
            // Wait for Wt pages (one tile-row of untilized sticks)
            cb_wait_front(cb_out_rm, Wt);
            uint32_t l1_read_addr = get_read_ptr(cb_out_rm);

            // Write 32 sticks per tile-row
            for (uint32_t s = 0; s < 32; ++s) {
                uint64_t noc_addr = output_accessor.get_noc_addr(stick_id);
                noc_async_write(l1_read_addr, noc_addr, output_stick_size);
                l1_read_addr += output_stick_size;
                stick_id++;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out_rm, Wt);
        }
    } else {
        // TILE output: write tiles one at a time from cb_out
        uint32_t tile_size = get_tile_size(cb_out);
        const auto output_accessor = TensorAccessor(output_ta_args, dst_addr, tile_size);

        uint32_t tile_id = 0;
        for (uint32_t row = 0; row < num_rows; ++row) {
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                cb_wait_front(cb_out, 1);
                uint32_t l1_read_addr = get_read_ptr(cb_out);
                uint64_t noc_addr = output_accessor.get_noc_addr(tile_id);
                noc_async_write(l1_read_addr, noc_addr, tile_size);
                noc_async_write_barrier();
                cb_pop_front(cb_out, 1);
                tile_id++;
            }
        }
    }
}
