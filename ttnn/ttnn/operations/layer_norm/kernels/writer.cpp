// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// LayerNorm - Writer Kernel
//
// Waits for untilized output (RM sticks) in cb_untilize_out, then writes
// each of the 32 sticks per tile-row to DRAM using TensorAccessor.
// Follows the untilize writer pattern from:
//   ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/
//   writer_unary_stick_layout_split_rows_single_core.cpp
//
// Compile-time args:
//   [0]  cb_id_out           -- output CB index (18 = c_18 = cb_untilize_out)
//   [1]  output_stick_size   -- bytes per output row (W * elem_size)
//   [2]  tile_height         -- 32
//   [3]  Ht                  -- tile-rows (N / 32)
//   [4]  num_tiles_per_block -- Wt (tiles per row)
//   [5]  block_width_size    -- Wt * 32 * elem_size
//   [6+] TensorAccessorArgs  -- bank mapping for output buffer
//
// Runtime args:
//   [0]  dst_addr  -- output buffer DRAM base address

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // ---- Compile-time args ----
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr uint32_t output_stick_size = get_compile_time_arg_val(1);
    constexpr uint32_t tile_height = get_compile_time_arg_val(2);
    constexpr uint32_t Ht = get_compile_time_arg_val(3);
    constexpr uint32_t num_tiles_per_block = get_compile_time_arg_val(4);
    constexpr uint32_t block_width_size = get_compile_time_arg_val(5);

    // TensorAccessor for the output tensor (args start at index 6)
    constexpr auto dst_args = TensorAccessorArgs<6>();

    // ---- Runtime args ----
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);

    const auto dst = TensorAccessor(dst_args, dst_addr, output_stick_size);

    // ---- Write Ht tile-rows back to DRAM ----
    // Each tile-row produces tile_height (32) RM sticks of output_stick_size bytes.
    // The untilize helper laid out all 32 sticks sequentially in L1.
    for (uint32_t tile_row = 0; tile_row < Ht; ++tile_row) {
        // Wait for the compute kernel to finish untilizing one tile-row
        cb_wait_front(cb_id_out, num_tiles_per_block);

        uint32_t l1_read_ptr = get_read_ptr(cb_id_out);

        // Pre-compute DRAM destination addresses for all 32 sticks
        uint64_t base_dst_noc_addr[32];
        for (uint32_t k = 0; k < tile_height; ++k) {
            uint32_t stick_id = tile_row * tile_height + k;
            base_dst_noc_addr[k] = get_noc_addr(stick_id, dst);
        }

        // Issue async writes: one strip per stick row
        for (uint32_t k = 0; k < tile_height; ++k) {
            noc_async_write(l1_read_ptr, base_dst_noc_addr[k], output_stick_size);
            l1_read_ptr += output_stick_size;
        }

        noc_async_write_barrier();
        cb_pop_front(cb_id_out, num_tiles_per_block);
    }
}
