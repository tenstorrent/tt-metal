// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// LayerNorm - Reader Kernel (Stage 1 Stub: data_pipeline)
//
// Reads row-major input sticks from DRAM and batches them as tile-sized RM pages
// into cb_in for the tilize compute helper.
//
// Stage 1 (data_pipeline): Only reads input data. Scaler/eps/weight/bias
// are left as stubs (filled in stages 2-4).
//
// Compile-time args:
//   [0]  stick_size         -- bytes per input row (W * elem_size)
//   [1+] TensorAccessorArgs -- interleaved DRAM bank mapping for input
//
// Runtime args:
//   [0]  src_addr          -- input buffer DRAM base address
//   [1]  num_rows          -- N (total rows)
//   [2]  Wt               -- tiles per row (W / 32)
//   [3]  block_width_size  -- W * elem_size per tile-row block
//   [4]  has_weight        -- 1 if gamma provided
//   [5]  has_bias          -- 1 if beta provided
//   [6]  weight_addr       -- weight buffer address (0 if none)
//   [7]  bias_addr         -- bias buffer address (0 if none)
//   [8]  eps_bits          -- epsilon as bit-cast uint32

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // ---- Runtime args ----
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);
    const uint32_t Wt = get_arg_val<uint32_t>(2);
    const uint32_t block_width_size = get_arg_val<uint32_t>(3);
    // has_weight, has_bias, weight_addr, bias_addr, eps_bits -- unused in Stage 1

    // ---- Compile-time args ----
    constexpr uint32_t stick_size = get_compile_time_arg_val(0);

    // TensorAccessor for the input tensor (args start at index 1)
    constexpr auto src_args = TensorAccessorArgs<1>();
    const auto src = TensorAccessor(src_args, src_addr, stick_size);

    // CB index for input
    constexpr uint32_t cb_in = 0;  // RM input pages

    // ---- Per tile-row: read 32 sticks and batch them as Wt tile-sized pages ----
    // This implements the tilize reader pattern from tilize_analysis.md:
    // For each of the 32 sticks in the tile-row, issue an async read of
    // block_width_size bytes. All reads target sequential L1 addresses inside
    // the CB, giving the compute kernel contiguous RM data for Wt tiles.
    const uint32_t num_tile_rows = num_rows / 32;

    for (uint32_t tile_row = 0; tile_row < num_tile_rows; ++tile_row) {
        // Reserve space for Wt tile-sized pages in cb_in
        cb_reserve_back(cb_in, Wt);

        uint32_t l1_write_ptr = get_write_ptr(cb_in);

        // Pre-compute NoC addresses for the 32 sticks in this tile-row
        uint64_t base_src_noc_addr[32];
        for (uint32_t k = 0; k < 32; ++k) {
            uint32_t stick_id = tile_row * 32 + k;
            base_src_noc_addr[k] = get_noc_addr(stick_id, src);
        }

        // Issue async reads: for each of the 32 sticks, read block_width_size bytes
        for (uint32_t k = 0; k < 32; ++k) {
            noc_async_read(base_src_noc_addr[k], l1_write_ptr, block_width_size);
            l1_write_ptr += block_width_size;
        }

        noc_async_read_barrier();
        cb_push_back(cb_in, Wt);
    }
}
