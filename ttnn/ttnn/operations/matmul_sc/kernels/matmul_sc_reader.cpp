// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_sc - Reader Kernel (Stage 1: data_pipeline)
// Reads Mt*Nt tiles from A sequentially into cb_in0 (ignoring B).
// Uses TensorAccessor for DRAM reads.
//
// Compile-time args (positional):
//   TensorAccessorArgs for A starting at index 0
//
// Runtime args:
//   [0] in0_addr  -- DRAM base address of A
//   [1] in1_addr  -- (unused in stage 1)
//   [2] Mt        -- tile rows of A/C
//   [3] Kt        -- inner dimension tiles (unused in stage 1)
//   [4] Nt        -- tile columns of B/C (= number of output tile columns)
//   [5] batch     -- always 1 for rank-2

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_in0 = 0;

void kernel_main() {
    uint32_t in0_addr = get_arg_val<uint32_t>(0);
    // in1_addr unused in stage 1
    uint32_t Mt = get_arg_val<uint32_t>(2);
    // Kt unused in stage 1
    uint32_t Nt = get_arg_val<uint32_t>(4);
    uint32_t batch = get_arg_val<uint32_t>(5);

    // TensorAccessor for A, compile-time args start at index 0
    constexpr auto s0_args = TensorAccessorArgs<0>();
    const auto s0 = TensorAccessor(s0_args, in0_addr, get_tile_size(cb_in0));

    // Read Mt*Nt tiles sequentially from A (tile indices 0..Mt*Nt-1)
    uint32_t tile_idx = 0;
    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t mt = 0; mt < Mt; ++mt) {
            for (uint32_t nt = 0; nt < Nt; ++nt) {
                cb_reserve_back(cb_in0, 1);
                noc_async_read_tile(tile_idx, s0, get_write_ptr(cb_in0));
                noc_async_read_barrier();
                cb_push_back(cb_in0, 1);
                tile_idx++;
            }
        }
    }
}
