// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_sc - Writer Kernel
// Writes output tiles from cb_out to DRAM.
// Implements the same logic as dataflow_kernel_lib::write_matmul_tiles
// but without including the full helper header (which causes compilation
// issues when read_matmul_tiles template constexpr is eagerly evaluated).
//
// Compile-time args (positional):
//   TensorAccessorArgs for C starting at index 0
//
// Runtime args:
//   [0] out_addr -- DRAM base address of C
//   [1] Mt       -- tile rows of C
//   [2] Nt       -- tile columns of C
//   [3] batch    -- always 1 for rank-2

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_out = 16;

void kernel_main() {
    uint32_t out_addr = get_arg_val<uint32_t>(0);
    uint32_t Mt = get_arg_val<uint32_t>(1);
    uint32_t Nt = get_arg_val<uint32_t>(2);
    uint32_t batch = get_arg_val<uint32_t>(3);

    constexpr auto s_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(s_args, out_addr, get_tile_size(cb_out));

    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t mt = 0; mt < Mt; ++mt) {
            for (uint32_t nt = 0; nt < Nt; ++nt) {
                uint32_t tile_index = b * Mt * Nt + mt * Nt + nt;
                cb_wait_front(cb_out, 1);
                noc_async_write_tile(tile_index, s, get_read_ptr(cb_out));
                noc_async_write_barrier();
                cb_pop_front(cb_out, 1);
            }
        }
    }
}
