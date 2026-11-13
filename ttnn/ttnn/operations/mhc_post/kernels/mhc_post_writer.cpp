// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// mhc_post writer (RISCV_0, NoC1). store_block: per unit (token-tile row t, column tile c) the n X' tiles
// X'_j -> page t*n*Ct + j*Ct + c, consumed per compute block
// (j-major, unit-minor). Writes stream (flush per block, one barrier at the end).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_mixed_out_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t n = get_compile_time_arg_val(1);
    constexpr uint32_t block_rows = get_compile_time_arg_val(2);
    constexpr uint32_t block_cols = get_compile_time_arg_val(3);
    constexpr uint32_t c_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t out_ring_tiles = get_compile_time_arg_val(5);  // cb_mixed_out_tiles capacity
    constexpr auto out_args = TensorAccessorArgs<6>();

    const uint32_t t_start = get_arg_val<uint32_t>(0);
    const uint32_t t_count = get_arg_val<uint32_t>(1);
    const uint32_t c_start = get_arg_val<uint32_t>(2);
    const uint32_t c_count = get_arg_val<uint32_t>(3);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);

    const uint32_t out_page = get_tile_size(cb_mixed_out_tiles);
    const auto out_acc = TensorAccessor(out_args, out_addr, out_page);
    // A block window may wrap the ring (a ragged pass publishes fewer tiles): address it modulo the ring.
    const uint32_t ring_base = get_read_ptr(cb_mixed_out_tiles);  // nothing consumed yet: the CB base
    uint32_t ring_slot = 0;

    for (uint32_t rb = 0; rb < t_count; rb += block_rows) {
        const uint32_t rows_in_block = (t_count - rb) < block_rows ? (t_count - rb) : block_rows;
        for (uint32_t r = 0; r < rows_in_block; ++r) {
            const uint32_t x_row = (t_start + rb + r) * n * c_tiles;
            for (uint32_t cb0 = 0; cb0 < c_count; cb0 += block_cols) {
                const uint32_t cols_in_pass = (c_count - cb0) < block_cols ? (c_count - cb0) : block_cols;
                // Same block decomposition as the compute: a full pass is one block, a ragged pass blocks of 1.
                const uint32_t b = (cols_in_pass == block_cols) ? block_cols : 1;
                for (uint32_t u0 = 0; u0 < cols_in_pass; u0 += b) {
                    const uint32_t c = c_start + cb0 + u0;
                    cb_wait_front(cb_mixed_out_tiles, b * n);
                    for (uint32_t j = 0; j < n; ++j) {  // block order: j-major, then unit
                        for (uint32_t u = 0; u < b; ++u) {
                            noc_async_write(
                                ring_base + ring_slot * out_page,
                                out_acc.get_noc_addr(x_row + j * c_tiles + c + u),
                                out_page);
                            ring_slot = (ring_slot + 1 == out_ring_tiles) ? 0 : ring_slot + 1;
                        }
                    }
                    noc_async_writes_flushed();  // the block's bytes have left L1: free the slots, no ack wait
                    for (uint32_t k = 0; k < b * n; ++k) {  // single-tile pops: a pop may not cross the ring end
                        cb_pop_front(cb_mixed_out_tiles, 1);
                    }
                }
            }
        }
    }
    noc_async_write_barrier();
}
