// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Writes tiles from output buffer to interleaved DRAM.
 * Kept simple for DIT LayerNorm pre/post all-gather.
 */

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t block_size = get_compile_time_arg_val(1);
    constexpr auto dst_args = TensorAccessorArgs<2>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t tile_row_start = get_arg_val<uint32_t>(1);
    const uint32_t tile_row_end = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = tt::CBIndex::c_8;
    const uint32_t tile_bytes = get_tile_size(cb_out);

    const auto s = TensorAccessor(dst_args, dst_addr, tile_bytes);

    uint32_t output_tile_idx = tile_row_start * Wt;
    for (uint32_t tile_row = tile_row_start; tile_row < tile_row_end; tile_row++) {
        for (uint32_t col_tile = 0; col_tile < Wt; col_tile += block_size) {
            cb_wait_front(cb_out, block_size);
            uint32_t l1_read_addr = get_read_ptr(cb_out);
            for (uint32_t i = 0; i < block_size && col_tile + i < Wt; i++) {
                noc_async_write_tile(output_tile_idx, s, l1_read_addr);
                output_tile_idx++;
                l1_read_addr += tile_bytes;
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_out, block_size);
        }
    }
    noc_async_write_barrier();
}
