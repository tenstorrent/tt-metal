// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t out_rm_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t C = get_compile_time_arg_val(1);
    constexpr uint32_t C_pad = get_compile_time_arg_val(2);
    constexpr uint32_t block_h_tiles = get_compile_time_arg_val(3);

    constexpr auto dst_args = TensorAccessorArgs<4>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_start = get_arg_val<uint32_t>(1);
    const uint32_t num_rows = get_arg_val<uint32_t>(2);

    const auto dst = TensorAccessor(dst_args, dst_addr, C * 4);

    constexpr uint32_t BLOCK_T = block_h_tiles * 32;
    constexpr uint32_t block_w_tiles = C_pad / 32;
    constexpr uint32_t block_num_tiles = block_h_tiles * block_w_tiles;
    constexpr uint32_t stick_bytes = C * 4;
    constexpr uint32_t padded_stick_bytes = C_pad * 4;

    const uint32_t num_blocks = (num_rows + BLOCK_T - 1) / BLOCK_T;

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        cb_wait_front(out_rm_cb_id, block_num_tiles);
        const uint32_t rptr = get_read_ptr(out_rm_cb_id);
        for (uint32_t i = 0; i < BLOCK_T; ++i) {
            const uint32_t local = blk * BLOCK_T + i;
            if (local >= num_rows) {
                break;
            }
            const uint32_t out_page = row_start + local;
            noc_async_write(rptr + i * padded_stick_bytes, dst.get_noc_addr(out_page), stick_bytes);
        }
        noc_async_write_barrier();
        cb_pop_front(out_rm_cb_id, block_num_tiles);
    }
}
