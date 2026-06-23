// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

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

    Noc noc;
    CircularBuffer out_rm_cb(out_rm_cb_id);

    const uint32_t num_blocks = (num_rows + BLOCK_T - 1) / BLOCK_T;

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        out_rm_cb.wait_front(block_num_tiles);
        for (uint32_t i = 0; i < BLOCK_T; ++i) {
            const uint32_t local = blk * BLOCK_T + i;
            if (local >= num_rows) {
                break;
            }
            const uint32_t out_page = row_start + local;
            // Write the real C channels (stick_bytes) to the output page, dropping the C_pad tail.
            noc.async_write(
                out_rm_cb, dst, stick_bytes, {.offset_bytes = i * padded_stick_bytes}, {.page_id = out_page});
        }
        noc.async_write_barrier();
        out_rm_cb.pop_front(block_num_tiles);
    }
}
