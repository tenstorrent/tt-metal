// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

// Block matmul, W side and output: the first core of each column reads that column's W blocks and multicasts them down
// the column; every core then writes its finished output block, subblock by subblock as compute packs it.
void kernel_main() {
    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t m_block = get_compile_time_arg_val(2);
    constexpr uint32_t n_block = get_compile_time_arg_val(3);
    constexpr uint32_t k_block = get_compile_time_arg_val(4);
    constexpr uint32_t k_blocks = get_compile_time_arg_val(5);
    constexpr uint32_t w_row_tiles = get_compile_time_arg_val(6);    // Nt of W
    constexpr uint32_t out_row_tiles = get_compile_time_arg_val(7);  // Nt of the output
    constexpr bool sends = get_compile_time_arg_val(8) != 0;
    constexpr uint32_t sub_w = get_compile_time_arg_val(9);
    constexpr uint32_t sub_h = get_compile_time_arg_val(10);
    constexpr auto w_args = TensorAccessorArgs<11>();
    constexpr auto out_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto mc = dataflow_kernel_lib::
        McastArgs<get_named_compile_time_arg_val("w_ct_offset"), get_named_compile_time_arg_val("w_rt_offset")>();
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t m0 = get_arg_val<uint32_t>(2);
    const uint32_t m_valid = get_arg_val<uint32_t>(3);
    const uint32_t n0 = get_arg_val<uint32_t>(4);
    const uint32_t n_valid = get_arg_val<uint32_t>(5);
    const uint32_t out_col0 = get_arg_val<uint32_t>(6);
    constexpr uint32_t block_tiles = k_block * n_block;
    const uint32_t w_tile_bytes = get_tile_size(cb_w);
    const auto w = TensorAccessor(w_args, w_addr, w_tile_bytes);
    Noc noc;
    if constexpr (sends) {
        // As the A reader: straight into the two-block circular buffer, the next block while this one goes out.
        auto pipe = mc.sender(noc);
        const uint32_t base = get_write_ptr(cb_w);
        const uint32_t block_bytes = block_tiles * w_tile_bytes;
        const auto issue = [&](uint32_t kb) {
            const uint32_t slot = base + (kb & 1) * block_bytes;
            for (uint32_t k = 0; k < k_block; ++k) {
                for (uint32_t n = 0; n < n_valid; ++n) {
                    noc_async_read_page(
                        (kb * k_block + k) * w_row_tiles + n0 + n, w, slot + (k * n_block + n) * w_tile_bytes);
                }
            }
        };
        cb_reserve_back(cb_w, block_tiles);
        issue(0);
        for (uint32_t kb = 0; kb < k_blocks; ++kb) {
            noc_async_read_barrier();
            const uint32_t slot = base + (kb & 1) * block_bytes;
            pipe.send(slot, slot, block_bytes);
            cb_push_back(cb_w, block_tiles);
            if (kb + 1 < k_blocks) {
                cb_reserve_back(cb_w, block_tiles);
                issue(kb + 1);
            }
        }
    } else {
        auto pipe = mc.receiver(noc);
        for (uint32_t kb = 0; kb < k_blocks; ++kb) {
            cb_reserve_back(cb_w, block_tiles);
            pipe.receive(kb);
            cb_push_back(cb_w, block_tiles);
        }
    }
    // The finished block: tile (r, c) at page r * n_block + c. Subblocks one tile high arrive in that order, each
    // written as it lands.
    const uint32_t out_tile_bytes = get_tile_size(cb_out);
    const auto out = TensorAccessor(out_args, out_addr, out_tile_bytes);
    if constexpr (sub_h != 1) {
        cb_wait_front(cb_out, m_block * n_block);
    }
    const uint32_t base = get_read_ptr(cb_out);
    for (uint32_t r = 0; r < m_block; ++r) {
        for (uint32_t c0 = 0; c0 < n_block; c0 += sub_w) {
            if constexpr (sub_h == 1) {
                cb_wait_front(cb_out, r * n_block + c0 + sub_w);
            }
            for (uint32_t c = c0; c < c0 + sub_w; ++c) {
                if (r < m_valid && c < n_valid) {
                    noc_async_write_page(
                        (m0 + r) * out_row_tiles + out_col0 + n0 + c, out, base + (r * n_block + c) * out_tile_bytes);
                }
            }
        }
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, m_block * n_block);
}
