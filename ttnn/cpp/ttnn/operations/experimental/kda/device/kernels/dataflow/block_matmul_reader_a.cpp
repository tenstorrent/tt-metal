// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

// Block matmul, A side: the first core of each row reads that row's A blocks and multicasts them along the row; the
// others receive. Blocks are double-buffered in the circular buffer.
void kernel_main() {
    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t m_block = get_compile_time_arg_val(1);
    constexpr uint32_t k_block = get_compile_time_arg_val(2);
    constexpr uint32_t k_blocks = get_compile_time_arg_val(3);
    constexpr uint32_t a_row_tiles = get_compile_time_arg_val(4);  // Kt
    constexpr bool sends = get_compile_time_arg_val(5) != 0;
    constexpr auto a_args = TensorAccessorArgs<6>();
    constexpr auto mc = dataflow_kernel_lib::
        McastArgs<get_named_compile_time_arg_val("a_ct_offset"), get_named_compile_time_arg_val("a_rt_offset")>();
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t m0 = get_arg_val<uint32_t>(1);
    const uint32_t m_valid = get_arg_val<uint32_t>(2);
    constexpr uint32_t block_tiles = m_block * k_block;
    const uint32_t tile_bytes = get_tile_size(cb_a);
    const auto a = TensorAccessor(a_args, a_addr, tile_bytes);
    Noc noc;
    if constexpr (sends) {
        // The row's first core reads straight into its two-block circular buffer: block kb + 1 into one slot while
        // block kb goes out of the other to the same address on the receivers.
        auto pipe = mc.sender(noc);
        const uint32_t base = get_write_ptr(cb_a);
        const uint32_t block_bytes = block_tiles * tile_bytes;
        const auto issue = [&](uint32_t kb) {
            const uint32_t slot = base + (kb & 1) * block_bytes;
            for (uint32_t r = 0; r < m_valid; ++r) {
                for (uint32_t k = 0; k < k_block; ++k) {
                    noc_async_read_page(
                        (m0 + r) * a_row_tiles + kb * k_block + k, a, slot + (r * k_block + k) * tile_bytes);
                }
            }
        };
        cb_reserve_back(cb_a, block_tiles);
        issue(0);
        for (uint32_t kb = 0; kb < k_blocks; ++kb) {
            noc_async_read_barrier();
            const uint32_t slot = base + (kb & 1) * block_bytes;
            pipe.send(slot, slot, block_bytes);
            cb_push_back(cb_a, block_tiles);
            if (kb + 1 < k_blocks) {
                cb_reserve_back(cb_a, block_tiles);
                issue(kb + 1);
            }
        }
    } else {
        auto pipe = mc.receiver(noc);
        for (uint32_t kb = 0; kb < k_blocks; ++kb) {
            cb_reserve_back(cb_a, block_tiles);
            pipe.receive(kb);
            cb_push_back(cb_a, block_tiles);
        }
    }
}
