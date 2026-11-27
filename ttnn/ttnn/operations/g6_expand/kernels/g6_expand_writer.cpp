// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_expand writer (BRISC, NoC1), on every core of a strip: store_block.
// Each built page of the block stays resident while it is written once per out tile row this core
// owns; x page q = n*Wt + c becomes out tile (n, r, c), page id (n*Ht + r)*Wt + c.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/debug/kernel_profiler.hpp"

constexpr uint32_t cb_row_bcast_tiles = get_compile_time_arg_val(0);
constexpr uint32_t page_bytes = get_compile_time_arg_val(1);
constexpr uint32_t block_pages = get_compile_time_arg_val(2);
constexpr uint32_t Wt = get_compile_time_arg_val(3);
constexpr uint32_t Ht = get_compile_time_arg_val(4);
constexpr auto out_args = TensorAccessorArgs<5>();

void kernel_main() {
    const uint32_t group_q_start = get_arg_val<uint32_t>(0);
    const uint32_t group_pages = get_arg_val<uint32_t>(1);
    const uint32_t core_r_start = get_arg_val<uint32_t>(2);
    const uint32_t core_rows_tiles = get_arg_val<uint32_t>(3);

    Noc noc;
    CircularBuffer cb(cb_row_bcast_tiles);
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(0), page_bytes);
    const uint32_t core_r_end = core_r_start + core_rows_tiles;

    // (n, c) of the first x page of the current block.
    uint32_t block_n = group_q_start / Wt;
    uint32_t block_c = group_q_start - block_n * Wt;
    for (uint32_t remaining = group_pages; remaining > 0;) {
        const uint32_t pages = remaining < block_pages ? remaining : block_pages;
        {
            DeviceZoneScopedN("G6E-store-wait");
            cb.wait_front(block_pages);
        }
        {
            DeviceZoneScopedN("G6E-store");
            for (uint32_t r = core_r_start; r < core_r_end; ++r) {
                uint32_t n = block_n;
                uint32_t c = block_c;
                for (uint32_t k = 0; k < pages; ++k) {
                    noc.async_write(
                        cb, out, page_bytes, {.offset_bytes = k * page_bytes}, {.page_id = (n * Ht + r) * Wt + c});
                    if (++c == Wt) {
                        c = 0;
                        ++n;
                    }
                }
            }
            // The slot may be refilled once every write has left L1, not once it has landed.
            noc.async_writes_flushed();
            cb.pop_front(block_pages);
        }
        block_c += pages;
        while (block_c >= Wt) {
            block_c -= Wt;
            ++block_n;
        }
        remaining -= pages;
    }
    DeviceZoneScopedN("G6E-drain");
    noc.async_write_barrier();
}
