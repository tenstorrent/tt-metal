// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reads, per output page (b, h, s_blk), the D/32 tiles that hold head h of keys
// [32 * s_blk, 32 * s_blk + 32) in the tiled (B, S, H*D) value.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t in_cb_index = get_compile_time_arg_val(0);
constexpr uint32_t TILES_PER_PAGE = get_compile_time_arg_val(1);  // D / 32
constexpr uint32_t NUM_HEADS = get_compile_time_arg_val(2);
constexpr uint32_t PAGES_PER_HEAD = get_compile_time_arg_val(3);  // ceil(S / 32)
constexpr uint32_t TILE_NBYTES = get_compile_time_arg_val(4);
constexpr auto in_args = TensorAccessorArgs<5>();

void kernel_main() {
    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t first_page = get_arg_val<uint32_t>(1);
    const uint32_t num_pages = get_arg_val<uint32_t>(2);

    const auto in_acc = TensorAccessor(in_args, in_addr, TILE_NBYTES);
    Noc noc;
    CircularBuffer in_cb(in_cb_index);

    for (uint32_t p = first_page; p < first_page + num_pages; ++p) {
        const uint32_t s_blk = p % PAGES_PER_HEAD;
        const uint32_t bh = p / PAGES_PER_HEAD;
        const uint32_t h = bh % NUM_HEADS;
        const uint32_t b = bh / NUM_HEADS;
        // Tile order of (B, S, H*D) is row-major over (b, s_blk, column tile),
        // and a page's s_blk is its tile row: PAGES_PER_HEAD is also the
        // tile-row count per batch, which the host validation enforces.
        const uint32_t first_tile = (b * PAGES_PER_HEAD + s_blk) * (NUM_HEADS * TILES_PER_PAGE) + h * TILES_PER_PAGE;

        in_cb.reserve_back(TILES_PER_PAGE);
        const uint32_t dst_l1 = in_cb.get_write_ptr();
        for (uint32_t k = 0; k < TILES_PER_PAGE; ++k) {
            noc.async_read(
                in_acc,
                CoreLocalMem<uint32_t>(dst_l1 + k * TILE_NBYTES),
                TILE_NBYTES,
                {.page_id = first_tile + k, .offset_bytes = 0},
                {.offset_bytes = 0});
        }
        noc.async_read_barrier();
        in_cb.push_back(TILES_PER_PAGE);
    }
}
