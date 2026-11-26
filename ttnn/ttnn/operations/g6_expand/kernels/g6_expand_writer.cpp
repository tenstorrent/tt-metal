// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_expand writer (BRISC, NoC1). For each resident block of pattern tiles, writes every one of this
// core's replica rows of those patterns to out as whole 2 KB pages: replica row outer, pattern inner,
// no barrier inside the block. A pattern page p (input page) maps to output page
//   (p / Wt) * (R * Wt) + row * Wt + p % Wt,   Wt = tensor_pattern_wt, R = tensor_replica_rows.
//
// store_block uses raw Noc::async_write: kernel_lib has no DRAM page writer.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

namespace {

template <uint32_t tile_bytes, uint32_t tensor_pattern_wt, uint32_t tensor_replica_rows, typename Accessor>
FORCE_INLINE void store_block(
    const Noc& noc,
    const Accessor& out,
    const CircularBuffer& cb,
    uint32_t first_pattern,
    uint32_t num_patterns,
    uint32_t replica_start,
    uint32_t replica_rows) {
    constexpr uint32_t image_pages = tensor_replica_rows * tensor_pattern_wt;
    const uint32_t first_image = first_pattern / tensor_pattern_wt;
    const uint32_t first_col = first_pattern - first_image * tensor_pattern_wt;
    for (uint32_t r = 0; r < replica_rows; ++r) {
        uint32_t row_base = first_image * image_pages + (replica_start + r) * tensor_pattern_wt;
        uint32_t col = first_col;
        for (uint32_t i = 0; i < num_patterns; ++i) {
            noc.async_write<NocOptions::DEFAULT, tile_bytes>(
                cb, out, tile_bytes, {.offset_bytes = i * tile_bytes}, {.page_id = row_base + col});
            if (++col == tensor_pattern_wt) {
                col = 0;
                row_base += image_pages;
            }
        }
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_pattern_tiles_id = get_compile_time_arg_val(0);
    constexpr uint32_t block_pattern_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t tensor_pattern_wt = get_compile_time_arg_val(3);
    constexpr uint32_t tensor_replica_rows = get_compile_time_arg_val(4);
    constexpr auto out_args = TensorAccessorArgs<5>();
    static_assert(tile_bytes <= NOC_MAX_BURST_SIZE);

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t pattern_start = get_arg_val<uint32_t>(1);
    const uint32_t core_pattern_tiles = get_arg_val<uint32_t>(2);
    const uint32_t replica_start = get_arg_val<uint32_t>(3);
    const uint32_t core_replica_rows = get_arg_val<uint32_t>(4);

    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);
    const uint32_t num_blocks = (core_pattern_tiles + block_pattern_tiles - 1) / block_pattern_tiles;

    Noc noc;
    CircularBuffer cb_pattern_tiles(cb_pattern_tiles_id);

    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t block_offset = block_idx * block_pattern_tiles;
        const uint32_t remaining = core_pattern_tiles - block_offset;
        const uint32_t this_block_pattern_tiles = remaining < block_pattern_tiles ? remaining : block_pattern_tiles;

        cb_pattern_tiles.wait_front(block_pattern_tiles);
        store_block<tile_bytes, tensor_pattern_wt, tensor_replica_rows>(
            noc,
            out,
            cb_pattern_tiles,
            pattern_start + block_offset,
            this_block_pattern_tiles,
            replica_start,
            core_replica_rows);
        // The pages may be overwritten once popped, so the NoC must have read them first.
        noc.async_writes_flushed();
        cb_pattern_tiles.pop_front(block_pattern_tiles);
    }

    noc.async_write_barrier();
}
