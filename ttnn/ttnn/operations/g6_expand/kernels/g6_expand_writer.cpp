// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_expand writer (BRISC, NoC1), every core of the rectangle: for each published pattern block,
// write core_replica_rows x this_block_pattern_tiles whole 2 KB output pages from the resident
// pattern pages, with no barrier inside the block.
//
// store_block uses raw Noc::async_write + TensorAccessor: kernel_lib has no DRAM page writer.
// The stateful write path is not used because it pins the destination NoC coordinates, and
// consecutive pages of an interleaved tensor live in different DRAM banks.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/debug/kernel_profiler.hpp"

namespace {

constexpr uint32_t cb_pattern_tiles_id = get_compile_time_arg_val(0);
constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
constexpr uint32_t block_pattern_tiles = get_compile_time_arg_val(2);
constexpr uint32_t tensor_pattern_wt = get_compile_time_arg_val(3);
constexpr uint32_t tensor_replica_rows = get_compile_time_arg_val(4);
constexpr auto out_args = TensorAccessorArgs<5>();

constexpr uint32_t image_pages = tensor_replica_rows * tensor_pattern_wt;

// Output page of (pattern p, replica row r): image(p) * image_pages + r * Wt + col(p). Replica row
// outer, pattern inner, so consecutive writes of a multi-pattern block hit different banks.
template <typename Accessor>
FORCE_INLINE void store_block(
    const Noc& noc,
    const Accessor& out,
    const CircularBuffer& cb,
    uint32_t first_pattern,
    uint32_t num_patterns,
    uint32_t replica_start,
    uint32_t core_replica_rows) {
    const uint32_t first_image_base = (first_pattern / tensor_pattern_wt) * image_pages;
    const uint32_t first_col = first_pattern % tensor_pattern_wt;
    for (uint32_t r = replica_start; r < replica_start + core_replica_rows; ++r) {
        uint32_t row_base = first_image_base + r * tensor_pattern_wt;
        uint32_t col = first_col;
        for (uint32_t j = 0; j < num_patterns; ++j) {
            noc.async_write<NocOptions::DEFAULT, tile_bytes>(
                cb, out, tile_bytes, {.offset_bytes = j * tile_bytes}, {.page_id = row_base + col});
            if (++col == tensor_pattern_wt) {
                col = 0;
                row_base += image_pages;
            }
        }
    }
}

}  // namespace

void kernel_main() {
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t pattern_start = get_arg_val<uint32_t>(0);
    const uint32_t core_pattern_tiles = get_arg_val<uint32_t>(1);
    const uint32_t replica_start = get_arg_val<uint32_t>(2);
    const uint32_t core_replica_rows = get_arg_val<uint32_t>(3);
    const uint32_t num_blocks = (core_pattern_tiles + block_pattern_tiles - 1) / block_pattern_tiles;

    Noc noc;
    CircularBuffer cb(cb_pattern_tiles_id);
    const auto out = TensorAccessor(out_args, out_addr);

    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const uint32_t block_offset = block_idx * block_pattern_tiles;
        const uint32_t remaining = core_pattern_tiles - block_offset;
        const uint32_t this_block_pattern_tiles = remaining < block_pattern_tiles ? remaining : block_pattern_tiles;
        {
            DeviceZoneScopedN("G6_CB_WAIT");
            cb.wait_front(block_pattern_tiles);
        }
        {
            DeviceZoneScopedN("G6_STORE");
            store_block(
                noc, out, cb, pattern_start + block_offset, this_block_pattern_tiles, replica_start, core_replica_rows);
        }
        {
            // The reader overwrites these pages on its next reserve: they must have left L1 first.
            DeviceZoneScopedN("G6_RELEASE");
            noc.async_writes_flushed();
            cb.pop_front(block_pattern_tiles);
        }
    }
    {
        DeviceZoneScopedN("G6_DRAIN");
        noc.async_write_barrier();
    }
}
