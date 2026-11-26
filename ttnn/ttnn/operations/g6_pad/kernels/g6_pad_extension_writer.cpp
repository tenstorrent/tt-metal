// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_pad regime E (tile_padding_extension), BRISC / NoC1. The only kernel of the regime.
//
// Writes c into rows [in_row_rem, out_row_end) of the last tile row of every slice of x, in
// place, as <= 3 face-row segments per tile sourced from the same offsets of a fill page. Reads
// nothing: the logical rows of those tiles stay where they are. No barrier inside the block; one
// write barrier at the end so no transaction outlives the kernel.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_fill_tile = get_compile_time_arg_val(0);
    constexpr uint32_t elem_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t num_segments = get_compile_time_arg_val(3);
    constexpr uint32_t seg_offset[3] = {
        get_compile_time_arg_val(4), get_compile_time_arg_val(6), get_compile_time_arg_val(8)};
    constexpr uint32_t seg_bytes[3] = {
        get_compile_time_arg_val(5), get_compile_time_arg_val(7), get_compile_time_arg_val(9)};
    constexpr uint32_t out_tensor_row_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t out_tensor_col_tiles = get_compile_time_arg_val(11);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(12);
    constexpr auto x_args = TensorAccessorArgs<13>();
    static_assert(num_segments >= 1 && num_segments <= 3);

    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t fill_bits = get_common_arg_val<uint32_t>(1);
    const uint32_t core_work_begin = get_arg_val<uint32_t>(0);
    const uint32_t core_work_tiles = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks_this_core = get_arg_val<uint32_t>(2);

    const auto x_accessor = TensorAccessor(x_args, x_addr, tile_bytes);

    // Self-owned scratch: written through the write pointer once, never pushed or popped.
    const uint32_t fill_base = get_write_ptr(cb_fill_tile);
    {
        DeviceZoneScopedN("G6PAD_FILL_BUILD");
#pragma GCC unroll 3
        for (uint32_t k = 0; k < num_segments; ++k) {
            dataflow_kernel_lib::fill_l1_range<elem_bytes>(fill_base + seg_offset[k], seg_bytes[k], fill_bits);
        }
    }

    // Touched tile w = (slice, tile column); its page is in the slice's last tile row.
    constexpr uint32_t slice_page_jump = (out_tensor_row_tiles - 1) * out_tensor_col_tiles;
    uint32_t tc = core_work_begin % out_tensor_col_tiles;
    const uint32_t slice = core_work_begin / out_tensor_col_tiles;
    uint32_t page = (slice * out_tensor_row_tiles + out_tensor_row_tiles - 1) * out_tensor_col_tiles + tc;

    uint32_t remaining = core_work_tiles;
    for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
        DeviceZoneScopedN("G6PAD_FILL_ROWS");
        const uint32_t extent = remaining < block_tiles ? remaining : block_tiles;
        for (uint32_t i = 0; i < extent; ++i) {
            const uint64_t page_noc_addr = x_accessor.get_noc_addr(page);
#pragma GCC unroll 3
            for (uint32_t k = 0; k < num_segments; ++k) {
                noc_async_write(fill_base + seg_offset[k], page_noc_addr + seg_offset[k], seg_bytes[k]);
            }
            ++page;
            if (++tc == out_tensor_col_tiles) {
                tc = 0;
                page += slice_page_jump;
            }
        }
        remaining -= extent;
    }

    {
        DeviceZoneScopedN("G6PAD_DRAIN");
        noc_async_write_barrier();
    }
}
