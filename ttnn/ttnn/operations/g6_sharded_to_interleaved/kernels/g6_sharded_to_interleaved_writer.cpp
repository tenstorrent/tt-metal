// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// One issue lane of the owner-writes store. The same source runs on BRISC (lane 0) and NCRISC
// (lane 1); each lane writes its contiguous slice of this core's resident shard straight from the
// shard slots to their interleaved output pages, one NoC write per tile.
//
// cb_shard is bound zero-copy on the input shard and used only as an address binding: no kernel
// pushes or pops it, so get_read_ptr() is the shard base for the whole program.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_shard = get_compile_time_arg_val(0);
    // Row stride of the shard in slots: the full shard width, also on a shorter last width shard.
    constexpr uint32_t shard_cols_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t tensor_cols_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t slot_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(4);
    constexpr auto out_args = TensorAccessorArgs<5>();

    constexpr uint32_t shard_row_bytes = shard_cols_tiles * slot_bytes;

    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);

    uint32_t rt = 0;
    const uint32_t num_blocks = get_arg_val<uint32_t>(rt++);

    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        DeviceZoneScopedN("store_block");
        // bind_resident_block: block origin in the tile grid, block width, and this lane's range of
        // local (row-major over block_extent_cols) tile indices.
        const uint32_t a0 = get_arg_val<uint32_t>(rt++);
        const uint32_t b0 = get_arg_val<uint32_t>(rt++);
        const uint32_t block_extent_cols = get_arg_val<uint32_t>(rt++);
        const uint32_t lane_start = get_arg_val<uint32_t>(rt++);
        const uint32_t lane_tiles = get_arg_val<uint32_t>(rt++);

        const auto out = TensorAccessor(out_args, out_addr);

        uint32_t r = lane_start / block_extent_cols;
        uint32_t c = lane_start - r * block_extent_cols;
        uint32_t src_row_addr = get_read_ptr(cb_shard) + r * shard_row_bytes;
        uint32_t dst_row_page = (a0 + r) * tensor_cols_tiles + b0;

        // store_block: whole pages only — a bfloat8_b tile carries its shared exponents, and an
        // interleaved neighbour page lives in another bank, so pages are neither split nor merged.
        for (uint32_t i = 0; i < lane_tiles; ++i) {
            noc_async_write_page(dst_row_page + c, out, src_row_addr + c * slot_bytes, page_bytes);
            if (++c == block_extent_cols) {
                c = 0;
                src_row_addr += shard_row_bytes;
                dst_row_page += tensor_cols_tiles;
            }
        }
    }

    {
        DeviceZoneScopedN("retire_writes");
        noc_async_write_barrier();
    }
}
