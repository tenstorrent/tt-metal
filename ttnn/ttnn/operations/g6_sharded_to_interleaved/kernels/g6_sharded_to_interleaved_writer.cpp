// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_sharded_to_interleaved — one writer lane.
//
// The core's whole shard is the block. Lane `writer_lane` of `num_writer_lanes` writes one contiguous
// part of it (local row-major tile order), read in place from the resident shard through a CB backed on
// the sharded input buffer, one NoC write per tile to its interleaved output page, then one barrier.
//
// Raw NoC calls: no kernel_lib helper writes resident tiles to interleaved pages by page id
// (local_copy_helpers only does self-aimed L1 reads; mcast has one payload for many receivers).
// The missing helper would be a `write_resident_block_to_pages` dataflow helper.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t cb_input_shard = get_compile_time_arg_val(0);
constexpr uint32_t writer_lane = get_compile_time_arg_val(1);
constexpr uint32_t num_writer_lanes = get_compile_time_arg_val(2);
constexpr uint32_t tensor_tile_rows = get_compile_time_arg_val(3);
constexpr uint32_t tensor_tile_cols = get_compile_time_arg_val(4);
constexpr uint32_t shard_tiles_h = get_compile_time_arg_val(5);
constexpr uint32_t shard_tiles_w = get_compile_time_arg_val(6);
// (shard_tiles_h, 0) for height sharding, (0, shard_tiles_w) for width sharding.
constexpr uint32_t shard_row_step = get_compile_time_arg_val(7);
constexpr uint32_t shard_col_step = get_compile_time_arg_val(8);
constexpr uint32_t num_shards = get_compile_time_arg_val(9);
constexpr bool orientation_row_major = get_compile_time_arg_val(10) != 0;
constexpr uint32_t num_core_ranges = get_compile_time_arg_val(11);
// (x0, y0, x1, y1) per range, in the shard grid's stored order.
constexpr uint32_t core_range_table = 12;
constexpr uint32_t core_range_fields = 4;

constexpr auto output_args = TensorAccessorArgs<core_range_table + core_range_fields * num_core_ranges>();
// Host guarantees the input shard slot stride equals the output page size (whole tiles, same dtype).
constexpr uint32_t page_bytes = output_args.get_aligned_page_size();

static_assert(num_writer_lanes >= 1 && writer_lane < num_writer_lanes);
static_assert(shard_tiles_h >= 1 && shard_tiles_w >= 1);

// Shard index this core holds: position in the buffer's walk of the shard grid (ranges in stored order,
// each walked row-wise or column-wise by orientation). num_shards when the core holds no shard.
FORCE_INLINE uint32_t locate_shard_index() {
    const uint32_t x = get_absolute_logical_x();
    const uint32_t y = get_absolute_logical_y();
    uint32_t cores_before = 0;
    for (uint32_t i = 0; i < num_core_ranges; ++i) {
        const uint32_t entry = core_range_table + core_range_fields * i;
        const uint32_t x0 = kernel_compile_time_args[entry + 0];
        const uint32_t y0 = kernel_compile_time_args[entry + 1];
        const uint32_t x1 = kernel_compile_time_args[entry + 2];
        const uint32_t y1 = kernel_compile_time_args[entry + 3];
        const uint32_t range_w = x1 - x0 + 1;
        const uint32_t range_h = y1 - y0 + 1;
        if (x >= x0 && x <= x1 && y >= y0 && y <= y1) {
            const uint32_t local =
                orientation_row_major ? (y - y0) * range_w + (x - x0) : (x - x0) * range_h + (y - y0);
            const uint32_t k = cores_before + local;
            return k < num_shards ? k : num_shards;
        }
        cores_before += range_w * range_h;
    }
    return num_shards;
}

FORCE_INLINE uint32_t min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }

}  // namespace

void kernel_main() {
    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);

    {
        DeviceZoneScopedN("S2I-STORE");

        // locate_shard_block
        const uint32_t shard_index = locate_shard_index();
        const uint32_t num_blocks_this_core = shard_index < num_shards ? 1 : 0;

        for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
            const uint32_t row0 = shard_index * shard_row_step;
            const uint32_t col0 = shard_index * shard_col_step;
            const uint32_t block_rows = min_u32(shard_tiles_h, tensor_tile_rows - row0);
            const uint32_t block_cols = min_u32(shard_tiles_w, tensor_tile_cols - col0);
            const uint32_t block_tiles = block_rows * block_cols;
            const uint32_t lane_tiles = (block_tiles + num_writer_lanes - 1) / num_writer_lanes;
            const uint32_t lane_begin = min_u32(writer_lane * lane_tiles, block_tiles);
            const uint32_t lane_end = min_u32(lane_begin + lane_tiles, block_tiles);
            if (lane_begin == lane_end) {
                continue;
            }

            const auto output = TensorAccessor(output_args, output_addr, page_bytes);
            const uint32_t shard_base = get_read_ptr(cb_input_shard);

            // store_shard_block: tile (r, c) of the block sits at shard slot r * shard_tiles_w + c (the
            // allocated shard width, also on a ragged last shard) and goes to page (row0 + r) * C + col0 + c.
            uint32_t r = lane_begin / block_cols;
            uint32_t c = lane_begin - r * block_cols;
            uint32_t src_addr = shard_base + (r * shard_tiles_w + c) * page_bytes;
            uint32_t page = (row0 + r) * tensor_tile_cols + col0 + c;
            const uint32_t src_row_wrap = (shard_tiles_w - block_cols + 1) * page_bytes;
            const uint32_t page_row_wrap = tensor_tile_cols - block_cols + 1;
            for (uint32_t t = lane_begin; t < lane_end; ++t) {
                noc_async_write_page(page, output, src_addr);
                if (++c == block_cols) {
                    c = 0;
                    src_addr += src_row_wrap;
                    page += page_row_wrap;
                } else {
                    src_addr += page_bytes;
                    page += 1;
                }
            }
        }
    }

    {
        // drain_writes: no write may be in flight when the kernel returns (the next program starts at once).
        DeviceZoneScopedN("S2I-DRAIN");
        noc_async_write_barrier();
    }
}
