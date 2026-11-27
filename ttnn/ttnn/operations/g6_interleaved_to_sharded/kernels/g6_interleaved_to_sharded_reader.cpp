// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Owner-fetch issuer: reads this issuer's rectangle of tile pages from DRAM-interleaved input straight
// into their slots of this core's output shard, then retires every read with one barrier.
//
// Raw NoC read + barrier instead of a kernel_lib helper: no helper reads interleaved DRAM pages into an
// arbitrary L1 address (local_copy_helpers are self-aimed L1->L1, compute helpers would route the bytes
// through unpack/DEST/pack and break bit-exactness). No CB exists: every read lands at its final address.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t tensor_tile_cols = get_compile_time_arg_val(0);  // input page row stride
    constexpr uint32_t shard_tile_cols = get_compile_time_arg_val(1);   // shard slot row stride
    constexpr uint32_t image_tile_rows = get_compile_time_arg_val(2);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(3);
    // A tile of an image's last tile row is read as [0, partial_span_bytes) plus, when
    // partial_span2_bytes > 0, [partial_span2_offset, +partial_span2_bytes): same offsets in page and slot.
    constexpr uint32_t partial_span_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t partial_span2_offset = get_compile_time_arg_val(5);
    constexpr uint32_t partial_span2_bytes = get_compile_time_arg_val(6);
    constexpr uint8_t issuer_noc = get_compile_time_arg_val(7);
    constexpr auto input_args = TensorAccessorArgs<8>();

    // One-packet issue path: a span longer than one burst would be truncated.
    static_assert(tile_bytes <= NOC_MAX_BURST_SIZE, "a tile page must fit one NoC packet");
    static_assert(partial_span_bytes > 0 && partial_span_bytes <= tile_bytes, "partial span outside the tile");
    static_assert(partial_span2_offset + partial_span2_bytes <= tile_bytes, "second partial span outside the tile");

    const uint32_t input_base = get_common_arg_val<uint32_t>(0);
    const uint32_t shard_base = get_common_arg_val<uint32_t>(1);

    const uint32_t block_row_begin = get_arg_val<uint32_t>(0);
    const uint32_t block_col_begin = get_arg_val<uint32_t>(1);
    const uint32_t block_tile_rows = get_arg_val<uint32_t>(2);
    const uint32_t block_tile_cols = get_arg_val<uint32_t>(3);
    const uint32_t block_slot_begin = get_arg_val<uint32_t>(4);    // shard slot of the block's first tile
    const uint32_t row_in_image_begin = get_arg_val<uint32_t>(5);  // block_row_begin mod image_tile_rows

    [[maybe_unused]] const auto input = TensorAccessor(input_args, input_base, tile_bytes);

    {
        DeviceZoneScopedN("I2S-FETCH");
        uint32_t row_in_image = row_in_image_begin;
        uint32_t row_page = block_row_begin * tensor_tile_cols + block_col_begin;
        uint32_t row_slot_addr = shard_base + block_slot_begin * tile_bytes;
        for (uint32_t r = 0; r < block_tile_rows; ++r) {
            // Only the last tile row of an image holds padding rows; its spans skip them.
            [[maybe_unused]] const bool partial_row = row_in_image == image_tile_rows - 1;
            [[maybe_unused]] const uint32_t span_bytes = partial_row ? partial_span_bytes : tile_bytes;
            uint32_t page = row_page;
            uint32_t slot_addr = row_slot_addr;
            for (uint32_t c = 0; c < block_tile_cols; ++c) {
#ifndef G6I2S_ABLATE_READS
                const uint64_t page_noc_addr = input.get_noc_addr(page, 0, issuer_noc);
                noc_async_read<tile_bytes>(page_noc_addr, slot_addr, span_bytes, issuer_noc);
                if constexpr (partial_span2_bytes > 0) {
                    if (partial_row) {
                        noc_async_read<tile_bytes>(
                            page_noc_addr + partial_span2_offset,
                            slot_addr + partial_span2_offset,
                            partial_span2_bytes,
                            issuer_noc);
                    }
                }
#endif
                ++page;
                slot_addr += tile_bytes;
            }
            row_page += tensor_tile_cols;
            row_slot_addr += shard_tile_cols * tile_bytes;
            row_in_image = (row_in_image + 1 == image_tile_rows) ? 0 : row_in_image + 1;
        }
    }

    {
        // The next program starts right after this one: no read may still be in flight at return.
        DeviceZoneScopedN("I2S-RETIRE");
        noc_async_read_barrier(issuer_noc);
    }
}
