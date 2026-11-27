// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_unsqueeze mover: the op's only kernel. Row 0 of each input tile becomes column 0 of the
// matching output tile, bit for bit.
//
// Per tile: face row 0 of input faces 0 and 1 (lanes 0..15, 16..31) is read into the bases of
// output faces 0 and 2 of a staging page; in each of those faces word j of face row 0 is copied
// to row j of column 0; the whole page is written as the output tile.
//
// Raw dataflow API, no kernel_lib helper:
// - copy_tile_column (tile_column_dataflow.hpp) applies one offset to source and destination, so
//   it copies column to column; here the source is a face row.
// - noc_async_read_page always transfers the accessor's full page; the load reads one face row.
// - The lane move is RISC-V word copies, not NoC: a NoC transfer keeps the source offset modulo
//   16 B, and lane j moves from byte 4j to byte 64j.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

// In one face whose face row 0 holds the lanes: word j of row 0 -> row j of column 0, j >= 1.
// Word 0 is already in place. Sources (row 0) and destinations (rows 1..) are disjoint.
template <uint32_t face_row_words>
FORCE_INLINE void face_row_to_column(volatile tt_l1_ptr uint32_t* face) {
    for (uint32_t j = 1; j < face_row_words; ++j) {
        face[j * face_row_words] = face[j];
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_w_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t staging_depth_blocks = get_compile_time_arg_val(2);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t face_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t face_row_bytes = get_compile_time_arg_val(5);
    constexpr auto input_args = TensorAccessorArgs<6>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();

    constexpr uint32_t face_row_words = face_row_bytes / sizeof(uint32_t);
    // Lanes 16..31 come from input face 1 and belong in output face 2.
    constexpr uint32_t upper_lanes_src_offset = face_bytes;
    constexpr uint32_t upper_lanes_land_offset = 2 * face_bytes;
    constexpr uint32_t slot_bytes = block_w_tiles * tile_bytes;
    static_assert(face_row_bytes % sizeof(uint32_t) == 0, "a lane is one 32-bit word");
    static_assert(face_bytes / face_row_bytes >= face_row_words, "a face needs one row per lane of its face row");
    static_assert(staging_depth_blocks >= 1, "at least one staging slot");

    uint32_t input_addr;
    uint32_t output_addr;
    uint32_t start_tile;
    uint32_t core_w_tiles;
    uint32_t num_blocks;
    uint32_t staging_base;
    {
        DeviceZoneScopedN("G6U-SETUP");
        input_addr = get_common_arg_val<uint32_t>(0);
        output_addr = get_common_arg_val<uint32_t>(1);
        start_tile = get_arg_val<uint32_t>(0);
        core_w_tiles = get_arg_val<uint32_t>(1);
        num_blocks = (core_w_tiles + block_w_tiles - 1) / block_w_tiles;
        staging_base = get_write_ptr(cb_output_tiles);
    }
    // Idle tail cores of the rectangle issue no NoC transaction, so nothing is left to drain.
    if (core_w_tiles == 0) {
        return;
    }

    const auto input = TensorAccessor(input_args, input_addr, tile_bytes);

    auto block_extent = [&](uint32_t block_idx) -> uint32_t {
        return block_idx + 1 < num_blocks ? block_w_tiles : core_w_tiles - block_idx * block_w_tiles;
    };
    auto slot_base = [&](uint32_t block_idx) -> uint32_t {
        return staging_base + (block_idx % staging_depth_blocks) * slot_bytes;
    };

    auto load_row_block = [&](uint32_t block_idx) {
        if (block_idx >= staging_depth_blocks) {
            // The slot still sources the writes of block block_idx - staging_depth_blocks.
            DeviceZoneScopedN("G6U-FLUSH-WAIT");
            noc_async_writes_flushed();
        }
        DeviceZoneScopedN("G6U-LOAD-ISSUE");
        const uint32_t first_tile = start_tile + block_idx * block_w_tiles;
        const uint32_t extent = block_extent(block_idx);
        uint32_t page_addr = slot_base(block_idx);
        for (uint32_t i = 0; i < extent; ++i, page_addr += tile_bytes) {
            const uint32_t tile_id = first_tile + i;
            noc_async_read(input.get_noc_addr(tile_id, 0), page_addr, face_row_bytes);
            noc_async_read(
                input.get_noc_addr(tile_id, upper_lanes_src_offset),
                page_addr + upper_lanes_land_offset,
                face_row_bytes);
        }
    };

    load_row_block(0);
    // Built while block 0's reads are in flight.
    const auto output = TensorAccessor(output_args, output_addr, tile_bytes);

    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        {
            DeviceZoneScopedN("G6U-LOAD-WAIT");
            noc_async_read_barrier();
        }
        if constexpr (staging_depth_blocks > 1) {
            if (block_idx + 1 < num_blocks) {
                load_row_block(block_idx + 1);
            }
        }
        {
            DeviceZoneScopedN("G6U-MOVE-STORE-ISSUE");
            const uint32_t first_tile = start_tile + block_idx * block_w_tiles;
            const uint32_t extent = block_extent(block_idx);
            uint32_t page_addr = slot_base(block_idx);
            for (uint32_t i = 0; i < extent; ++i, page_addr += tile_bytes) {
                auto* page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page_addr);
                face_row_to_column<face_row_words>(page);
                face_row_to_column<face_row_words>(page + upper_lanes_land_offset / sizeof(uint32_t));
                // Orders this tile's stores before the NoC reads the page (BH L1 cache).
                invalidate_l1_cache();
                noc_async_write_page(first_tile + i, output, page_addr);
            }
        }
        if constexpr (staging_depth_blocks == 1) {
            if (block_idx + 1 < num_blocks) {
                load_row_block(block_idx + 1);
            }
        }
    }

    {
        DeviceZoneScopedN("G6U-STORE-WAIT");
        noc_async_write_barrier();
    }
}
