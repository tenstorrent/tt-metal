// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// g6_expand reader (NCRISC, NoC0). Produces cb_pattern_tiles: one block of pattern tiles, each tile
// holding row 0 of its input tile in all 32 rows.
//
//   injector (top core of a grid column): fetch row 0 of the block's input tiles from DRAM, replicate
//       it into rows 1..31 in place, multicast the block down the column, publish.
//   receiver (every other core of the column): reserve, receive the multicast block, publish.
//
// Every core of a column walks the same blocks and reserves/pushes the nominal block_pattern_tiles
// pages per block, so the multicast lands at the same CB write pointer on every receiver. The ragged
// last block narrows only the fetch, replicate and send size.
//
// fetch_row0_block uses raw Noc::async_read: kernel_lib has no DRAM sub-page reader (local_copy
// helpers address this core's own L1 only).

#include <cstdint>
#include <optional>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tile_constants.h"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

using namespace dataflow_kernel_lib;

namespace {

using tt::constants::FACE_HEIGHT;
using tt::constants::FACE_HW;
using tt::constants::TILE_HW;

// 32x32 tile = four 16x16 faces: face 0 / 1 hold rows 0..15, faces 2 / 3 rows 16..31.
constexpr uint32_t NUM_FACES = TILE_HW / FACE_HW;

// Row 0 of a tile is row 0 of face 0 and row 0 of face 1. Each lands at the same in-page offset it
// has in x: Blackhole DRAM reads need source and destination congruent modulo 64 B.
template <uint32_t tile_bytes, uint32_t face_bytes, uint32_t face_row_bytes, typename Accessor>
FORCE_INLINE void fetch_row0_block(
    const Noc& noc, const Accessor& x, const CircularBuffer& cb, uint32_t first_page, uint32_t num_pages) {
    for (uint32_t i = 0; i < num_pages; ++i) {
        for (uint32_t half = 0; half < 2; ++half) {
            const uint32_t offset = half * face_bytes;
            noc.async_read<NocOptions::DEFAULT, face_row_bytes>(
                x,
                cb,
                face_row_bytes,
                {.page_id = first_page + i, .offset_bytes = offset},
                {.offset_bytes = i * tile_bytes + offset});
        }
    }
    noc.async_read_barrier();
}

// Fill rows 1..31 of every page from its own row 0: face h row 0 -> face h rows 1..15 and face h+2
// rows 0..15, for h in {0, 1}. One stateful-read setup per (page, face half), one barrier per block.
template <uint32_t tile_bytes, uint32_t face_bytes, uint32_t face_row_bytes>
FORCE_INLINE void replicate_block(const Noc& noc, uint32_t block_base, uint32_t num_pages) {
    for (uint32_t i = 0; i < num_pages; ++i) {
        const uint32_t page = block_base + i * tile_bytes;
        for (uint32_t half = 0; half < 2; ++half) {
            const uint32_t row0 = page + half * face_bytes;
            set_read_state<face_row_bytes>(noc, row0);
            for (uint32_t r = 1; r < FACE_HEIGHT; ++r) {
                read_with_state<face_row_bytes>(noc, row0 + r * face_row_bytes, row0);
            }
            const uint32_t lower_face = row0 + 2 * face_bytes;
            for (uint32_t r = 0; r < FACE_HEIGHT; ++r) {
                read_with_state<face_row_bytes>(noc, lower_face + r * face_row_bytes, row0);
            }
        }
    }
    noc.async_read_barrier();
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_pattern_tiles_id = get_compile_time_arg_val(0);
    constexpr uint32_t block_pattern_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr bool is_injector = get_compile_time_arg_val(3) != 0;
    constexpr auto x_args = TensorAccessorArgs<4>();
    constexpr auto pattern_mcast = McastArgs<
        get_named_compile_time_arg_val("pattern_mcast_ct_offset"),
        get_named_compile_time_arg_val("pattern_mcast_rt_offset")>();

    constexpr uint32_t face_bytes = tile_bytes / NUM_FACES;
    constexpr uint32_t face_row_bytes = face_bytes / FACE_HEIGHT;
    static_assert(face_row_bytes <= NOC_MAX_BURST_SIZE);

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t pattern_start = get_arg_val<uint32_t>(1);
    const uint32_t core_pattern_tiles = get_arg_val<uint32_t>(2);

    const uint32_t num_blocks = (core_pattern_tiles + block_pattern_tiles - 1) / block_pattern_tiles;

    Noc noc;
    CircularBuffer cb_pattern_tiles(cb_pattern_tiles_id);

    if constexpr (is_injector) {
        const auto x = TensorAccessor(x_args, x_addr, tile_bytes);
        // Absent when the column has no receivers (one replica group).
        auto sender = pattern_mcast.optional_sender(noc);
        for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
            const uint32_t block_offset = block_idx * block_pattern_tiles;
            const uint32_t remaining = core_pattern_tiles - block_offset;
            const uint32_t this_block_pattern_tiles = remaining < block_pattern_tiles ? remaining : block_pattern_tiles;

            cb_pattern_tiles.reserve_back(block_pattern_tiles);
            const uint32_t block_base = cb_pattern_tiles.get_write_ptr();
            fetch_row0_block<tile_bytes, face_bytes, face_row_bytes>(
                noc, x, cb_pattern_tiles, pattern_start + block_offset, this_block_pattern_tiles);
            replicate_block<tile_bytes, face_bytes, face_row_bytes>(noc, block_base, this_block_pattern_tiles);
            if (sender) {
                sender->send(block_base, block_base, this_block_pattern_tiles * tile_bytes);
            }
            cb_pattern_tiles.push_back(block_pattern_tiles);
        }
    } else {
        auto receiver = pattern_mcast.optional_receiver(noc);
        ASSERT(receiver.has_value());
        for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
            cb_pattern_tiles.reserve_back(block_pattern_tiles);
            receiver->receive();
            cb_pattern_tiles.push_back(block_pattern_tiles);
        }
    }

    // Multicast data/flag writes and ready-signal increments must land before the next program.
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
