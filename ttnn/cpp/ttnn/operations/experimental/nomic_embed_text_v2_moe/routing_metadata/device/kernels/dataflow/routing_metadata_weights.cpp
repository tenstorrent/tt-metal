// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Copies the core's tile rows of the window's top-k weights into combine_weights, (1, top_k,
// padded_tokens, 1) FLOAT32 TILE: column 0 of tile (k, tile row) holds choice k's weights for those
// 32 tokens; padding tokens and columns 1-31 hold 0. A 32-bit copy, so every weight stays bit-exact.

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "ckernel.h"
#include "routing_metadata_faces.hpp"

namespace {

using namespace routing_metadata;

constexpr uint32_t kFaceRows = 16;
constexpr uint32_t kTileWords = 32 * 32;
constexpr uint32_t kTileBytes = kTileWords * kWordBytes;
// Rows whose weights are loaded before any is stored, so the L1 load latencies overlap.
constexpr uint32_t kRowGroup = 8;

// Word of element (row, 0) in a 32 x 32 tile.
constexpr uint32_t column_zero_word(uint32_t row) {
    return row < kFaceRows ? row * 16 : 2 * kFaceWords + (row - kFaceRows) * 16;
}

}  // namespace

void kernel_main() {
    constexpr uint32_t top_k = get_named_compile_time_arg_val("top_k");
    constexpr auto values_args = TensorAccessorArgs<0>();
    constexpr auto combine_weights_args = TensorAccessorArgs<values_args.next_compile_time_args_offset()>();

    const uint32_t values_address = get_arg_val<uint32_t>(0);
    const uint32_t combine_weights_address = get_arg_val<uint32_t>(1);
    const uint32_t token_start = get_arg_val<uint32_t>(2);
    const uint32_t num_tokens = get_arg_val<uint32_t>(3);
    const uint32_t tile_rows = get_arg_val<uint32_t>(4);
    const uint32_t first_tile_row = get_arg_val<uint32_t>(5);
    const uint32_t num_tile_rows = get_arg_val<uint32_t>(6);
    // Tile rows of weight tiles held at once, at least 2 when the core has 2 or more tile rows.
    const uint32_t weight_slots = get_arg_val<uint32_t>(7);
    const uint32_t core_index = get_arg_val<uint32_t>(8);

    const auto values = TensorAccessor(values_args, values_address);
    const auto combine_weights = TensorAccessor(combine_weights_args, combine_weights_address);

    Noc noc;
    CircularBuffer weight_tiles_cb(get_named_compile_time_arg_val("weight_tiles_cb"));
    const uint32_t weight_tiles = weight_tiles_cb.get_write_ptr();
    const uint32_t value_faces = CircularBuffer(get_named_compile_time_arg_val("value_faces_cb")).get_write_ptr();

    read_face_pairs(noc, values, value_faces, token_start / kBlockRows + first_tile_row, num_tile_rows);
    // Columns 1-31 stay zero throughout: each tile row rewrites column 0 only.
    noc.async_write_zeros(weight_tiles_cb, weight_slots * top_k * kTileBytes);
    noc.write_zeros_l1_barrier();
    noc.async_read_barrier();

    volatile tt_l1_ptr uint32_t* tiles = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(weight_tiles);
    // Pages are spread over the DRAM banks in page order, and every core runs this loop at about the
    // same pace. Starting each core at a different tile row spreads the cores' writes over the banks
    // instead of sending them all to the same few at once.
    const uint32_t first_index = core_index % num_tile_rows;
    for (uint32_t step = 0; step < num_tile_rows; ++step) {
        const uint32_t index = (first_index + step) % num_tile_rows;
        const uint32_t slot = step % weight_slots;
        const volatile tt_l1_ptr uint32_t* weights =
            reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(value_faces + index * kFacePairBytes);
        volatile tt_l1_ptr uint32_t* slot_tiles = tiles + slot * top_k * kTileWords;
        const uint32_t tile_row = first_tile_row + index;
        const uint32_t remaining_tokens = num_tokens - tile_row * kBlockRows;
        const uint32_t valid_rows = remaining_tokens < kBlockRows ? remaining_tokens : kBlockRows;
        for (uint32_t k = 0; k < top_k; ++k) {
#pragma GCC unroll 4
            for (uint32_t first_row = 0; first_row < kBlockRows; first_row += kRowGroup) {
                uint32_t group[kRowGroup];
#pragma GCC unroll 8
                for (uint32_t member = 0; member < kRowGroup; ++member) {
                    group[member] = weights[(first_row + member) * kWordsPerToken + k];
                }
#pragma GCC unroll 8
                for (uint32_t member = 0; member < kRowGroup; ++member) {
                    const uint32_t row = first_row + member;
                    slot_tiles[k * kTileWords + column_zero_word(row)] = row < valid_rows ? group[member] : 0;
                }
            }
        }
        // Order the stores above ahead of the NoC reading them; see routing_metadata_indices.cpp.
        (void)ckernel::load_blocking(&slot_tiles[(top_k - 1) * kTileWords + column_zero_word(kBlockRows - 1)]);

        // Waits for the previous tile row's writes to leave L1, so the next step can refill their
        // slot; they were in flight while this row was built in another slot.
        noc.async_writes_flushed();
        for (uint32_t k = 0; k < top_k; ++k) {
            noc.async_write(
                CoreLocalMem<uint32_t>(weight_tiles + (slot * top_k + k) * kTileBytes),
                combine_weights,
                kTileBytes,
                {},
                {.page_id = k * tile_rows + tile_row});
        }
    }
    noc.async_write_barrier();
}
