// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared by the latent engine's gather ports (gather_port_staging_reader, gather_port_fabric_sender; one port per
// direction and link, in the grid column right of the engine). A port (direction d, link l) sends K half l of its
// blocks (own first, then the ones it relays) to the next chip in direction d, straight into the engine cores' landing
// slots there: share by share (row tile of the block: its step there; engine column c: its share's K tiles), packets
// of up to tiles_per_packet tiles. A row tile's last packet of a group of landed_group_columns columns (the watcher's)
// is a fused write + increment of the watcher's landed word of that step - flushed behind the row tile's other packets
// on that link, so the watcher counts the step landed for its group's other cores. A share (own or relayed) is read
// from this chip's owner core once it landed / was tilized there.

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/moe_block_args.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/tile_bytes.hpp"

// compile-time args (moe_tail_program_factory engine_workload, the ports' list)
constexpr uint32_t row_tiles_per_chunk = get_compile_time_arg_val(0), engine_columns = get_compile_time_arg_val(1);
constexpr uint32_t tiles_per_packet = get_compile_time_arg_val(2);  // (the fabric's max payload / a tile)
constexpr uint32_t max_share_tiles = get_compile_time_arg_val(3);
// (compile-time arg 4: the latent's tiles, not read here)
// packets a group: the reader's barrier, the sender's flush
constexpr uint32_t packets_per_group = get_compile_time_arg_val(5);
constexpr uint32_t num_pairs = get_compile_time_arg_val(6), landed_group_columns = get_compile_time_arg_val(7);
// (the staging CB, the group packet counts CB, the poll scratch CB, and the engine's landing CB - also allocated here,
// at the same address as on the engine cores, so its base is the landing slots' address on every engine core)
constexpr uint32_t staging_cb = get_compile_time_arg_val(8), packet_count_cb = get_compile_time_arg_val(9);
constexpr uint32_t poll_cb = get_compile_time_arg_val(10), landing_cb = get_compile_time_arg_val(11);
constexpr uint32_t row_blocks = get_compile_time_arg_val(12);  // (row blocks: the kernels' own loop)
// landing slots and staging pages: bfloat16 tiles
constexpr uint32_t tile_bytes = moe_block_tiles::kBf16TileBytes;
// the engine's step words (this call's, CB_STEP_WORDS after the landing CB: one address on every engine and port
// core; a step's landed word at 16 B x step) and the zero gate's go a port kernel waits for before it reads or writes
// any of them (engine_common.hpp); BLOCK_GATE_COUNT_SEM: the engine lead's gate count a port reader counts in on
constexpr uint32_t CB_STEP_WORDS = 16, STEP_WORD_BYTES = 16;
constexpr uint32_t PORT_READER_GO_SEM = 0, PORT_WRITER_GO_SEM = 1, BLOCK_GATE_COUNT_SEM = 5;
// (the go: the engine's lead increments it once a gate - a row block's (engine_common.hpp); re-armed here)
FORCE_INLINE void wait_zero_gate(uint32_t semaphore_id) {
    volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(semaphore_id));
    noc_semaphore_wait_min(go, 1);
    noc_semaphore_inc(get_noc_addr(get_semaphore(semaphore_id)), 0u - 1u);
    noc_async_atomic_barrier();
}

// the shares of a unit (block at chunk position `position` on the chip that holds the slots), in the walk both kernels
// follow: callback(pair, step, column, share_k_first, packet_k_first, tiles, last packet of the share); part 0 every
// column, 1 / 2 the first / second half of them; share_k_ranges: the args index of the columns' K ranges (first, end)
template <typename Callback>
FORCE_INLINE void walk(uint32_t part, uint32_t position, uint32_t share_k_ranges, Callback&& callback) {
    for (uint32_t chunk_row_tile = 0; chunk_row_tile < row_tiles_per_chunk; ++chunk_row_tile) {
        const uint32_t row_tile = position * row_tiles_per_chunk + chunk_row_tile;
        for (uint32_t column = 0; column < engine_columns; ++column) {
            if ((part == 1 && column >= engine_columns / 2) || (part == 2 && column < engine_columns / 2)) {
                continue;
            }
            const uint32_t k_first = moe_block_args::arg(share_k_ranges + 2 * column);
            const uint32_t k_end = moe_block_args::arg(share_k_ranges + 2 * column + 1);
            for (uint32_t packet_k_first = k_first; packet_k_first < k_end; packet_k_first += tiles_per_packet) {
                callback(
                    row_tile % num_pairs,
                    row_tile / num_pairs,
                    column,
                    k_first,
                    packet_k_first,
                    (k_end - packet_k_first < tiles_per_packet ? k_end - packet_k_first : tiles_per_packet),
                    packet_k_first + tiles_per_packet >= k_end);
            }
        }
    }
}
