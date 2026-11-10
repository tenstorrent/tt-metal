// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The x all-gather (fabric_port_stager / fabric_port_sender): its compile-time args, the row chunks' real row tiles (a
// padded forward), the walk over a chunk's shard and link, where each packet lands (the destination tables) and the
// chips a port tells "started".
#pragma once

#include <cstdint>
#include <type_traits>
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/row_block.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/tile_bytes.hpp"
#ifdef SKIP_PADDING_CHUNKS
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_head/device/kernels/padded_rows.hpp"
#endif

// ---- the compile-time args: the same list for both kernels (head_gather_program.cpp ChipProgram::gather,
// `port_compile_time_args`) ----
constexpr uint32_t k_step_tiles = get_compile_time_arg_val(0);        // a K step's tiles
constexpr uint32_t row_tiles_per_core = get_compile_time_arg_val(1);  // a matmul core's row tiles of a chunk
constexpr uint32_t k_steps_per_shard = get_compile_time_arg_val(2);   // the K steps of one chip's x shard
constexpr uint32_t tiles_per_packet = get_compile_time_arg_val(3);    // a fabric packet's bf16 tiles
constexpr uint32_t chunk_row_tiles = get_compile_time_arg_val(4);     // a row chunk's row tiles
constexpr uint32_t row_chunks = get_compile_time_arg_val(5);          // the call's row chunks
constexpr uint32_t gather_links = get_compile_time_arg_val(6);        // the fabric links a direction gathers on
// CB 0's batch: the stager pushes and the sender pops whole groups of this many packets (the sender's data headers)
constexpr uint32_t packets_per_group = get_compile_time_arg_val(7);
// one arrival increment on the downstream port every this many packets of an entry (and on its last)
constexpr uint32_t packets_per_arrival_increment = get_compile_time_arg_val(8);
constexpr uint32_t engine_columns = get_compile_time_arg_val(9);         // the matmul cores' columns
constexpr uint32_t engine_rows = get_compile_time_arg_val(10);           // the matmul cores' rows
constexpr uint32_t num_family_0_columns = get_compile_time_arg_val(11);  // the x family 0's owner columns
constexpr uint32_t num_family_1_columns = get_compile_time_arg_val(12);  // the x family 1's owner columns
constexpr uint32_t tp_size = get_compile_time_arg_val(13);               // the chips of the TP ring / line
// (the stager's x accessor args follow them)
constexpr uint32_t x_gather_num_compile_time_args = 14;

constexpr uint32_t packets_per_k_step = (k_step_tiles + tiles_per_packet - 1) / tiles_per_packet;
constexpr uint32_t packets_per_row_tile = k_steps_per_shard * packets_per_k_step;

// a core's tiles of block `block` from its counts word: a byte a block, or past TP 4 two (block, count) pairs
// (ReduceScatterPlan::counts_word)
FORCE_INLINE uint32_t tiles_of_block(uint32_t counts_word, uint32_t block) {
    if constexpr (tp_size > 4) {
        return ((counts_word & 0xFF) == block ? (counts_word >> 8) & 0xFF : 0) +
               (((counts_word >> 16) & 0xFF) == block ? counts_word >> 24 : 0);
    } else {
        return (counts_word >> (8 * block)) & 0xFF;
    }
}

// ---- the row chunks' real row tiles ----
// the call's live row chunks (SKIP_PADDING_CHUNKS, a padded forward: up to the last with a real row - the kernel sets
// it at the start of each row block; else every chunk)
inline uint32_t x_gather_num_live_chunks = row_chunks;
#ifdef SKIP_PADDING_CHUNKS
// The common args a padded forward's live chunks come from (head_chip_program.hpp ChipProgram::live_args:
// padded_rows.hpp's [0 .. 9], then [10] the "valid" scratch word), read once in kernel_main
// (read_x_gather_live_chunk_args)
struct XGatherLiveChunkArgs {
    PaddedForwardArgs forward;       // [0 .. 4] where the metadata comes from, my SP row
    uint32_t first_row;              // [5] the call's first row in my SP row (row block 0's)
    uint32_t owner_chunk_row_tiles;  // [8] global_row_tile's own chunk row tiles (0: chunk-major)
    uint32_t owner_row_tiles;        // [9] its own row tiles
    uint32_t scratch_address;        // [10] rounded up to 64 B: the metadata is read into it
};
FORCE_INLINE XGatherLiveChunkArgs read_x_gather_live_chunk_args() {
    XGatherLiveChunkArgs args;
    args.forward = read_padded_forward_args(moe_block_args::common_pointer(0));
    args.first_row = moe_block_args::common(5);
    args.owner_chunk_row_tiles = moe_block_args::common(8);
    args.owner_row_tiles = moe_block_args::common(9);
    args.scratch_address = (moe_block_args::common(10) + 63) & ~63u;
    return args;
}
static_assert(chunk_row_tiles <= 64 && row_chunks <= 16, "a chunk's real row tiles: one 64-bit mask a chunk");
using x_gather_mask_t =
    std::conditional_t<(chunk_row_tiles <= 32), uint32_t, uint64_t>;  // (a register on the 32-bit RISCs when it fits)
inline x_gather_mask_t x_gather_real_row_tiles[row_chunks];  // (a chunk's real chunk-local row tiles, a bit each)
// (each row block) this block's live chunks and their real row tiles
FORCE_INLINE void x_gather_find_live_chunks(const XGatherLiveChunkArgs& args) {
    const PaddedRows padded_rows = fetch_padded_rows(args.forward, args.scratch_address);
    const uint32_t first_row =
        args.first_row + row_block::index() * ROW_BLOCK_PADDED_ROW_STRIDE;  // (row block b's rows)
    x_gather_num_live_chunks = 1;
    for (uint32_t chunk = 0; chunk < row_chunks; ++chunk) {
        x_gather_mask_t real = 0;
        for (uint32_t chunk_row_tile = 0; chunk_row_tile < chunk_row_tiles; ++chunk_row_tile) {
            if (padded_tile_real(
                    padded_rows,
                    first_row,
                    padded_global_row_tile(
                        chunk, chunk_row_tile, chunk_row_tiles, args.owner_chunk_row_tiles, args.owner_row_tiles))) {
                real |= x_gather_mask_t(1) << chunk_row_tile;
            }
        }
        x_gather_real_row_tiles[chunk] = real;
        if (real) {
            x_gather_num_live_chunks = chunk + 1;
        }
    }
}
// is chunk `chunk`'s chunk-local row tile real (a padding tile's x / reduce-scatter payload is not sent);
// x_gather_mask(chunk) once a chunk, the bit tested a packet
FORCE_INLINE bool x_gather_tile_real(uint32_t chunk, uint32_t chunk_row_tile) {
    return (x_gather_real_row_tiles[chunk] >> chunk_row_tile) & 1;
}
FORCE_INLINE x_gather_mask_t x_gather_mask(uint32_t chunk) { return x_gather_real_row_tiles[chunk]; }
#else
constexpr bool x_gather_tile_real(uint32_t, uint32_t) { return true; }
using x_gather_mask_t = uint32_t;
constexpr x_gather_mask_t x_gather_mask(uint32_t) { return ~x_gather_mask_t(0); }
#endif
// (down + reduce-scatter) does matmul core `core`'s sub-block of block `block` in chunk `chunk` hold a real row (its
// rows: its chunk-local row tiles core / engine_columns * row_tiles_per_core + r, in block block's
// chunk_row_tiles / tp_size)
FORCE_INLINE bool x_gather_block_real(uint32_t chunk, uint32_t core, uint32_t block) {
#ifndef SKIP_PADDING_CHUNKS
    return true;  // (every row real - and a latent's blocks are columns)
#endif
    const uint32_t core_row = core / engine_columns;
    for (uint32_t r = 0; r < row_tiles_per_core; ++r) {
        const uint32_t chunk_row_tile = core_row * row_tiles_per_core + r;
        if (chunk_row_tile / (chunk_row_tiles / tp_size) == block && x_gather_tile_real(chunk, chunk_row_tile)) {
            return true;
        }
    }
    return false;  // (computed once a sub-block, not a packet)
}

// ---- the walk ----
// The packets of chunk `chunk`'s shard `shard` on link `link`, in the order both kernels walk them: per row tile of the
// chunk on the link (its chunk-local row's pair: (row / 2) % gather_links - the host counts packets so; part 0: every
// such row, 1 / 2: its even / odd chunk-local rows), per K step of the shard, per packet of tiles_per_packet tiles.
// visit(row_tile, k_step, shard_tile, step_tile, num_tiles, packet_index): the row tile of the call, the K step (the
// shard's steps after the earlier shards'), the packet's first tile within the shard's K and within the step, its
// tiles, and its index in the link's full (part 0) walk of the chunk's shard.
template <typename Visit>
FORCE_INLINE void x_gather_walk(uint32_t chunk, uint32_t shard, uint32_t part, uint32_t link, Visit&& visit) {
    uint32_t link_rows = 0;
    for (uint32_t row_tile = chunk * chunk_row_tiles; row_tile < (chunk + 1) * chunk_row_tiles; ++row_tile) {
        const uint32_t chunk_row_tile = row_tile - chunk * chunk_row_tiles;
        if ((chunk_row_tile / 2) % gather_links != link) {
            continue;
        }
        if (part == 0 || (chunk_row_tile & 1) == part - 1) {
            for (uint32_t step = 0; step < k_steps_per_shard; ++step) {
                for (uint32_t step_tile = 0, packet = 0; step_tile < k_step_tiles;
                     step_tile += tiles_per_packet, ++packet) {
                    visit(
                        row_tile,
                        shard * k_steps_per_shard + step,
                        step * k_step_tiles + step_tile,
                        step_tile,
                        k_step_tiles - step_tile < tiles_per_packet ? k_step_tiles - step_tile : tiles_per_packet,
                        link_rows * packets_per_row_tile + step * packets_per_k_step + packet);
                }
            }
        }
        ++link_rows;
    }
}

// ---- where a packet lands ----
struct XGatherDestination {
    uint32_t noc_x, noc_y, address;
};
// A chip's destination tables in a kernel's args, in place (head_model.cpp HeadModel::tables - the stager's: its own
// chip's, the sender's: the neighbour's): the chip's shard ranks [tp_size]; the family 0 owner columns
// [num_family_0_columns]; the family 1 owner columns [num_family_1_columns]; per engine column (its owned slots, its
// family 1 base slot); per engine column its NoC x; per engine row its NoC y
struct XGatherDestinationTables {
    static constexpr uint32_t num_words =
        tp_size + num_family_0_columns + num_family_1_columns + 3 * engine_columns + engine_rows;
    static constexpr uint32_t columns_at = tp_size + num_family_0_columns + num_family_1_columns;
    volatile tt_l1_ptr uint32_t* words;

    // x K step `k_step` of chunk `chunk`, row tile `row_tile` on the chip: its processing position there (the chip's
    // rank of the step's shard, then the step within the shard), the position's owner core (NoC) and its x-own address
    // (the matmul cores' arena layout) for the packet starting at tile `step_tile` of the step
    XGatherDestination destination(
        uint32_t x_own_address, uint32_t chunk, uint32_t row_tile, uint32_t k_step, uint32_t step_tile) const {
        const uint32_t position = words[k_step / k_steps_per_shard] * k_steps_per_shard + k_step % k_steps_per_shard;
        const uint32_t family = position & 1, index_in_family = position / 2;
        const uint32_t column = family ? words[tp_size + num_family_0_columns + index_in_family % num_family_1_columns]
                                       : words[tp_size + index_in_family % num_family_0_columns];
        const uint32_t owned_slots = words[columns_at + 2 * column],
                       family_1_base_slot = words[columns_at + 2 * column + 1];
        const uint32_t slot = family ? family_1_base_slot + index_in_family / num_family_1_columns
                                     : index_in_family / num_family_0_columns;
        const uint32_t chunk_row_tile = row_tile - chunk * chunk_row_tiles;
        const uint32_t address =
            x_own_address +
            (((chunk * owned_slots + slot) * row_tiles_per_core + chunk_row_tile % row_tiles_per_core) * k_step_tiles +
             step_tile) *
                moe_block_tiles::kBf16TileBytes;
        return {
            words[columns_at + 2 * engine_columns + column],
            words[columns_at + 3 * engine_columns + chunk_row_tile / row_tiles_per_core],
            address};
    }
};

// The chips a port tells "started" (the latent's / the logits' end-to-end fence), in place: per chip (its fabric chip
// id | its mesh id << 16, its port's NoC x | y << 8) - head_gather_program.cpp latent_block / logits_block, `sends`
struct FabricStartedTargets {
    volatile tt_l1_ptr uint32_t* words;
    uint32_t count;
    uint16_t chip_id(uint32_t i) const { return static_cast<uint16_t>(words[2 * i] & 0xFFFF); }
    uint16_t mesh_id(uint32_t i) const { return static_cast<uint16_t>(words[2 * i] >> 16); }
    uint32_t port_noc_x(uint32_t i) const { return words[2 * i + 1] & 0xFF; }
    uint32_t port_noc_y(uint32_t i) const { return (words[2 * i + 1] >> 8) & 0xFF; }
};
