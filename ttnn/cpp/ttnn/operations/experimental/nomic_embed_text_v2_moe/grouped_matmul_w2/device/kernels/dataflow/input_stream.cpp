// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RISCV_1 (NOC1) of every compute core and of the dedicated input senders: the input stream of the core's lane, and,
// where writes_here, the core's outputs.
// A block is input_pieces pieces of piece_tiles; the input ring holds input_slots pieces, so a block need not wait
// for a whole free block. Every core pushes the same pieces in the same order, so piece g of the lane sits at slot
// g % input_slots on every core. The lane's sender reads each piece from the lane's own cores (the input's lane-local
// layout: piece j of block b on lane core (input_pieces * (b / num_lanes) + j) % lane_cores, as shard
// (input_pieces * b + j) / (num_lanes * lane_cores) there), and multicasts it once every receiver has granted it;
// after a block's last piece it sets the receivers' count of blocks. Counters only grow:
//   granted_base + s (on the sender, one a slot): +1 from each receiver for each piece of slot s whose slot is free;
//   arrived (on the receivers): the blocks delivered.
// A receiver grants a piece once its slot is free, up to the end of the block after the one it waits for. Grants are
// counted per slot: a receiver can run up to input_slots - 1 pieces ahead of the piece the sender waits for, so one
// counter's total could be met by a fast receiver's later grants while a slow one still reads that slot. Granting
// piece g + input_slots needs piece g popped, which the sender has not sent while it waits for g, so each slot's
// counter holds at most one grant ahead from any receiver.
// A sender that computes reads into its own ring and pushes to its own compute; a dedicated one reads into the same
// slots. Each piece is multicast as it lands, unlinked: the sender waits for reads and grants between them, whose
// responses travel the lane's links a linked multicast would hold. The block's last piece is linked, ahead of the
// count.
// Output block d - 2 is written once block d is delivered: block d's grants needed compute past block d - 1's start,
// so block d - 2 is packed and the write waits only for its pack.

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "output_rows.hpp"
#include "w2_schedule.hpp"

void kernel_main() {
    constexpr uint32_t num_experts = get_named_compile_time_arg_val("num_experts");
    constexpr uint32_t num_lanes = get_named_compile_time_arg_val("num_lanes");
    constexpr uint32_t lane_cores = get_named_compile_time_arg_val("lane_cores");
    constexpr uint32_t k_tiles = get_named_compile_time_arg_val("k_tiles");
    constexpr uint32_t n_tiles = get_named_compile_time_arg_val("n_tiles");
    constexpr uint32_t input_pieces = get_named_compile_time_arg_val("input_pieces");
    constexpr uint32_t input_slots = get_named_compile_time_arg_val("input_slots");
    constexpr uint32_t counts_page_bytes = get_named_compile_time_arg_val("counts_page_bytes");
    constexpr uint32_t input_cb_id = get_named_compile_time_arg_val("input_cb");
    constexpr uint32_t output_cb_id = get_named_compile_time_arg_val("output_cb");
    constexpr uint32_t granted_base = get_named_compile_time_arg_val("granted_base");
    constexpr auto output_args = TensorAccessorArgs<0>();
    constexpr auto counts_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile_bytes = get_tile_size(input_cb_id);
    constexpr uint32_t piece_tiles = k_tiles / input_pieces;
    constexpr uint32_t piece_bytes = piece_tiles * tile_bytes;
    constexpr uint32_t shard_cores = num_lanes * lane_cores;
    constexpr uint32_t segment_bytes = n_tiles * get_tile_size(output_cb_id) / grouped_matmul::kBlockRows;
    // Reads in flight a sender keeps, one transaction id each.
    constexpr uint32_t read_depth = input_pieces;
    static_assert(k_tiles % input_pieces == 0 && input_pieces <= 8);
    // A block's pieces fit the ring with room for the next block's first, and the lagged write's block is then
    // always past its last K step.
    static_assert(input_slots > input_pieces && input_slots <= 2 * input_pieces);

    uint32_t arg = 0;
    const uint32_t input_address = get_arg_val<uint32_t>(arg++);
    const uint32_t output_address = get_arg_val<uint32_t>(arg++);
    const uint32_t counts_address = get_arg_val<uint32_t>(arg++);
    const uint32_t input_blocks = get_arg_val<uint32_t>(arg++);
    const uint32_t lane = get_arg_val<uint32_t>(arg++);
    const uint32_t share = get_arg_val<uint32_t>(arg++);
    // 0: a receiver; 1: the lane's sender, computing; 2: a dedicated sender.
    const uint32_t role = get_arg_val<uint32_t>(arg++);
    const bool writes_here = get_arg_val<uint32_t>(arg++) == 1;
    const uint32_t sender_x = get_arg_val<uint32_t>(arg++);
    const uint32_t sender_y = get_arg_val<uint32_t>(arg++);
    const uint32_t start_x = get_arg_val<uint32_t>(arg++);
    const uint32_t start_y = get_arg_val<uint32_t>(arg++);
    const uint32_t end_x = get_arg_val<uint32_t>(arg++);
    const uint32_t end_y = get_arg_val<uint32_t>(arg++);
    const uint32_t receivers = get_arg_val<uint32_t>(arg++);
    // The lane's cores' NoC coordinates, (x, y) in lane order.
    const uint32_t lane_cores_arg = arg;

    const Noc noc;
    DataflowBuffer input_cb(input_cb_id);
    DataflowBuffer counts_cb(get_named_compile_time_arg_val("counts_cb"));
    Semaphore<> arrived(get_named_compile_time_arg_val("arrived"));
    auto granted = [](uint32_t slot) { return Semaphore<>(granted_base + slot); };

    const auto counts_accessor = TensorAccessor(counts_args, counts_address);
    noc.async_read(counts_accessor, counts_cb, counts_page_bytes, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    const grouped_matmul_w2::Walk<num_experts> walk(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counts_cb.get_write_ptr()), input_blocks);
    grouped_matmul_w2::LaneWalk<num_experts, num_lanes> blocks{walk, lane};
    const uint32_t ring_base = input_cb.get_write_ptr();
    auto slot_address = [&](uint32_t piece) { return ring_base + piece % input_slots * piece_bytes; };

    if (role != 0) {
        // A sender. The lane's pieces in order: piece g is piece g % input_pieces of the lane's block g /
        // input_pieces; reads go up to read_depth pieces ahead, each into its slot once the slot is free.
        constexpr uint32_t kKnownBlocks = 2;  // the block being sent and the next
        uint32_t block_ids[kKnownBlocks] = {0, 0};
        uint32_t known_blocks = 0;
        auto learn_next = [&]() {
            uint32_t block = 0;
            if (blocks.next(block)) {
                block_ids[known_blocks % kKnownBlocks] = block;
                ++known_blocks;
            }
        };
        learn_next();
        learn_next();
        auto known = [&](uint32_t index) { return index < known_blocks; };
        const bool computing = role == 1;
        uint32_t issued = 0;
        uint32_t multicast = 0;  // pieces whose multicast is issued
        uint32_t flushed = 0;    // pieces whose multicast has left the core
        // Piece `target`'s slot is free once the piece input_slots back has left the core and, on a sender that
        // computes, compute has popped it; `current`, the piece being sent, and those after it are not pushed yet.
        auto slot_free = [&](uint32_t target, uint32_t current) {
            if (target >= input_slots && target - input_slots >= flushed) {
                if (target - input_slots >= multicast) {
                    return false;
                }
                noc.async_writes_flushed();
                flushed = multicast;
            }
            return !computing || input_cb.pages_reservable_at_back((target - current + 1) * piece_tiles);
        };
        auto issue = [&](uint32_t piece) {
            const uint32_t block = block_ids[piece / input_pieces % kKnownBlocks];
            const uint32_t piece_in_block = piece % input_pieces;
            const uint32_t shard = input_pieces * block + piece_in_block;
            const uint32_t core = (input_pieces * (block / num_lanes) + piece_in_block) % lane_cores;
            noc.async_read<NocOptions::TXN_ID>(
                UnicastEndpoint{},
                CoreLocalMem<uint32_t>(slot_address(piece)),
                piece_bytes,
                {.noc_x = get_arg_val<uint32_t>(lane_cores_arg + 2 * core),
                 .noc_y = get_arg_val<uint32_t>(lane_cores_arg + 2 * core + 1),
                 .addr = input_address + (shard / shard_cores) * piece_bytes},
                {},
                {.trid = piece % read_depth + 1});
            ++issued;
        };
        for (uint32_t piece = 0; known(piece / input_pieces); ++piece) {
            // This piece (waiting for its slot if need be), then read-ahead into free slots.
            while (issued <= piece) {
                while (!slot_free(issued, piece)) {
                }
                issue(issued);
            }
            while (issued < piece + read_depth && known(issued / input_pieces) && slot_free(issued, piece)) {
                issue(issued);
            }
            granted(piece % input_slots).wait_min((piece / input_slots + 1) * receivers);
            noc.async_read_barrier<NocOptions::TXN_ID>({.trid = piece % read_depth + 1});
            if (computing) {
                input_cb.push_back(piece_tiles);
            }
            const bool last_piece = piece % input_pieces == input_pieces - 1;
            noc.async_write_multicast(
                CoreLocalMem<uint32_t>(slot_address(piece)),
                MulticastEndpoint{},
                piece_bytes,
                receivers,
                {},
                {.noc_x_start = start_x,
                 .noc_y_start = start_y,
                 .noc_x_end = end_x,
                 .noc_y_end = end_y,
                 .addr = slot_address(piece)},
                last_piece);
            ++multicast;
            if (last_piece) {
                noc.async_writes_flushed();
                flushed = multicast;
                arrived.set(piece / input_pieces + 1);
                arrived.set_multicast(noc, start_x, start_y, end_x, end_y, receivers);
                // The block after next is known only now: its id replaces this block's.
                learn_next();
            }
        }
        noc.async_write_barrier();
        noc.async_atomic_barrier();
        return;
    }

    // A receiver: grants each piece once its slot is free, up to the end of the block after the one it waits for and
    // never past the lane's last piece.
    uint32_t lane_pieces = 0;
    {
        grouped_matmul_w2::LaneWalk<num_experts, num_lanes> count_walk{walk, lane};
        uint32_t block = 0;
        while (count_walk.next(block)) {
            lane_pieces += input_pieces;
        }
    }
    DataflowBuffer output_cb(output_cb_id);
    const auto output = TensorAccessor(output_args, output_address);
    grouped_matmul_w2::LaneWalk<num_experts, num_lanes> outputs{walk, lane};
    uint32_t output_block = 0;
    bool output_pending = writes_here && outputs.next(output_block);
    auto write_next = [&]() {
        output_cb.wait_front(n_tiles);
        grouped_matmul_w2::write_output_block<n_tiles, segment_bytes>(
            noc, output, output_cb, output_block, share * segment_bytes);
        output_pending = outputs.next(output_block);
    };
    uint32_t delivered = 0;
    uint32_t grants = 0;
    uint32_t block = 0;
    while (blocks.next(block)) {
        const uint32_t grant_end =
            (delivered + 2) * input_pieces < lane_pieces ? (delivered + 2) * input_pieces : lane_pieces;
        while (arrived.value() < delivered + 1) {
            if (grants < grant_end &&
                input_cb.pages_reservable_at_back((grants - delivered * input_pieces + 1) * piece_tiles)) {
                granted(grants % input_slots).up(noc, sender_x, sender_y, 1);
                ++grants;
            }
        }
        // Piece by piece: the ring's write pointer wraps only when a push lands exactly on its end.
        for (uint32_t piece = 0; piece < input_pieces; ++piece) {
            input_cb.push_back(piece_tiles);
        }
        ++delivered;
        if (output_pending && delivered >= 3) {
            write_next();
        }
    }
    while (output_pending) {
        write_next();
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
