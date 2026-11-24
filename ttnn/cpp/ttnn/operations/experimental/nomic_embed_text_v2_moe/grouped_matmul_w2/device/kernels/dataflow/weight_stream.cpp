// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RISCV_0 (NOC0) of every compute core: the weight stream of the core's share, and compute's schedule. The share's
// sender (its top core) reads each expert's slice from DRAM in pieces, `depth` reads in flight, and multicasts each
// piece south down the column, then the count of pieces sent. Receivers grant each piece once its place in the ring
// is free and push it as its count arrives, so compute starts on a slice before all of it is in. A forwarder in the
// column passes each piece on east to its share's cores in the column lanes once they have granted it; the space it
// forwards from is granted upstream again only after the forward is flushed.
// A slice is two K halves of slice_tiles / 2, each in its own ring (compute/grouped_matmul_w2.cpp), the first of
// first_slots halves and the second of second_slots. Piece j of a slice is in half j / pieces_per_half; every core
// pushes every piece, so a piece sits at the same address on every core.
// Grants go to one of two counters by the piece's parity, and a receiver grants a piece only once the piece two
// before it has arrived, so a counter cannot reach the sender's wait for piece p before every receiver has granted
// p. A receiver grants up to two pieces past the last it has as soon as that one arrives, ahead of a forward.
// Where writes_here (the lanes' computing input senders, whose RISCV_1 is busy with the lane), this RISC also writes
// the core's outputs, and every wait of a receiver polls: a receiver blocked on its ring while outputs pile up would
// stop compute, which frees that ring.
// Chunk j of share s's slice of expert e is the contiguous run of piece_tiles at tile
//     bank * shard_tiles + ((e * pieces_per_slice + j) * bank_sharers + s / 8) * piece_tiles, bank (s + j + e) % 8
// of the prepared weight.

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "ckernel.h"
#include "output_rows.hpp"
#include "w2_schedule.hpp"
#include "ttnn/operations/experimental/nomic_embed_text_v2_moe/pirotski_matmul/device/kernels/dataflow/stream_protocol.hpp"

void kernel_main() {
    constexpr uint32_t num_experts = get_named_compile_time_arg_val("num_experts");
    constexpr uint32_t num_lanes = get_named_compile_time_arg_val("num_lanes");
    constexpr uint32_t slice_tiles = get_named_compile_time_arg_val("slice_tiles");
    constexpr uint32_t piece_tiles = get_named_compile_time_arg_val("piece_tiles");
    constexpr uint32_t first_slots = get_named_compile_time_arg_val("first_slots");
    constexpr uint32_t second_slots = get_named_compile_time_arg_val("second_slots");
    constexpr uint32_t shard_tiles = get_named_compile_time_arg_val("shard_tiles");
    constexpr uint32_t bank_sharers = get_named_compile_time_arg_val("bank_sharers");
    constexpr uint32_t depth = get_named_compile_time_arg_val("depth");
    constexpr uint32_t counts_page_bytes = get_named_compile_time_arg_val("counts_page_bytes");
    constexpr uint32_t first_half_cb_id = get_named_compile_time_arg_val("first_half_cb");
    constexpr uint32_t n_tiles = get_named_compile_time_arg_val("n_tiles");
    constexpr uint32_t output_cb_id = get_named_compile_time_arg_val("output_cb");
    constexpr auto weight_args = TensorAccessorArgs<0>();
    constexpr auto counts_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<counts_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile_bytes = get_tile_size(first_half_cb_id);
    constexpr uint32_t segment_bytes = n_tiles * get_tile_size(output_cb_id) / grouped_matmul::kBlockRows;
    constexpr uint32_t half_tiles = slice_tiles / 2;
    constexpr uint32_t pieces_per_half = half_tiles / piece_tiles;
    constexpr uint32_t pieces_per_slice = 2 * pieces_per_half;
    static_assert(depth >= 1 && depth <= 8);
    static_assert(half_tiles % piece_tiles == 0 && first_slots >= 1 && second_slots >= 1);

    uint32_t arg = 0;
    const uint32_t weight_address = get_arg_val<uint32_t>(arg++);
    const uint32_t counts_address = get_arg_val<uint32_t>(arg++);
    const uint32_t output_address = get_arg_val<uint32_t>(arg++);
    const uint32_t input_blocks = get_arg_val<uint32_t>(arg++);
    const uint32_t lane = get_arg_val<uint32_t>(arg++);
    const uint32_t share = get_arg_val<uint32_t>(arg++);
    const bool is_sender = get_arg_val<uint32_t>(arg++) == 1;
    const bool writes_here = get_arg_val<uint32_t>(arg++) == 1;
    const uint32_t sender_x = get_arg_val<uint32_t>(arg++);
    const uint32_t sender_y = get_arg_val<uint32_t>(arg++);
    // A braced list evaluates in order: start x, start y, end x, end y, count.
    const pirotski_matmul::Multicast column{
        get_arg_val<uint32_t>(arg++),
        get_arg_val<uint32_t>(arg++),
        get_arg_val<uint32_t>(arg++),
        get_arg_val<uint32_t>(arg++),
        get_arg_val<uint32_t>(arg++)};
    const pirotski_matmul::Multicast forward{
        get_arg_val<uint32_t>(arg++),
        get_arg_val<uint32_t>(arg++),
        get_arg_val<uint32_t>(arg++),
        get_arg_val<uint32_t>(arg++),
        get_arg_val<uint32_t>(arg++)};

    const Noc noc;
    DataflowBuffer rings[2] = {
        DataflowBuffer(first_half_cb_id), DataflowBuffer(get_named_compile_time_arg_val("second_half_cb"))};
    constexpr uint32_t slots[2] = {first_slots, second_slots};
    DataflowBuffer counts_cb(get_named_compile_time_arg_val("counts_cb"));
    DataflowBuffer schedule_cb(get_named_compile_time_arg_val("schedule_cb"));
    Semaphore<> granted(get_named_compile_time_arg_val("granted"));
    Semaphore<> granted_odd(get_named_compile_time_arg_val("granted_odd"));
    Semaphore<> arrived(get_named_compile_time_arg_val("arrived"));
    // The forwarder's count word; its own `arrived` keeps the sender's count.
    Semaphore<> forward_flag(get_named_compile_time_arg_val("forward_flag"));

    const auto counts_accessor = TensorAccessor(counts_args, counts_address);
    noc.async_read(counts_accessor, counts_cb, counts_page_bytes, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    const grouped_matmul_w2::Walk<num_experts> walk(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counts_cb.get_write_ptr()), input_blocks);

    // Compute's schedule in the walk order: this core's blocks of each expert, or kSkipExpert for an expert with no
    // rows, whose slice is never sent. Every lane takes every slice its share sends.
    schedule_cb.reserve_back(1);
    auto* schedule = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(schedule_cb.get_write_ptr());
    uint32_t taken[num_experts];
    uint32_t num_taken = 0;
    for (uint32_t position = 0; position < num_experts; ++position) {
        const uint32_t expert = walk.order[position];
        if (walk.num_blocks[expert] > 0) {
            schedule[position] = walk.template lane_blocks<num_lanes>(position, lane).count;
            taken[num_taken++] = expert;
        } else {
            schedule[position] = grouped_matmul::kSkipExpert;
        }
    }
    // A store can retire before L1 has it; a blocking load of the last word returns once all have landed.
    (void)ckernel::load_blocking(&schedule[num_experts - 1]);
    schedule_cb.push_back(1);

    const uint32_t bases[2] = {rings[0].get_write_ptr(), rings[1].get_write_ptr()};
    const uint32_t pieces = num_taken * pieces_per_slice;
    uint32_t pushed[2] = {0, 0};
    auto half_of = [](uint32_t piece) { return piece % pieces_per_slice / pieces_per_half; };
    // Piece p's place in its ring is free once compute has popped that ring up to the piece's end, a ring back:
    // reserve_back(n) with n its end less what was pushed there.
    auto piece_request = [&](uint32_t piece) {
        const uint32_t half = half_of(piece);
        return (piece / pieces_per_slice) * half_tiles + (piece % pieces_per_half + 1) * piece_tiles - pushed[half];
    };
    auto piece_address = [&](uint32_t piece) {
        const uint32_t half = half_of(piece);
        return bases[half] + ((piece / pieces_per_slice) % slots[half]) * half_tiles * tile_bytes +
               (piece % pieces_per_half) * piece_tiles * tile_bytes;
    };
    auto push_piece = [&](uint32_t piece) {
        rings[half_of(piece)].push_back(piece_tiles);
        pushed[half_of(piece)] += piece_tiles;
    };

    if (!is_sender) {
        DataflowBuffer output_cb(output_cb_id);
        const auto output = TensorAccessor(output_args, output_address);
        grouped_matmul_w2::LaneWalk<num_experts, num_lanes> outputs{walk, lane};
        uint32_t output_block = 0;
        bool output_pending = writes_here && outputs.next(output_block);
        auto try_write = [&]() {
            if (output_pending && output_cb.pages_available_at_front(n_tiles)) {
                grouped_matmul_w2::write_output_block<n_tiles, segment_bytes>(
                    noc, output, output_cb, output_block, share * segment_bytes);
                output_pending = outputs.next(output_block);
            }
        };
        uint32_t grants = 0;
        auto grant = [&](uint32_t piece) {
            while (!rings[half_of(piece)].pages_reservable_at_back(piece_request(piece))) {
                try_write();
            }
            (piece % 2 == 0 ? granted : granted_odd).up(noc, sender_x, sender_y, 1);
            ++grants;
        };
        for (uint32_t piece = 0; piece < pieces; ++piece) {
            if (grants == piece) {
                grant(piece);
            }
            if (grants == piece + 1 && grants < pieces &&
                rings[half_of(grants)].pages_reservable_at_back(piece_request(grants))) {
                grant(grants);
            }
            while (arrived.value() < piece + 1) {
                try_write();
            }
            push_piece(piece);
            // With piece p in, this core waits for p + 1, so it may grant up to p + 2 now, ahead of its forward:
            // granted next iteration, p + 2 would wait for that forward and hold up the sender's multicast.
            while (grants <= piece + 2 && grants < pieces &&
                   rings[half_of(grants)].pages_reservable_at_back(piece_request(grants))) {
                grant(grants);
            }
            // Its downstream cores grant this core as receivers grant the sender. The forward is flushed before the
            // loop moves on, and the space it was read from is granted to the sender only in a later iteration.
            if (forward.count > 0) {
                (piece % 2 == 0 ? granted : granted_odd).wait_min((piece / 2 + 1) * forward.count);
                const uint32_t address = piece_address(piece);
                noc.async_write_multicast(
                    CoreLocalMem<uint32_t>(address),
                    MulticastEndpoint{},
                    piece_tiles * tile_bytes,
                    forward.count,
                    {},
                    {.noc_x_start = forward.start_x,
                     .noc_y_start = forward.start_y,
                     .noc_x_end = forward.end_x,
                     .noc_y_end = forward.end_y,
                     .addr = address},
                    true);
                noc.async_writes_flushed();
                forward_flag.set(piece + 1);
                forward_flag.relay_multicast(
                    noc, arrived, forward.start_x, forward.start_y, forward.end_x, forward.end_y, forward.count);
            }
        }
        while (output_pending) {
            output_cb.wait_front(n_tiles);
            try_write();
        }
        noc.async_write_barrier();
        noc.async_atomic_barrier();
        return;
    }

    const auto weight = TensorAccessor(weight_args, weight_address);
    // Read-ahead takes a piece only once its place is already free: waiting for it would hold up the current
    // piece's multicast until this core's compute has popped the ring that far.
    auto issuable = [&](uint32_t piece) {
        const uint32_t request = piece_request(piece);
        return request <= slots[half_of(piece)] * half_tiles && rings[half_of(piece)].pages_reservable_at_back(request);
    };
    auto issue = [&](uint32_t piece) {
        rings[half_of(piece)].reserve_back(piece_request(piece));
        const uint32_t expert = taken[piece / pieces_per_slice];
        const uint32_t chunk = piece % pieces_per_slice;
        const uint32_t bank = (share + chunk + expert) % 8;
        const uint32_t tile =
            bank * shard_tiles + ((expert * pieces_per_slice + chunk) * bank_sharers + share / 8) * piece_tiles;
        noc.async_read<NocOptions::TXN_ID>(
            weight,
            CoreLocalMem<uint32_t>(piece_address(piece)),
            piece_tiles * tile_bytes,
            {.page_id = tile},
            {},
            {.trid = piece % depth + 1});
    };

    uint32_t next = 0;
    while (next < depth && next < pieces && issuable(next)) {
        issue(next++);
    }
    for (uint32_t piece = 0; piece < pieces; ++piece) {
        // Every earlier piece is pushed, so this one's request fits its ring; issue() may still wait for its place.
        while (next <= piece) {
            issue(next++);
        }
        noc.async_read_barrier<NocOptions::TXN_ID>({.trid = piece % depth + 1});
        push_piece(piece);
        (piece % 2 == 0 ? granted : granted_odd).wait_min((piece / 2 + 1) * column.count);
        pirotski_matmul::multicast_block(
            noc, arrived, column, piece_address(piece), piece_tiles * tile_bytes, piece + 1);
        while (next < pieces && next <= piece + depth && issuable(next)) {
            issue(next++);
        }
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
