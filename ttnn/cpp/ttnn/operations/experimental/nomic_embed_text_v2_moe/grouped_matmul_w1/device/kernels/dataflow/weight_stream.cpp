// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RISCV_0 (NOC0) of every compute core: the weight stream of the core's share, and compute's schedule. The share's
// sender (the top core of its column) reads each expert's slice in chunks, `depth` reads in flight, and multicasts
// each chunk south down the column, then the count of chunks sent. Receivers grant each chunk once its place in the
// ring is reserved and push it as its count arrives, so compute can start on a slice before all of it is in. Compute
// pops each chunk of an expert's slice during the expert's last block, so the slice two on can stream into it then.
// Grants go to one of two counters by the piece's parity, and a receiver grants at most one piece past the one it
// waits for: a grant of piece p + 2 needs piece p to have arrived, so a counter can only reach the sender's wait for
// piece p once every receiver has granted p, and the next grant is already in while p is multicast. Two counts can
// then be in flight to a receiver at once; both take the same multicast route and virtual channel, so they arrive in
// order.
//
// A slice holds slice_tiles of the slot's slot_tiles in slot order: each output subblock's K rows of subblock_w
// tiles, then its bias tiles. Every core pushes slot_tiles a slice, so slot n sits at the same address on every
// core. From a chunked weight, chunk j of share s's slice of expert e is one contiguous run in one DRAM bank:
//     tile bank * shard_tiles + ((e * max_chunks + j) * bank_sharers + s / 8) * chunk_tiles, bank (s + j + e) % 8.
// From a plain TILE weight each tile is read on its own, in the same order (the fallback).
// With per-expert lanes every core reads its own lane's expert only and has no receivers.

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "ckernel.h"
#include "expert_schedule.hpp"
#include "ttnn/operations/experimental/nomic_embed_text_v2_moe/pirotski_matmul/device/kernels/dataflow/stream_protocol.hpp"

void kernel_main() {
    constexpr uint32_t num_experts = get_named_compile_time_arg_val("num_experts");
    constexpr uint32_t num_lanes = get_named_compile_time_arg_val("num_lanes");
    constexpr bool per_expert_lanes = get_named_compile_time_arg_val("per_expert_lanes") == 1;
    constexpr uint32_t k_tiles = get_named_compile_time_arg_val("k_tiles");
    constexpr bool has_bias = get_named_compile_time_arg_val("has_bias") == 1;
    constexpr uint32_t subblock_w = get_named_compile_time_arg_val("subblock_w");
    constexpr uint32_t slot_tiles = get_named_compile_time_arg_val("slot_tiles");
    constexpr uint32_t slots = get_named_compile_time_arg_val("slots");
    constexpr bool chunked = get_named_compile_time_arg_val("chunked") == 1;
    constexpr uint32_t chunk_tiles = get_named_compile_time_arg_val("chunk_tiles");
    constexpr uint32_t max_chunks = get_named_compile_time_arg_val("max_chunks");
    constexpr uint32_t shard_tiles = get_named_compile_time_arg_val("shard_tiles");
    constexpr uint32_t bank_sharers = get_named_compile_time_arg_val("bank_sharers");
    // A plain TILE weight: tile (k, n) of expert e is page k * stride_k + n * stride_n + e * stride_expert, the bias
    // tile n of expert e page e * output_n_tiles + n of the bias.
    constexpr uint32_t stride_k = get_named_compile_time_arg_val("stride_k");
    constexpr uint32_t stride_n = get_named_compile_time_arg_val("stride_n");
    constexpr uint32_t stride_expert = get_named_compile_time_arg_val("stride_expert");
    constexpr uint32_t output_n_tiles = get_named_compile_time_arg_val("output_n_tiles");
    constexpr uint32_t depth = get_named_compile_time_arg_val("depth");
    constexpr uint32_t counts_page_bytes = get_named_compile_time_arg_val("counts_page_bytes");
    constexpr bool tail_split = get_named_compile_time_arg_val("tail_split") == 1;
    constexpr uint32_t weight_cb_id = get_named_compile_time_arg_val("weight_cb");
    constexpr auto weight_args = TensorAccessorArgs<0>();
    constexpr auto counts_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();
    // Without a bias the accessor is never built; the weight's arguments hold its place.
    [[maybe_unused]] constexpr auto bias_args = TensorAccessorArgs<counts_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile_bytes = get_tile_size(weight_cb_id);
    constexpr uint32_t slot_bytes = slot_tiles * tile_bytes;
    static_assert(depth >= 1 && depth <= 8);
    static_assert(!tail_split || num_experts == 1);

    uint32_t arg = 0;
    const uint32_t weight_address = get_arg_val<uint32_t>(arg++);
    const uint32_t counts_address = get_arg_val<uint32_t>(arg++);
    const uint32_t bias_address = get_arg_val<uint32_t>(arg++);
    const uint32_t input_blocks = get_arg_val<uint32_t>(arg++);
    const uint32_t lane = get_arg_val<uint32_t>(arg++);
    const uint32_t share = get_arg_val<uint32_t>(arg++);
    const uint32_t first_n_tile = get_arg_val<uint32_t>(arg++);
    const uint32_t slice_tiles = get_arg_val<uint32_t>(arg++);
    const bool is_sender = get_arg_val<uint32_t>(arg++) == 1;
    const uint32_t sender_x = get_arg_val<uint32_t>(arg++);
    const uint32_t sender_y = get_arg_val<uint32_t>(arg++);
    // A braced list evaluates in order: start x, start y, end x, end y, count.
    const pirotski_matmul::Multicast column{
        get_arg_val<uint32_t>(arg++),
        get_arg_val<uint32_t>(arg++),
        get_arg_val<uint32_t>(arg++),
        get_arg_val<uint32_t>(arg++),
        get_arg_val<uint32_t>(arg++)};
    const uint32_t receivers = column.count;
    // A forwarder (the folded layout) passes each piece it receives on to these cores, count 0 elsewhere.
    const pirotski_matmul::Multicast forward{
        get_arg_val<uint32_t>(arg++),
        get_arg_val<uint32_t>(arg++),
        get_arg_val<uint32_t>(arg++),
        get_arg_val<uint32_t>(arg++),
        get_arg_val<uint32_t>(arg++)};

    const Noc noc;
    DataflowBuffer weight_cb(weight_cb_id);
    DataflowBuffer counts_cb(get_named_compile_time_arg_val("counts_cb"));
    DataflowBuffer schedule_cb(get_named_compile_time_arg_val("schedule_cb"));
    Semaphore<> granted(get_named_compile_time_arg_val("granted"));
    Semaphore<> granted_odd(get_named_compile_time_arg_val("granted_odd"));
    Semaphore<> arrived(get_named_compile_time_arg_val("arrived"));
    // The forwarder's count word; its own `arrived` keeps the sender's count.
    Semaphore<> forward_flag(get_named_compile_time_arg_val("forward_flag"));

    if constexpr (get_named_compile_time_arg_val("has_counts") == 1) {
        const auto counts_accessor = TensorAccessor(counts_args, counts_address);
        noc.async_read(counts_accessor, counts_cb, counts_page_bytes, {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
    } else {
        // No counts: one expert takes every row.
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counts_cb.get_write_ptr()) =
            input_blocks * grouped_matmul::kBlockRows;
    }
    uint32_t first_block[num_experts];
    uint32_t num_blocks[num_experts];
    grouped_matmul::expert_regions<num_experts>(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counts_cb.get_write_ptr()),
        input_blocks,
        first_block,
        num_blocks);

    // Compute's schedule in the walk order: this core's blocks of each expert, or kSkipExpert where no slice comes.
    // In the shared schedule every lane takes every slice its share sends, also of an expert it has no block of.
    uint32_t order[num_experts];
    grouped_matmul::expert_order<num_experts>(num_blocks, order);
    schedule_cb.reserve_back(1);
    auto* schedule = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(schedule_cb.get_write_ptr());
    uint32_t taken[num_experts];
    uint32_t num_taken = 0;
    for (uint32_t position = 0; position < num_experts; ++position) {
        const uint32_t expert = order[position];
        const uint32_t count = grouped_matmul::lane_blocks<num_lanes, per_expert_lanes>(
                                   expert, first_block[expert], num_blocks[expert], lane)
                                   .count;
        if (per_expert_lanes ? count > 0 : num_blocks[expert] > 0) {
            schedule[position] = count;
            taken[num_taken++] = expert;
        } else {
            schedule[position] = grouped_matmul::kSkipExpert;
        }
    }
    if constexpr (tail_split) {
        // Word 0 the expert's full blocks, words 1 and 2 the lane's subblocks of its tail block, [begin, end).
        if (num_blocks[0] > 0) {
            const auto work = grouped_matmul::lane_work(
                num_blocks[0], num_lanes, lane, get_named_compile_time_arg_val("subblocks_max"));
            schedule[0] = work.full_count;
            schedule[1] = work.tail_subblock_begin;
            schedule[2] = work.tail_subblock_end;
            (void)ckernel::load_blocking(&schedule[2]);
        }
    }
    // A store can retire before L1 has it; a blocking load of the last word returns once all have landed, so
    // compute never reads the schedule ahead of it.
    (void)ckernel::load_blocking(&schedule[num_experts - 1]);
    schedule_cb.push_back(1);

    const uint32_t chunks = (slice_tiles + chunk_tiles - 1) / chunk_tiles;
    auto chunk_tiles_of = [&](uint32_t chunk) {
        return chunk + 1 < chunks ? chunk_tiles : slice_tiles - chunk * chunk_tiles;
    };
    // A narrower share pushes the rest of its slot unfilled after its last chunk.
    uint32_t pushed = 0;
    auto push_chunk = [&](uint32_t chunk) {
        const uint32_t tiles = chunk_tiles_of(chunk) + (chunk + 1 == chunks ? slot_tiles - slice_tiles : 0);
        weight_cb.push_back(tiles);
        pushed += tiles;
    };
    const uint32_t slot_base = weight_cb.get_write_ptr();
    const uint32_t pieces = num_taken * chunks;
    // cb_reserve_back(n) waits for acked >= pushed + n - slots * slot_tiles. Piece p's place is free once compute has
    // popped the ring up to the piece's end, one ring back: n = piece_end(p) - pushed, which the ring can grant only
    // once n <= slots * slot_tiles. A slice's last piece ends at its slot's end, with a narrower share's padding.
    auto piece_end = [&](uint32_t piece) {
        const uint32_t chunk = piece % chunks;
        return (piece / chunks) * slot_tiles + (chunk + 1 == chunks ? slot_tiles : (chunk + 1) * chunk_tiles);
    };
    auto piece_request = [&](uint32_t piece) { return piece_end(piece) - pushed; };
    auto piece_address = [&](uint32_t piece) {
        return slot_base + ((piece / chunks) % slots) * slot_bytes + (piece % chunks) * chunk_tiles * tile_bytes;
    };

    if (!is_sender) {
        uint32_t grants = 0;
        auto grant = [&](uint32_t piece) {
            weight_cb.reserve_back(piece_request(piece));
            (piece % 2 == 0 ? granted : granted_odd).up(noc, sender_x, sender_y, 1);
            ++grants;
        };
        for (uint32_t piece = 0; piece < pieces; ++piece) {
            if (grants == piece) {
                grant(piece);
            }
            if (grants == piece + 1 && grants < pieces && weight_cb.pages_reservable_at_back(piece_request(grants))) {
                grant(grants);
            }
            arrived.wait_min(piece + 1);
            push_chunk(piece % chunks);
            // Its downstream cores grant this core as receivers grant the sender. The forward is flushed before
            // the loop moves on, and the space it was read from is granted to the sender only in a later
            // iteration, so the sender cannot overwrite it first.
            if (forward.count > 0) {
                ASSERT(slots * chunks >= 2);
                (piece % 2 == 0 ? granted : granted_odd).wait_min((piece / 2 + 1) * forward.count);
                const uint32_t address = piece_address(piece);
                noc.async_write_multicast(
                    CoreLocalMem<uint32_t>(address),
                    MulticastEndpoint{},
                    chunk_tiles_of(piece % chunks) * tile_bytes,
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
        noc.async_write_barrier();
        noc.async_atomic_barrier();
        return;
    }

    const auto weight = TensorAccessor(weight_args, weight_address);
    // Read-ahead takes a piece only once its place is already free: waiting for it would hold up the current
    // piece's multicast until this core's compute has popped the ring that far.
    auto issuable = [&](uint32_t piece) {
        const uint32_t request = piece_request(piece);
        return request <= slots * slot_tiles && weight_cb.pages_reservable_at_back(request);
    };
    auto issue = [&](uint32_t piece) {
        const uint32_t slice = piece / chunks;
        const uint32_t chunk = piece % chunks;
        weight_cb.reserve_back(piece_request(piece));
        const uint32_t expert = taken[slice];
        const uint32_t trid = piece % depth + 1;
        const uint32_t address = piece_address(piece);
        if constexpr (chunked) {
            const uint32_t bank = (share + chunk + expert) % 8;
            const uint32_t tile =
                bank * shard_tiles + ((expert * max_chunks + chunk) * bank_sharers + share / 8) * chunk_tiles;
            noc.async_read<NocOptions::TXN_ID>(
                weight,
                CoreLocalMem<uint32_t>(address),
                chunk_tiles_of(chunk) * tile_bytes,
                {.page_id = tile},
                {},
                {.trid = trid});
        } else {
            // Slot position i: subblock i / group_tiles, row (i % group_tiles) / subblock_w (k, or the bias row),
            // column first_n_tile + subblock * subblock_w + i % subblock_w.
            constexpr uint32_t group_tiles = (k_tiles + (has_bias ? 1 : 0)) * subblock_w;
            uint32_t position = chunk * chunk_tiles;
            const uint32_t end = position + chunk_tiles_of(chunk);
            for (uint32_t offset = 0; position < end; ++position, offset += tile_bytes) {
                const uint32_t row = position % group_tiles / subblock_w;
                const uint32_t n = first_n_tile + position / group_tiles * subblock_w + position % subblock_w;
                if (row < k_tiles) {
                    noc.async_read<NocOptions::TXN_ID>(
                        weight,
                        CoreLocalMem<uint32_t>(address + offset),
                        tile_bytes,
                        {.page_id = row * stride_k + n * stride_n + expert * stride_expert},
                        {},
                        {.trid = trid});
                } else if constexpr (has_bias) {
                    const auto bias = TensorAccessor(bias_args, bias_address);
                    noc.async_read<NocOptions::TXN_ID>(
                        bias,
                        CoreLocalMem<uint32_t>(address + offset),
                        tile_bytes,
                        {.page_id = expert * output_n_tiles + n},
                        {},
                        {.trid = trid});
                }
            }
        }
    };

    uint32_t next = 0;
    while (next < depth && next < pieces && issuable(next)) {
        issue(next++);
    }
    for (uint32_t piece = 0; piece < pieces; ++piece) {
        const uint32_t chunk = piece % chunks;
        // Every earlier piece is pushed, so this one's request fits the ring; issue() may still wait for its place.
        while (next <= piece) {
            issue(next++);
        }
        noc.async_read_barrier<NocOptions::TXN_ID>({.trid = piece % depth + 1});
        push_chunk(chunk);
        if (receivers > 0) {
            (piece % 2 == 0 ? granted : granted_odd).wait_min((piece / 2 + 1) * receivers);
            pirotski_matmul::multicast_block(
                noc, arrived, column, piece_address(piece), chunk_tiles_of(chunk) * tile_bytes, piece + 1);
        }
        while (next < pieces && next <= piece + depth && issuable(next)) {
            issue(next++);
        }
    }
    noc.async_write_barrier();
    noc.async_atomic_barrier();
}
