// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A worker core's side of the exchange, which is entirely local: park this rank's
// statistics in its own slot of the shared statistics tensor, tell the gather core
// this core is done, and wait to be told every rank's slot is filled. Where rows are
// split into chunks, a chunk's statistics are partial: they go to the row's leader
// (its chunk 0 core), whose compute sums them, and only the leader parks the row.
//
// Everything a general all-gather does for this payload collapses because the
// statistics are row-local and every tensor-parallel rank holds the same token
// rows: rank p's contribution for a token row lands at a slot indexed by p, and
// summing the slots is the reduction. No scatter, no chunking, no reordering.
//
// The statistics tensor is a mesh buffer, so a page of it has the same address on
// every chip. That is what lets the gather core address a peer's slot without
// anything being exchanged first.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/experimental/deepseek_prefill/attn_res_gather_softmax/device/kernels/dataflow/attn_res_stats_layout.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/attn_res_gather_softmax/device/kernels/attn_res_core_runs.hpp"

void kernel_main() {
    // compile-time args
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t ring_size = get_compile_time_arg_val(1);
    // Token row-tiles, which is what sizes a plane of the statistics tensor.
    constexpr uint32_t Ht = get_compile_time_arg_val(2);
    constexpr uint32_t ready_sem_id = get_compile_time_arg_val(3);
    constexpr uint32_t done_sem_id = get_compile_time_arg_val(4);
    // A deferred residual write settled in pass one. Where there is none the accessor
    // below aliases the merged output, so the offsets hold either way and nothing is
    // written through it.
    constexpr bool fuse_add = get_compile_time_arg_val(5) == 1;
    constexpr uint32_t chunk_sem_id = get_compile_time_arg_val(6);
    constexpr auto stats_args = TensorAccessorArgs<7>();
    constexpr auto shift_args = TensorAccessorArgs<stats_args.next_compile_time_args_offset()>();
    constexpr auto mass_args = TensorAccessorArgs<shift_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<mass_args.next_compile_time_args_offset()>();
    constexpr auto total_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    using Split = AttnResSplit<total_args.next_compile_time_args_offset()>;
    constexpr uint32_t chunk_tiles = Split::chunk_tiles;
    constexpr uint32_t chunks_per_row = Split::chunks_per_row;
    const auto [first_unit, num_units] = Split::core_units();

    // Everything else is common: the site's offsets into
    // shift and mass, the buffer addresses, and this chip's rank and gather core. A
    // program-cache hit patches these once per program instead of once per core.
    const auto shift_page_offset = get_common_arg_val<uint32_t>(0);
    const auto mass_page_offset = get_common_arg_val<uint32_t>(1);
    const auto stats_addr = get_common_arg_val<uint32_t>(2);
    const auto shift_addr = get_common_arg_val<uint32_t>(3);
    const auto mass_addr = get_common_arg_val<uint32_t>(4);
    const auto out_addr = get_common_arg_val<uint32_t>(5);
    const auto total_addr = get_common_arg_val<uint32_t>(6);
    const auto my_rank = get_common_arg_val<uint32_t>(7);
    const auto gather_core_x = get_common_arg_val<uint32_t>(8);
    const auto gather_core_y = get_common_arg_val<uint32_t>(9);
    // Then, where rows are split, each row's leader as (x << 16) | y.
    constexpr uint32_t kLeaderArgs = 10;

    constexpr uint32_t cb_id_scalars = 5;
    constexpr uint32_t cb_id_local_stats = 7;
    constexpr uint32_t cb_id_out = 16;
    constexpr uint32_t cb_id_total = 12;
    constexpr uint32_t cb_id_packed = 13;
    constexpr uint32_t cb_id_chunk_stats = 14;
    constexpr uint32_t cb_id_row_stats = 15;

    constexpr uint32_t stat_tile_bytes = get_tile_size(cb_id_local_stats);
    constexpr uint32_t scalar_tile_bytes = get_tile_size(cb_id_scalars);
    constexpr uint32_t out_tile_bytes = get_tile_size(cb_id_out);
    constexpr uint32_t kFixedScalars = 2;
    constexpr uint32_t kStatsPerPartial = 2;
    constexpr uint32_t kScalars = kFixedScalars + kStatsPerPartial * ring_size;
    constexpr uint32_t kStatsPerRow = 2;
    constexpr uint32_t onetile = 1;

    // The statistics cross the fabric packed by column, which turns a plane from Ht
    // packets into one. Both the pack below and the un-pack in pass two are element
    // strides through L1, which is why they sit here on the workers — one core per token
    // row — rather than on the single core that runs the exchange.
    constexpr uint32_t packed_row_bytes = stats_row_bytes(scalar_tile_bytes);

    constexpr uint32_t total_tile_bytes = get_tile_size(cb_id_total);

    Noc noc;
    CircularBuffer cb_scalars(cb_id_scalars);
    DataflowBuffer local_stats_buf(cb_id_local_stats);
    DataflowBuffer out_buf(cb_id_out);
    DataflowBuffer total_buf(cb_id_total);
    DataflowBuffer row_stats_out_buf(cb_id_row_stats);
    // Scratch for the packed form, on both sides of the exchange: the columns this core
    // parks in pass one, the columns it collects in pass two. Never pushed, so the two
    // pointers stay at its base and it can be addressed directly.
    DataflowBuffer packed_buf(cb_id_packed);
    const uint32_t packed_base = packed_buf.get_write_ptr();

    // Page size is given explicitly: the accessor's compile-time value can be stale
    // on a program-cache hit, and the gather core reuses it as the fabric payload size.
    auto stats_accessor = TensorAccessor(stats_args, stats_addr, stat_tile_bytes);
    auto shift_accessor = TensorAccessor(shift_args, shift_addr);
    auto mass_accessor = TensorAccessor(mass_args, mass_addr);
    auto out_accessor = TensorAccessor(out_args, out_addr);
    auto total_accessor = TensorAccessor(total_args, total_addr);

    Semaphore<> ready_sem(ready_sem_id);
    Semaphore<> done_sem(done_sem_id);
    Semaphore<> chunk_sem(chunk_sem_id);

    // A row's leader collects its other chunks' pairs in its own copy of this buffer,
    // which sits at the same address on every core; it is reserved once and filled
    // remotely, so a sender addresses it by this core's own base.
    DataflowBuffer chunk_stats_buf(cb_id_chunk_stats);
    const uint32_t chunk_stats_base = chunk_stats_buf.get_write_ptr();
    const bool leads = chunks_per_row == 1 || Split::unit(first_unit).chunk == 0;
    if constexpr (chunks_per_row > 1) {
        if (leads) {
            chunk_stats_buf.reserve_back(kStatsPerRow * (chunks_per_row - 1));
        }
    }

    // Pass one: this rank's whole share, parked before anything blocks. Compute
    // produces the two statistics of a run together, so they are written together.
    for (uint32_t u = first_unit; u < first_unit + num_units; ++u) {
        const auto [g, chunk] = Split::unit(u);
        const uint32_t first_tile = Split::template first_tile<Wt>(u);

        // The settled stream, parked before the statistics it feeds are announced: the
        // caller carries it forward as the live stream. Compute packs the run before it
        // reduces it, so this is the order the buffers fill in.
        if constexpr (fuse_add) {
            total_buf.wait_front(chunk_tiles);
            for (uint32_t k = 0; k < chunk_tiles; ++k) {
                noc.async_write(
                    total_buf,
                    total_accessor,
                    total_tile_bytes,
                    {.offset_bytes = k * total_tile_bytes},
                    {.page_id = first_tile + k});
            }
            noc.async_write_barrier();
            total_buf.pop_front(chunk_tiles);
        }

        if constexpr (chunks_per_row > 1) {
            if (chunk != 0) {
                // A partial pair, to the row's leader: the data, then the count it waits on.
                const uint32_t leader = get_common_arg_val<uint32_t>(kLeaderArgs + g);
                const uint32_t leader_x = leader >> 16;
                const uint32_t leader_y = leader & 0xFFFF;
                local_stats_buf.wait_front(kStatsPerRow);
                noc_async_write(
                    local_stats_buf.get_read_ptr(),
                    get_noc_addr(leader_x, leader_y, chunk_stats_base + (chunk - 1) * kStatsPerRow * stat_tile_bytes),
                    kStatsPerRow * stat_tile_bytes);
                noc_async_write_barrier();
                noc_semaphore_inc(get_noc_addr(leader_x, leader_y, get_semaphore(chunk_sem_id)), 1);
                local_stats_buf.pop_front(kStatsPerRow);
                continue;
            }
            // The leader: hand compute the other chunks' pairs once they have all landed,
            // and park the sum it makes of them.
            chunk_sem.wait(chunks_per_row - 1);
            chunk_sem.set(0);
            chunk_stats_buf.push_back(kStatsPerRow * (chunks_per_row - 1));
        }

        DataflowBuffer& row_stats_buf = chunks_per_row > 1 ? row_stats_out_buf : local_stats_buf;
        row_stats_buf.wait_front(kStatsPerRow);
        const uint32_t tiles_base = row_stats_buf.get_read_ptr();
        for (uint32_t s = 0; s < kStatsPerRow; ++s) {
            stats_pack_column<stat_tile_bytes>(tiles_base + s * stat_tile_bytes, packed_base + s * packed_row_bytes);
        }
        for (uint32_t s = 0; s < kStatsPerRow; ++s) {
            noc.async_write(
                packed_buf,
                stats_accessor,
                packed_row_bytes,
                {.offset_bytes = s * packed_row_bytes},
                {.page_id = stats_page_of(2 * my_rank + s, g, Ht, stat_tile_bytes),
                 .offset_bytes = stats_page_offset_of(g, stat_tile_bytes)});
        }
        noc.async_write_barrier();
        row_stats_buf.pop_front(kStatsPerRow);
    }

    // Only a core that parked rows has anything to announce, and the gather core counts
    // exactly those. Every core is still held here: the scalar sets it is about to read
    // are the gathered statistics.
    if (leads) {
        ready_sem.up(noc, gather_core_x, gather_core_y, 1);
    }

    // The gather core signals only once every rank's slot for every row is filled,
    // so this single wait is the whole of a worker's participation in the collective.
    done_sem.wait(1);
    // Reset for the next dispatch: a program-cache hit reuses this semaphore.
    done_sem.set(0);

    // Pass two: hand compute a complete scalar set per run, then drain its output. The
    // set is rank-major — shift, mass, then each rank's sum of squares and dots — the
    // layout a gathering collective leaves, so the fold's weight derivation reads it
    // without reordering.
    for (uint32_t u = first_unit; u < first_unit + num_units; ++u) {
        const uint32_t g = Split::unit(u).row;
        const uint32_t first_tile = Split::template first_tile<Wt>(u);

        cb_scalars.reserve_back(kScalars);
        noc.async_read(
            shift_accessor, cb_scalars, scalar_tile_bytes, {.page_id = g + shift_page_offset}, {.offset_bytes = 0});
        noc.async_read(
            mass_accessor,
            cb_scalars,
            scalar_tile_bytes,
            {.page_id = g + mass_page_offset},
            {.offset_bytes = scalar_tile_bytes});

        // The gathered statistics arrive packed — one page per rank and statistic, one
        // value per token — so they come in as a block rather than a page each.
        for (uint32_t k = 0; k < kStatsPerPartial * ring_size; ++k) {
            noc.async_read(
                stats_accessor,
                packed_buf,
                packed_row_bytes,
                {.page_id = stats_page_of(k, g, Ht, scalar_tile_bytes),
                 .offset_bytes = stats_page_offset_of(g, scalar_tile_bytes)},
                {.offset_bytes = k * packed_row_bytes});
        }
        noc.async_read_barrier();

        // Spread them back over the tiles compute reads.
        const uint32_t scalars_base = cb_scalars.get_write_ptr();
        for (uint32_t k = 0; k < kStatsPerPartial * ring_size; ++k) {
            stats_unpack_column<scalar_tile_bytes>(
                packed_base + k * packed_row_bytes, scalars_base + (kFixedScalars + k) * scalar_tile_bytes);
        }
        cb_scalars.push_back(kScalars);

        for (uint32_t k = 0; k < chunk_tiles; ++k) {
            out_buf.wait_front(onetile);
            noc.async_write(out_buf, out_accessor, out_tile_bytes, {.offset_bytes = 0}, {.page_id = first_tile + k});
            noc.async_write_barrier();
            out_buf.pop_front(onetile);
        }
    }
}
