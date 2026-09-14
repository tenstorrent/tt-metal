// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A worker core's side of the exchange, which is entirely local: park this rank's
// statistics in its own slot of the shared statistics tensor, tell the gather core
// this core is done, and wait to be told every rank's slot is filled.
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
    constexpr auto stats_args = TensorAccessorArgs<6>();
    constexpr auto shift_args = TensorAccessorArgs<stats_args.next_compile_time_args_offset()>();
    constexpr auto mass_args = TensorAccessorArgs<shift_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<mass_args.next_compile_time_args_offset()>();
    constexpr auto total_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    // runtime args
    const auto stats_addr = get_arg_val<uint32_t>(0);
    const auto shift_addr = get_arg_val<uint32_t>(1);
    const auto mass_addr = get_arg_val<uint32_t>(2);
    const auto out_addr = get_arg_val<uint32_t>(3);
    // Statistics are split by token row and the fold by output tile, so a core's two
    // runs are unrelated. `num_stat_rows` is zero on a core that only folds.
    const auto num_stat_rows = get_arg_val<uint32_t>(4);
    const auto stat_start_row = get_arg_val<uint32_t>(5);
    const auto num_fold_tiles = get_arg_val<uint32_t>(6);
    const auto fold_start_tile = get_arg_val<uint32_t>(7);
    const auto my_rank = get_arg_val<uint32_t>(8);
    const auto gather_core_x = get_arg_val<uint32_t>(9);
    const auto gather_core_y = get_arg_val<uint32_t>(10);
    const auto total_addr = get_arg_val<uint32_t>(11);

    // Every core reads the same site out of shift and mass, so the two offsets are
    // common rather than per-core; they are also what the host re-patches on a
    // program-cache hit.
    const auto shift_page_offset = get_common_arg_val<uint32_t>(0);
    const auto mass_page_offset = get_common_arg_val<uint32_t>(1);

    constexpr uint32_t cb_id_scalars = 5;
    constexpr uint32_t cb_id_local_stats = 7;
    constexpr uint32_t cb_id_out = 16;
    constexpr uint32_t cb_id_total = 12;
    constexpr uint32_t cb_id_packed = 13;

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

    // Pass one: this rank's whole share, parked before anything blocks. Compute
    // produces the two statistics of a row together, so they are written together.
    for (uint32_t r = 0; r < num_stat_rows; ++r) {
        const uint32_t g = stat_start_row + r;

        // The settled stream, parked before the statistics it feeds are announced: the
        // caller carries it forward as the live stream, and the fold below never reads
        // it back. Compute packs the row before it reduces it, so this is the order the
        // buffers fill in.
        if constexpr (fuse_add) {
            total_buf.wait_front(Wt);
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                noc.async_write(
                    total_buf,
                    total_accessor,
                    total_tile_bytes,
                    {.offset_bytes = wt * total_tile_bytes},
                    {.page_id = g * Wt + wt});
            }
            noc.async_write_barrier();
            total_buf.pop_front(Wt);
        }

        local_stats_buf.wait_front(kStatsPerRow);
        const uint32_t tiles_base = local_stats_buf.get_read_ptr();
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
        local_stats_buf.pop_front(kStatsPerRow);
    }

    // Only a core that carried rows has anything to announce, and the gather core
    // counts exactly those. A fold-only core has nothing to park but still has to be
    // held here: the scalar sets it is about to read are the gathered statistics.
    if (num_stat_rows > 0) {
        ready_sem.up(noc, gather_core_x, gather_core_y, 1);
    }

    // The gather core signals only once every rank's slot for every row is filled,
    // so this single wait is the whole of a worker's participation in the collective.
    done_sem.wait(1);
    // Reset for the next dispatch: a program-cache hit reuses this semaphore.
    done_sem.set(0);

    // Pass two: hand compute a complete scalar set per token row this core's tile run
    // touches, then drain its output. The set is rank-major — shift, mass, then each
    // rank's sum of squares and dots — the layout a gathering collective leaves, so
    // the fold's weight derivation reads it without reordering.
    //
    // The run is contiguous in output-tile order, so `i % Wt == 0` is exactly a token
    // row boundary and the Wt tiles between two boundaries share one scalar set.
    uint32_t g = fold_start_tile / Wt;
    for (uint32_t i = fold_start_tile; i < fold_start_tile + num_fold_tiles; ++i) {
        if (i == fold_start_tile || i % Wt == 0) {
            cb_scalars.reserve_back(kScalars);
            noc.async_read(
                shift_accessor, cb_scalars, scalar_tile_bytes, {.page_id = g + shift_page_offset}, {.offset_bytes = 0});
            noc.async_read(
                mass_accessor,
                cb_scalars,
                scalar_tile_bytes,
                {.page_id = g + mass_page_offset},
                {.offset_bytes = scalar_tile_bytes});

            // The gathered statistics arrive packed — one page per rank and statistic,
            // one value per token — so they come in as a block rather than a page each.
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
            ++g;
        }

        out_buf.wait_front(onetile);
        noc.async_write(out_buf, out_accessor, out_tile_bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        out_buf.pop_front(onetile);
    }
}
