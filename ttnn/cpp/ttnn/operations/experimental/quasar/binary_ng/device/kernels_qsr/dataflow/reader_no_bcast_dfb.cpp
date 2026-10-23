// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 / DataflowBuffer (DFB) reader for binary_ng's no-broadcast binary op, Quasar-native.
//
// Diverges from kernels_dfb/dataflow/reader_no_bcast_dfb.cpp in three ways, all licensed by
// matches_quasar_native_slice:
//   - the nD stride cascade is gone: page = start_tile_id + k, or, beside a borrowed operand, the tile order of
//     the program's anchor (SHARD_WALK_WIDTH, BANK_WALK_STRIDE).
//   - the tile loop is per-thread. Thread t of N takes the STRIDED share {t, t+N, t+2N, ...}, which
//     is the slot assignment the DFB gives producer thread t.
//   - a borrowed input is published once per tile counter, and a shard's tiles past the borrowed rings are
//     copied into the tail rings.
// Each input is read over the NoC or borrowed from this core's L1 on its own (SRC_SHARDED, SRC_SHARDED_B): its
// shard, or its slice of an L1-interleaved tensor.
//
// "no_bcast" means no SUBTILE broadcast. The cascade this replaces also carried OUTER-dim broadcast,
// indexing each operand through strides the factory zeroes for unit input dims, so the linear form here
// is the identity only while every input dim equals the output's. The gate enforces that, and the
// factory's a_dims_are_output_dims TT_FATAL fires if it is relaxed without restoring the cascade.

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/local_tensor_accessor.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // A borrowed ring drains when its object leaves scope, so it lives to kernel exit: the compute pops a borrowed
    // entry only after the matching entry of the other input, which this thread may still read over the NoC.
#if HAS_MAIN_RING && SRC_SHARDED
    DataflowBuffer borrowed_in0(dfb::in0);
#endif
#if HAS_MAIN_RING && SRC_SHARDED_B
    DataflowBuffer borrowed_in1(dfb::in1);
#endif
#if SRC_SHARDED || SRC_SHARDED_B
    // Borrowed inputs: each borrowed DFB is the resident L1 shard or slice, so this thread publishes its credits, once
    // per tile counter. One reserve_back for the whole shard, as in the legacy reader, cannot work: the active counter
    // holds only its share, and push_back credits one counter and then rotates.
    static_assert(!TAIL_TILES || (SRC_SHARDED && SRC_SHARDED_B), "the tail rings run only beside two borrowed inputs");
    {
        const uint32_t dst_num_tiles = get_arg(args::dst_num_tiles);
        constexpr uint32_t num_tcs = get_arg(args::num_tcs);
        const uint32_t num_threads = get_num_threads();

#if HAS_MAIN_RING
        {
            const uint32_t tile_step = num_tcs * num_threads;
            // Counter c holds the tiles from thread_id + c*num_threads spaced tile_step apart. The factory sizes
            // the borrowed part of the shard, which every core processes, to divide by that spacing, so each
            // count is exactly the counter's capacity.
            uint32_t first = get_my_thread_id();
            for (uint32_t c = 0; c < num_tcs && first < dst_num_tiles; ++c, first += num_threads) {
                const uint32_t n = (dst_num_tiles - first + tile_step - 1) / tile_step;
#if SRC_SHARDED
                borrowed_in0.reserve_back(n);
                borrowed_in0.push_back(n);
#endif
#if SRC_SHARDED_B
                borrowed_in1.reserve_back(n);
                borrowed_in1.push_back(n);
#endif
            }
        }
#endif
#if TAIL_TILES
        // Copy the shard's TAIL_TILES tiles past the borrowed part (all of it without a borrowed ring) into the
        // tail rings, one entry per compute thread, through the local NoC loopback. Entries past TAIL_TILES stay
        // padding: the compute runs on them, the writer drops them.
        {
            Noc noc;
            DataflowBuffer tail_in0(dfb::in0_tail);
            DataflowBuffer tail_in1(dfb::in1_tail);
            const LocalTensorAccessor<uint32_t> a_shard(tensor::in0);
            const LocalTensorAccessor<uint32_t> b_shard(tensor::in1);
            const uint32_t a_tile_bytes = tail_in0.get_entry_size();
            const uint32_t b_tile_bytes = tail_in1.get_entry_size();
            uint32_t slot = get_my_thread_id();
            for (uint32_t c = 0; c < num_tcs; ++c, slot += num_threads) {
                tail_in0.reserve_back(1);
                tail_in1.reserve_back(1);
                if (slot < TAIL_TILES) {
                    noc.async_read(
                        a_shard,
                        tail_in0,
                        a_tile_bytes,
                        {.offset_bytes = (dst_num_tiles + slot) * a_tile_bytes},
                        {.offset_bytes = 0});
                    noc.async_read(
                        b_shard,
                        tail_in1,
                        b_tile_bytes,
                        {.offset_bytes = (dst_num_tiles + slot) * b_tile_bytes},
                        {.offset_bytes = 0});
                    noc.async_read_barrier();
                }
                tail_in0.push_back(1);
                tail_in1.push_back(1);
            }
        }
#endif
    }
#endif
#if !SRC_SHARDED || !SRC_SHARDED_B
    const uint32_t start_tile_id = get_arg(args::start_tile_id);
    const uint32_t dst_num_tiles = get_arg(args::dst_num_tiles);

    constexpr uint32_t dm_batch = get_arg(args::dm_batch);

    Noc noc;
#if !SRC_SHARDED
    DataflowBuffer dfb_in0(dfb::in0);
#endif
#if !SRC_SHARDED_B
    DataflowBuffer dfb_in1(dfb::in1);
#endif

#if !SRC_SHARDED
    const uint32_t src_tile_bytes = dfb_in0.get_entry_size();
#endif
#if !SRC_SHARDED_B
    const uint32_t src_tile_bytes_b = dfb_in1.get_entry_size();
#endif
#if !SRC_SHARDED
    const auto src = TensorAccessor(tensor::in0);
#endif
#if !SRC_SHARDED_B
    const auto src_b = TensorAccessor(tensor::in1);
#endif

    const uint32_t thread_id = get_my_thread_id();
    const uint32_t num_threads = get_num_threads();

    // Each thread reads every input that moves for its own k, as the legacy reader reads both inputs.
    // Entry i of a reserved batch sits at i * stride_size, NOT i * entry_size: with more than one
    // producer or consumer the ring is STRIDED and a thread's entries interleave with its siblings'.
#if !SRC_SHARDED
    const uint32_t in0_stride = dfb_in0.get_stride_size();
#endif
#if !SRC_SHARDED_B
    const uint32_t in1_stride = dfb_in1.get_stride_size();
#endif

#if SHARD_WALK_WIDTH
    // Beside a borrowed shard, this core computes the anchor's shard: tile k of the shard is page
    // start_tile_id + (k / w) * Wt + k % w, with w = SHARD_WALK_WIDTH, a constant, so the divide is cheap.
    const uint32_t row_tiles = get_arg(args::Wt);
    auto anchor_page = [&](uint32_t k) {
        return start_tile_id + (k / SHARD_WALK_WIDTH) * row_tiles + k % SHARD_WALK_WIDTH;
    };
#elif BANK_WALK_STRIDE
    // Beside borrowed bank slices, this core computes the pages of its bank: tile k is page
    // start_tile_id + k * BANK_WALK_STRIDE, where start_tile_id is the bank.
    auto anchor_page = [&](uint32_t k) { return start_tile_id + k * BANK_WALK_STRIDE; };
#endif
#if SHARD_WALK_WIDTH || BANK_WALK_STRIDE
#if WALK_PADS
    // An uneven anchor gives this core pad slots: tiles past the tensor's last row or column. A pad slot moves
    // nothing; its ring entry keeps stale data, and its result is never read back. A bank slice is a column of
    // pages, one tile wide.
    const uint32_t valid_rows = get_arg(args::valid_rows);
#if SHARD_WALK_WIDTH
    const uint32_t valid_cols = get_arg(args::valid_cols);
    auto in_tensor = [&](uint32_t k) { return k / SHARD_WALK_WIDTH < valid_rows && k % SHARD_WALK_WIDTH < valid_cols; };
#else
    auto in_tensor = [&](uint32_t k) { return k < valid_rows; };
#endif
#else
    auto in_tensor = [](uint32_t) { return true; };
#endif
#endif

    // n of this thread's tiles starting at tile index k, spaced tile_step apart. In the linear walk, page
    // and ring offset walk by addition: num_threads is thread_local runtime state, so indexing off i would
    // be a multiply per read. The anchor walks compute each page from its tile index.
    auto read_batch = [&](uint32_t k, uint32_t n, uint32_t tile_step) {
    // DeviceZoneScopedSum* feeds the work-split gate (per-thread RD_RSV / RD_BAR). Zero-cost
    // unless TT_METAL_PROFILER_SUM=1, and compiled OUT under PROFILER_OPT_DO_ACCUMULATE.
    // Not "nativeness" -- keep when reconciling against kernels_dfb/.
#if !SRC_SHARDED
        {
            DeviceZoneScopedSumN1("RD_RSV");
            dfb_in0.reserve_back(n);
        }
        {
#if SHARD_WALK_WIDTH || BANK_WALK_STRIDE
            uint32_t tile = k;
            uint32_t offset = 0;
            for (uint32_t i = 0; i < n; ++i) {
                if (in_tensor(tile)) {
                    noc.async_read(
                        src, dfb_in0, src_tile_bytes, {.page_id = anchor_page(tile)}, {.offset_bytes = offset});
                }
                tile += tile_step;
                offset += in0_stride;
            }
#else
            uint32_t page = start_tile_id + k;
            uint32_t offset = 0;
            for (uint32_t i = 0; i < n; ++i) {
                noc.async_read(src, dfb_in0, src_tile_bytes, {.page_id = page}, {.offset_bytes = offset});
                page += tile_step;
                offset += in0_stride;
            }
#endif
        }
#endif
#if !SRC_SHARDED_B
        {
            DeviceZoneScopedSumN1("RD_RSV");
            dfb_in1.reserve_back(n);
        }
        {
#if SHARD_WALK_WIDTH || BANK_WALK_STRIDE
            uint32_t tile = k;
            uint32_t offset = 0;
            for (uint32_t i = 0; i < n; ++i) {
                if (in_tensor(tile)) {
                    noc.async_read(
                        src_b, dfb_in1, src_tile_bytes_b, {.page_id = anchor_page(tile)}, {.offset_bytes = offset});
                }
                tile += tile_step;
                offset += in1_stride;
            }
#else
            uint32_t page = start_tile_id + k;
            uint32_t offset = 0;
            for (uint32_t i = 0; i < n; ++i) {
                noc.async_read(src_b, dfb_in1, src_tile_bytes_b, {.page_id = page}, {.offset_bytes = offset});
                page += tile_step;
                offset += in1_stride;
            }
#endif
        }
#endif
        {
            DeviceZoneScopedSumN2("RD_BAR");
            noc.async_read_barrier();
        }
#if !SRC_SHARDED
        dfb_in0.push_back(n);
#endif
#if !SRC_SHARDED_B
        dfb_in1.push_back(n);
#endif
    };

    // This thread owns tiles {thread_id, thread_id + num_threads, ...} and round-robins num_tcs tile
    // counters, one rotation per push_back. Counter c therefore holds the tiles starting at
    // thread_id + c*num_threads and spaced num_tcs*num_threads apart -- so a batch drawn from ONE
    // counter strides by that product, not by num_threads.
    if constexpr (dm_batch == 1) {
        // One tile per push rotates the counter every tile, so the counter-major order and the plain
        // per-tile order coincide for every num_tcs. Keep the plain loop: the nested walk below is
        // slower per tile, and at one tile per push it buys nothing.
        for (uint32_t k = thread_id; k < dst_num_tiles; k += num_threads) {
            read_batch(k, 1, num_threads);
        }
    } else {
        // Tile counters this thread round-robins: the factory mirrors the DFB's num_tcs_to_rr for this
        // role. push_back rotates by exactly one, so a batch fills one counter and the next batch fills
        // the next.
        constexpr uint32_t num_tcs = get_arg(args::num_tcs);
        const uint32_t tile_step = num_tcs * num_threads;
        const uint32_t round_span = dm_batch * tile_step;
        // A batch that starts at or past full_limit has fewer than dm_batch tiles left in its counter,
        // so only there is the count derived. Every earlier batch is full without counting.
        const uint32_t batch_span = (dm_batch - 1) * tile_step;
        const uint32_t full_limit = dst_num_tiles > batch_span ? dst_num_tiles - batch_span : 0;

        // Counter 0 starts earliest, so once it is out of range every counter is, and the rotation
        // never has to resume after a skipped push.
        for (uint32_t round_base = thread_id; round_base < dst_num_tiles; round_base += round_span) {
            uint32_t first = round_base;
            for (uint32_t c = 0; c < num_tcs && first < dst_num_tiles; ++c, first += num_threads) {
                uint32_t n = dm_batch;
                if (first >= full_limit) {
                    // The tiles this counter has left: first, first + tile_step, ... below dst_num_tiles.
                    // full_limit guarantees fewer than dm_batch of them, and the loop bound at least one.
                    n = (dst_num_tiles - first + tile_step - 1) / tile_step;
                }
                read_batch(first, n, tile_step);
            }
        }
    }
    // Each ring's destructor drains this thread's credits at kernel exit. Keep explicit sync here: the implicit-sync
    // overloads (NocOptions::TXN_ID with a DataflowBuffer endpoint) put a thread barrier in that drain,
    // which deadlocks a thread with zero tiles.
#endif
}
