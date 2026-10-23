// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 / DataflowBuffer (DFB) writer for binary_ng's no-broadcast binary op, Quasar-native.
//
// Diverges from kernels_dfb/dataflow/writer_no_bcast_dfb.cpp in three ways, all licensed by
// matches_quasar_native_slice:
//   - the nD stride cascade is gone: page = start_tile_id + k, or, beside a borrowed operand, the tile order of
//     the program's anchor (SHARD_WALK_WIDTH, BANK_WALK_STRIDE).
//   - the tile loop is per-thread. Thread t of N drains the STRIDED share {t, t+N, t+2N, ...}, which
//     is the slot assignment the DFB gives consumer thread t.
//   - for a borrowed output it writes back only the tail rings' results.
// The output is written over the NoC, or borrowed from this core's L1 (DST_SHARDED): its shard, or its
// slice of an L1-interleaved tensor, which the compute packs in place.
//
// The writer's cascade is the milder case: an output is never broadcast, so its strides are dense. Without an
// anchor each core writes a linear page range, and the sharding-aware TensorAccessor maps each page to its
// shard. The reader's cascade is the load-bearing one; see the note in reader_no_bcast_dfb.cpp before widening
// the gate.

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
#if DST_SHARDED
    // Borrowed output: the compute packs it in place, so nothing of the borrowed part is written, as in the legacy
    // writer. This thread still pops every entry, once per tile counter as the reader publishes a borrowed input: the
    // compute's out destructor waits until the ring is empty.
#if HAS_MAIN_RING
    {
        const uint32_t dst_num_tiles = get_arg(args::dst_num_tiles);
        constexpr uint32_t num_tcs = get_arg(args::num_tcs);
        const uint32_t num_threads = get_num_threads();
        const uint32_t tile_step = num_tcs * num_threads;
        DataflowBuffer dfb_out(dfb::out);
        uint32_t first = get_my_thread_id();
        for (uint32_t c = 0; c < num_tcs && first < dst_num_tiles; ++c, first += num_threads) {
            const uint32_t n = (dst_num_tiles - first + tile_step - 1) / tile_step;
            dfb_out.wait_front(n);
            dfb_out.pop_front(n);
        }
    }
#endif
#if TAIL_TILES
    // Copy the real tail results into the shard just past the borrowed part, and drop the padding.
    {
        const uint32_t dst_num_tiles = get_arg(args::dst_num_tiles);
        constexpr uint32_t num_tcs = get_arg(args::num_tcs);
        Noc noc;
        DataflowBuffer tail_out(dfb::out_tail);
        const LocalTensorAccessor<uint32_t> out_shard(tensor::out);
        const uint32_t tile_bytes = tail_out.get_entry_size();
        const uint32_t num_threads = get_num_threads();
        uint32_t slot = get_my_thread_id();
        for (uint32_t c = 0; c < num_tcs; ++c, slot += num_threads) {
            tail_out.wait_front(1);
            if (slot < TAIL_TILES) {
                noc.async_write(
                    tail_out,
                    out_shard,
                    tile_bytes,
                    {.offset_bytes = 0},
                    {.offset_bytes = (dst_num_tiles + slot) * tile_bytes});
                noc.async_write_barrier();
            }
            tail_out.pop_front(1);
        }
    }
#endif
#else
    const uint32_t start_tile_id = get_arg(args::start_tile_id);
    const uint32_t dst_num_tiles = get_arg(args::dst_num_tiles);

    constexpr uint32_t dm_batch = get_arg(args::dm_batch);

    Noc noc;
    DataflowBuffer dfb_out(dfb::out);

    const uint32_t dst_tile_bytes = dfb_out.get_entry_size();
    const auto dst = TensorAccessor(tensor::out);

    const uint32_t thread_id = get_my_thread_id();
    const uint32_t num_threads = get_num_threads();

    // Entry i of a batch sits at i * stride_size, not i * entry_size -- see the reader.
    const uint32_t out_stride = dfb_out.get_stride_size();

#if SHARD_WALK_WIDTH
    // Beside borrowed inputs, this core writes the anchor's shard in its tile order -- see the reader.
    const uint32_t row_tiles = get_arg(args::Wt);
    auto anchor_page = [&](uint32_t k) {
        return start_tile_id + (k / SHARD_WALK_WIDTH) * row_tiles + k % SHARD_WALK_WIDTH;
    };
#elif BANK_WALK_STRIDE
    // Beside borrowed bank slices, this core writes the pages of its bank -- see the reader.
    auto anchor_page = [&](uint32_t k) { return start_tile_id + k * BANK_WALK_STRIDE; };
#endif
#if SHARD_WALK_WIDTH || BANK_WALK_STRIDE
#if WALK_PADS
    // A pad slot must not be written: in a block or width shard, column Wt + j of row r is the page of column j
    // of row r + 1, a REAL page. See the reader for the bounds.
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

    // n of this thread's tiles starting at tile index k, spaced tile_step apart -- see the reader.
    auto write_batch = [&](uint32_t k, uint32_t n, uint32_t tile_step) {
        // DeviceZoneScopedSum* feeds the work-split gate (per-thread WR_WAIT / WR_BAR). Zero-cost
        // unless TT_METAL_PROFILER_SUM=1, and compiled OUT under PROFILER_OPT_DO_ACCUMULATE.
        // Not "nativeness" -- keep when reconciling against kernels_dfb/.
        {
            DeviceZoneScopedSumN1("WR_WAIT");
            dfb_out.wait_front(n);
        }
#if SHARD_WALK_WIDTH || BANK_WALK_STRIDE
        uint32_t tile = k;
        uint32_t offset = 0;
        for (uint32_t i = 0; i < n; ++i) {
            if (in_tensor(tile)) {
                noc.async_write(dfb_out, dst, dst_tile_bytes, {.offset_bytes = offset}, {.page_id = anchor_page(tile)});
            }
            tile += tile_step;
            offset += out_stride;
        }
#else
        uint32_t page = start_tile_id + k;
        uint32_t offset = 0;
        for (uint32_t i = 0; i < n; ++i) {
            noc.async_write(dfb_out, dst, dst_tile_bytes, {.offset_bytes = offset}, {.page_id = page});
            page += tile_step;
            offset += out_stride;
        }
#endif
        {
            DeviceZoneScopedSumN2("WR_BAR");
            noc.async_write_barrier();
        }
        dfb_out.pop_front(n);
    };

    // Counter-major walk: counter c holds the tiles from thread_id + c*num_threads spaced
    // num_tcs*num_threads apart, so a batch from one counter strides by that product -- see the reader.
    if constexpr (dm_batch == 1) {
        // Plain per-tile loop: at one tile per pop the two orders coincide -- see the reader.
        for (uint32_t k = thread_id; k < dst_num_tiles; k += num_threads) {
            write_batch(k, 1, num_threads);
        }
    } else {
        // Tile counters this thread round-robins -- see the reader. pop_front rotates by exactly one.
        constexpr uint32_t num_tcs = get_arg(args::num_tcs);
        const uint32_t tile_step = num_tcs * num_threads;
        const uint32_t round_span = dm_batch * tile_step;
        // Only a batch at or past full_limit can be short, so only there is the count derived.
        const uint32_t batch_span = (dm_batch - 1) * tile_step;
        const uint32_t full_limit = dst_num_tiles > batch_span ? dst_num_tiles - batch_span : 0;

        for (uint32_t round_base = thread_id; round_base < dst_num_tiles; round_base += round_span) {
            uint32_t first = round_base;
            for (uint32_t c = 0; c < num_tcs && first < dst_num_tiles; ++c, first += num_threads) {
                uint32_t n = dm_batch;
                if (first >= full_limit) {
                    // The tiles this counter has left -- see the reader.
                    n = (dst_num_tiles - first + tile_step - 1) / tile_step;
                }
                write_batch(first, n, tile_step);
            }
        }
    }
    // dfb_out's destructor drains this thread's credits at kernel exit. Keep explicit sync here: the implicit-sync
    // overloads (NocOptions::TXN_ID with a DataflowBuffer endpoint) put a thread barrier in that drain,
    // which deadlocks a thread with zero tiles.
#endif
}
