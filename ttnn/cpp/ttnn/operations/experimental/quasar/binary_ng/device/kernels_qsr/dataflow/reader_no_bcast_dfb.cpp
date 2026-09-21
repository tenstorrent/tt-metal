// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 / DataflowBuffer (DFB) reader for binary_ng's no-broadcast binary op, Quasar-native.
//
// Diverges from kernels_dfb/dataflow/reader_no_bcast_dfb.cpp in two ways, both licensed by
// matches_quasar_native_slice:
//   - the nD stride cascade is gone: page = start_tile_id + k.
//   - the tile loop is per-thread. Thread t of N takes the STRIDED share {t, t+N, t+2N, ...}, which
//     is the slot assignment the DFB gives producer thread t.
// Both operands are interleaved: the gate rejects sharded inputs, so no borrowed-shard branch exists.
//
// "no_bcast" means no SUBTILE broadcast. The cascade this replaces also carried OUTER-dim broadcast,
// indexing each operand through strides the factory zeroes for unit input dims, so the linear form here
// is the identity only while every input dim equals the output's. The gate enforces that, and the
// factory's a_dims_are_output_dims TT_FATAL fires if it is relaxed without restoring the cascade.

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t start_tile_id = get_arg(args::start_tile_id);
    const uint32_t dst_num_tiles = get_arg(args::dst_num_tiles);

    constexpr uint32_t dm_batch = get_arg(args::dm_batch);

    Noc noc;
    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);

    const uint32_t src_tile_bytes = dfb_in0.get_entry_size();
    const uint32_t src_tile_bytes_b = dfb_in1.get_entry_size();
    const auto src = TensorAccessor(tensor::in0);
    const auto src_b = TensorAccessor(tensor::in1);

    const uint32_t thread_id = get_my_thread_id();
    const uint32_t num_threads = get_num_threads();

    // Each thread reads BOTH operands for its own k. Splitting them across two MULTI-THREAD reader
    // kernels collides on barrier slot 0: the thread barriers are a fixed pair keyed by role rather
    // than allocated per group, which is a current DFB/LLK implementation limit, not a design one.
    // Entry i of a reserved batch sits at i * stride_size, NOT i * entry_size: with more than one
    // producer or consumer the ring is STRIDED and a thread's entries interleave with its siblings'.
    const uint32_t in0_stride = dfb_in0.get_stride_size();
    const uint32_t in1_stride = dfb_in1.get_stride_size();

    // n of this thread's tiles starting at tile index k, spaced tile_step apart. Page and ring offset
    // walk by addition: num_threads is thread_local runtime state, so indexing off i would be a
    // multiply per read.
    auto read_batch = [&](uint32_t k, uint32_t n, uint32_t tile_step) {
        // DeviceZoneScopedSum* feeds the work-split gate (per-thread RD_RSV / RD_BAR). Zero-cost
        // unless TT_METAL_PROFILER_SUM=1, and compiled OUT under PROFILER_OPT_DO_ACCUMULATE.
        // Not "nativeness" -- keep when reconciling against kernels_dfb/.
        {
            DeviceZoneScopedSumN1("RD_RSV");
            dfb_in0.reserve_back(n);
        }
        uint32_t page = start_tile_id + k;
        uint32_t offset = 0;
        for (uint32_t i = 0; i < n; ++i) {
            noc.async_read(src, dfb_in0, src_tile_bytes, {.page_id = page}, {.offset_bytes = offset});
            page += tile_step;
            offset += in0_stride;
        }
        {
            DeviceZoneScopedSumN1("RD_RSV");
            dfb_in1.reserve_back(n);
        }
        page = start_tile_id + k;
        offset = 0;
        for (uint32_t i = 0; i < n; ++i) {
            noc.async_read(src_b, dfb_in1, src_tile_bytes_b, {.page_id = page}, {.offset_bytes = offset});
            page += tile_step;
            offset += in1_stride;
        }
        {
            DeviceZoneScopedSumN2("RD_BAR");
            noc.async_read_barrier();
        }
        dfb_in0.push_back(n);
        dfb_in1.push_back(n);
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
    // Drains this thread's outstanding credits; the ack wait is unguarded and runs even at zero tiles,
    // which is benign there (posted == acked == 0). No deadlock because finish()'s thread barrier sits
    // inside handle_final_credits, reached only via the NocOptions::TXN_ID overloads this kernel avoids.
    dfb_in0.finish();
    dfb_in1.finish();
}
