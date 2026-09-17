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

    constexpr uint32_t onetile = 1;

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
    for (uint32_t k = thread_id; k < dst_num_tiles; k += num_threads) {
        const uint32_t page_id = start_tile_id + k;
        // DeviceZoneScopedSum* feeds the work-split gate (per-thread RD_RSV / RD_BAR). Zero-cost
        // unless TT_METAL_PROFILER_SUM=1, and compiled OUT under PROFILER_OPT_DO_ACCUMULATE.
        // Not "nativeness" -- keep when reconciling against kernels_dfb/.
        {
            DeviceZoneScopedSumN1("RD_RSV");
            dfb_in0.reserve_back(onetile);
        }
        noc.async_read(src, dfb_in0, src_tile_bytes, {.page_id = page_id}, {.offset_bytes = 0});
        {
            DeviceZoneScopedSumN1("RD_RSV");
            dfb_in1.reserve_back(onetile);
        }
        noc.async_read(src_b, dfb_in1, src_tile_bytes_b, {.page_id = page_id}, {.offset_bytes = 0});
        {
            DeviceZoneScopedSumN2("RD_BAR");
            noc.async_read_barrier();
        }
        dfb_in0.push_back(onetile);
        dfb_in1.push_back(onetile);
    }
    // Drains outstanding credits before exit. Its sync_threads is reached only when a thread did
    // work, so a zero-work thread would skip the barrier while its siblings block -- the gate's even
    // divisibility is what keeps every thread non-empty.
    dfb_in0.finish();
    dfb_in1.finish();
}
