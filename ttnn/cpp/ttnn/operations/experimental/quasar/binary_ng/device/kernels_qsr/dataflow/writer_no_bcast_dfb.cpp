// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 / DataflowBuffer (DFB) writer for binary_ng's no-broadcast binary op, Quasar-native.
//
// Diverges from kernels_dfb/dataflow/writer_no_bcast_dfb.cpp in two ways, both licensed by
// matches_quasar_native_slice:
//   - the nD stride cascade is gone: page = start_tile_id + k.
//   - the tile loop is per-thread. Thread t of N drains the STRIDED share {t, t+N, t+2N, ...}, which
//     is the slot assignment the DFB gives consumer thread t.
// The output is interleaved: the gate rejects a sharded output, so no borrowed-shard branch exists.
//
// The writer's cascade is the milder case -- an output is never broadcast, so its strides are always
// dense and only the sharded-row wrap (dst_shard_width) is lost, which the gate rejects. The reader's
// is the load-bearing one; see the note in reader_no_bcast_dfb.cpp before widening the gate.

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
    DataflowBuffer dfb_out(dfb::out);

    const uint32_t dst_tile_bytes = dfb_out.get_entry_size();
    const auto dst = TensorAccessor(tensor::out);

    const uint32_t thread_id = get_my_thread_id();
    const uint32_t num_threads = get_num_threads();

    // Entry i of a batch sits at i * stride_size, not i * entry_size -- see the reader.
    const uint32_t out_stride = dfb_out.get_stride_size();

    // n of this thread's tiles starting at k. Page and ring offset walk by addition -- see the reader.
    auto write_batch = [&](uint32_t k, uint32_t n) {
        // DeviceZoneScopedSum* feeds the work-split gate (per-thread WR_WAIT / WR_BAR). Zero-cost
        // unless TT_METAL_PROFILER_SUM=1, and compiled OUT under PROFILER_OPT_DO_ACCUMULATE.
        // Not "nativeness" -- keep when reconciling against kernels_dfb/.
        {
            DeviceZoneScopedSumN1("WR_WAIT");
            dfb_out.wait_front(n);
        }
        uint32_t page = start_tile_id + k;
        uint32_t offset = 0;
        for (uint32_t i = 0; i < n; ++i) {
            noc.async_write(dfb_out, dst, dst_tile_bytes, {.offset_bytes = offset}, {.page_id = page});
            page += num_threads;
            offset += out_stride;
        }
        {
            DeviceZoneScopedSumN2("WR_BAR");
            noc.async_write_barrier();
        }
        dfb_out.pop_front(n);
    };

    // Full batches then at most one short one -- see the reader for why the tail cannot exceed one.
    const uint32_t batch_span = (dm_batch - 1) * num_threads;
    const uint32_t full_limit = dst_num_tiles > batch_span ? dst_num_tiles - batch_span : 0;
    const uint32_t k_step = dm_batch * num_threads;

    uint32_t k = thread_id;
    for (; k < full_limit; k += k_step) {
        write_batch(k, dm_batch);
    }
    if (k < dst_num_tiles) {
        uint32_t n = 1;
        for (uint32_t t = k + num_threads; t < dst_num_tiles; t += num_threads) {
            ++n;
        }
        write_batch(k, n);
    }
    // Drains this thread's outstanding credits; the ack wait is unguarded and runs even at zero tiles,
    // which is benign there (posted == acked == 0). No deadlock because finish()'s thread barrier sits
    // inside handle_final_credits, reached only via the NocOptions::TXN_ID overloads this kernel avoids.
    dfb_out.finish();
}
