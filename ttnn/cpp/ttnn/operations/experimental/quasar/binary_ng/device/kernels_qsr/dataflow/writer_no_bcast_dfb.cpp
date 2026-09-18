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

    constexpr uint32_t onetile = 1;

    Noc noc;
    DataflowBuffer dfb_out(dfb::out);

    const uint32_t dst_tile_bytes = dfb_out.get_entry_size();
    const auto dst = TensorAccessor(tensor::out);

    const uint32_t thread_id = get_my_thread_id();
    const uint32_t num_threads = get_num_threads();

    for (uint32_t k = thread_id; k < dst_num_tiles; k += num_threads) {
        // DeviceZoneScopedSum* feeds the work-split gate (per-thread WR_WAIT / WR_BAR). Zero-cost
        // unless TT_METAL_PROFILER_SUM=1, and compiled OUT under PROFILER_OPT_DO_ACCUMULATE.
        // Not "nativeness" -- keep when reconciling against kernels_dfb/.
        {
            DeviceZoneScopedSumN1("WR_WAIT");
            dfb_out.wait_front(onetile);
        }
        noc.async_write(dfb_out, dst, dst_tile_bytes, {}, {.page_id = start_tile_id + k});
        {
            DeviceZoneScopedSumN2("WR_BAR");
            noc.async_write_barrier();
        }
        dfb_out.pop_front(onetile);
    }
    // Drains outstanding credits before exit. Its sync_threads is reached only when a thread did
    // work, so a zero-work thread would skip the barrier while its siblings block -- the gate's even
    // divisibility is what keeps every thread non-empty.
    dfb_out.finish();
}
