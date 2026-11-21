// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// add_ reader (block operation `credit_shard_block`, regime `sharded_resident`).
//
// Both operands are height-sharded in L1, and the rings cb_a_shard / cb_b_shard are BORROWED from the
// resident shards: the bytes are already where the compute kernel unpacks them. This kernel moves no data
// (no NoC transfer, no TensorAccessor, no transaction tag); it only posts the credits that say the
// resident tiles are there. A ring has exactly one producer, and there is no "already full" state a ring
// can be declared in, so resident data needs a producer that writes nothing.
//
// Raw DFB credit calls, no helper: the kernel library has no helper that credits a resident ring
// (dfb_helpers_dataflow.hpp has only tile-dim getters; local_copy_helpers_dataflow.hpp moves bytes, which
// is exactly the traffic the borrowed ring removes). Recorded in op_design.md, "Helpers considered".
//
// Thread plan: hart p of R serves NEOs p, p+R, ... A STRIDED DM-side reserve/push credits the current
// NEO's tile counter and rotates to the next, so the k-th pair of calls of hart p credits NEO p + k*R.
// NEO t's share is the round-robin deal of this core's tiles: core_tiles / C, plus one for t < core_tiles % C.
// Shares are non-increasing in t, so only trailing zero shares are skipped and the rotation stays aligned.
// The rings are listed in this kernel's disable_dfb_implicit_sync_for (explicit sync).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t core_tiles = get_arg(args::core_tiles);
    // The plan's counts, not get_num_threads(): were a thread not to run, its tiles would be missing.
    constexpr uint32_t R = get_arg(args::reader_threads);
    constexpr uint32_t C = get_arg(args::compute_threads);

    DataflowBuffer dfb_a(dfb::cb_a_shard);
    DataflowBuffer dfb_b(dfb::cb_b_shard);

    for (uint32_t t = get_my_thread_id(); t < C; t += R) {
        const uint32_t share = core_tiles / C + (t < core_tiles % C ? 1u : 0u);
        if (share == 0) {
            break;  // trailing NEOs with nothing to do: their counters stay untouched
        }
        // Nothing has been pushed yet, so the whole per-NEO capacity is free: these return immediately.
        dfb_a.reserve_back(share);
        dfb_a.push_back(share);
        dfb_b.reserve_back(share);
        dfb_b.push_back(share);
    }
}
