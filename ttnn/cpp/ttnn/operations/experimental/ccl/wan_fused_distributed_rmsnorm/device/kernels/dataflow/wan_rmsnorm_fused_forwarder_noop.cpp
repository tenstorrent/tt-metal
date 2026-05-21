// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * No-op forwarder kernel for TP=1 validation of the fused Wan2.2 distributed
 * RMSNorm op.
 *
 * For TP=1 the AG is degenerate: each device's local stats ARE the global
 * stats. This kernel simply moves tiles from stats_local_cb (produced by the
 * compute kernel's pre phase) into stats_gathered_cb (consumed by the same
 * compute kernel's post phase) without any fabric activity.
 *
 * Compute writes 1 stat tile per row; for TP=1, stats_tiles_cols == 1, so we
 * push exactly the same number of tiles into the gathered CB.
 *
 * Real fabric forwarder for ring TP>1 lives in a separate kernel
 * (next session) and uses the fabric APIs from all_gather_minimal_matmul_async.
 */

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t stats_local_cb = get_compile_time_arg_val(0);
    constexpr uint32_t stats_gathered_cb = get_compile_time_arg_val(1);

    const uint32_t num_tile_rows = get_arg_val<uint32_t>(0);

    const uint32_t stats_tile_bytes = get_tile_size(stats_local_cb);

    for (uint32_t row = 0; row < num_tile_rows; row++) {
        cb_wait_front(stats_local_cb, 1);
        cb_reserve_back(stats_gathered_cb, 1);

        const uint32_t src_addr = get_read_ptr(stats_local_cb);
        const uint32_t dst_addr = get_write_ptr(stats_gathered_cb);

        // Same-core L1 copy via NoC (loopback). Simplest path for TP=1; real
        // forwarder uses fabric writes for cross-chip propagation.
        noc_async_write(src_addr, get_noc_addr(dst_addr), stats_tile_bytes);
        noc_async_write_barrier();

        cb_pop_front(stats_local_cb, 1);
        cb_push_back(stats_gathered_cb, 1);
    }
}
