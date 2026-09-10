// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// writer_dual_risc bench — THE DECISIVE CHECK (do this before anything else).
//
// Isolates the writer's own store loop with NO reader and NO compute at all:
// each active core seeds its own `block_width_tiles` (8) output tiles once,
// UNMEASURED, then the measured section is exactly the real op's
// `store_block` payload — 8x `noc_async_write<out_tile_bytes>` behind ONE
// barrier, `write_rows_per_barrier == 1` — with NOTHING else running on the
// core. Varying `num_active_cores` while holding EVERY core's own payload
// fixed at 8 tiles / 16 KiB (identical to the focus shape's per-core block)
// answers: is the store issue-bound (per-core time flat as more cores join)
// or DRAM-bandwidth-bound (per-core time grows as more cores join and start
// contending for the same banks)? See report for the measured numbers.
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_width_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t out_tile_bytes = get_compile_time_arg_val(2);
    constexpr auto out_args = TensorAccessorArgs<3>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t col_base = get_arg_val<uint32_t>(1);  // this core's page_base (row 0, C == num_active_cores*bw)

    const auto out_acc = TensorAccessor(out_args, dst_addr);

    // Seed ONE batch's worth of tiles, UNMEASURED and UNFILLED. This bench
    // measures TIMING ONLY (it never claims correctness — see the report), so
    // the L1 bytes the write below moves are whatever garbage already sits
    // there. A per-word RISC store loop here would itself cost thousands of ns
    // (4096 words/core) and get counted in `DEVICE KERNEL DURATION [ns]`
    // (a whole-kernel-span metric, NOT a sum of the zones below) — silently
    // swamping the very NoC cost this bench exists to isolate.
    cb_reserve_back(cb_output_tiles, block_width_tiles);
    cb_push_back(cb_output_tiles, block_width_tiles);

    // --- measured: the real writer's own payload, nothing else in flight ---
    {
        MaybeDeviceZoneScope("writer_wait_out");
        cb_wait_front(cb_output_tiles, block_width_tiles);
    }
    uint32_t l1_read_addr = get_read_ptr(cb_output_tiles);
    {
        MaybeDeviceZoneScope("writer_issue");
        for (uint32_t i = 0; i < block_width_tiles; ++i) {
            noc_async_write<out_tile_bytes>(l1_read_addr, out_acc.get_noc_addr(col_base + i), out_tile_bytes);
            l1_read_addr += out_tile_bytes;
        }
    }
    {
        MaybeDeviceZoneScope("writer_barrier");
        noc_async_write_barrier();
    }
    cb_pop_front(cb_output_tiles, block_width_tiles);
}
