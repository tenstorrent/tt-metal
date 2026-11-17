// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute-core writer (RISCV_0, NoC1).
//   export_partial  (q == p) col 0 of each cb_partial tile (32 fp32 row sums, bit-exact) -> my stats[p][rows] -> the
//                   hub's stats[p][rows] (one NoC write per block), then own_partials++ on the hub
//   store_block     cb_out in chunk_tiles quanta -> output tiles (m*block_h + r, q*Wl_t + j); one flush per chunk, one
//   barrier at the end
//
// Raw dataflow: no helper moves a tile column (export), and TensorAccessor is the store helper.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "all_gather_rms_norm_kimi_common.hpp"
#include "all_gather_rms_norm_kimi_perf.hpp"

using namespace agrn;

constexpr uint32_t C = NUM_GEO_CT;
constexpr uint32_t cb_partial = get_compile_time_arg_val(C + 0);
constexpr uint32_t cb_out = get_compile_time_arg_val(C + 1);
constexpr uint32_t sem_own_partials = get_compile_time_arg_val(C + 2);
constexpr uint32_t tile_bytes = get_compile_time_arg_val(C + 3);
constexpr uint32_t sem_landed = get_compile_time_arg_val(C + 4);
constexpr uint32_t own_block_reads =
    get_compile_time_arg_val(C + 5);  // OWN_STORE_GATE: ports that release this own block; 0 = off
constexpr auto out_args = TensorAccessorArgs<C + 6>();

void kernel_main() {
    // common args: out_addr, stats_addr, p, hub xy, row_base[grid_y]
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t stats_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t p = get_common_arg_val<uint32_t>(2);
    const uint32_t hub_xy = get_common_arg_val<uint32_t>(3);
    constexpr uint32_t row_base_idx = 4;
    const BlockId b = block_id(get_absolute_logical_x(), get_absolute_logical_y(), row_base_idx, [](uint32_t i) {
        return get_common_arg_val<uint32_t>(i);
    });

    if (b.q == p) {
        {
            MaybeDeviceZoneScope("writer_partial_wait");
            cb_wait_front(cb_partial, b.h);
        }
        MaybeDeviceZoneScope("writer_export");
        const uint32_t src_base = get_read_ptr(cb_partial);
        const uint32_t slot = stats_addr + (p * Mt + b.m * block_h) * STATS_ROW_BYTES;
        for (uint32_t r = 0; r < b.h; ++r) {
            col0_to_stats_row(src_base + r * FP32_TILE_BYTES, slot + r * STATS_ROW_BYTES);
        }
        cb_pop_front(cb_partial, b.h);
        noc_async_write(slot, get_noc_addr(hub_xy >> 8, hub_xy & 0xFF, slot), b.h * STATS_ROW_BYTES);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(hub_xy >> 8, hub_xy & 0xFF, get_semaphore(sem_own_partials)), 1);
    }

    // ---- store_block ---- (an own block waits until every port sending it has issued its reads)
    if (own_block_reads > 0 && b.q == p) {
        MaybeDeviceZoneScope("writer_gate_wait");
        noc_semaphore_wait_min(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_landed)), own_block_reads);
    }
    MaybeDeviceZoneScope("writer_store");  // per-chunk cb_out waits inside
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);
    const uint32_t n_chunks = b.h * block_w / chunk_tiles;
    uint32_t t = 0;
    for (uint32_t k = 0; k < n_chunks; ++k) {
        cb_wait_front(cb_out, chunk_tiles);
        uint32_t src = get_read_ptr(cb_out);
        for (uint32_t i = 0; i < chunk_tiles; ++i, ++t) {
            const uint32_t r = t / block_w;
            const uint32_t j = t - r * block_w;
            const uint32_t page = (b.m * block_h + r) * Wt + b.q * Wl_t + j;
            noc_async_write(src, out.get_noc_addr(page), tile_bytes);
            src += tile_bytes;
        }
        noc_async_writes_flushed();  // the chunk's bytes have left L1: its cb_out pages can be reused
        cb_pop_front(cb_out, chunk_tiles);
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
