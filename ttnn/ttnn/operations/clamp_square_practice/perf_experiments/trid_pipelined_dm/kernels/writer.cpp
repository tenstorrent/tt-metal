// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// clamp_square_practice writer (BRISC / NoC1) — CANDIDATE: transaction-id pipelined store.
//
// Baseline (op's writer): wait, `valid` writes, noc_async_writes_flushed(), pop — the flush waits
// until EVERY outstanding write has left L1 before the next block is issued.
// Candidate: block k's writes are tagged with trid (k % T) + 1; once T blocks are outstanding, the
// OLDEST id is waited on with noc_async_write_flushed_with_trid (its L1 source reads are done) and
// its slot popped, while the T-1 newer blocks' writes stay queued in the NIU.
//
// READ SLOT: the CB read pointer only advances on cb_pop_front, so it lags the in-flight blocks.
// Block k is read from get_read_ptr() + in_flight slots (wrapped at the CB end), after waiting for
// (in_flight + 1) blocks.
//
// RAW NOC (no helper): kernel_lib has no write-side trid wrapper; noc_async_write_set_trid +
// noc_async_write_flushed_with_trid are used directly. A plain noc_async_write never writes
// NOC_PACKET_TAG, so the tag set once rides on every subsequent write on write_cmd_buf.
// Defines: TRID_DEPTH (T), CB_BLOCKS (CB capacity in blocks), ABLATE_WRITE (stub the NoC writes).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

#ifndef ABLATE_WRITE
#define ABLATE_WRITE 0
#endif

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr auto output_args = TensorAccessorArgs<3>();

    constexpr uint32_t cb_blocks = CB_BLOCKS;
    constexpr uint32_t depth = TRID_DEPTH < cb_blocks ? TRID_DEPTH : cb_blocks;
    static_assert(depth >= 1 && depth <= NOC_MAX_TRANSACTION_ID, "trid depth out of range");
    static_assert(block_tiles <= NOC_MAX_TRANSACTION_ID_COUNT, "block exceeds per-id counter");

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    const uint32_t core_tiles = get_arg_val<uint32_t>(2);

    const auto output_accessor = TensorAccessor(output_args, dst_addr, page_bytes);

    constexpr uint32_t block_bytes = block_tiles * page_bytes;
    constexpr uint32_t cb_bytes = cb_blocks * block_bytes;
    const uint32_t cb_base = get_read_ptr(cb_output_tiles);
    const uint32_t cb_end = cb_base + cb_bytes;
    auto slot = [=](uint32_t from, uint32_t j) {
        const uint32_t a = from + j * block_bytes;
        return a >= cb_end ? a - cb_bytes : a;
    };

    const uint32_t num_blocks = (core_tiles + block_tiles - 1) / block_tiles;
    {
        MaybeDeviceZoneScope("writer_write");
        for (uint32_t k = 0; k < num_blocks; ++k) {
            const uint32_t trid = (k % depth) + 1;
            if (k >= depth) {
                noc_async_write_flushed_with_trid(trid);  // oldest block's L1 reads done -> recycle
                cb_pop_front(cb_output_tiles, block_tiles);
            }
            const uint32_t in_flight = (k < depth) ? k : (depth - 1);
            cb_wait_front(cb_output_tiles, (in_flight + 1) * block_tiles);
            uint32_t l1_read_addr = slot(get_read_ptr(cb_output_tiles), in_flight);
            const uint32_t done = k * block_tiles;
            const uint32_t remaining = core_tiles - done;
            const uint32_t valid = remaining < block_tiles ? remaining : block_tiles;
            const uint32_t page0 = start_tile + done;
            noc_async_write_set_trid(trid);
            for (uint32_t i = 0; i < valid; ++i) {
                if constexpr (!ABLATE_WRITE) {
                    noc_async_write_page(page0 + i, output_accessor, l1_read_addr);
                }
                l1_read_addr += page_bytes;
            }
        }
        const uint32_t to_drain = num_blocks < depth ? num_blocks : depth;
        for (uint32_t d = 0; d < to_drain; ++d) {
            noc_async_write_flushed_with_trid(((num_blocks - to_drain + d) % depth) + 1);
            cb_pop_front(cb_output_tiles, block_tiles);
        }
        noc_async_write_set_trid(0);  // restore untagged
    }
    {
        MaybeDeviceZoneScope("writer_final_barrier");
        noc_async_write_barrier();
    }
}
