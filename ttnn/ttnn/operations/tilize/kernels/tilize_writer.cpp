// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tilize writer (BRISC / NoC1).
//
//   alias_mode == 1  (Path B, zero-copy)
//       cb_tiled_output is aliased onto the resident L1 TILE shard, so compute
//       has already written the bytes to their final address. One
//       cb_wait_front / cb_pop_front drains the shard; no NoC traffic.
//
//   alias_mode == 0  (Path A / C)
//       Whole-TILE-page writes through the output TensorAccessor, chunk_wt
//       writes per barrier (lever B7).
//
// RAW-API NOTE (helper substitution, deliberate): there is no kernel_lib
// dataflow helper that moves TILE pages from a CB to a TensorAccessor-addressed
// buffer. The only write helper, dataflow_kernel_lib::write_sticks_after_untilize
// (tilize_helpers_dataflow.hpp:129-135), writes ROW_MAJOR *sticks* — its inner
// loop issues one noc_async_write of row_bytes per row and advances the L1
// pointer by padded_row_bytes (inl:232-236), i.e. it de-interleaves a tile back
// into 32 sticks. It is the untilize partner, the wrong direction; using it here
// would write tile bytes to stick addresses and destroy the layout.
//
// The iteration order MUST match the reader's: chunk-outer, tile-row-inner.
// read_sticks_for_tilize loops over tile-row blocks internally, so the chunk
// loop is the caller's outer loop. Reversing it keeps every CB count balanced
// and silently transposes the output blocks.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_tiled_output = 16;

    constexpr uint32_t alias_mode = get_compile_time_arg_val(0);
    constexpr uint32_t chunk_wt = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t wt = get_compile_time_arg_val(3);
    constexpr uint32_t shard_tiles = get_compile_time_arg_val(4);
    // Perf-ablation only (TILIZE_SKIP_DM=1) — see the reader for the contract.
    constexpr uint32_t skip_dm = get_compile_time_arg_val(5);
    constexpr auto dst_args = TensorAccessorArgs<6>();

    if constexpr (alias_mode) {
        cb_wait_front(cb_tiled_output, shard_tiles);
        cb_pop_front(cb_tiled_output, shard_tiles);
        return;
    } else {
        const uint32_t dst_addr = get_arg_val<uint32_t>(0);
        const uint32_t row_start = get_arg_val<uint32_t>(1);
        const uint32_t row_count = get_arg_val<uint32_t>(2);
        const uint32_t chunk_start = get_arg_val<uint32_t>(3);
        const uint32_t chunk_count = get_arg_val<uint32_t>(4);

        const auto accessor = TensorAccessor(dst_args, dst_addr);

        for (uint32_t c = 0; c < chunk_count; ++c) {
            const uint32_t col0 = (chunk_start + c) * chunk_wt;
            for (uint32_t r = 0; r < row_count; ++r) {
                const uint32_t base_page = (row_start + r) * wt + col0;

                cb_wait_front(cb_tiled_output, chunk_wt);
                uint32_t l1_addr = get_read_ptr(cb_tiled_output);
                for (uint32_t k = 0; k < chunk_wt; ++k) {
                    const uint64_t noc_addr = accessor.get_noc_addr(base_page + k);
                    if constexpr (skip_dm) {
                        volatile uint32_t sink = static_cast<uint32_t>(noc_addr);
                        (void)sink;
                    } else {
                        noc_async_write(l1_addr, noc_addr, tile_bytes);
                    }
                    l1_addr += tile_bytes;
                }
                // Recycling the CB pages only requires the writes to have DEPARTED
                // (the data read out of L1), not to have been acked by the
                // destination — that is exactly noc_async_writes_flushed
                // (dataflow_api.h:1802 "wait for ... calls to depart, but will not
                // wait for them to complete"). A full barrier per block would idle
                // BRISC for the round-trip latency of the last tile of every block.
                // One barrier after the loop still guarantees completion before the
                // kernel ends.
                noc_async_writes_flushed();
                cb_pop_front(cb_tiled_output, chunk_wt);
            }
        }
        noc_async_write_barrier();
    }
}
