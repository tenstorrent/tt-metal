// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tilize reader (NCRISC / NoC0).
//
// Modes, selected by compile-time args:
//
//   alias_mode == 1  (Path B, zero-copy)
//       cb_rm_input is aliased onto the resident L1 ROW_MAJOR shard, so the
//       bytes are already at the CB's address. One cb_reserve_back /
//       cb_push_back arms the whole shard; there is no NoC traffic at all.
//
//   alias_mode == 0, split_read == 0  (Path A / C, single reader)
//       Chunk-outer, tile-row-inner. For each column chunk we hand a whole
//       tile-row band to dataflow_kernel_lib::read_sticks_for_tilize in TILE
//       granularity, which owns cb_reserve_back / 32 strided reads / one
//       noc_async_read_barrier / cb_push_back per block. `stateful_read`
//       selects StickReadMode::Stateful (lever B13) inside the helper.
//
//   alias_mode == 0, split_read == 1  (Path A / C, split reader — lever C7)
//       The 32 stick reads of each block are shared with BRISC (the writer
//       kernel), which parks in cb_wait_front for the whole read window
//       otherwise. NCRISC keeps sole ownership of the CB — a circular buffer
//       must have exactly ONE producer, so BRISC never reserves or pushes
//       cb_rm_input; it is handed the reserved window through two counting
//       semaphores:
//           NCRISC: reserve -> sem_reserve = blk+1 -> read half -> barrier
//                   -> wait sem_done >= blk+1 -> push
//           BRISC : wait sem_reserve >= blk+1 -> read half -> barrier
//                   -> sem_done = blk+1
//       Both semaphores are monotonic per-launch counters and both live in this
//       core's own L1, so set/wait are plain local loads and stores (no NoC
//       round trip). Requires depth == 1 so the reserved window is always the
//       CB base address, which is what BRISC's untouched get_write_ptr returns
//       (see the host gate in tilize_program_descriptor.py).
//
//       When the source is ROW_MAJOR-*sharded* with more than one page per
//       logical row (`row_page_stride > 1`) neither helper path can be used:
//       their page index advances by exactly 1 per row, hard-coding "one page ==
//       one full logical row", and the signature exposes no row-stride
//       parameter. The raw fallback below mirrors the helper's block structure
//       exactly (reserve chunk_wt, 32 reads, one barrier, push chunk_wt) so
//       lever B7 (one barrier per block) still holds.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_rm_input = 0;
    constexpr uint32_t tile_height = 32;  // rows per tile-row block

    constexpr uint32_t alias_mode = get_compile_time_arg_val(0);
    constexpr uint32_t chunk_wt = get_compile_time_arg_val(1);
    constexpr uint32_t chunk_row_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t row_page_stride = get_compile_time_arg_val(3);
    constexpr uint32_t source_page_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t shard_tiles = get_compile_time_arg_val(5);
    // Perf-ablation only (TILIZE_SKIP_DM=1): drop the noc_async_read *payload* and
    // keep every CB op, barrier, handshake and loop trip count, so /perf-measure
    // can attribute time to the read stage. Never set on a correctness run.
    constexpr uint32_t skip_dm = get_compile_time_arg_val(6);
    constexpr uint32_t stateful_read = get_compile_time_arg_val(7);  // lever B13
    constexpr uint32_t split_read = get_compile_time_arg_val(8);     // lever C7
    constexpr uint32_t sem_reserve_id = get_compile_time_arg_val(9);
    constexpr uint32_t sem_done_id = get_compile_time_arg_val(10);
    constexpr auto src_args = TensorAccessorArgs<11>();

    using dataflow_kernel_lib::StickReadMode;
    constexpr StickReadMode read_mode = stateful_read ? StickReadMode::Stateful : StickReadMode::Generic;
    static_assert(!split_read || row_page_stride == 1, "the split reader needs one source page per logical row");

    if constexpr (alias_mode) {
        // Data is already resident at the CB address — just hand it to compute.
        cb_reserve_back(cb_rm_input, shard_tiles);
        cb_push_back(cb_rm_input, shard_tiles);
        return;
    } else {
        const uint32_t src_addr = get_arg_val<uint32_t>(0);
        const uint32_t start_row = get_arg_val<uint32_t>(1);
        const uint32_t num_rows = get_arg_val<uint32_t>(2);
        const uint32_t chunk_start = get_arg_val<uint32_t>(3);
        const uint32_t chunk_count = get_arg_val<uint32_t>(4);

        const auto accessor = TensorAccessor(src_args, src_addr);

        for (uint32_t c = 0; c < chunk_count; ++c) {
            const uint32_t byte_offset = (chunk_start + c) * chunk_row_bytes;

            if constexpr (row_page_stride == 1 && !split_read && !skip_dm) {
                dataflow_kernel_lib::
                    read_sticks_for_tilize<cb_rm_input, dataflow_kernel_lib::TilizeGranularity::TILE, read_mode>(
                        accessor, num_rows, chunk_row_bytes, start_row, byte_offset);
            } else if constexpr (row_page_stride == 1 && split_read) {
                // Lever C7. The CB dance stays here (single producer); the row
                // band is split with BRISC by bank group inside the helper.
                volatile tt_l1_ptr uint32_t* sem_reserve =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_reserve_id));
                volatile tt_l1_ptr uint32_t* sem_done =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_done_id));
                const uint32_t blocks = num_rows / tile_height;

                for (uint32_t block = 0; block < blocks; ++block) {
                    const uint32_t first_page = start_row + block * tile_height;
                    cb_reserve_back(cb_rm_input, chunk_wt);
                    const uint32_t l1_addr = get_write_ptr(cb_rm_input);
                    // The window is free: hand it to BRISC. Sequence numbers are
                    // per (chunk, block) so they stay monotonic across chunks.
                    const uint32_t seq = c * blocks + block + 1;
                    noc_semaphore_set(sem_reserve, seq);

                    if constexpr (!skip_dm) {
                        dataflow_kernel_lib::read_stick_rows_for_tilize<read_mode, 2>(
                            accessor,
                            first_page,
                            chunk_row_bytes,
                            byte_offset,
                            l1_addr,
                            chunk_row_bytes,
                            tile_height,
                            /*split_id=*/0);
                    }
                    noc_async_read_barrier();
                    noc_semaphore_wait_min(sem_done, seq);
                    cb_push_back(cb_rm_input, chunk_wt);
                }
            } else {
                // A chunk never straddles a source page (host guarantees
                // chunk_row_bytes divides source_page_bytes), so the whole
                // chunk lives in one page at a fixed intra-page offset.
                const uint32_t page_col = byte_offset / source_page_bytes;
                const uint32_t offset_in_page = byte_offset - page_col * source_page_bytes;
                const uint32_t blocks = num_rows / tile_height;

                for (uint32_t block = 0; block < blocks; ++block) {
                    const uint32_t row0 = start_row + block * tile_height;
                    cb_reserve_back(cb_rm_input, chunk_wt);
                    uint32_t l1_addr = get_write_ptr(cb_rm_input);
                    for (uint32_t row = 0; row < tile_height; ++row) {
                        const uint64_t noc_addr =
                            accessor.get_noc_addr((row0 + row) * row_page_stride + page_col, offset_in_page);
                        if constexpr (skip_dm) {
                            // Ablation: keep the address-gen observable so dead-code
                            // elimination cannot delete the loop being timed.
                            volatile uint32_t sink = static_cast<uint32_t>(noc_addr);
                            (void)sink;
                        } else {
                            noc_async_read(noc_addr, l1_addr, chunk_row_bytes);
                        }
                        l1_addr += chunk_row_bytes;
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_rm_input, chunk_wt);
                }
            }
        }
    }
}
