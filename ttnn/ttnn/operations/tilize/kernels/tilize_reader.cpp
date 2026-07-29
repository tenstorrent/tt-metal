// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// tilize reader (NCRISC / NoC0).
//
// Two modes, selected by the `alias_mode` compile-time arg:
//
//   alias_mode == 1  (Path B, zero-copy)
//       cb_rm_input is aliased onto the resident L1 ROW_MAJOR shard, so the
//       bytes are already at the CB's address. One cb_reserve_back /
//       cb_push_back arms the whole shard; there is no NoC traffic at all.
//
//   alias_mode == 0  (Path A / C)
//       Chunk-outer, tile-row-inner. For each column chunk we hand a whole
//       tile-row band to dataflow_kernel_lib::read_sticks_for_tilize in TILE
//       granularity, which owns cb_reserve_back / 32 strided reads / one
//       noc_async_read_barrier / cb_push_back per block.
//
//       When the source is ROW_MAJOR-*sharded* with more than one page per
//       logical row (`row_page_stride > 1`) the helper cannot be used: its
//       page index advances by exactly 1 per row
//       (tilize_helpers_dataflow.inl:121), hard-coding "one page == one full
//       logical row", and the signature exposes no row-stride parameter. The
//       raw fallback below mirrors the helper's block structure exactly
//       (reserve chunk_wt, 32 reads, one barrier, push chunk_wt) so lever B7
//       (one barrier per block) still holds.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_rm_input = 0;

    constexpr uint32_t alias_mode = get_compile_time_arg_val(0);
    constexpr uint32_t chunk_wt = get_compile_time_arg_val(1);
    constexpr uint32_t chunk_row_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t row_page_stride = get_compile_time_arg_val(3);
    constexpr uint32_t source_page_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t shard_tiles = get_compile_time_arg_val(5);
    constexpr auto src_args = TensorAccessorArgs<6>();

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

            if constexpr (row_page_stride == 1) {
                dataflow_kernel_lib::read_sticks_for_tilize<cb_rm_input, dataflow_kernel_lib::TilizeGranularity::TILE>(
                    accessor, num_rows, chunk_row_bytes, start_row, byte_offset);
            } else {
                // A chunk never straddles a source page (host guarantees
                // chunk_row_bytes divides source_page_bytes), so the whole
                // chunk lives in one page at a fixed intra-page offset.
                const uint32_t page_col = byte_offset / source_page_bytes;
                const uint32_t offset_in_page = byte_offset - page_col * source_page_bytes;
                const uint32_t blocks = num_rows / 32;

                for (uint32_t block = 0; block < blocks; ++block) {
                    const uint32_t row0 = start_row + block * 32;
                    cb_reserve_back(cb_rm_input, chunk_wt);
                    uint32_t l1_addr = get_write_ptr(cb_rm_input);
                    for (uint32_t row = 0; row < 32; ++row) {
                        const uint64_t noc_addr =
                            accessor.get_noc_addr((row0 + row) * row_page_stride + page_col, offset_in_page);
                        noc_async_read(noc_addr, l1_addr, chunk_row_bytes);
                        l1_addr += chunk_row_bytes;
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_rm_input, chunk_wt);
                }
            }
        }
    }
}
