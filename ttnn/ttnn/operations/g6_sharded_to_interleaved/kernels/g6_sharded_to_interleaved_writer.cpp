// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Owner-writes sharded -> interleaved: this core's resident shard rectangle is written straight to its
// interleaved output pages, one NoC write per tile, and drained by one barrier. No NoC read, no CB handshake.
//
// Helper deviation: raw dataflow NoC writes. No kernel_lib helper writes from a resident L1 shard to
// NoC-addressed interleaved pages (eltwise copy stages through DEST and re-derives bfp8 exponents; the
// local-copy helpers target local L1). The design's API Mapping records the same gap.
//
// One instance runs per write lane. The host hands each lane a disjoint rectangle of the shard, so the
// kernel has no lane-specific code: the processor and NoC come from the kernel config.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

// Writes one block: `rows x cols` tiles. Source slots keep the nominal shard stride even in a clipped
// last shard (the buffer's slot map advances by the nominal shard shape), so the row stride is the
// shard's, never the block's. Consecutive pages of a row sit in different banks, so every tile is its
// own write; nothing waits here, the caller drains once. With two write NoCs consecutive writes alternate
// between them, so each NoC carries half the traffic over its own links.
template <uint32_t page_size, uint32_t num_write_nocs, typename Accessor>
FORCE_INLINE void store_block(
    const Accessor& output,
    uint8_t& noc,
    uint32_t src_addr,
    uint32_t src_row_stride_bytes,
    uint32_t dst_page,
    uint32_t dst_row_stride_pages,
    uint32_t rows,
    uint32_t cols) {
    for (uint32_t r = 0; r < rows; ++r) {
        uint32_t src = src_addr;
        uint32_t page = dst_page;
        for (uint32_t c = 0; c < cols; ++c) {
            noc_async_write<page_size>(src, output.get_noc_addr(page, 0, noc), page_size, noc);
            if constexpr (num_write_nocs == 2) {
                noc ^= 1;
            }
            src += page_size;
            ++page;
        }
        src_addr += src_row_stride_bytes;
        dst_page += dst_row_stride_pages;
    }
}

// Walks a `rows x cols` rectangle as blocks of at most `block_rows x block_cols`, issuing every write.
template <uint32_t page_size, uint32_t num_write_nocs, typename Accessor>
FORCE_INLINE void store_rect(
    const Accessor& output,
    uint8_t& noc,
    uint32_t src_addr,
    uint32_t src_row_stride_bytes,
    uint32_t dst_page,
    uint32_t dst_row_stride_pages,
    uint32_t rows,
    uint32_t cols,
    uint32_t block_rows,
    uint32_t block_cols) {
    for (uint32_t block_row = 0; block_row < rows; block_row += block_rows) {
        const uint32_t brows = rows - block_row < block_rows ? rows - block_row : block_rows;
        for (uint32_t block_col = 0; block_col < cols; block_col += block_cols) {
            DeviceZoneScopedN("S2I-STORE-BLOCK");
            const uint32_t bcols = cols - block_col < block_cols ? cols - block_col : block_cols;
            store_block<page_size, num_write_nocs>(
                output,
                noc,
                src_addr + block_row * src_row_stride_bytes + block_col * page_size,
                src_row_stride_bytes,
                dst_page + block_row * dst_row_stride_pages + block_col,
                dst_row_stride_pages,
                brows,
                bcols);
        }
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_input_shard = get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    // 1 = this kernel's NoC only; 2 = both NoCs (the host compiles the kernel in dynamic NoC mode).
    constexpr uint32_t num_write_nocs = get_compile_time_arg_val(2);
    constexpr auto output_args = TensorAccessorArgs<3>();
    static_assert(num_write_nocs == 1 || num_write_nocs == 2);

    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t src_row_stride_slots = get_common_arg_val<uint32_t>(1);  // nominal shard_tile_cols
    const uint32_t dst_row_stride_pages = get_common_arg_val<uint32_t>(2);  // tensor_tile_cols

    // This lane's rectangle of the shard, and the block extents that tile it (Phase 0: one block).
    const uint32_t src_first_slot = get_arg_val<uint32_t>(0);
    const uint32_t dst_first_page = get_arg_val<uint32_t>(1);
    const uint32_t rect_tile_rows = get_arg_val<uint32_t>(2);
    const uint32_t rect_tile_cols = get_arg_val<uint32_t>(3);
    const uint32_t block_tile_rows = get_arg_val<uint32_t>(4);
    const uint32_t block_tile_cols = get_arg_val<uint32_t>(5);
    // Rows are issued from this one to the end, then from row 0: the host rotates cores whose rectangles start
    // on the same bank apart, so each write step lands on distinct banks. Two segments keep the walk modulo-free.
    const uint32_t issue_first_row = get_arg_val<uint32_t>(6);

    const auto output = TensorAccessor(output_args, output_addr, page_size);
    const uint32_t src_row_stride_bytes = src_row_stride_slots * page_size;
    const uint32_t rect_src_addr = get_read_ptr(cb_input_shard) + src_first_slot * page_size;

    uint8_t noc = noc_index;
    store_rect<page_size, num_write_nocs>(
        output,
        noc,
        rect_src_addr + issue_first_row * src_row_stride_bytes,
        src_row_stride_bytes,
        dst_first_page + issue_first_row * dst_row_stride_pages,
        dst_row_stride_pages,
        rect_tile_rows - issue_first_row,
        rect_tile_cols,
        block_tile_rows,
        block_tile_cols);
    store_rect<page_size, num_write_nocs>(
        output,
        noc,
        rect_src_addr,
        src_row_stride_bytes,
        dst_first_page,
        dst_row_stride_pages,
        issue_first_row,
        rect_tile_cols,
        block_tile_rows,
        block_tile_cols);

    {
        // Also on a lane with no tiles: every issued write is acked before the next program starts.
        DeviceZoneScopedN("S2I-WRITE-BARRIER");
        noc_async_write_barrier(noc_index);
        if constexpr (num_write_nocs == 2) {
            noc_async_write_barrier(1 - noc_index);
        }
    }
}
