// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Owner-writes sharded -> interleaved: this core's resident shard rectangle is written straight to its
// interleaved output pages, one NoC write per tile, and drained by one barrier. No NoC read, no CB handshake.
//
// Helper deviation: raw dataflow NoC writes. No kernel_lib helper writes from a resident L1 shard to
// NoC-addressed interleaved pages (eltwise copy stages through DEST and re-derives bfp8 exponents; the
// local-copy helpers target local L1). The design's API Mapping records the same gap.
//
// One instance runs per write lane; each takes a disjoint rectangle of the shard. The processor and NoC come
// from the kernel config.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

FORCE_INLINE uint32_t clip(uint32_t extent, uint32_t limit) { return extent < limit ? extent : limit; }

// This core's shard index: the buffer's own shard -> core walk (`corerange_to_cores`), i.e. the ranges in their
// given order, each walked along rows (row-major orientation) or columns. Each range is 4 common args
// (x0, y0, x1, y1) starting at `first_range_arg`.
FORCE_INLINE uint32_t shard_index(uint32_t first_range_arg, uint32_t num_ranges, bool row_wise) {
    const uint32_t x = get_absolute_logical_x();
    const uint32_t y = get_absolute_logical_y();
    uint32_t k = 0;
    for (uint32_t i = 0; i < num_ranges; ++i) {
        const uint32_t arg = first_range_arg + 4 * i;
        const uint32_t x0 = get_common_arg_val<uint32_t>(arg);
        const uint32_t y0 = get_common_arg_val<uint32_t>(arg + 1);
        const uint32_t x1 = get_common_arg_val<uint32_t>(arg + 2);
        const uint32_t y1 = get_common_arg_val<uint32_t>(arg + 3);
        const uint32_t width = x1 - x0 + 1;
        const uint32_t height = y1 - y0 + 1;
        if (x >= x0 && x <= x1 && y >= y0 && y <= y1) {
            return k + (row_wise ? (y - y0) * width + (x - x0) : (x - x0) * height + (y - y0));
        }
        k += width * height;
    }
    return k;
}

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
    // This instance takes piece `lane` of `num_write_lanes`, the block cut along its longer tile axis with the
    // larger pieces first (the host's `_lane_rects`).
    constexpr uint32_t lane = get_compile_time_arg_val(3);
    constexpr uint32_t num_write_lanes = get_compile_time_arg_val(4);
    constexpr auto output_args = TensorAccessorArgs<5>();
    static_assert(num_write_nocs == 1 || num_write_nocs == 2);
    static_assert(lane < num_write_lanes);

    // Every core gets the same (common) args and derives its own block from its logical coordinates: per-core
    // unique args would cost the dispatcher one write per core on every call.
    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t shard_tile_rows = get_common_arg_val<uint32_t>(1);
    const uint32_t shard_tile_cols = get_common_arg_val<uint32_t>(2);  // nominal: also the source row stride in slots
    const uint32_t tensor_tile_rows = get_common_arg_val<uint32_t>(3);
    const uint32_t tensor_tile_cols = get_common_arg_val<uint32_t>(4);  // also the destination row stride in pages

    uint32_t src_first_slot = 0;
    uint32_t dst_first_page = 0;
    uint32_t rect_tile_rows = 0;
    uint32_t rect_tile_cols = 0;
    uint32_t block_tile_rows = 0;
    uint32_t block_tile_cols = 0;
    uint32_t issue_first_row = 0;
    {
        DeviceZoneScopedN("S2I-LOCATE-BLOCK");
        // Shards stack along one tile axis only, so shard k starts k steps along it (one step is 0).
        const uint32_t shard_row_step = get_common_arg_val<uint32_t>(5);
        const uint32_t shard_col_step = get_common_arg_val<uint32_t>(6);
        const uint32_t num_shards = get_common_arg_val<uint32_t>(7);
        const uint32_t block_cap_rows = get_common_arg_val<uint32_t>(8);  // 0 = the whole rectangle
        const uint32_t block_cap_cols = get_common_arg_val<uint32_t>(9);
        const uint32_t issue_period = get_common_arg_val<uint32_t>(10);  // 0 = no rotation
        const bool row_wise = get_common_arg_val<uint32_t>(11) != 0;
        const uint32_t num_ranges = get_common_arg_val<uint32_t>(12);

        const uint32_t k = shard_index(13, num_ranges, row_wise);
        if (k < num_shards) {
            const uint32_t first_row = k * shard_row_step;
            const uint32_t first_col = k * shard_col_step;
            const uint32_t rows = clip(shard_tile_rows, tensor_tile_rows - first_row);
            const uint32_t cols = clip(shard_tile_cols, tensor_tile_cols - first_col);
            const bool cut_rows = rows >= cols;
            const uint32_t extent = cut_rows ? rows : cols;
            const uint32_t base = extent / num_write_lanes;
            const uint32_t rem = extent % num_write_lanes;
            const uint32_t piece = base + (lane < rem ? 1 : 0);
            const uint32_t offset = lane * base + (lane < rem ? lane : rem);
            const uint32_t row_off = cut_rows ? offset : 0;
            const uint32_t col_off = cut_rows ? 0 : offset;
            rect_tile_rows = cut_rows ? piece : rows;
            rect_tile_cols = cut_rows ? cols : piece;
            src_first_slot = row_off * shard_tile_cols + col_off;
            dst_first_page = (first_row + row_off) * tensor_tile_cols + first_col + col_off;
            block_tile_rows = block_cap_rows == 0 ? rect_tile_rows : clip(block_cap_rows, rect_tile_rows);
            block_tile_cols = block_cap_cols == 0 ? rect_tile_cols : clip(block_cap_cols, rect_tile_cols);
            // Rows are issued from this one to the end, then from row 0. Cores whose rectangles start on the same
            // bank recur every `issue_period` shards; rotating each by its class index puts them on distinct
            // banks at every write step (the host's `issue_first_row`). Two segments keep the walk modulo-free.
            if (issue_period != 0 && rect_tile_rows != 0) {
                issue_first_row = (k / issue_period) % rect_tile_rows;
            }
        }
    }

    const auto output = TensorAccessor(output_args, output_addr, page_size);
    const uint32_t src_row_stride_bytes = shard_tile_cols * page_size;
    const uint32_t rect_src_addr = get_read_ptr(cb_input_shard) + src_first_slot * page_size;

    uint8_t noc = noc_index;
    store_rect<page_size, num_write_nocs>(
        output,
        noc,
        rect_src_addr + issue_first_row * src_row_stride_bytes,
        src_row_stride_bytes,
        dst_first_page + issue_first_row * tensor_tile_cols,
        tensor_tile_cols,
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
        tensor_tile_cols,
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
