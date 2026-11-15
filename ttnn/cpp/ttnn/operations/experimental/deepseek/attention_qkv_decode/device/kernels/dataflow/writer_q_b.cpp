// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "experimental/kernel_args.h"
#include "rope_table_rows.hpp"

// BLOCK_ROWS: every tile is (rows x 32), one token per row; q's rows are head-major (head h's row r at
// h * rows + r), the RoPE tables hold one row per token, and comp_output is (rows x 32)-tiled.
//
// A q_b core: the RoPE table rows its compute needs (do_rope: it holds a head's last tiles), then its
// Nc result tiles -- row-major bf16, a 1x32 tile is 32 contiguous values -- into row q_row of q at
// byte q_col_bytes.
//
// COMP (on is_comp cores): before q, its compressor-column tile into comp_output's one row at byte
// comp_col_bytes.
void kernel_main() {
    constexpr uint32_t nc_tiles = get_arg(args::nc_tiles);
    constexpr uint32_t rope_tiles = get_arg(args::rope_tiles);
    constexpr uint32_t row_tile_bytes = get_arg(args::row_tile_bytes);
    const bool do_rope = get_arg(args::do_rope) != 0;
    const uint32_t q_row = get_arg(args::q_row);
    const uint32_t q_col_bytes = get_arg(args::q_col_bytes);

    Noc noc;
    CircularBuffer cs_cb(dfb::cs);
    CircularBuffer cs_raw_cb(dfb::cs_raw);
    CircularBuffer out_cb(dfb::out);

#ifdef BLOCK_ROWS
    constexpr uint32_t rows = get_arg(args::rows);
    if (do_rope) {
        const auto cos = TensorAccessor(tensor::cos);
        const auto sin = TensorAccessor(tensor::sin);
        read_rope_table_tiles(noc, cos, sin, 0, rope_tiles, cs_cb, row_tile_bytes);
    }
    CircularBuffer ones_cb(dfb::ones);
    fill_ones_tile(ones_cb);
#else
    if (do_rope) {
        const auto cos = TensorAccessor(tensor::cos);
        const auto sin = TensorAccessor(tensor::sin);
        read_rope_table_rows(noc, cos, sin, 0, rope_tiles, cs_cb, cs_raw_cb, row_tile_bytes);
    }
#endif

#ifdef COMP
    if (get_arg(args::is_comp) != 0) {
        CircularBuffer comp_x_cb(dfb::comp_x);
        const auto comp_output = TensorAccessor(tensor::comp_output);
        comp_x_cb.wait_front(1);
#ifdef BLOCK_ROWS
        // comp_output is (rows x 32)-tiled: this core's column tile is one whole page.
        noc_async_write(comp_x_cb.get_read_ptr(), comp_output.get_noc_addr(get_arg(args::comp_page)), row_tile_bytes);
#else
        noc_async_write(
            comp_x_cb.get_read_ptr(), comp_output.get_noc_addr(0, get_arg(args::comp_col_bytes)), row_tile_bytes);
#endif
        noc.async_write_barrier();
        comp_x_cb.pop_front(1);
    }
#endif

    const auto q = TensorAccessor(tensor::q);
#ifdef BLOCK_ROWS
    // Row r of head q_row is q's row q_row * rows + r; row r of a (rows x 32) tile is row r of each of its two
    // 16-wide faces. The rows are put together in L1, tile by tile as compute hands them over (the plain tiles
    // while it rotates the rope tiles), so each row goes out as one write.
    constexpr uint32_t face_row_bytes = 16 * 2;
    constexpr uint32_t q_row_bytes = nc_tiles * 2 * face_row_bytes;
    CircularBuffer q_rows_cb(dfb::q_rows);
    q_rows_cb.reserve_back(1);
    const uint32_t rows_addr = q_rows_cb.get_write_ptr();
    const uint32_t plain_tiles = do_rope ? nc_tiles - rope_tiles : nc_tiles;
    const uint32_t ready_tiles[2] = {plain_tiles, nc_tiles};
    uint32_t t = 0;
    for (const uint32_t ready : ready_tiles) {
        out_cb.wait_front(ready);
        const uint32_t out_addr = out_cb.get_read_ptr();
        for (; t < ready; ++t) {
            for (uint32_t r = 0; r < rows; ++r) {
                const uint32_t src = out_addr + t * row_tile_bytes + r * face_row_bytes;
                const uint32_t dst = rows_addr + r * q_row_bytes + t * 2 * face_row_bytes;
                copy_face_row(dst, src);
                copy_face_row(dst + face_row_bytes, src + rows * face_row_bytes);
            }
        }
    }
    for (uint32_t r = 0; r < rows; ++r) {
        noc_async_write(rows_addr + r * q_row_bytes, q.get_noc_addr(q_row * rows + r, q_col_bytes), q_row_bytes);
    }
    noc.async_write_barrier();
    q_rows_cb.push_back(1);
    q_rows_cb.wait_front(1);
    q_rows_cb.pop_front(1);
#else
    out_cb.wait_front(nc_tiles);
    noc_async_write(out_cb.get_read_ptr(), q.get_noc_addr(q_row, q_col_bytes), nc_tiles * row_tile_bytes);
    noc.async_write_barrier();
#endif
    out_cb.pop_front(nc_tiles);
}
