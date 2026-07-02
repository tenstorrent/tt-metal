// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "llk_defs.h"
#include <tt-metalium/constants.hpp>
#include "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reduce_rm_dataflow_common.hpp"

//
// Dense RM reduce writer (handles both W reduce and H reduce; branched on REDUCE_DIM).
//
// Compute packs one or more output tiles per work unit into cb_id_tile (c_3). The writer extracts
// the meaningful datums from those tiles and emits them into the corresponding RM output pages.
//
// W reduce path (REDUCE_DIM == REDUCE_ROW):
//   Output: one scalar per reduced logical row, one RM page per scalar (page_id == logical row).
//   Each compute output tile carries up to TILE_HEIGHT row reductions in column 0 of the tile
//   (intra-tile via get_tilized_idx(r, 0)). Writer emits up to TILE_HEIGHT separate
//   datum_bytes-sized writes per popped tile, each to its own page.
//
// H reduce path (REDUCE_DIM == REDUCE_COL):
//   Output: (N, C, 1, W) row-major. One page per (n, c) → page_id == nc, page width == W. Compute
//   produces wt_in_chunk output tiles per (nc, w_chunk) work unit, each tile holding the reduced
//   row of W datums in tile-row 0 (rows 1..TILE_HEIGHT-1 unused). Writer extracts that row
//   face-by-face (1–2 wide writes per tile, each up to half_tile_width datums) into the (n, c)
//   page at offset (w_base_col * datum_bytes), clamping the last W tile to W_logical so we don't
//   overflow the destination page.
//

template <ckernel::ReduceDim DIM>
void reduce_rm_writer() {
    //
    // Runtime args. Slots shared between paths; semantics differ:
    //   W reduce: (dst_addr, num_rows, start_page)
    //   H reduce: (dst_addr, num_output_tiles_local, start_output_tile_id)
    //
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t rt_count = get_arg_val<uint32_t>(1);
    const uint32_t rt_start = get_arg_val<uint32_t>(2);

    //
    // Compile-time args. Slot 0 (datum_bytes) is shared. The H reduce path adds Wt, W_logical, and
    // wt_tiles_per_chunk at slots 1-3, so the dst TensorAccessor args start at slot 1 (W) or slot 4 (H).
    //
    constexpr uint32_t datum_bytes = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<(DIM == ckernel::ReduceDim::REDUCE_ROW) ? 1 : 4>();

    constexpr uint32_t cb_id_tile = tt::CBIndex::c_3;
    constexpr uint32_t onetile = 1;

    const uint32_t tile_size_bytes = get_tile_size(cb_id_tile);
    const auto dst_accessor = TensorAccessor(dst_args, dst_addr);

    Noc noc;
    DataflowBuffer cb_tile(cb_id_tile);

    if constexpr (DIM == ckernel::ReduceDim::REDUCE_ROW) {
        //
        // === W reduce path ===
        //
        // One scalar per logical row, one page per scalar. Compute emits one tile at a time
        // (single-shot path for MAX with wt_tiles_per_chunk == Wt, or chunked SUM that ends on
        // is_last_chunk for each ht). Either way the writer sees one tile per pop.
        //
        const uint32_t num_rows = rt_count;
        const uint32_t start_page = rt_start;

        uint32_t rows_written = 0;
        while (rows_written < num_rows) {
            cb_tile.wait_front(onetile);
            const uint32_t rows_this_tile = ((num_rows - rows_written) < tt::constants::TILE_HEIGHT)
                                                ? (num_rows - rows_written)
                                                : tt::constants::TILE_HEIGHT;
            for (uint32_t r = 0; r < rows_this_tile; ++r) {
                const uint32_t tile_scalar_idx = get_tilized_idx(r, 0);
                noc.async_write(
                    cb_tile,
                    dst_accessor,
                    datum_bytes,
                    {.offset_bytes = tile_scalar_idx * datum_bytes},
                    {.page_id = start_page + rows_written + r, .offset_bytes = 0});
            }
            noc.async_write_barrier();
            cb_tile.pop_front(onetile);
            rows_written += rows_this_tile;
        }
    } else {
        //
        // === H reduce path ===
        //
        // One page per (n, c), W datums per page. Each output tile contributes one row-stripe of
        // up to TILE_WIDTH datums (1–2 face-wise wide writes per tile).
        //
        // Wt, W_logical, and wt_tiles_per_chunk are only consumed here, so they live in this branch.
        // The indices embed DIM to make them value-dependent: a literal `get_compile_time_arg_val(N)`
        // is non-dependent and would be eagerly instantiated even in this discarded branch, tripping
        // the index range check for the W path (which never passes these slots).
        constexpr uint32_t Wt = get_compile_time_arg_val((DIM == ckernel::ReduceDim::REDUCE_COL) ? 1 : 0);
        constexpr uint32_t W_logical = get_compile_time_arg_val((DIM == ckernel::ReduceDim::REDUCE_COL) ? 2 : 0);
        constexpr uint32_t wt_tiles_per_chunk =
            get_compile_time_arg_val((DIM == ckernel::ReduceDim::REDUCE_COL) ? 3 : 0);
        constexpr uint32_t face_w = tt::constants::TILE_WIDTH / 2;
        const uint32_t num_output_tiles_local = rt_count;
        const uint32_t start_output_tile_id = rt_start;

        uint32_t outputs_remaining = num_output_tiles_local;
        uint32_t current_nc = start_output_tile_id / Wt;
        uint32_t wt_in_nc = start_output_tile_id % Wt;

        while (outputs_remaining > 0) {
            // Pick the largest chunk that stays within one NC and within remaining work.
            uint32_t wt_in_chunk = wt_tiles_per_chunk;
            if (wt_in_chunk > Wt - wt_in_nc) {
                wt_in_chunk = Wt - wt_in_nc;
            }
            if (wt_in_chunk > outputs_remaining) {
                wt_in_chunk = outputs_remaining;
            }

            cb_tile.wait_front(wt_in_chunk);

            for (uint32_t wt = 0; wt < wt_in_chunk; ++wt) {
                const uint32_t w_tile_col = wt_in_nc + wt;
                const uint32_t w_base_col = w_tile_col * tt::constants::TILE_WIDTH;
                // Clamp the last W tile to W_logical so we don't write into padding.
                uint32_t valid_cols = tt::constants::TILE_WIDTH;
                if (w_base_col + valid_cols > W_logical) {
                    valid_cols = (w_base_col >= W_logical) ? 0 : (W_logical - w_base_col);
                }
                if (valid_cols == 0) {
                    continue;
                }

                // Emit at most 2 wide writes per tile (one per face along W), each up to face_w datums.
                for (uint32_t face_col = 0; face_col < valid_cols; face_col += face_w) {
                    const uint32_t face_valid = (valid_cols - face_col) < face_w ? (valid_cols - face_col) : face_w;
                    const uint32_t src_idx_in_tile = get_tilized_idx(0, face_col);
                    noc.async_write(
                        cb_tile,
                        dst_accessor,
                        face_valid * datum_bytes,
                        {.offset_bytes = wt * tile_size_bytes + src_idx_in_tile * datum_bytes},
                        {.page_id = current_nc, .offset_bytes = (w_base_col + face_col) * datum_bytes});
                }
            }

            noc.async_write_barrier();
            cb_tile.pop_front(wt_in_chunk);

            wt_in_nc += wt_in_chunk;
            outputs_remaining -= wt_in_chunk;
            if (wt_in_nc == Wt) {
                wt_in_nc = 0;
                ++current_nc;
            }
        }
    }
}

void kernel_main() { reduce_rm_writer<REDUCE_DIM>(); }
