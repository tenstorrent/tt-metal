// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"
#include <tt-metalium/constants.hpp>

//
// Dense RM H reduce writer.
//
// Compute packs `wt_in_chunk` output tiles per (nc, w_chunk) work unit into cb_out (cb_id_tile). Each tile
// stores the W datums of one reduced tile-column in **tile-row 0**; rows 1..TILE_HEIGHT-1 are unused.
// The writer extracts that row 0 (one face at a time so we issue 1-2 wide writes per tile instead of
// 32 single-datum writes) and emits it into the output RM page for the corresponding (n, c) batch.
//
// Output layout: (N, C, 1, W) in row-major. One output page per (n, c) → page_id == nc, page width == W.
//

FORCE_INLINE uint32_t tile_row0_face_offset_datums(uint32_t w_idx) {
    // Mirror of writer_reduce_w_rm_scalar.cpp's get_tilized_idx(0, w_idx). Within the tilized layout, the
    // first 16 columns of tile-row 0 are contiguous at offsets [0..15]; the next 16 columns start at offset
    // 16*16 == 256 (face 1, top-right). This helper returns the starting index for a face's leftmost column.
    constexpr uint32_t half_tile_width = tt::constants::TILE_WIDTH / 2;
    constexpr uint32_t half_tile_height = tt::constants::TILE_HEIGHT / 2;
    if (w_idx < half_tile_width) {
        return w_idx;
    }
    return (w_idx - half_tile_width) + half_tile_height * half_tile_width;
}

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_output_tiles_local = get_arg_val<uint32_t>(1);
    const uint32_t start_output_tile_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t datum_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t W_logical = get_compile_time_arg_val(2);
    constexpr uint32_t wt_tiles_per_chunk = get_compile_time_arg_val(3);
    constexpr auto dst_args = TensorAccessorArgs<4>();

    constexpr uint32_t cb_id_tile = tt::CBIndex::c_3;
    constexpr uint32_t face_w = tt::constants::TILE_WIDTH / 2;

    const uint32_t tile_size_bytes = get_tile_size(cb_id_tile);
    const auto dst_accessor = TensorAccessor(dst_args, dst_addr);

    experimental::Noc noc;
    experimental::CircularBuffer cb_tile(cb_id_tile);

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

            // Emit at most 2 wide writes per tile (one per face along W), each up to half_tile_width datums.
            for (uint32_t face_col = 0; face_col < valid_cols; face_col += face_w) {
                const uint32_t face_valid = (valid_cols - face_col) < face_w ? (valid_cols - face_col) : face_w;
                const uint32_t src_idx_in_tile = tile_row0_face_offset_datums(face_col);
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
