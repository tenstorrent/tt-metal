// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// R4 row reduce-scatter, sender side, shared by the writer (NoC1) and the reader (NoC0).
//
// cb_combine_slots of an output-owning column holds k_cols slots of [block_m_tiles, out_n_tiles] fp32 tiles,
// slot c written by grid column c. The partials are [block_m_tiles, block_n_tiles] tiles, row-major. Both
// CBs are always pushed and popped whole, so their pointers never leave the CB base: the base addresses
// read at kernel start are valid on every core, for every M-block, from either data-movement RISC.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace conv2d_3x3 {

// Writes N-slice `col` of the partials into slot `my_col` of grid column `col`, for col in
// [col_begin, col_end), on the calling kernel's NoC. The caller fences with noc_async_write_barrier().
template <uint32_t block_m_tiles, uint32_t block_n_tiles, uint32_t out_n_tiles, uint32_t tensor_n_tiles>
inline void scatter_partial_slices(
    uint32_t partials_l1,
    uint32_t combine_base_l1,
    uint32_t tile_bytes,
    uint32_t my_col,
    uint32_t col_begin,
    uint32_t col_end,
    uint32_t row_peer_coords_arg) {
#ifdef CONV2D_3X3_ABLATE_SCATTER
    return;
#endif
    constexpr uint32_t slot_tiles = block_m_tiles * out_n_tiles;
    const uint32_t my_slot_l1 = combine_base_l1 + my_col * slot_tiles * tile_bytes;
    for (uint32_t col = col_begin; col < col_end; ++col) {
        const uint32_t n0 = col * out_n_tiles;
        const uint32_t cols_valid = tensor_n_tiles - n0 < out_n_tiles ? tensor_n_tiles - n0 : out_n_tiles;
        const uint32_t x = get_arg_val<uint32_t>(row_peer_coords_arg + 2 * col);
        const uint32_t y = get_arg_val<uint32_t>(row_peer_coords_arg + 2 * col + 1);
        for (uint32_t t = 0; t < block_m_tiles; ++t) {
            noc_async_write(
                partials_l1 + (t * block_n_tiles + n0) * tile_bytes,
                get_noc_addr(x, y, my_slot_l1 + t * out_n_tiles * tile_bytes),
                cols_valid * tile_bytes);
        }
    }
}

// N-slice-outer K split (perf experiment): writes one owner N-slice `col` ([block_m_tiles, out_n_tiles] fp32,
// row-major, at slice_l1) into slot `my_col` of grid column `col`, on the calling kernel's NoC. A full slice is
// one contiguous write (slice and slot share the layout); a short last slice writes its valid columns per row.
// The caller fences with noc_async_write_barrier().
template <uint32_t block_m_tiles, uint32_t out_n_tiles, uint32_t tensor_n_tiles>
inline void send_slice(
    uint32_t slice_l1,
    uint32_t combine_base_l1,
    uint32_t tile_bytes,
    uint32_t my_col,
    uint32_t col,
    uint32_t row_peer_coords_arg) {
#ifdef CONV2D_3X3_ABLATE_SCATTER
    return;
#endif
    constexpr uint32_t slot_tiles = block_m_tiles * out_n_tiles;
    const uint32_t my_slot_l1 = combine_base_l1 + my_col * slot_tiles * tile_bytes;
    const uint32_t n0 = col * out_n_tiles;
    const uint32_t cols_valid = tensor_n_tiles - n0 < out_n_tiles ? tensor_n_tiles - n0 : out_n_tiles;
    const uint32_t x = get_arg_val<uint32_t>(row_peer_coords_arg + 2 * col);
    const uint32_t y = get_arg_val<uint32_t>(row_peer_coords_arg + 2 * col + 1);
    if (cols_valid == out_n_tiles) {
        noc_async_write(slice_l1, get_noc_addr(x, y, my_slot_l1), slot_tiles * tile_bytes);
    } else {
        for (uint32_t t = 0; t < block_m_tiles; ++t) {
            noc_async_write(
                slice_l1 + t * out_n_tiles * tile_bytes,
                get_noc_addr(x, y, my_slot_l1 + t * out_n_tiles * tile_bytes),
                cols_valid * tile_bytes);
        }
    }
}

}  // namespace conv2d_3x3
