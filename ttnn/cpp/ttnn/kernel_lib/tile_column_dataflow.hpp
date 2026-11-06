// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file tile_column_dataflow.hpp
 * @brief Writing one column of a tile from a dataflow kernel.
 *
 * A tile is stored face by face: FACE_HEIGHT x FACE_WIDTH faces, row-major within a face and
 * face-major across the tile, tile_c_dim / FACE_WIDTH faces to a face row. So column `col` of a
 * 32x32 tile is sixteen elements at a stride of FACE_WIDTH in each of two faces: 0 and 2 for the
 * left half of the tile, 1 and 3 for the right.
 *
 * A column is what a column broadcast reads its operand from (column 0), so these build that
 * operand in L1: a column of a constant over a zeroed tile (`zero_tile` in l1_helpers.hpp), or a
 * column of one tile moved into column 0 of another.
 *
 * The tile shape is the dataflow buffer's (`get_tile_r_dim` / `get_tile_c_dim`), and the element
 * type `T` is the caller's: uint16_t for bfloat16, uint32_t for float32.
 *
 * These are ordinary RISC-V stores. On Quasar a data-movement core's stores land in its data cache,
 * so the caller flushes the tile before a TRISC or the NoC reads it.
 */

#include <stdint.h>

#include "api/tile_constants.h"
#include "ttnn/cpp/ttnn/kernel_lib/dfb_helpers_dataflow.hpp"

namespace dataflow_kernel_lib {

namespace detail {

template <uint32_t dfb_id>
struct TileColumnLayout {
    static constexpr uint32_t tile_r_dim = get_tile_r_dim<dfb_id>();
    static constexpr uint32_t tile_c_dim = get_tile_c_dim<dfb_id>();
    static_assert(tile_r_dim % tt::constants::FACE_HEIGHT == 0, "tile height must be a multiple of FACE_HEIGHT");
    static_assert(tile_c_dim % tt::constants::FACE_WIDTH == 0, "tile width must be a multiple of FACE_WIDTH");

    static constexpr uint32_t face_rows = tile_r_dim / tt::constants::FACE_HEIGHT;
    // Elements between a face and the one below it.
    static constexpr uint32_t face_row_stride = (tile_c_dim / tt::constants::FACE_WIDTH) * tt::constants::FACE_HW;

    // Offset of row 0 of column `col`.
    static constexpr uint32_t column_base(uint32_t col) {
        return (col / tt::constants::FACE_WIDTH) * tt::constants::FACE_HW + col % tt::constants::FACE_WIDTH;
    }

    // Offset of row `row` of a column, from that column's row 0.
    static constexpr uint32_t row_offset(uint32_t face_row, uint32_t row_in_face) {
        return face_row * face_row_stride + row_in_face * tt::constants::FACE_WIDTH;
    }
};

}  // namespace detail

/**
 * @brief Set every row of column `col` of a tile to `value`, leaving the rest of the tile as it is.
 *
 * @tparam dfb_id Dataflow buffer whose tile shape the tile has
 * @tparam T      Element type: uint16_t (bfloat16) or uint32_t (float32)
 * @param tile_addr L1 address of the tile
 * @param col       Column to set, in [0, tile_c_dim)
 * @param value     Element bit pattern to write
 */
template <uint32_t dfb_id, typename T>
FORCE_INLINE void fill_tile_column(uint32_t tile_addr, uint32_t col, T value) {
    using Layout = detail::TileColumnLayout<dfb_id>;
    auto* column = reinterpret_cast<volatile tt_l1_ptr T*>(tile_addr) + Layout::column_base(col);
    for (uint32_t face_row = 0; face_row < Layout::face_rows; ++face_row) {
        for (uint32_t row = 0; row < tt::constants::FACE_HEIGHT; ++row) {
            column[Layout::row_offset(face_row, row)] = value;
        }
    }
}

/**
 * @brief Copy column `src_col` of one tile into column `dst_col` of another, leaving the rest of the
 *        destination as it is.
 *
 * @tparam dfb_id Dataflow buffer whose tile shape both tiles have
 * @tparam T      Element type: uint16_t (bfloat16) or uint32_t (float32)
 * @param src_addr L1 address of the source tile
 * @param src_col  Column to read, in [0, tile_c_dim)
 * @param dst_addr L1 address of the destination tile
 * @param dst_col  Column to write, in [0, tile_c_dim)
 */
template <uint32_t dfb_id, typename T>
FORCE_INLINE void copy_tile_column(uint32_t src_addr, uint32_t src_col, uint32_t dst_addr, uint32_t dst_col) {
    using Layout = detail::TileColumnLayout<dfb_id>;
    const auto* src = reinterpret_cast<const volatile tt_l1_ptr T*>(src_addr) + Layout::column_base(src_col);
    auto* dst = reinterpret_cast<volatile tt_l1_ptr T*>(dst_addr) + Layout::column_base(dst_col);
    for (uint32_t face_row = 0; face_row < Layout::face_rows; ++face_row) {
        for (uint32_t row = 0; row < tt::constants::FACE_HEIGHT; ++row) {
            const uint32_t offset = Layout::row_offset(face_row, row);
            dst[offset] = src[offset];
        }
    }
}

}  // namespace dataflow_kernel_lib
