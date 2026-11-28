// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"
#include "g6_to_layout_block_walk.hpp"

// reorder_block of row_major_to_tile: the 32-B self-aimed copies that turn one staged tile-row of
// rows into its tiles. Shared by the reader and the writer, which split a tile-row's face halves.

namespace g6_to_layout {

// Face rows of half h of block tile-row i: face 2h + (0 | 1) row fr <- stage row 16h + fr, byte
// j * 64 + (0 | 32) of the tile-row's slot, for every tile j of the block. Only live rows are
// copied, the right face only when live_cols > 16. Two trids alternate per tile so no trid ever
// holds more than one tile's copies (the per-trid counter holds 255); every copy has retired when
// this returns.
template <uint32_t first_half, uint32_t end_half>
FORCE_INLINE void copy_tile_row_faces(
    Noc noc, const Block& block, uint32_t i, uint32_t slot, uint32_t tiles_base, uint32_t trid_base, uint32_t& parity) {
    namespace dkl = dataflow_kernel_lib;
    const uint32_t live_rows = live_rows_of(block.tile_row + i);
    if (live_rows <= first_half * FACE_HW) {
        return;
    }
    // Per tile-row: other reads on this NoC reprogram the same read cmd_buf in between.
    dkl::set_read_state<FACE_ROW_BYTES>(noc, slot);
    uint32_t tile = tiles_base + i * Geometry::block_cols_tiles * TILE_BYTES;
    uint32_t column = slot;
    for (uint32_t j = 0; j < block.cols; ++j) {
        const bool right_face = live_cols_of(block.col + j) > FACE_HW;
        const uint32_t trid = trid_base + parity;
        dkl::set_read_trid(noc, trid);
        {
            DeviceZoneScopedSumN2("wait_reorder_copies");
            dkl::async_read_barrier_with_trid(noc, trid);
        }
        for (uint32_t half = first_half; half < end_half; ++half) {
            const uint32_t half_base = half * FACE_HW;
            if (live_rows <= half_base) {
                break;
            }
            const uint32_t rows = min_u32(live_rows - half_base, FACE_HW);
            uint32_t src = column + half_base * Geometry::stage_row_bytes;
            uint32_t dst = tile + half * 2 * FACE_BYTES;
            for (uint32_t r = 0; r < rows; ++r) {
                dkl::read_with_state<FACE_ROW_BYTES>(noc, dst, src);
                if (right_face) {
                    dkl::read_with_state<FACE_ROW_BYTES>(noc, dst + FACE_BYTES, src + FACE_ROW_BYTES);
                }
                src += Geometry::stage_row_bytes;
                dst += FACE_ROW_BYTES;
            }
        }
        parity ^= 1;
        tile += TILE_BYTES;
        column += TILE_ROW_BYTES;
    }
    DeviceZoneScopedSumN2("wait_reorder_copies");
    dkl::async_read_barrier_with_trid(noc, trid_base);
    dkl::async_read_barrier_with_trid(noc, trid_base + 1);
}

}  // namespace g6_to_layout
