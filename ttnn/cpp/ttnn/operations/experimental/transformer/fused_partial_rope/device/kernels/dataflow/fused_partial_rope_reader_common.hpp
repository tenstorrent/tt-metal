// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"

// Issues the reads that land the 32x32 trans_mat in `trans_mat_cb` as one tiled page. The caller
// barriers and pushes one page.
//
// A ROW_MAJOR trans_mat is 32 row pages; matmul needs it in face order (four 16x16 faces, each
// row-major), so the rows land in the CB's second page and are reordered into the first. Row r's
// left / right halves go to row r % 16 of face (r / 16) * 2 + {0, 1}. The CB holds two pages for this.
template <bool trans_mat_rm, typename Accessor>
FORCE_INLINE void read_trans_mat(
    Noc& noc, CircularBuffer& trans_mat_cb, const Accessor& s_trans_mat, uint32_t tile_bytes) {
    constexpr uint32_t rows = 32;
    constexpr uint32_t face_rows = 16;
    if constexpr (!trans_mat_rm) {
        trans_mat_cb.reserve_back(1);
        noc.async_read(
            s_trans_mat, CoreLocalMem<uint32_t>(trans_mat_cb.get_write_ptr()), tile_bytes, {.page_id = 0}, {});
    } else {
        trans_mat_cb.reserve_back(2);
        const uint32_t tile_l1 = trans_mat_cb.get_write_ptr();
        const uint32_t rows_l1 = tile_l1 + tile_bytes;
        const uint32_t row_bytes = tile_bytes / rows;
        for (uint32_t r = 0; r < rows; ++r) {
            noc.async_read(s_trans_mat, CoreLocalMem<uint32_t>(rows_l1 + r * row_bytes), row_bytes, {.page_id = r}, {});
        }
        noc.async_read_barrier();

        const uint32_t half_row_words = row_bytes / (2 * sizeof(uint32_t));
        auto* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows_l1);
        auto* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_l1);
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t face = (r / face_rows) * 2 + half;
                auto* d = dst + (face * face_rows + r % face_rows) * half_row_words;
                auto* s = src + (r * 2 + half) * half_row_words;
                for (uint32_t w = 0; w < half_row_words; ++w) {
                    d[w] = s[w];
                }
            }
        }
    }
}
