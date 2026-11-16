// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

// Issues the W tiles of K-block b (k_block_tiles rows x core_n_tiles columns) into cb_w at offset_bytes past
// its write pointer. The core owns columns n_start + j * n_stride; n_stride > 1 means bank-strided columns,
// so the core's tiles of one k row are adjacent in one DRAM bank and arrive in one transfer.
template <typename Accessor>
FORCE_INLINE void read_w_block(
    Noc& noc,
    const Accessor& w,
    CircularBuffer& cb_w,
    uint32_t w_page_bytes,
    uint32_t b,
    uint32_t offset_bytes,
    uint32_t k_block_tiles,
    uint32_t tensor_n_tiles,
    uint32_t n_start,
    uint32_t core_n_tiles,
    uint32_t n_stride) {
    uint32_t offset = offset_bytes;
    uint32_t row_page = b * k_block_tiles * tensor_n_tiles + n_start;
    for (uint32_t kk = 0; kk < k_block_tiles; ++kk) {
        if (n_stride > 1) {
            const uint32_t run_bytes = core_n_tiles * w_page_bytes;
            noc.async_read(w, cb_w, run_bytes, {.page_id = row_page}, {.offset_bytes = offset});
            offset += run_bytes;
        } else {
            for (uint32_t n = 0; n < core_n_tiles; ++n) {
                noc.async_read(w, cb_w, w_page_bytes, {.page_id = row_page + n}, {.offset_bytes = offset});
                offset += w_page_bytes;
            }
        }
        row_page += tensor_n_tiles;
    }
}
