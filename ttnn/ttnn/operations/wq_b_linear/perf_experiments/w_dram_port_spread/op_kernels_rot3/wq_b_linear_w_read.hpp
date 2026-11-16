// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"

// Blackhole DRAM banks have three NOC2AXI ports (translated y = y0 .. y0 + 2, y0 = 12 + 3i), on different NoC
// rows. TensorAccessor always targets the bank's preferred NoC0 endpoint, so every bank's whole stream leaves on
// one row and two banks share row links. Rotating a core's reads over the three ports by k row spreads each
// bank's egress over three rows (W stream alone 19.5 -> 16.1 us). A port other than the preferred endpoint only
// forwards to GDDR while its NIU is in NOC2AXI mode; with programmable DRAM cores the firmware puts those NIUs
// in stream mode (and fills tensor_prefetcher_signal_noc_xy), so then every read stays on the preferred port.
FORCE_INLINE bool dram_port_spread_enabled() {
#if defined(ARCH_BLACKHOLE)
    return tensor_prefetcher_signal_noc_xy[0] == 0;
#else
    return false;
#endif
}

FORCE_INLINE uint64_t dram_port_addr(uint64_t preferred, uint32_t k_row) {
    constexpr uint32_t kDramTranslatedY0 = 12;
    constexpr uint32_t kYShift = NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS;
    const uint32_t y = (preferred >> kYShift) & NOC_NODE_ID_MASK;
    const uint32_t port_y = y - (y - kDramTranslatedY0) % 3 + k_row % 3;
    return (preferred & ~(NOC_NODE_ID_MASK << kYShift)) | (static_cast<uint64_t>(port_y) << kYShift);
}

template <typename Accessor>
FORCE_INLINE void read_w_page_run(
    Noc& noc,
    const Accessor& w,
    CircularBuffer& cb_w,
    uint32_t bytes,
    uint32_t page_id,
    uint32_t offset,
    uint32_t k_row,
    bool spread) {
    if (spread) {
        const uint64_t src = dram_port_addr(w.get_noc_addr(page_id), k_row);
        noc.async_read(PrecomposedUnicastEndpoint{}, cb_w, bytes, {.noc_addr = src}, {.offset_bytes = offset});
    } else {
        noc.async_read(w, cb_w, bytes, {.page_id = page_id}, {.offset_bytes = offset});
    }
}

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
    const bool spread = dram_port_spread_enabled();
    uint32_t offset = offset_bytes;
    uint32_t k_row = b * k_block_tiles;
    uint32_t row_page = k_row * tensor_n_tiles + n_start;
    for (uint32_t kk = 0; kk < k_block_tiles; ++kk) {
        if (n_stride > 1) {
            const uint32_t run_bytes = core_n_tiles * w_page_bytes;
            read_w_page_run(noc, w, cb_w, run_bytes, row_page, offset, k_row, spread);
            offset += run_bytes;
        } else {
            for (uint32_t n = 0; n < core_n_tiles; ++n) {
                read_w_page_run(noc, w, cb_w, w_page_bytes, row_page + n, offset, k_row, spread);
                offset += w_page_bytes;
            }
        }
        row_page += tensor_n_tiles;
        ++k_row;
    }
}
