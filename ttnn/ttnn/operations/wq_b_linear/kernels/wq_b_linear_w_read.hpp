// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"

// Blackhole DRAM banks have three NOC2AXI ports (translated y0 .. y0 + 2, y0 = 12 + 3i): the preferred NoC0
// endpoint, the preferred NoC1 endpoint and a free port. TensorAccessor always targets the NoC0 endpoint, and
// one port caps a bank at ~50 GB/s (W stream alone 19.5 us on any single port). Alternating a core's reads by
// k row between the NoC0 endpoint and the free port lifts the stream to 16.2 us; the NoC1 endpoint is left to
// the x prefetch, which queues behind W when it shares that port. A non-endpoint port only reaches GDDR while
// its NIU is in NOC2AXI mode; with programmable DRAM cores the firmware puts it in stream mode (and fills
// tensor_prefetcher_signal_noc_xy), so then every read stays on the NoC0 endpoint.
constexpr uint32_t kDramTranslatedX0 = 17;
constexpr uint32_t kDramTranslatedY0 = 12;
constexpr uint32_t kDramPortBlocks = 4;
constexpr uint32_t kNocYShift = NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS;
constexpr uint32_t kNocXShift = NOC_ADDR_LOCAL_BITS;

// dram_free_port_y[column][block]: translated y of the free port of the bank whose ports are that block; 0 = none.
inline uint8_t dram_free_port_y[2][kDramPortBlocks];
// 0 = not yet probed, 1 = spread, 2 = preferred endpoint only.
inline uint8_t dram_port_spread_state;

FORCE_INLINE bool dram_port_spread_enabled() {
#if defined(ARCH_BLACKHOLE)
    if (dram_port_spread_state == 0) {
        dram_port_spread_state = tensor_prefetcher_signal_noc_xy[0] == 0 ? 1 : 2;
        for (uint32_t b = 0; b < NUM_DRAM_BANKS && dram_port_spread_state == 1; ++b) {
            const uint32_t xy0 = dram_bank_to_noc_xy[0][b];
            const uint32_t xy1 = dram_bank_to_noc_xy[1][b];
            const uint32_t x = xy0 & NOC_NODE_ID_MASK;
            const uint32_t y0 = (xy0 >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
            const uint32_t y1 = (xy1 >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
            const uint32_t blk = (y0 - kDramTranslatedY0) / 3;
            const uint32_t base = kDramTranslatedY0 + 3 * blk;
            if (x < kDramTranslatedX0 || x > kDramTranslatedX0 + 1 || y0 < kDramTranslatedY0 ||
                blk >= kDramPortBlocks || (xy1 & NOC_NODE_ID_MASK) != x || y1 < base || y1 >= base + 3 || y1 == y0) {
                dram_port_spread_state = 2;
                break;
            }
            dram_free_port_y[x - kDramTranslatedX0][blk] = base + 3 - (y0 - base) - (y1 - base);
        }
    }
    return dram_port_spread_state == 1;
#else
    return false;
#endif
}

FORCE_INLINE uint64_t dram_port_addr(uint64_t preferred, uint32_t k_row) {
    if ((k_row & 1) == 0) {
        return preferred;
    }
    const uint32_t x = (preferred >> kNocXShift) & NOC_NODE_ID_MASK;
    const uint32_t y = (preferred >> kNocYShift) & NOC_NODE_ID_MASK;
    const uint32_t port_y = dram_free_port_y[x - kDramTranslatedX0][(y - kDramTranslatedY0) / 3];
    return (preferred & ~(NOC_NODE_ID_MASK << kNocYShift)) | (static_cast<uint64_t>(port_y) << kNocYShift);
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
    const bool spread = Accessor::is_dram && dram_port_spread_enabled();
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
