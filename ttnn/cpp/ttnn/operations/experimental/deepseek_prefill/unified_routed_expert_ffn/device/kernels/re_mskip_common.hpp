// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// RE_MSKIP count-sparsity row-map helpers, shared by the reader, writer, and compute
// kernels. The SPREAD map and the per-core valid-row count MUST agree across all three
// kernels or FFN outputs are silently written to the wrong token rows, so they are
// defined here exactly once. Pure integer arithmetic (no NoC/CB/reg APIs) so the header
// is valid in BRISC, NCRISC, and TRISC translation units alike.
//
// SPREAD map: valid token tile-rows are dealt round-robin across the GRID_Y M-row cores
// so a sparse chunk fills the fewest cores' M-loops. Core gy (in [0, GRID_Y)) owns rows
// { base + m*GRID_Y : m < per_core_M }; the valid rows are the prefix m < m_valid.
namespace re_mskip {

// Number of M-row cores a chunk's rows are spread across.
constexpr uint32_t grid_y(uint32_t chunk_m_tiles, uint32_t per_core_m) { return chunk_m_tiles / per_core_m; }

// First (m == 0) global tile-row this core owns in this chunk.
constexpr uint32_t spread_base(uint32_t chunk, uint32_t chunk_m_tiles, uint32_t gy) {
    return chunk * chunk_m_tiles + gy;
}

// The m-th owned tile-row for this core.
constexpr uint32_t spread_row(uint32_t base, uint32_t m, uint32_t gy_stride) { return base + m * gy_stride; }

// Count of valid subblock-rows this core owns: the prefix of owned rows whose spread
// index is < count_tiles, clamped to per_core_m. Zero when the core owns no valid row.
constexpr uint32_t m_valid(uint32_t base, uint32_t count_tiles, uint32_t gy_stride, uint32_t per_core_m) {
    if (base >= count_tiles) {
        return 0;
    }
    const uint32_t budget = count_tiles - base;
    const uint32_t mv = (budget + gy_stride - 1) / gy_stride;
    return mv < per_core_m ? mv : per_core_m;
}

}  // namespace re_mskip
