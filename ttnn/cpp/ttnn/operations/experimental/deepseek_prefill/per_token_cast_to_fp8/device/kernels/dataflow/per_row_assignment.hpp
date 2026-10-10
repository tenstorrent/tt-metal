// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Per-row-scale work assignment. The grid is split into `segments` groups of `group` cores; a row's width
// is cut into `segments` equal column segments. Core c works segment c / group of tile-rows
// c % group, c % group + group, ... — so the `segments` cores sharing a row are fixed partners that
// exchange their partial row maxima. Cores at or past segments * group idle. The host mirrors this.
struct PerRowAssignment {
    uint32_t seg;       // column segment of this core
    uint32_t grp;       // first tile-row (and partner-group index)
    uint32_t num_rows;  // tile-rows grp, grp + group, ... below tile_rows
};

inline PerRowAssignment per_row_assignment(uint32_t core, uint32_t tile_rows, uint32_t segments, uint32_t group) {
    if (core >= segments * group) {
        return {0, 0, 0};
    }
    const uint32_t grp = core % group;
    return {core / group, grp, tile_rows > grp ? (tile_rows - grp + group - 1) / group : 0};
}
