// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>

// Per-core tile-row range of the interleaved multi-core untilize, derived on device from the common
// row count; shared by reader, compute and writer. Cores past the end of a short input get an empty range.
inline std::pair<uint32_t, uint32_t> untilize_row_range(uint32_t core_idx, uint32_t num_tile_rows, uint32_t num_cores) {
    const uint32_t per_core = (num_tile_rows + num_cores - 1) / num_cores;
    return {std::min(core_idx * per_core, num_tile_rows), std::min((core_idx + 1) * per_core, num_tile_rows)};
}
