// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

#include "ttnn/operations/common/shared_with_host/runtime_length.hpp"

// Per-core tile-row range of the interleaved multi-core untilize, derived on device from the common
// row count: the same split as split_blocks_for_tilize (ceil(rows / cores) rows per core, the last
// active core takes the remainder, trailing cores none), shared by reader, compute and writer.
inline std::pair<uint32_t, uint32_t> untilize_row_range(uint32_t core_idx, uint32_t num_tile_rows, uint32_t num_cores) {
    const uint32_t per_core = ttnn::runtime_length::tiles_per_core(num_tile_rows, num_cores);
    return {
        ttnn::runtime_length::core_start(core_idx, per_core, num_tile_rows),
        ttnn::runtime_length::core_end(core_idx, per_core, num_tile_rows)};
}
