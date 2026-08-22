// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/operations/wavelet/device/protocol/lwt_config.hpp"
#include "api/dataflow/dataflow_api.h"

#ifndef ALWI
#define ALWI inline __attribute__((always_inline))
#endif

namespace ttnn::operations::wavelet::kernels::primitives {

struct WorkspaceIndexCursor {
    uint32_t group{0};
    uint32_t row{0};
    uint32_t block{0};
    uint32_t lane{0};
    uint32_t physical{0};

    ALWI explicit WorkspaceIndexCursor(const uint32_t logical_index) {
        constexpr uint32_t group_elements = device_protocol::kLwtGroupOutputElements;
        constexpr uint32_t block_elements = device_protocol::kLwtHalfStickElements;
        constexpr uint32_t blocks_per_row = device_protocol::kLwtOutputBlocksPerRow;
        constexpr uint32_t narrow_tile_elements = device_protocol::kLwtNarrowTileElements;
        group = logical_index / group_elements;
        const uint32_t group_index = logical_index - group * group_elements;
        row = group_index / (blocks_per_row * block_elements);
        const uint32_t row_index = group_index - row * blocks_per_row * block_elements;
        block = row_index / block_elements;
        lane = row_index - block * block_elements;
        physical = group * group_elements + block * narrow_tile_elements + row * block_elements + lane;
    }

    ALWI void advance() {
        constexpr uint32_t group_elements = device_protocol::kLwtGroupOutputElements;
        constexpr uint32_t block_elements = device_protocol::kLwtHalfStickElements;
        constexpr uint32_t blocks_per_row = device_protocol::kLwtOutputBlocksPerRow;
        constexpr uint32_t rows_per_group = device_protocol::kLwtRowsPerGroup;
        constexpr uint32_t narrow_tile_elements = device_protocol::kLwtNarrowTileElements;
        ++lane;
        ++physical;
        if (lane == block_elements) {
            lane = 0;
            ++block;
            if (block == blocks_per_row) {
                block = 0;
                ++row;
                if (row == rows_per_group) {
                    row = 0;
                    ++group;
                }
            }
            physical = group * group_elements + block * narrow_tile_elements + row * block_elements;
        }
    }

    ALWI void advance_block() {
        constexpr uint32_t group_elements = device_protocol::kLwtGroupOutputElements;
        constexpr uint32_t block_elements = device_protocol::kLwtHalfStickElements;
        constexpr uint32_t blocks_per_row = device_protocol::kLwtOutputBlocksPerRow;
        constexpr uint32_t rows_per_group = device_protocol::kLwtRowsPerGroup;
        constexpr uint32_t narrow_tile_elements = device_protocol::kLwtNarrowTileElements;
        ++block;
        if (block == blocks_per_row) {
            block = 0;
            ++row;
            if (row == rows_per_group) {
                row = 0;
                ++group;
            }
        }
        physical = group * group_elements + block * narrow_tile_elements + row * block_elements + lane;
    }
};

template <bool TileNative>
[[nodiscard]] ALWI uint32_t workspace_physical_index(const uint32_t logical_index) {
    if constexpr (!TileNative) {
        return logical_index;
    }
    return WorkspaceIndexCursor(logical_index).physical;
}

}  // namespace ttnn::operations::wavelet::kernels::primitives
