// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_buffer.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

namespace convert_to_hwc {

constexpr uint32_t TILE_SIZE = 32;

template <uint32_t StickSize, uint32_t PaddedStickSize, uint32_t NumSticks>
FORCE_INLINE void copy_padded_sticks(Noc noc, uint32_t l1_read_addr, uint32_t& l1_write_addr) {
    experimental::set_read_state<StickSize>(noc, l1_read_addr);
    for (uint32_t row = 0; row < NumSticks; row++) {
        experimental::read_with_state(noc, l1_write_addr, l1_read_addr);
        l1_read_addr += PaddedStickSize;
        l1_write_addr += StickSize;
    }
}

template <
    uint32_t NumOutputChannelsPadded,
    uint32_t NumFullTiles,
    uint32_t TotalTilesPerBlock,
    bool IsPrimaryWriter,
    uint32_t ElementSizeBytes,
    uint32_t L1WriteOutputAddrStride>
FORCE_INLINE void write_transposed_block(Noc noc, DataflowBuffer& transpose, uint32_t& l1_output_write_addr) {
    constexpr uint32_t channel_size = NumOutputChannelsPadded * ElementSizeBytes;
    constexpr uint32_t tile_size_stick_bytes = TILE_SIZE * ElementSizeBytes;
    for (uint32_t i = 0; i < NumFullTiles; i++) {
        transpose.wait_front(1);
        const uint32_t l1_read_addr = transpose.get_read_ptr();
        copy_padded_sticks<channel_size, tile_size_stick_bytes, TILE_SIZE>(noc, l1_read_addr, l1_output_write_addr);
        noc.async_read_barrier();
        transpose.pop_front(1);
        // Stride by a number of sticks when splitting writers across cores
        l1_output_write_addr += L1WriteOutputAddrStride;
    }
    if constexpr ((TotalTilesPerBlock % 2) != 0) {
        // One full tile extent, which is half of the two-writer 64-stick stride cycle.
        constexpr uint32_t tile_extent_bytes = TILE_SIZE * NumOutputChannelsPadded * ElementSizeBytes;
        if constexpr (IsPrimaryWriter) {
            l1_output_write_addr -= tile_extent_bytes;
        } else {
            l1_output_write_addr += tile_extent_bytes;
        }
    }
}

}  // namespace convert_to_hwc
