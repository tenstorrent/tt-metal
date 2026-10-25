// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::kv_cache {

inline std::uint32_t user_groups_in_batch_tile(
    std::uint32_t batch_size, std::uint32_t batch_start_id, std::uint32_t tile_offset, std::uint32_t granularity) {
    constexpr std::uint32_t tile_height = 32;
    const std::uint32_t tiles_per_head = (batch_size + tile_height - 1) / tile_height;
    const std::uint32_t tile_in_head = (batch_start_id / tile_height + tile_offset) % tiles_per_head;
    const std::uint32_t users = batch_size - tile_in_head * tile_height;
    return (users < tile_height ? users : tile_height) / granularity;
}

}  // namespace ttnn::kv_cache
