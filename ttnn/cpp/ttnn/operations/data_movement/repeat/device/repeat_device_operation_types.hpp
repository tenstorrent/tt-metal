// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct RepeatParams {
    uint32_t m_num_repeats{};
    bool m_is_last_dim{};
    tt::tt_metal::MemoryConfig m_output_mem_config;

    // Tile-native path: when m_tile_page_size_bytes > 0, the higher-dim factory
    // uses these explicit tile-space dimensions instead of computing from the
    // tensor's logical shape. This allows repeating TILE-layout tensors without
    // converting to ROW_MAJOR first.
    uint32_t m_tile_higher_pages{0};
    uint32_t m_tile_rep_dim_pages{0};
    uint32_t m_tile_lower_pages{0};
    uint32_t m_tile_page_size_bytes{0};
    int32_t m_repeat_dim{-1};
};

struct RepeatInputs {
    Tensor input;
};

}  // namespace ttnn::prim
