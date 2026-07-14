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

    // Tile-native: higher-dim factory uses explicit tile-space dims from host.
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
