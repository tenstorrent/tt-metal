// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

namespace ttnn::prim::llama31_8b_rms_norm_residual_decode::table {

struct Row {
    std::uint32_t grid_x;
    std::uint32_t grid_y;
    std::uint32_t shard_w;
    std::uint32_t subblock_w;
};

inline constexpr std::uint32_t kWidth = 4096;
inline constexpr std::uint32_t kResidualRow = 1;
inline constexpr std::array<Row, 2> kRows{{{8, 8, 64, 2}, {8, 4, 128, 4}}};

}  // namespace ttnn::prim::llama31_8b_rms_norm_residual_decode::table
