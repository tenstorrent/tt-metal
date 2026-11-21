// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

namespace ttnn::experimental::llama31_8b_v3::kv_rows_update_table {

inline constexpr uint32_t kGridX = 11;
inline constexpr uint32_t kGridY = 10;
inline constexpr uint32_t kKvHeads = 8;
inline constexpr uint32_t kHeadDim = 128;
inline constexpr uint32_t kBlock = 32;
inline constexpr uint32_t kMaxBlocks = 1024;
inline constexpr uint32_t kMetaBytes = 5184;
inline constexpr uint32_t kBf16TileBytes = 2048;
inline constexpr uint32_t kBf8TileBytes = 1088;
inline constexpr std::array<uint32_t, 4> kRows = {1, 2, 4, 8};
inline constexpr std::array<uint32_t, 4> kTiles = {1, 2, 2, 2};
inline constexpr std::array<uint32_t, 4> kScrBytes = {1152, 2240, 2240, 2240};

struct Core {
    uint8_t x;
    uint8_t y;
    uint8_t kv;
    uint8_t head;
    uint8_t col;
};

// clang-format off
inline constexpr std::array<Core, 64> kCores = {{
    {0, 0, 0, 0, 0},
    {1, 0, 0, 1, 0},
    {2, 0, 0, 2, 0},
    {3, 0, 0, 3, 0},
    {4, 0, 0, 4, 0},
    {5, 0, 0, 5, 0},
    {6, 0, 0, 6, 0},
    {7, 0, 0, 7, 0},
    {0, 1, 0, 0, 1},
    {1, 1, 0, 1, 1},
    {2, 1, 0, 2, 1},
    {3, 1, 0, 3, 1},
    {4, 1, 0, 4, 1},
    {5, 1, 0, 5, 1},
    {6, 1, 0, 6, 1},
    {7, 1, 0, 7, 1},
    {0, 2, 0, 0, 2},
    {1, 2, 0, 1, 2},
    {2, 2, 0, 2, 2},
    {3, 2, 0, 3, 2},
    {4, 2, 0, 4, 2},
    {5, 2, 0, 5, 2},
    {6, 2, 0, 6, 2},
    {7, 2, 0, 7, 2},
    {0, 3, 0, 0, 3},
    {1, 3, 0, 1, 3},
    {2, 3, 0, 2, 3},
    {3, 3, 0, 3, 3},
    {4, 3, 0, 4, 3},
    {5, 3, 0, 5, 3},
    {6, 3, 0, 6, 3},
    {7, 3, 0, 7, 3},
    {0, 4, 1, 0, 0},
    {1, 4, 1, 1, 0},
    {2, 4, 1, 2, 0},
    {3, 4, 1, 3, 0},
    {4, 4, 1, 4, 0},
    {5, 4, 1, 5, 0},
    {6, 4, 1, 6, 0},
    {7, 4, 1, 7, 0},
    {0, 5, 1, 0, 1},
    {1, 5, 1, 1, 1},
    {2, 5, 1, 2, 1},
    {3, 5, 1, 3, 1},
    {4, 5, 1, 4, 1},
    {5, 5, 1, 5, 1},
    {6, 5, 1, 6, 1},
    {7, 5, 1, 7, 1},
    {0, 6, 1, 0, 2},
    {1, 6, 1, 1, 2},
    {2, 6, 1, 2, 2},
    {3, 6, 1, 3, 2},
    {4, 6, 1, 4, 2},
    {5, 6, 1, 5, 2},
    {6, 6, 1, 6, 2},
    {7, 6, 1, 7, 2},
    {0, 7, 1, 0, 3},
    {1, 7, 1, 1, 3},
    {2, 7, 1, 2, 3},
    {3, 7, 1, 3, 3},
    {4, 7, 1, 4, 3},
    {5, 7, 1, 5, 3},
    {6, 7, 1, 6, 3},
    {7, 7, 1, 7, 3},
}};
// clang-format on

}  // namespace ttnn::experimental::llama31_8b_v3::kv_rows_update_table
