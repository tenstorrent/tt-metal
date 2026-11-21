// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <span>

namespace ttnn::experimental::llama31_8b::rotary_bf8_table {

struct CoreWork {
    std::uint32_t x;
    std::uint32_t y;
    std::uint32_t seq_start;
    std::uint32_t seq_end;
    std::uint32_t head_start;
    std::uint32_t head_end;
};

struct Entry {
    std::uint32_t grid_x;
    std::uint32_t grid_y;
    std::uint32_t n_heads;
    std::uint32_t seq_tiles;
    std::uint32_t head_dim_tiles;
    std::uint32_t cos_seq_tiles;
    std::uint32_t rotary_seq_tiles;
    std::uint32_t reload;
    std::uint32_t in_pages;
    std::uint32_t cos_sin_pages;
    std::uint32_t interm_pages;
    std::uint32_t out_pages;
    std::span<const CoreWork> cores;
};

inline constexpr std::array<CoreWork, 32> CORES_11X10_H32_S32_W4_C32{{
    {0, 0, 0, 1, 0, 32},    {1, 0, 1, 2, 0, 32},    {2, 0, 2, 3, 0, 32},   {3, 0, 3, 4, 0, 32},   {4, 0, 4, 5, 0, 32},
    {5, 0, 5, 6, 0, 32},    {6, 0, 6, 7, 0, 32},    {7, 0, 7, 8, 0, 32},   {8, 0, 8, 9, 0, 32},   {9, 0, 9, 10, 0, 32},
    {10, 0, 10, 11, 0, 32}, {0, 1, 11, 12, 0, 32},  {1, 1, 12, 13, 0, 32}, {2, 1, 13, 14, 0, 32}, {3, 1, 14, 15, 0, 32},
    {4, 1, 15, 16, 0, 32},  {5, 1, 16, 17, 0, 32},  {6, 1, 17, 18, 0, 32}, {7, 1, 18, 19, 0, 32}, {8, 1, 19, 20, 0, 32},
    {9, 1, 20, 21, 0, 32},  {10, 1, 21, 22, 0, 32}, {0, 2, 22, 23, 0, 32}, {1, 2, 23, 24, 0, 32}, {2, 2, 24, 25, 0, 32},
    {3, 2, 25, 26, 0, 32},  {4, 2, 26, 27, 0, 32},  {5, 2, 27, 28, 0, 32}, {6, 2, 28, 29, 0, 32}, {7, 2, 29, 30, 0, 32},
    {8, 2, 30, 31, 0, 32},  {9, 2, 31, 32, 0, 32},
}};

inline constexpr std::array<CoreWork, 32> CORES_11X10_H8_S32_W4_C32{{
    {0, 0, 0, 1, 0, 8},    {1, 0, 1, 2, 0, 8},    {2, 0, 2, 3, 0, 8},   {3, 0, 3, 4, 0, 8},   {4, 0, 4, 5, 0, 8},
    {5, 0, 5, 6, 0, 8},    {6, 0, 6, 7, 0, 8},    {7, 0, 7, 8, 0, 8},   {8, 0, 8, 9, 0, 8},   {9, 0, 9, 10, 0, 8},
    {10, 0, 10, 11, 0, 8}, {0, 1, 11, 12, 0, 8},  {1, 1, 12, 13, 0, 8}, {2, 1, 13, 14, 0, 8}, {3, 1, 14, 15, 0, 8},
    {4, 1, 15, 16, 0, 8},  {5, 1, 16, 17, 0, 8},  {6, 1, 17, 18, 0, 8}, {7, 1, 18, 19, 0, 8}, {8, 1, 19, 20, 0, 8},
    {9, 1, 20, 21, 0, 8},  {10, 1, 21, 22, 0, 8}, {0, 2, 22, 23, 0, 8}, {1, 2, 23, 24, 0, 8}, {2, 2, 24, 25, 0, 8},
    {3, 2, 25, 26, 0, 8},  {4, 2, 26, 27, 0, 8},  {5, 2, 27, 28, 0, 8}, {6, 2, 28, 29, 0, 8}, {7, 2, 29, 30, 0, 8},
    {8, 2, 30, 31, 0, 8},  {9, 2, 31, 32, 0, 8},
}};

inline constexpr std::array<Entry, 2> ENTRIES{{
    {11, 10, 32, 32, 4, 32, 32, 1, 8, 8, 4, 8, CORES_11X10_H32_S32_W4_C32},
    {11, 10, 8, 32, 4, 32, 32, 0, 8, 8, 4, 8, CORES_11X10_H8_S32_W4_C32},
}};

inline const Entry* find(
    std::uint32_t grid_x,
    std::uint32_t grid_y,
    std::uint32_t n_heads,
    std::uint32_t seq_tiles,
    std::uint32_t head_dim_tiles,
    std::uint32_t cos_seq_tiles) {
    for (const auto& e : ENTRIES) {
        if (e.grid_x == grid_x && e.grid_y == grid_y && e.n_heads == n_heads && e.seq_tiles == seq_tiles &&
            e.head_dim_tiles == head_dim_tiles && e.cos_seq_tiles == cos_seq_tiles) {
            return &e;
        }
    }
    return nullptr;
}

}  // namespace ttnn::experimental::llama31_8b::rotary_bf8_table
