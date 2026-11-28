// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <span>

namespace ttnn::experimental::llama31_8b::rms_norm_residual_table {

struct CoreWork {
    std::uint32_t x;
    std::uint32_t y;
    std::uint32_t tile_rows;
    std::uint32_t tile_offset;
};

struct Entry {
    std::uint32_t grid_x;
    std::uint32_t grid_y;
    std::uint32_t rows;
    std::uint32_t width;
    std::uint32_t row_tiles;
    std::uint32_t block;
    std::uint32_t in_pages;
    std::uint32_t out_pages;
    std::uint32_t hout_pages;
    std::uint32_t fusion_pages;
    std::uint32_t ex2pe_pages;
    std::span<const CoreWork> cores;
};

inline constexpr std::array<CoreWork, 32> CORES_11X10_1024X4096{{
    {0, 0, 1, 0},    {1, 0, 1, 128},  {2, 0, 1, 256},  {3, 0, 1, 384},   {4, 0, 1, 512},   {5, 0, 1, 640},
    {6, 0, 1, 768},  {7, 0, 1, 896},  {8, 0, 1, 1024}, {9, 0, 1, 1152},  {10, 0, 1, 1280}, {0, 1, 1, 1408},
    {1, 1, 1, 1536}, {2, 1, 1, 1664}, {3, 1, 1, 1792}, {4, 1, 1, 1920},  {5, 1, 1, 2048},  {6, 1, 1, 2176},
    {7, 1, 1, 2304}, {8, 1, 1, 2432}, {9, 1, 1, 2560}, {10, 1, 1, 2688}, {0, 2, 1, 2816},  {1, 2, 1, 2944},
    {2, 2, 1, 3072}, {3, 2, 1, 3200}, {4, 2, 1, 3328}, {5, 2, 1, 3456},  {6, 2, 1, 3584},  {7, 2, 1, 3712},
    {8, 2, 1, 3840}, {9, 2, 1, 3968},
}};

inline constexpr std::array<Entry, 1> ENTRIES{{
    {11, 10, 1024, 4096, 128, 4, 64, 8, 8, 8, 8, CORES_11X10_1024X4096},
}};

inline const Entry* find(std::uint32_t grid_x, std::uint32_t grid_y, std::uint32_t rows, std::uint32_t width) {
    for (const auto& e : ENTRIES) {
        if (e.grid_x == grid_x && e.grid_y == grid_y && e.rows == rows && e.width == width) {
            return &e;
        }
    }
    return nullptr;
}

struct TileBytes {
    std::uint32_t in;
    std::uint32_t inb;
    std::uint32_t out;
    std::uint32_t gamma;
    std::uint32_t interm;
    std::uint32_t bf16;
};

inline std::uint64_t cb_bytes(const Entry& e, const TileBytes& t) {
    const std::uint64_t in_pages = e.in_pages, out_pages = e.out_pages, hout_pages = e.hout_pages;
    const std::uint64_t row_tiles = e.row_tiles, ex2pe_pages = e.ex2pe_pages, fusion_pages = e.fusion_pages;
    return in_pages * t.in + out_pages * t.out + 2 * t.bf16 + 2 * t.bf16 + 2 * t.interm + row_tiles * t.in +
           hout_pages * t.in + row_tiles * t.interm + ex2pe_pages * t.interm + fusion_pages * t.interm +
           row_tiles * t.gamma + in_pages * t.inb;
}

}  // namespace ttnn::experimental::llama31_8b::rms_norm_residual_table
