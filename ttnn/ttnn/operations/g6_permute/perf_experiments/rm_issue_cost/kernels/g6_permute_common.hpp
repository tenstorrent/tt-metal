// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Regime rows_tile_scatter: out[n, b, a, w] = x[n, a, b, w] on bfloat16 tiles of four 16x16 faces.
// Work unit = one input tile; a core owns a contiguous range of input pages
//     p_in(n, a, bt, wt) = ((n * tensor_a + a) * tensor_bt + bt) * tensor_wt + wt
// and walks it with InTileCursor (wt innermost) instead of dividing the page id per tile.
namespace g6_permute {

constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t FACE_ROWS = 16;

template <uint32_t face_row_bytes>
struct TileGeometry {
    static constexpr uint32_t face_bytes = FACE_ROWS * face_row_bytes;
    static constexpr uint32_t tile_bytes = 4 * face_bytes;

    // Half h (columns 0-15 / 16-31) of tile row t: faces 0/1 hold rows 0-15, faces 2/3 rows 16-31.
    static constexpr uint32_t face_off(uint32_t t, uint32_t h) {
        return ((t / FACE_ROWS) * 2 + h) * face_bytes + (t % FACE_ROWS) * face_row_bytes;
    }

    // Shortest byte prefix of a tile that holds both halves of rows 0..rows-1. Reads start at the
    // page base so local and remote addresses agree mod 64 (Blackhole DRAM read alignment).
    static constexpr uint32_t prefix_bytes(uint32_t rows) {
        const uint32_t top = (rows - 1) / FACE_ROWS;
        return top * 2 * face_bytes + face_bytes + (rows - FACE_ROWS * top) * face_row_bytes;
    }
};

// Live rows of input tile row bt: the last one holds tensor_b - 32 * (tensor_bt - 1) rows.
template <uint32_t tensor_b, uint32_t tensor_bt>
struct RowsLive {
    static constexpr uint32_t last = tensor_b - TILE_ROWS * (tensor_bt - 1);
    static_assert(last >= 1 && last <= TILE_ROWS, "tensor_bt = ceil(tensor_b / 32)");
};

template <uint32_t tensor_a, uint32_t tensor_bt, uint32_t tensor_wt>
struct InTileCursor {
    uint32_t n, a, bt, wt;

    void advance() {
        if (++wt == tensor_wt) {
            wt = 0;
            if (++bt == tensor_bt) {
                bt = 0;
                if (++a == tensor_a) {
                    a = 0;
                    ++n;
                }
            }
        }
    }
};

// Per-block step zones are kept for the first blocks of a core only: zone executions are capped
// per RISC and a zone past the cap vanishes silently. The plain build has no zones, so it compiles
// the block loop once instead of a zoned copy beside the tail copy.
#if defined(PROFILE_KERNEL)
constexpr uint32_t ZONED_BLOCKS = 16;
#else
constexpr uint32_t ZONED_BLOCKS = 0;
#endif

}  // namespace g6_permute

#define G6_ZONED_STEP(zoned, name, ...) \
    if constexpr (zoned) {              \
        DeviceZoneScopedN(name);        \
        __VA_ARGS__;                    \
    } else {                            \
        __VA_ARGS__;                    \
    }
