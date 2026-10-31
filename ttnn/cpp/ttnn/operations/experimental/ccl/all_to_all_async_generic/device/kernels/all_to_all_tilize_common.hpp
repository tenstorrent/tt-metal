// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Row-major input (TILIZE_INPUT): the tilize kernel fills the packet buffer a tile row at a time and the writer takes
// it a packet at a time. After each destination range, the tilize kernel pushes empty pages up to a multiple of
// range_align_tiles (a multiple of both a tile row and a packet, as is the buffer's size), and the writer pops them
// without sending, so every range starts aligned and no tile row or packet crosses the buffer's end.
constexpr uint32_t range_pad_tiles(uint32_t range_tiles, uint32_t range_align_tiles) {
    return (range_align_tiles - range_tiles % range_align_tiles) % range_align_tiles;
}
