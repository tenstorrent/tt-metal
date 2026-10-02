// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

// SDPA input preparation splits `batches` tile batches over `cores` cores numbered row-major on a
// `grid_x`-wide grid; the first batches % cores cores take one extra batch. Every kernel derives its
// own range from its logical core, so the host passes only common runtime args.
struct PrepareRange {
    uint32_t start;  // first tile
    uint32_t count;  // tiles
};

inline PrepareRange prepare_range(uint32_t x, uint32_t y, uint32_t grid_x, uint32_t batches, uint32_t cores, uint32_t batch) {
    const uint32_t index = y * grid_x + x;
    const uint32_t base = batches / cores, extra = batches % cores;
    const uint32_t first = index * base + (index < extra ? index : extra);
    return {first * batch, (base + (index < extra ? 1u : 0u)) * batch};
}
