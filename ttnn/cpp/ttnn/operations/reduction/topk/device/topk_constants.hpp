// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

namespace ttnn::prim::constants {

// Minimum reduced-dim width for multi-core execution regardless of tensor height.
constexpr uint32_t multi_core_min_width = 8192;

// Ht-aware relaxation of multi_core_min_width: inputs with at most
// multi_core_low_ht_max_tile_rows tile rows leave most of the grid idle on the
// row-parallel single-core factory, so the column-split multi-core path is
// enabled from multi_core_low_ht_min_width up. 1024 is the smallest pow2 width
// that still gives every local core >= min_dim_per_core elements at a useful
// core count (measured ~4x on 32x2048 k=32, p150a).
constexpr uint32_t multi_core_low_ht_min_width = 1024;
constexpr uint32_t multi_core_low_ht_max_tile_rows = 2;

// The multi-core bitonic sort network addresses elements with 16-bit indices,
// so the reduced dim must stay below 65535 (exclusive bound, uint16 max).
constexpr uint32_t multi_core_max_width_exclusive = std::numeric_limits<uint16_t>::max();

// Largest K the multi-core local-topk/gather/final-topk pipeline supports.
constexpr uint32_t multi_core_max_k = 64;

constexpr uint32_t min_dim_per_core = 64;  // Minimum dimension size per core required

}  // namespace ttnn::prim::constants
