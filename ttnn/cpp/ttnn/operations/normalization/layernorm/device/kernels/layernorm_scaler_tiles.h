// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Shared source of truth for the layernorm reduce-scaler CB tile count, so the reader's push
// count and the compute kernel's pop count are derived from one formula (issue #48487).

#pragma once

#include <cstdint>

namespace norm::layernorm {

// Number of reduce-scaler tiles the reader pushes into cb_scaler (and compute must pop): one
// full-row tile, plus a second tile when the last column tile is partial (W not a multiple of
// tile_width). Both sides must pass the same tile_width.
constexpr uint32_t reduce_scaler_tile_count(uint32_t W, uint32_t tile_width) { return (W % tile_width > 0) ? 2 : 1; }

}  // namespace norm::layernorm
