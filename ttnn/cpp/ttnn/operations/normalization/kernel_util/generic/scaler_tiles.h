// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file scaler_tiles.h
 * @brief Single source of truth for the layernorm reduce-scaler CB tile count.
 */

#pragma once

#include <cstdint>

namespace norm::kernel_util::generic {

/**
 * @brief Number of reduce-scaler tiles the layernorm reader pushes into cb_scaler
 *        and the compute kernel must pop.
 *
 * The reader always generates one full-row scaler tile, plus a second
 * partial-column scaler tile when the row width @p W is not a multiple of the
 * tile width. The reader push count and the compute pop count MUST agree; if the
 * two ever diverge, cb_scaler drifts and the op eventually deadlocks or reads
 * stale data (see issue #48487). Both sides derive the count from this helper so
 * the formula lives in exactly one place.
 *
 * @param W          Row width in elements (last dim).
 * @param tile_width Tile width in elements; both sides must pass the same value.
 * @return 2 when the last column tile is partial, otherwise 1.
 */
constexpr uint32_t reduce_scaler_tile_count(uint32_t W, uint32_t tile_width) { return (W % tile_width > 0) ? 2 : 1; }

}  // namespace norm::kernel_util::generic
