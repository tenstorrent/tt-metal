// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Supported geometry of the named SDPA precision recipes (docs/sdpa_precision.md).
//
// `recipe_geometry_rejection` is the only place that knows which (op, recipe, Q, K, D) geometries
// the kernels support; the ring / exp-ring entry points and device operations validate through
// `validate_recipe_geometry`. L1 fit is checked when the program is built.

#include <cstdint>
#include <optional>
#include <string>

#include "sdpa_precision_policy.hpp"

namespace ttnn::operations::transformer::sdpa::detail {

enum class RecipeOp : uint8_t { Dense, Joint, Ring, ExpRing };

// Why (op, recipe, q_tiles, k_tiles, d_tiles) is not a supported recipe geometry, or nullopt if it is.
// This is shape support only; L1 fit is checked when the program is built.
std::optional<std::string> recipe_geometry_rejection(
    RecipeOp op, const PrecisionPolicy& policy, uint32_t q_tiles, uint32_t k_tiles, uint32_t d_tiles);

// TT_FATAL unless the chunk sizes and head dim are tile-aligned and recipe_geometry_rejection accepts them.
void validate_recipe_geometry(
    RecipeOp op, const PrecisionPolicy& policy, uint32_t q_chunk_size, uint32_t k_chunk_size, uint32_t head_dim);

}  // namespace ttnn::operations::transformer::sdpa::detail
