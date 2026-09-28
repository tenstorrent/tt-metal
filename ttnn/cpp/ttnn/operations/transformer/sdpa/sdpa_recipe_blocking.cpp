// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "sdpa_recipe_blocking.hpp"

#include <fmt/format.h>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::transformer::sdpa::detail {
namespace {

constexpr uint32_t kTile = 32;

}  // namespace

std::optional<std::string> recipe_geometry_rejection(
    RecipeOp op, const PrecisionPolicy& policy, uint32_t q_tiles, uint32_t k_tiles, uint32_t d_tiles) {
    // The only statement of supported recipe geometry: the ring / exp-ring entry points and device
    // operations validate through validate_recipe_geometry below. Dense/joint also mirror it in
    // recipe_dense_q_tiles / recipe_dense_k_tiles (sdpa_recipe.cpp).
    if (q_tiles == 0 || k_tiles == 0 || d_tiles == 0) {
        return std::string("recipes require tile-aligned, nonzero Q/K chunks and head dims");
    }
    // B-E on every op: any tile-aligned Q chunk up to the recurrent-state arrays (32 tile rows), any
    // tile-aligned K chunk and head dim; L1 fit is checked when the program is built.
    if (q_tiles > 32) {
        return fmt::format("Q chunk {} exceeds 1024 rows", q_tiles * kTile);
    }
    if (op == RecipeOp::Dense || op == RecipeOp::Joint || policy.selection.recipe != Recipe::A) {
        return std::nullopt;
    }
    // FAST ring / exp ring keep the legacy ring kernels, qualified only at these geometries.
    if (d_tiles != 2 && d_tiles != 4 && d_tiles != 8) {
        return fmt::format("head dim {} is not 64, 128 or 256", d_tiles * kTile);
    }
    if (q_tiles < 4 || q_tiles > 10) {
        return fmt::format("Q chunk {} is outside 128-320 rows", q_tiles * kTile);
    }
    if (k_tiles != 8 && k_tiles != 12 && k_tiles != 16) {
        return fmt::format("K chunk {} is not 256, 384 or 512 rows", k_tiles * kTile);
    }
    if (op == RecipeOp::ExpRing && (k_tiles != 16 || d_tiles != 4)) {
        return std::string("exp ring recipes require K512/D128");
    }
    return std::nullopt;
}

void validate_recipe_geometry(
    RecipeOp op, const PrecisionPolicy& policy, uint32_t q_chunk_size, uint32_t k_chunk_size, uint32_t head_dim) {
    const char* name = op == RecipeOp::Ring      ? "ring"
                       : op == RecipeOp::ExpRing ? "exp ring"
                       : op == RecipeOp::Joint   ? "joint"
                                                 : "dense";
    TT_FATAL(
        q_chunk_size % kTile == 0 && k_chunk_size % kTile == 0 && head_dim % kTile == 0,
        "Named {} SDPA recipes require tile-aligned Q/K chunks and head dims, got Q{}/K{}/D{}",
        name,
        q_chunk_size,
        k_chunk_size,
        head_dim);
    const auto rejection =
        recipe_geometry_rejection(op, policy, q_chunk_size / kTile, k_chunk_size / kTile, head_dim / kTile);
    TT_FATAL(
        !rejection,
        "Named {} SDPA recipes do not support Q{}/K{}/D{}: {}",
        name,
        q_chunk_size,
        k_chunk_size,
        head_dim,
        rejection.value_or(""));
}

}  // namespace ttnn::operations::transformer::sdpa::detail
