// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Op-selected blocking for the named SDPA precision recipes (docs/sdpa_precision.md, "Blocking").
//
// A recipe caller may leave the Q and/or K chunk size unset (no program_config for dense SDPA, or a
// chunk size of 0 in SDPAProgramConfig); the op then chooses them, and for exp ring also the worker
// grid width, from the shape, recipe, op, grid and L1. Explicit chunk sizes are always honored and
// validated as before. Blocking never changes a recipe's arithmetic; it can change the rounding
// order (docs/sdpa_precision.md, "Q blocking"). Q256/K512/D128 remains the frozen geometry.
//
// Layering: `recipe_geometry_rejection` is the only place that knows which (op, recipe, Q, K, D)
// geometries the kernels support, and `recipe_l1_bytes` the only place that sizes their circular
// buffers (the recipe part is `recipe_compute_program`, the program factories' own layout). The
// chooser consults nothing else, so widening the supported geometry needs no chooser change.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "sdpa_precision_policy.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::transformer::sdpa::detail {

enum class RecipeOp : uint8_t { Dense, Joint, Ring, ExpRing };

// Q/K chunk tile counts the chooser enumerates. Candidates outside the supported geometry are
// filtered by `recipe_geometry_rejection`; this range only bounds the search to where the cost
// model is fitted: Q from 128 rows and K from 256 to 512 rows (shorter only when the whole
// sequence is shorter). Smaller blocks are overhead-dominated and K > 512 is not yet measured;
// a caller may still pass any supported chunk explicitly.
inline constexpr uint32_t kRecipeSearchMinQTiles = 4;
inline constexpr uint32_t kRecipeSearchMaxQTiles = 32;
inline constexpr uint32_t kRecipeSearchMinKTiles = 8;
inline constexpr uint32_t kRecipeSearchMaxKTiles = 16;

// Why (op, recipe, q_tiles, k_tiles, d_tiles) is not a supported recipe geometry, or nullopt if it is.
// This is shape support only; L1 fit is `recipe_l1_bytes`.
std::optional<std::string> recipe_geometry_rejection(
    RecipeOp op, const PrecisionPolicy& policy, uint32_t q_tiles, uint32_t k_tiles, uint32_t d_tiles);

inline bool recipe_geometry_supported(
    RecipeOp op, const PrecisionPolicy& policy, uint32_t q_tiles, uint32_t k_tiles, uint32_t d_tiles) {
    return !recipe_geometry_rejection(op, policy, q_tiles, k_tiles, d_tiles).has_value();
}

// Schedule facts that change the circular-buffer layout.
struct RecipeL1Context {
    uint32_t q_blocks_per_worker = 2;  // ring: most Q chunks one worker owns (1 lets FAST single-buffer Q)
    uint32_t passes = 1;               // exp ring: head-segments per core row (FAST keeps one Q per pass)
};

// Circular-buffer bytes per core. `preferred` is the op's first-choice layout; `minimum` is the
// smallest layout the op falls back to when `preferred` does not fit (ring: single-slot Q;
// FAST exp ring: streamed Q). A geometry fits iff minimum <= available.
struct RecipeL1Estimate {
    uint64_t preferred = 0;
    uint64_t minimum = 0;
};

// Requires a supported geometry (see recipe_geometry_rejection).
RecipeL1Estimate recipe_l1_bytes(
    RecipeOp op,
    const PrecisionPolicy& policy,
    uint32_t q_tiles,
    uint32_t k_tiles,
    uint32_t d_tiles,
    const RecipeL1Context& context = {});

// Everything the chooser needs, independent of tensors and devices (host-testable).
struct RecipeBlockingProblem {
    RecipeOp op = RecipeOp::Dense;
    PrecisionPolicy policy = resolve_precision_policy({Recipe::A});
    uint32_t batch = 1;
    uint32_t q_heads = 1;
    // Dense/joint: padded rows of the primary and joint segments. Ring/exp ring: the local (per
    // device) primary rows and the joint rows; K rows are those one ring iteration processes.
    uint32_t q_rows = 0;
    uint32_t joint_q_rows = 0;
    uint32_t k_rows = 0;
    uint32_t joint_k_rows = 0;
    uint32_t ring_size = 1;
    uint32_t d_tiles = 4;
    // Dense/joint: the grid the op may use. Ring: the SDPA worker grid. Exp ring: the program
    // config grid including the fabric MUX column (the chooser may narrow its width).
    CoreCoord grid{1, 1};
    uint32_t max_cores_per_head_batch = 16;
    uint64_t l1_bytes = 0;  // unreserved L1 per core available to circular buffers
    // Nonzero pins that dimension (a caller-provided chunk); zero lets the chooser pick.
    uint32_t fixed_q_tiles = 0;
    uint32_t fixed_k_tiles = 0;
    bool exp_mux_on_bottom_row = false;
};

struct RecipeBlocking {
    uint32_t q_chunk_size = 0;
    uint32_t k_chunk_size = 0;
    CoreCoord grid{0, 0};
    double cost = 0.0;           // modeled makespan, arbitrary units (comparable within one problem)
    uint32_t jobs_per_core = 0;  // Q chunks on the busiest core (exp ring: passes)
    RecipeL1Estimate l1{};
};

// Every feasible blocking with its modeled cost, cheapest first (the chooser's candidate list).
std::vector<RecipeBlocking> recipe_blocking_candidates(const RecipeBlockingProblem& problem);

// The chosen blocking, or nullopt if no supported geometry fits.
std::optional<RecipeBlocking> choose_recipe_blocking(const RecipeBlockingProblem& problem);

// True when the caller left a chunk size to the op (no config, or a zero chunk size).
bool recipe_blocking_requested(const std::optional<SDPAProgramConfig>& program_config);

// Legacy (no precision) calls must pass explicit chunk sizes.
void reject_auto_blocking_without_recipe(const std::optional<SDPAProgramConfig>& program_config);

// Tensor-level hooks: return the caller's config when both chunk sizes are explicit, otherwise a
// copy with the chosen chunk sizes (and, for exp ring, grid). If no candidate fits, unset chunks
// fall back to Q256/K512 so the op's own validation reports why.
std::optional<SDPAProgramConfig> resolve_dense_recipe_blocking(
    const PrecisionPolicy& policy,
    const Tensor& q,
    const Tensor& k,
    const Tensor* joint_q,
    const Tensor* joint_k,
    const std::optional<SDPAProgramConfig>& program_config);

SDPAProgramConfig resolve_ring_recipe_blocking(
    const PrecisionPolicy& policy,
    const Tensor& q,
    const Tensor& k,
    const std::optional<Tensor>& joint_q,
    const std::optional<Tensor>& joint_k,
    uint32_t ring_size,
    const SDPAProgramConfig& program_config);

SDPAProgramConfig resolve_exp_ring_recipe_blocking(
    const PrecisionPolicy& policy,
    const Tensor& q,
    const Tensor& k,
    const std::optional<Tensor>& joint_q,
    uint32_t ring_size,
    const SDPAProgramConfig& program_config);

}  // namespace ttnn::operations::transformer::sdpa::detail
