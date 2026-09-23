// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Shared host/kernel CB contract for exp-ring named precision recipes (SDPA_RECIPE_EXP_RING).
// The recipe compute owns fixed CB indices 0-16 (Q=0, K=1, V=2, reduce scaler=3, column identity=4,
// ...; see recipe_compute_program in sdpa_recipe.cpp). The exp-ring dataflow-only CBs that would collide
// with that layout move to free indices above it. Only the K/V fabric-writer aliases survive in recipe
// mode: the lightweight mask, scale, stats, state-FIFO and derived CBs are not allocated (the recipe
// masks key tails in the pack thread and keeps its own recurrent state).
namespace ttnn::operations::transformer::sdpa::exp_ring {

// Second handles on the recipe K (1) and V (2) CBs so the MUX writer pops independently of compute.
inline constexpr uint32_t kRecipeKWriterAliasCb = 19;
inline constexpr uint32_t kRecipeVWriterAliasCb = 20;

}  // namespace ttnn::operations::transformer::sdpa::exp_ring
