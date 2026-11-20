// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <compare>
#include <cstdint>
#include <tt_stl/assert.hpp>

#include <tt-metalium/base_types.hpp>

namespace ttnn::operations::transformer::sdpa::detail {

// Recipe B-E is SDPAPrecision STANDARD, BALANCED, ACCURATE, FAST.
enum class Recipe : uint8_t { B, C, D, E };
enum class KVStorage : uint8_t { BF16, BFP8, BFP4 };

struct RecipeSelection {
    Recipe recipe;
    KVStorage kv_storage = KVStorage::BF16;
    auto operator<=>(const RecipeSelection&) const = default;
};

// The numerical choices the host passes to the recipe kernels. QK fidelity, exp and
// normalization are fixed per recipe inside the kernels: STANDARD and FAST take their score exp at one
// SFPLOADMACRO per 32 scores (kernels/compute/recipe/recipe_sfpu.hpp, init_sdpa_score_exp), BALANCED and
// ACCURATE the refined FP32 grid exp; see tech_reports/FlashAttention/SDPAPrecisionRecipes.md.
struct PrecisionPolicy {
    RecipeSelection selection;
    tt::tt_metal::MathFidelity pv_fidelity;  // compute-config fidelity; C runs QK at HiFi3 in-kernel
    // False (STANDARD, FAST): BF16 scores against a reference row max, O and l accumulated in FP32 in L1.
    bool fp32_destination;
    // STANDARD, FAST: the most K tiles whose PV products accumulate in the 16-bit DEST before the packer adds them
    // to O in FP32 (the fused chunks split a longer K chunk's PV into pieces), so the O error does not depend on the
    // K chunk the planner picks. The intervals are the hand-tuned Blackhole SDPA loop's, measured there (one core,
    // Q512 x K8192, randn): STANDARD folding every 8 blocks of 64 keys lost 0.22 points of rel-L2 to the 16-bit O
    // accumulation against every 4 (2.50% vs 2.28%), FAST at 8 blocks matched the FAST recipe (3.06% vs 3.00%).
    // 0: FP32 DEST, no limit.
    uint32_t pv_fold_tiles;
    auto operator<=>(const PrecisionPolicy&) const = default;
};

// Every recipe takes BF16, BFP8 or BFP4 K/V: the unpacker expands packed K/V into the recipe's BF16 source
// registers, so K/V storage changes the inputs (and the CB formats), not the recipe's arithmetic.
constexpr PrecisionPolicy resolve_precision_policy(RecipeSelection selection) {
    using Fidelity = tt::tt_metal::MathFidelity;
    switch (selection.recipe) {
        case Recipe::B: return {selection, Fidelity::HiFi2, false, 8};
        case Recipe::C: return {selection, Fidelity::HiFi2, true, 0};
        case Recipe::D: return {selection, Fidelity::HiFi4, true, 0};
        case Recipe::E: return {selection, Fidelity::LoFi, false, 16};
    }
    TT_THROW("Unknown SDPA precision recipe");
}

}  // namespace ttnn::operations::transformer::sdpa::detail
