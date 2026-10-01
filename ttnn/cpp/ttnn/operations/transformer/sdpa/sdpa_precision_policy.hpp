// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tt_stl/assert.hpp>

#include <tt-metalium/base_types.hpp>

namespace ttnn::operations::transformer::sdpa::detail {

// Recipe A-E is SDPAPrecision FAST, STANDARD, BALANCED, ACCURATE, LOW_PRECISION.
enum class Recipe : uint8_t { A, B, C, D, E };
enum class KVStorage : uint8_t { BF16, BFP8, BFP4 };
// Online-softmax running state (row max, row sum, output accumulator) between K chunks.
// ReferenceMaxFP32: BF16 scores against a reference row max, O and l accumulated in FP32 in L1.
enum class RecurrentState : uint8_t { BF16, ReferenceMaxFP32, FP32 };

struct RecipeSelection {
    Recipe recipe;
    KVStorage kv_storage = KVStorage::BF16;
    bool operator==(const RecipeSelection&) const = default;
};

// The numerical choices the host passes to the recipe kernels. QK fidelity, exp and
// normalization are fixed per recipe inside the kernels; see
// tech_reports/FlashAttention/SDPAPrecisionRecipes.md.
struct PrecisionPolicy {
    RecipeSelection selection;
    tt::tt_metal::MathFidelity pv_fidelity;  // compute-config fidelity; C runs QK at HiFi4 in-kernel
    bool fp32_destination;
    RecurrentState recurrent_state;
    bool operator==(const PrecisionPolicy&) const = default;
};

constexpr PrecisionPolicy resolve_precision_policy(RecipeSelection selection) {
    using Fidelity = tt::tt_metal::MathFidelity;
    if (selection.recipe != Recipe::E && selection.kv_storage != KVStorage::BF16) {
        TT_THROW("Only LOW_PRECISION accepts BFP8/BFP4 K/V; other SDPA recipes require BF16 K/V");
    }
    switch (selection.recipe) {
        case Recipe::A: return {selection, Fidelity::HiFi2, false, RecurrentState::BF16};
        case Recipe::B: return {selection, Fidelity::HiFi2, false, RecurrentState::ReferenceMaxFP32};
        case Recipe::C: return {selection, Fidelity::HiFi2, true, RecurrentState::FP32};
        case Recipe::D: return {selection, Fidelity::HiFi4, true, RecurrentState::FP32};
        case Recipe::E: return {selection, Fidelity::LoFi, false, RecurrentState::ReferenceMaxFP32};
    }
    TT_THROW("Unknown SDPA precision recipe");
}

}  // namespace ttnn::operations::transformer::sdpa::detail
