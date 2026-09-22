// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tt_stl/assert.hpp>

#include <tt-metalium/base_types.hpp>

namespace ttnn::operations::transformer::sdpa::detail {

// Internal names preserve the frozen research identities. This is not a public
// preset API and does not establish device/shape eligibility or dispatch.
enum class Recipe : uint8_t { A, B, C, D, E };
enum class KVStorage : uint8_t { BF16, BFP8, BFP4 };
enum class RecurrentState : uint8_t { BF16, CompensatedBF16, FP32 };
enum class SoftmaxArithmetic : uint8_t { Baseline, CompensatedBF16, BalancedFP32, AccurateFP32, CompensatedLoFi };
enum class InputPreparation : uint8_t { None, Rne7, Rne5, Rne5ThenBfp8, Bfp4GridRneSaturate };

struct RecipeSelection {
    Recipe recipe;
    KVStorage kv_storage = KVStorage::BF16;
    bool operator==(const RecipeSelection&) const = default;
};

// Numerical intent only. Buffer depth, grid, chunks, fusion/grouping eligibility
// and addresses are deliberately not numerical-policy fields. SoftmaxArithmetic
// identifies a complete qualified subtraction/exp/normalization recipe, not
// merely the low-level math_approx_mode flag (which is true even for D).
struct PrecisionPolicy {
    RecipeSelection selection;
    tt::tt_metal::MathFidelity qk_fidelity;
    tt::tt_metal::MathFidelity pv_fidelity;
    bool fp32_destination;
    RecurrentState recurrent_state;
    SoftmaxArithmetic softmax_arithmetic;
    InputPreparation q_preparation;
    InputPreparation kv_preparation;
    bool operator==(const PrecisionPolicy&) const = default;
};

constexpr PrecisionPolicy resolve_precision_policy(RecipeSelection selection) {
    using Fidelity = tt::tt_metal::MathFidelity;
    switch (selection.kv_storage) {
        case KVStorage::BF16:
        case KVStorage::BFP8:
        case KVStorage::BFP4: break;
        default: TT_THROW("Unknown SDPA KV storage policy");
    }
    if (selection.recipe != Recipe::E && selection.kv_storage != KVStorage::BF16) {
        TT_THROW("Packed KV is qualified only for explicit SDPA recipe E");
    }
    switch (selection.recipe) {
        case Recipe::A:
            return {
                selection,
                Fidelity::HiFi2,
                Fidelity::HiFi2,
                false,
                RecurrentState::BF16,
                SoftmaxArithmetic::Baseline,
                InputPreparation::None,
                InputPreparation::None};
        case Recipe::B:
            return {
                selection,
                Fidelity::HiFi2,
                Fidelity::HiFi2,
                false,
                RecurrentState::CompensatedBF16,
                SoftmaxArithmetic::CompensatedBF16,
                InputPreparation::None,
                InputPreparation::None};
        case Recipe::C:
            return {
                selection,
                Fidelity::HiFi4,
                Fidelity::HiFi2,
                true,
                RecurrentState::FP32,
                SoftmaxArithmetic::BalancedFP32,
                InputPreparation::None,
                InputPreparation::None};
        case Recipe::D:
            return {
                selection,
                Fidelity::HiFi4,
                Fidelity::HiFi4,
                true,
                RecurrentState::FP32,
                SoftmaxArithmetic::AccurateFP32,
                InputPreparation::None,
                InputPreparation::None};
        case Recipe::E: {
            const auto preparation = selection.kv_storage == KVStorage::BF16   ? InputPreparation::Rne5
                                     : selection.kv_storage == KVStorage::BFP8 ? InputPreparation::Rne5ThenBfp8
                                                                               : InputPreparation::Bfp4GridRneSaturate;
            return {
                selection,
                Fidelity::LoFi,
                Fidelity::LoFi,
                false,
                RecurrentState::CompensatedBF16,
                SoftmaxArithmetic::CompensatedLoFi,
                InputPreparation::Rne7,
                preparation};
        }
    }
    TT_THROW("Unknown SDPA precision recipe");
}

}  // namespace ttnn::operations::transformer::sdpa::detail
