// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "ttnn/operations/wavelet/planner/step.hpp"

namespace ttnn::operations::wavelet {

template <StepType Type, int32_t Shift, uint32_t... CoeffBits>
struct StaticStep {
    static constexpr StepType type = Type;
    static constexpr int32_t shift = Shift;
    static constexpr uint32_t k = sizeof...(CoeffBits);
    static constexpr std::array<uint32_t, k> coeff_bits = {CoeffBits...};
};

template <typename Scheme, size_t Index>
using SchemeStep = typename Scheme::template step<Index>::type;

[[nodiscard]] constexpr bool is_predict_update_step(const StepType type) noexcept {
    return type == StepType::kPredict || type == StepType::kUpdate;
}

[[nodiscard]] constexpr bool is_scale_step(const StepType type) noexcept {
    return type == StepType::kScaleEven || type == StepType::kScaleOdd;
}

template <typename Step>
[[nodiscard]] constexpr bool is_executable_step() noexcept {
    return is_predict_update_step(Step::type) || is_scale_step(Step::type);
}

template <typename Scheme, size_t Index = 0>
[[nodiscard]] constexpr uint32_t executable_step_count() noexcept {
    if constexpr (Index >= Scheme::num_steps) {
        return 0;
    } else {
        using Step = SchemeStep<Scheme, Index>;
        return (is_executable_step<Step>() ? 1U : 0U) + executable_step_count<Scheme, Index + 1>();
    }
}

}  // namespace ttnn::operations::wavelet
