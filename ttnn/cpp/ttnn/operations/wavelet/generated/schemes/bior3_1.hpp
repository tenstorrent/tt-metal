// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct bior3_1_inverse;

struct bior3_1 {
    static constexpr const char* name = "bior3.1";
    static constexpr uint32_t tap_size = 4U;
    static constexpr int32_t delay_even = 1;
    static constexpr int32_t delay_odd = 1;
    static constexpr uint32_t num_steps = 5U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/bior3_1.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::bior3_1";
    using inverse = bior3_1_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct bior3_1::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0xbeaaaaabU>;
    static_assert(type::k == 1U);
};

template <>
struct bior3_1::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f900000U, 0xbec00000U>;
    static_assert(type::k == 2U);
};

template <>
struct bior3_1::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0xbee38e39U>;
    static_assert(type::k == 1U);
};

template <>
struct bior3_1::step<3> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3f715befU>;
    static_assert(type::k == 1U);
};

template <>
struct bior3_1::step<4> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3f87c3b6U>;
    static_assert(type::k == 1U);
};

struct bior3_1_inverse {
    static constexpr const char* name = "bior3.1-inverse";
    static constexpr uint32_t tap_size = 4U;
    static constexpr uint32_t num_steps = 5U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/bior3_1.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::bior3_1_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct bior3_1_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3f715bf0U>;
    static_assert(type::k == 1U);
};

template <>
struct bior3_1_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3f87c3b6U>;
    static_assert(type::k == 1U);
};

template <>
struct bior3_1_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0x3ee38e39U>;
    static_assert(type::k == 1U);
};

template <>
struct bior3_1_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf900000U, 0x3ec00000U>;
    static_assert(type::k == 2U);
};

template <>
struct bior3_1_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x3eaaaaabU>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
