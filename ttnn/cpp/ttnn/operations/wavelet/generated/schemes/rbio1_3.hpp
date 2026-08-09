// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct rbio1_3_inverse;

struct rbio1_3 {
    static constexpr const char* name = "rbio1.3";
    static constexpr uint32_t tap_size = 6U;
    static constexpr int32_t delay_even = 1;
    static constexpr int32_t delay_odd = 2;
    static constexpr uint32_t num_steps = 5U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/rbio1_3.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::rbio1_3";
    using inverse = rbio1_3_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct rbio1_3::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0x3f800000U>;
    static_assert(type::k == 1U);
};

template <>
struct rbio1_3::step<1> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct rbio1_3::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xbd800000U, 0xbf000000U, 0x3d800000U>;
    static_assert(type::k == 3U);
};

template <>
struct rbio1_3::step<3> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3f3504f3U>;
    static_assert(type::k == 1U);
};

template <>
struct rbio1_3::step<4> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xbfb504f3U>;
    static_assert(type::k == 1U);
};

struct rbio1_3_inverse {
    static constexpr const char* name = "rbio1.3-inverse";
    static constexpr uint32_t tap_size = 6U;
    static constexpr uint32_t num_steps = 5U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/rbio1_3.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::rbio1_3_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct rbio1_3_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xbf3504f3U>;
    static_assert(type::k == 1U);
};

template <>
struct rbio1_3_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3fb504f3U>;
    static_assert(type::k == 1U);
};

template <>
struct rbio1_3_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0x3d800000U, 0x3f000000U, 0xbd800000U>;
    static_assert(type::k == 3U);
};

template <>
struct rbio1_3_inverse::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct rbio1_3_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, 0, 0xbf800000U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
