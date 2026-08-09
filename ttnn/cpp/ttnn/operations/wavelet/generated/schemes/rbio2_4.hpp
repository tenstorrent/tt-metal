// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct rbio2_4_inverse;

struct rbio2_4 {
    static constexpr const char* name = "rbio2.4";
    static constexpr uint32_t tap_size = 10U;
    static constexpr int32_t delay_even = 2;
    static constexpr int32_t delay_odd = 3;
    static constexpr uint32_t num_steps = 5U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/rbio2_4.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::rbio2_4";
    using inverse = rbio2_4_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct rbio2_4::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0x3f000000U, 0x3f000000U>;
    static_assert(type::k == 2U);
};

template <>
struct rbio2_4::step<1> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct rbio2_4::step<2> {
    using type = StaticStep<StepType::kPredict, -2, 0x3d400000U, 0xbe980000U, 0xbe980000U, 0x3d400000U>;
    static_assert(type::k == 4U);
};

template <>
struct rbio2_4::step<3> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3f3504f3U>;
    static_assert(type::k == 1U);
};

template <>
struct rbio2_4::step<4> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xbfb504f3U>;
    static_assert(type::k == 1U);
};

struct rbio2_4_inverse {
    static constexpr const char* name = "rbio2.4-inverse";
    static constexpr uint32_t tap_size = 10U;
    static constexpr uint32_t num_steps = 5U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/rbio2_4.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::rbio2_4_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct rbio2_4_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xbf3504f3U>;
    static_assert(type::k == 1U);
};

template <>
struct rbio2_4_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3fb504f3U>;
    static_assert(type::k == 1U);
};

template <>
struct rbio2_4_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, -2, 0xbd400000U, 0x3e980000U, 0x3e980000U, 0xbd400000U>;
    static_assert(type::k == 4U);
};

template <>
struct rbio2_4_inverse::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct rbio2_4_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, 0, 0xbf000000U, 0xbf000000U>;
    static_assert(type::k == 2U);
};

}  // namespace ttnn::operations::wavelet::schemes
