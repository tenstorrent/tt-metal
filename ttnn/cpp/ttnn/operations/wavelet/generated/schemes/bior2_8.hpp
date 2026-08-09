// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct bior2_8_inverse;

struct bior2_8 {
    static constexpr const char* name = "bior2.8";
    static constexpr uint32_t tap_size = 18U;
    static constexpr int32_t delay_even = 4;
    static constexpr int32_t delay_odd = 5;
    static constexpr uint32_t num_steps = 5U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/bior2_8.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::bior2_8";
    using inverse = bior2_8_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct bior2_8::step<0> {
    using type = StaticStep<StepType::kUpdate, -1, 0xbf000000U, 0xbf000000U>;
    static_assert(type::k == 2U);
};

template <>
struct bior2_8::step<1> {
    using type = StaticStep<StepType::kPredict, -3, 0xbb0c0000U, 0x3ca78000U, 0xbdc36000U, 0x3ea77800U, 0x3ea77800U, 0xbdc36000U, 0x3ca78000U, 0xbb0c0000U>;
    static_assert(type::k == 8U);
};

template <>
struct bior2_8::step<2> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct bior2_8::step<3> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3fb504f3U>;
    static_assert(type::k == 1U);
};

template <>
struct bior2_8::step<4> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xbf3504f3U>;
    static_assert(type::k == 1U);
};

struct bior2_8_inverse {
    static constexpr const char* name = "bior2.8-inverse";
    static constexpr uint32_t tap_size = 18U;
    static constexpr uint32_t num_steps = 5U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/bior2_8.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::bior2_8_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct bior2_8_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xbfb504f3U>;
    static_assert(type::k == 1U);
};

template <>
struct bior2_8_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3f3504f3U>;
    static_assert(type::k == 1U);
};

template <>
struct bior2_8_inverse::step<2> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct bior2_8_inverse::step<3> {
    using type = StaticStep<StepType::kPredict, -3, 0x3b0c0000U, 0xbca78000U, 0x3dc36000U, 0xbea77800U, 0xbea77800U, 0x3dc36000U, 0xbca78000U, 0x3b0c0000U>;
    static_assert(type::k == 8U);
};

template <>
struct bior2_8_inverse::step<4> {
    using type = StaticStep<StepType::kUpdate, -1, 0x3f000000U, 0x3f000000U>;
    static_assert(type::k == 2U);
};

}  // namespace ttnn::operations::wavelet::schemes
