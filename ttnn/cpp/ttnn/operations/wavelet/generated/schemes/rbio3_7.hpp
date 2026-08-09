// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct rbio3_7_inverse;

struct rbio3_7 {
    static constexpr const char* name = "rbio3.7";
    static constexpr uint32_t tap_size = 16U;
    static constexpr int32_t delay_even = 4;
    static constexpr int32_t delay_odd = 4;
    static constexpr uint32_t num_steps = 5U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/rbio3_7.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::rbio3_7";
    using inverse = rbio3_7_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct rbio3_7::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0x3eaaaaabU>;
    static_assert(type::k == 1U);
};

template <>
struct rbio3_7::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f900000U, 0x3ec00000U>;
    static_assert(type::k == 2U);
};

template <>
struct rbio3_7::step<2> {
    using type = StaticStep<StepType::kPredict, -3, 0xbb78e38eU, 0x3d055555U, 0xbe0c5555U, 0xbee38e39U, 0x3e0c5555U, 0xbd055555U, 0x3b78e38eU>;
    static_assert(type::k == 7U);
};

template <>
struct rbio3_7::step<3> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3ef15befU>;
    static_assert(type::k == 1U);
};

template <>
struct rbio3_7::step<4> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x4007c3b6U>;
    static_assert(type::k == 1U);
};

struct rbio3_7_inverse {
    static constexpr const char* name = "rbio3.7-inverse";
    static constexpr uint32_t tap_size = 16U;
    static constexpr uint32_t num_steps = 5U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/rbio3_7.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::rbio3_7_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct rbio3_7_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3ef15bf0U>;
    static_assert(type::k == 1U);
};

template <>
struct rbio3_7_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x4007c3b6U>;
    static_assert(type::k == 1U);
};

template <>
struct rbio3_7_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, -3, 0x3b78e38eU, 0xbd055555U, 0x3e0c5555U, 0x3ee38e39U, 0xbe0c5555U, 0x3d055555U, 0xbb78e38eU>;
    static_assert(type::k == 7U);
};

template <>
struct rbio3_7_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf900000U, 0xbec00000U>;
    static_assert(type::k == 2U);
};

template <>
struct rbio3_7_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xbeaaaaabU>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
