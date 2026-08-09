// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct bior3_9_inverse;

struct bior3_9 {
    static constexpr const char* name = "bior3.9";
    static constexpr uint32_t tap_size = 20U;
    static constexpr int32_t delay_even = 5;
    static constexpr int32_t delay_odd = 5;
    static constexpr uint32_t num_steps = 6U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/bior3_9.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::bior3_9";
    using inverse = bior3_9_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct bior3_9::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0xbeaaaaabU>;
    static_assert(type::k == 1U);
};

template <>
struct bior3_9::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf900000U, 0xbec00000U>;
    static_assert(type::k == 2U);
};

template <>
struct bior3_9::step<2> {
    using type = StaticStep<StepType::kPredict, -4, 0xba600000U, 0x3c1238e4U, 0xbd365555U, 0x3e189555U, 0x3ee38e39U, 0xbe189555U, 0x3d365555U, 0xbc1238e4U, 0x3a600000U>;
    static_assert(type::k == 9U);
};

template <>
struct bior3_9::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct bior3_9::step<4> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x4007c3b6U>;
    static_assert(type::k == 1U);
};

template <>
struct bior3_9::step<5> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xbef15befU>;
    static_assert(type::k == 1U);
};

struct bior3_9_inverse {
    static constexpr const char* name = "bior3.9-inverse";
    static constexpr uint32_t tap_size = 20U;
    static constexpr uint32_t num_steps = 6U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/bior3_9.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::bior3_9_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct bior3_9_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xc007c3b6U>;
    static_assert(type::k == 1U);
};

template <>
struct bior3_9_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3ef15bf0U>;
    static_assert(type::k == 1U);
};

template <>
struct bior3_9_inverse::step<2> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct bior3_9_inverse::step<3> {
    using type = StaticStep<StepType::kPredict, -4, 0x3a600000U, 0xbc1238e4U, 0x3d365555U, 0xbe189555U, 0xbee38e39U, 0x3e189555U, 0xbd365555U, 0x3c1238e4U, 0xba600000U>;
    static_assert(type::k == 9U);
};

template <>
struct bior3_9_inverse::step<4> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f900000U, 0x3ec00000U>;
    static_assert(type::k == 2U);
};

template <>
struct bior3_9_inverse::step<5> {
    using type = StaticStep<StepType::kPredict, -1, 0x3eaaaaabU>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
