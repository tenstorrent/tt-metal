// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct bior1_3_inverse;

struct bior1_3 {
    static constexpr const char* name = "bior1.3";
    static constexpr uint32_t tap_size = 6U;
    static constexpr int32_t delay_even = 1;
    static constexpr int32_t delay_odd = 2;
    static constexpr uint32_t num_steps = 4U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/bior1_3.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::bior1_3";
    using inverse = bior1_3_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct bior1_3::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0xbf800000U>;
    static_assert(type::k == 1U);
};

template <>
struct bior1_3::step<1> {
    using type = StaticStep<StepType::kUpdate, -1, 0x3d800000U, 0x3f000000U, 0xbd800000U>;
    static_assert(type::k == 3U);
};

template <>
struct bior1_3::step<2> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3fb504f3U>;
    static_assert(type::k == 1U);
};

template <>
struct bior1_3::step<3> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3f3504f3U>;
    static_assert(type::k == 1U);
};

struct bior1_3_inverse {
    static constexpr const char* name = "bior1.3-inverse";
    static constexpr uint32_t tap_size = 6U;
    static constexpr uint32_t num_steps = 4U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/bior1_3.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::bior1_3_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct bior1_3_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3fb504f3U>;
    static_assert(type::k == 1U);
};

template <>
struct bior1_3_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3f3504f3U>;
    static_assert(type::k == 1U);
};

template <>
struct bior1_3_inverse::step<2> {
    using type = StaticStep<StepType::kUpdate, -1, 0xbd800000U, 0xbf000000U, 0x3d800000U>;
    static_assert(type::k == 3U);
};

template <>
struct bior1_3_inverse::step<3> {
    using type = StaticStep<StepType::kPredict, 0, 0x3f800000U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
