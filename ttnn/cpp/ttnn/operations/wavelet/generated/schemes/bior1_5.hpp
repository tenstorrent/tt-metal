// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct bior1_5_inverse;

struct bior1_5 {
    static constexpr const char* name = "bior1.5";
    static constexpr uint32_t tap_size = 10U;
    static constexpr int32_t delay_even = 2;
    static constexpr int32_t delay_odd = 3;
    static constexpr uint32_t num_steps = 4U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/bior1_5.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::bior1_5";
    using inverse = bior1_5_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct bior1_5::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0xbf800000U>;
    static_assert(type::k == 1U);
};

template <>
struct bior1_5::step<1> {
    using type = StaticStep<StepType::kUpdate, -2, 0xbc400000U, 0x3db00000U, 0x3f000000U, 0xbdb00000U, 0x3c400000U>;
    static_assert(type::k == 5U);
};

template <>
struct bior1_5::step<2> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3fb504f3U>;
    static_assert(type::k == 1U);
};

template <>
struct bior1_5::step<3> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3f3504f3U>;
    static_assert(type::k == 1U);
};

struct bior1_5_inverse {
    static constexpr const char* name = "bior1.5-inverse";
    static constexpr uint32_t tap_size = 10U;
    static constexpr uint32_t num_steps = 4U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/bior1_5.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::bior1_5_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct bior1_5_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3fb504f3U>;
    static_assert(type::k == 1U);
};

template <>
struct bior1_5_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3f3504f3U>;
    static_assert(type::k == 1U);
};

template <>
struct bior1_5_inverse::step<2> {
    using type = StaticStep<StepType::kUpdate, -2, 0x3c400000U, 0xbdb00000U, 0xbf000000U, 0x3db00000U, 0xbc400000U>;
    static_assert(type::k == 5U);
};

template <>
struct bior1_5_inverse::step<3> {
    using type = StaticStep<StepType::kPredict, 0, 0x3f800000U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
