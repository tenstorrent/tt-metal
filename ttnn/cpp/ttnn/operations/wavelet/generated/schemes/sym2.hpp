// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct sym2_inverse;

struct sym2 {
    static constexpr const char* name = "sym2";
    static constexpr uint32_t tap_size = 4U;
    static constexpr int32_t delay_even = 1;
    static constexpr int32_t delay_odd = 1;
    static constexpr uint32_t num_steps = 5U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/sym2.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::sym2";
    using inverse = sym2_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct sym2::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf13cd3aU>;
    static_assert(type::k == 1U);
};

template <>
struct sym2::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e4dc8f4U, 0x3eddb3d7U>;
    static_assert(type::k == 2U);
};

template <>
struct sym2::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0xbeaaaaabU>;
    static_assert(type::k == 1U);
};

template <>
struct sym2::step<3> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3f8ec3f4U>;
    static_assert(type::k == 1U);
};

template <>
struct sym2::step<4> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3f6585f8U>;
    static_assert(type::k == 1U);
};

struct sym2_inverse {
    static constexpr const char* name = "sym2-inverse";
    static constexpr uint32_t tap_size = 4U;
    static constexpr uint32_t num_steps = 5U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/sym2.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::sym2_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct sym2_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3f8ec3f5U>;
    static_assert(type::k == 1U);
};

template <>
struct sym2_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3f6585f9U>;
    static_assert(type::k == 1U);
};

template <>
struct sym2_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0x3eaaaaabU>;
    static_assert(type::k == 1U);
};

template <>
struct sym2_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe4dc8f4U, 0xbeddb3d7U>;
    static_assert(type::k == 2U);
};

template <>
struct sym2_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f13cd3aU>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
