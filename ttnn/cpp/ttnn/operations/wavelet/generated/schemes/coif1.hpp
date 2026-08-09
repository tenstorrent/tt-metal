// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct coif1_inverse;

struct coif1 {
    static constexpr const char* name = "coif1";
    static constexpr uint32_t tap_size = 6U;
    static constexpr int32_t delay_even = 1;
    static constexpr int32_t delay_odd = 2;
    static constexpr uint32_t num_steps = 7U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/coif1.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::coif1";
    using inverse = coif1_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct coif1::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0xc094a9ffU>;
    static_assert(type::k == 1U);
};

template <>
struct coif1::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e52a7fbU, 0xbc891b5bU>;
    static_assert(type::k == 2U);
};

template <>
struct coif1::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0x40eefefeU, 0xc2b799efU>;
    static_assert(type::k == 2U);
};

template <>
struct coif1::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct coif1::step<4> {
    using type = StaticStep<StepType::kPredict, 0, 0x3c2a9218U>;
    static_assert(type::k == 1U);
};

template <>
struct coif1::step<5> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xbd41e5fcU>;
    static_assert(type::k == 1U);
};

template <>
struct coif1::step<6> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x41a8fee9U>;
    static_assert(type::k == 1U);
};

struct coif1_inverse {
    static constexpr const char* name = "coif1-inverse";
    static constexpr uint32_t tap_size = 6U;
    static constexpr uint32_t num_steps = 7U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/coif1.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::coif1_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct coif1_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3d41e5fcU>;
    static_assert(type::k == 1U);
};

template <>
struct coif1_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xc1a8fee9U>;
    static_assert(type::k == 1U);
};

template <>
struct coif1_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0xbc2a9218U>;
    static_assert(type::k == 1U);
};

template <>
struct coif1_inverse::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct coif1_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0eefefeU, 0x42b799efU>;
    static_assert(type::k == 2U);
};

template <>
struct coif1_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe52a7fbU, 0x3c891b5bU>;
    static_assert(type::k == 2U);
};

template <>
struct coif1_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, 0, 0x4094a9ffU>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
