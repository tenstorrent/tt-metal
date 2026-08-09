// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct sym4_inverse;

struct sym4 {
    static constexpr const char* name = "sym4";
    static constexpr uint32_t tap_size = 8U;
    static constexpr int32_t delay_even = 2;
    static constexpr int32_t delay_odd = 2;
    static constexpr uint32_t num_steps = 7U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/sym4.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::sym4";
    using inverse = sym4_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct sym4::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0x40239f12U>;
    static_assert(type::k == 1U);
};

template <>
struct sym4::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c9be755U, 0xbeadb163U>;
    static_assert(type::k == 2U);
};

template <>
struct sym4::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0bc0fcbU, 0xbf878f30U>;
    static_assert(type::k == 2U);
};

template <>
struct sym4::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e479e94U, 0x3d8722f0U>;
    static_assert(type::k == 2U);
};

template <>
struct sym4::step<4> {
    using type = StaticStep<StepType::kPredict, 0, 0xc08b0231U>;
    static_assert(type::k == 1U);
};

template <>
struct sym4::step<5> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3fadfb43U>;
    static_assert(type::k == 1U);
};

template <>
struct sym4::step<6> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3f3c5785U>;
    static_assert(type::k == 1U);
};

struct sym4_inverse {
    static constexpr const char* name = "sym4-inverse";
    static constexpr uint32_t tap_size = 8U;
    static constexpr uint32_t num_steps = 7U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/sym4.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::sym4_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct sym4_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3fadfb43U>;
    static_assert(type::k == 1U);
};

template <>
struct sym4_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3f3c5785U>;
    static_assert(type::k == 1U);
};

template <>
struct sym4_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0x408b0231U>;
    static_assert(type::k == 1U);
};

template <>
struct sym4_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe479e94U, 0xbd8722f0U>;
    static_assert(type::k == 2U);
};

template <>
struct sym4_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x40bc0fcbU, 0x3f878f30U>;
    static_assert(type::k == 2U);
};

template <>
struct sym4_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc9be755U, 0x3eadb163U>;
    static_assert(type::k == 2U);
};

template <>
struct sym4_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0239f12U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
