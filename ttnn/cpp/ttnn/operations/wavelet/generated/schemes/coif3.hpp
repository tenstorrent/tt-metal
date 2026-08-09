// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct coif3_inverse;

struct coif3 {
    static constexpr const char* name = "coif3";
    static constexpr uint32_t tap_size = 18U;
    static constexpr int32_t delay_even = 4;
    static constexpr int32_t delay_odd = 5;
    static constexpr uint32_t num_steps = 13U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/coif3.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::coif3";
    using inverse = coif3_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct coif3::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0xc0034ca9U>;
    static_assert(type::k == 1U);
};

template <>
struct coif3::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ec9a7b1U, 0x3e5bfed9U>;
    static_assert(type::k == 2U);
};

template <>
struct coif3::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xc025a27dU, 0x3f183278U>;
    static_assert(type::k == 2U);
};

template <>
struct coif3::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbdd589f7U, 0xbd565e5cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif3::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ea037cbU, 0xbf88ed47U>;
    static_assert(type::k == 2U);
};

template <>
struct coif3::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e1beb05U, 0xbf43c563U>;
    static_assert(type::k == 2U);
};

template <>
struct coif3::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f8b0c78U, 0xc082d55bU>;
    static_assert(type::k == 2U);
};

template <>
struct coif3::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e76e790U, 0xbede9ff9U>;
    static_assert(type::k == 2U);
};

template <>
struct coif3::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x40128a88U, 0xc18590d4U>;
    static_assert(type::k == 2U);
};

template <>
struct coif3::step<9> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct coif3::step<10> {
    using type = StaticStep<StepType::kPredict, 0, 0x3d754fdeU>;
    static_assert(type::k == 1U);
};

template <>
struct coif3::step<11> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xbcbf7691U>;
    static_assert(type::k == 1U);
};

template <>
struct coif3::step<12> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x422b252cU>;
    static_assert(type::k == 1U);
};

struct coif3_inverse {
    static constexpr const char* name = "coif3-inverse";
    static constexpr uint32_t tap_size = 18U;
    static constexpr uint32_t num_steps = 13U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/coif3.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::coif3_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct coif3_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3cbf7691U>;
    static_assert(type::k == 1U);
};

template <>
struct coif3_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xc22b252cU>;
    static_assert(type::k == 1U);
};

template <>
struct coif3_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0xbd754fdeU>;
    static_assert(type::k == 1U);
};

template <>
struct coif3_inverse::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct coif3_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0128a88U, 0x418590d4U>;
    static_assert(type::k == 2U);
};

template <>
struct coif3_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe76e790U, 0x3ede9ff9U>;
    static_assert(type::k == 2U);
};

template <>
struct coif3_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf8b0c78U, 0x4082d55bU>;
    static_assert(type::k == 2U);
};

template <>
struct coif3_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe1beb05U, 0x3f43c563U>;
    static_assert(type::k == 2U);
};

template <>
struct coif3_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xbea037cbU, 0x3f88ed47U>;
    static_assert(type::k == 2U);
};

template <>
struct coif3_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3dd589f7U, 0x3d565e5cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif3_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x4025a27dU, 0xbf183278U>;
    static_assert(type::k == 2U);
};

template <>
struct coif3_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbec9a7b1U, 0xbe5bfed9U>;
    static_assert(type::k == 2U);
};

template <>
struct coif3_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, 0, 0x40034ca9U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
