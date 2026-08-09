// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct coif2_inverse;

struct coif2 {
    static constexpr const char* name = "coif2";
    static constexpr uint32_t tap_size = 12U;
    static constexpr int32_t delay_even = 3;
    static constexpr int32_t delay_odd = 3;
    static constexpr uint32_t num_steps = 9U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/coif2.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::coif2";
    using inverse = coif2_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct coif2::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0x3eca58e6U>;
    static_assert(type::k == 1U);
};

template <>
struct coif2::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ef91d7eU, 0xbeaf0315U>;
    static_assert(type::k == 2U);
};

template <>
struct coif2::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xbdd1a039U, 0xbefcf5acU>;
    static_assert(type::k == 2U);
};

template <>
struct coif2::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfbd67c0U, 0x3e061068U>;
    static_assert(type::k == 2U);
};

template <>
struct coif2::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0011aadU, 0x3edb80aaU>;
    static_assert(type::k == 2U);
};

template <>
struct coif2::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3b60a05aU, 0x3ef75f02U>;
    static_assert(type::k == 2U);
};

template <>
struct coif2::step<6> {
    using type = StaticStep<StepType::kPredict, 0, 0xbf101f4eU>;
    static_assert(type::k == 1U);
};

template <>
struct coif2::step<7> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x406501acU>;
    static_assert(type::k == 1U);
};

template <>
struct coif2::step<8> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3e8f166fU>;
    static_assert(type::k == 1U);
};

struct coif2_inverse {
    static constexpr const char* name = "coif2-inverse";
    static constexpr uint32_t tap_size = 12U;
    static constexpr uint32_t num_steps = 9U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/coif2.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::coif2_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct coif2_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x406501abU>;
    static_assert(type::k == 1U);
};

template <>
struct coif2_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3e8f166eU>;
    static_assert(type::k == 1U);
};

template <>
struct coif2_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0x3f101f4eU>;
    static_assert(type::k == 1U);
};

template <>
struct coif2_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbb60a05aU, 0xbef75f02U>;
    static_assert(type::k == 2U);
};

template <>
struct coif2_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x40011aadU, 0xbedb80aaU>;
    static_assert(type::k == 2U);
};

template <>
struct coif2_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fbd67c0U, 0xbe061068U>;
    static_assert(type::k == 2U);
};

template <>
struct coif2_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x3dd1a039U, 0x3efcf5acU>;
    static_assert(type::k == 2U);
};

template <>
struct coif2_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbef91d7eU, 0x3eaf0315U>;
    static_assert(type::k == 2U);
};

template <>
struct coif2_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xbeca58e6U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
