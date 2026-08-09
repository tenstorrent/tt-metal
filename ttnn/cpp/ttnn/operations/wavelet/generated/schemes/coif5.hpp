// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct coif5_inverse;

struct coif5 {
    static constexpr const char* name = "coif5";
    static constexpr uint32_t tap_size = 30U;
    static constexpr int32_t delay_even = 7;
    static constexpr int32_t delay_odd = 8;
    static constexpr uint32_t num_steps = 19U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/coif5.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::coif5";
    using inverse = coif5_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct coif5::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0xbfd86a8cU>;
    static_assert(type::k == 1U);
};

template <>
struct coif5::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ee05830U, 0x3ee3f342U>;
    static_assert(type::k == 2U);
};

template <>
struct coif5::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xbfd6bd56U, 0x3f956097U>;
    static_assert(type::k == 2U);
};

template <>
struct coif5::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf0104ffU, 0x3e2f2e76U>;
    static_assert(type::k == 2U);
};

template <>
struct coif5::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf53715aU, 0x3e893f4eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif5::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd822a44U, 0xbd6e5945U>;
    static_assert(type::k == 2U);
};

template <>
struct coif5::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e7c02f5U, 0xbf3fe30eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif5::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e2240c4U, 0xbe4eca6cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif5::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f63ad32U, 0xc00dd825U>;
    static_assert(type::k == 2U);
};

template <>
struct coif5::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e851b85U, 0xbf6a07e3U>;
    static_assert(type::k == 2U);
};

template <>
struct coif5::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f842790U, 0xc07ba8b6U>;
    static_assert(type::k == 2U);
};

template <>
struct coif5::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e81ad17U, 0xbe40ad1eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif5::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x40a8d036U, 0xc188ba58U>;
    static_assert(type::k == 2U);
};

template <>
struct coif5::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d6f7d2aU, 0xbe9d2b7fU>;
    static_assert(type::k == 2U);
};

template <>
struct coif5::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x40507b7eU, 0xc211d243U>;
    static_assert(type::k == 2U);
};

template <>
struct coif5::step<15> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct coif5::step<16> {
    using type = StaticStep<StepType::kPredict, 0, 0x3ce0b69bU>;
    static_assert(type::k == 1U);
};

template <>
struct coif5::step<17> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xbb66fad0U>;
    static_assert(type::k == 1U);
};

template <>
struct coif5::step<18> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x438ddd81U>;
    static_assert(type::k == 1U);
};

struct coif5_inverse {
    static constexpr const char* name = "coif5-inverse";
    static constexpr uint32_t tap_size = 30U;
    static constexpr uint32_t num_steps = 19U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/coif5.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::coif5_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct coif5_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3b66fad1U>;
    static_assert(type::k == 1U);
};

template <>
struct coif5_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xc38ddd82U>;
    static_assert(type::k == 1U);
};

template <>
struct coif5_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0xbce0b69bU>;
    static_assert(type::k == 1U);
};

template <>
struct coif5_inverse::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct coif5_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0507b7eU, 0x4211d243U>;
    static_assert(type::k == 2U);
};

template <>
struct coif5_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd6f7d2aU, 0x3e9d2b7fU>;
    static_assert(type::k == 2U);
};

template <>
struct coif5_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0a8d036U, 0x4188ba58U>;
    static_assert(type::k == 2U);
};

template <>
struct coif5_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe81ad17U, 0x3e40ad1eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif5_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf842790U, 0x407ba8b6U>;
    static_assert(type::k == 2U);
};

template <>
struct coif5_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe851b85U, 0x3f6a07e3U>;
    static_assert(type::k == 2U);
};

template <>
struct coif5_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf63ad32U, 0x400dd825U>;
    static_assert(type::k == 2U);
};

template <>
struct coif5_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe2240c4U, 0x3e4eca6cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif5_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe7c02f5U, 0x3f3fe30eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif5_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d822a44U, 0x3d6e5945U>;
    static_assert(type::k == 2U);
};

template <>
struct coif5_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f53715aU, 0xbe893f4eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif5_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f0104ffU, 0xbe2f2e76U>;
    static_assert(type::k == 2U);
};

template <>
struct coif5_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x3fd6bd56U, 0xbf956097U>;
    static_assert(type::k == 2U);
};

template <>
struct coif5_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbee05830U, 0xbee3f342U>;
    static_assert(type::k == 2U);
};

template <>
struct coif5_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, 0, 0x3fd86a8cU>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
