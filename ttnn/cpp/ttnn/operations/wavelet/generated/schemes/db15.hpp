// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct db15_inverse;

struct db15 {
    static constexpr const char* name = "db15";
    static constexpr uint32_t tap_size = 30U;
    static constexpr int32_t delay_even = 7;
    static constexpr int32_t delay_odd = 8;
    static constexpr uint32_t num_steps = 19U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db15.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db15";
    using inverse = db15_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct db15::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0x4124c99bU>;
    static_assert(type::k == 1U);
};

template <>
struct db15::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbdc508a8U, 0xbb36dcbaU>;
    static_assert(type::k == 2U);
};

template <>
struct db15::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0x41ee65beU, 0xc24cfc4fU>;
    static_assert(type::k == 2U);
};

template <>
struct db15::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3b89a246U, 0xbbcf871aU>;
    static_assert(type::k == 2U);
};

template <>
struct db15::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x42736caeU, 0xc2afd218U>;
    static_assert(type::k == 2U);
};

template <>
struct db15::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3bd35730U, 0xbc18938bU>;
    static_assert(type::k == 2U);
};

template <>
struct db15::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x429d2fb4U, 0xc2eaebd4U>;
    static_assert(type::k == 2U);
};

template <>
struct db15::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3befaa91U, 0xbc403658U>;
    static_assert(type::k == 2U);
};

template <>
struct db15::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x42a0434aU, 0xc30fc773U>;
    static_assert(type::k == 2U);
};

template <>
struct db15::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3bdf7b07U, 0xbc6b028bU>;
    static_assert(type::k == 2U);
};

template <>
struct db15::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x428acfa4U, 0xc333f5bbU>;
    static_assert(type::k == 2U);
};

template <>
struct db15::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3bb5f6c1U, 0xbc99cdb4U>;
    static_assert(type::k == 2U);
};

template <>
struct db15::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x425509e3U, 0xc37c8b23U>;
    static_assert(type::k == 2U);
};

template <>
struct db15::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3b81c067U, 0xbcf389f3U>;
    static_assert(type::k == 2U);
};

template <>
struct db15::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x42068ca3U, 0xc405656fU>;
    static_assert(type::k == 2U);
};

template <>
struct db15::step<15> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct db15::step<16> {
    using type = StaticStep<StepType::kPredict, 0, 0x3af5a4e5U>;
    static_assert(type::k == 1U);
};

template <>
struct db15::step<17> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3926dfd1U>;
    static_assert(type::k == 1U);
};

template <>
struct db15::step<18> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xc5c45d07U>;
    static_assert(type::k == 1U);
};

struct db15_inverse {
    static constexpr const char* name = "db15-inverse";
    static constexpr uint32_t tap_size = 30U;
    static constexpr uint32_t num_steps = 19U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db15.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db15_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct db15_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xb926dfd1U>;
    static_assert(type::k == 1U);
};

template <>
struct db15_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x45c45d07U>;
    static_assert(type::k == 1U);
};

template <>
struct db15_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0xbaf5a4e5U>;
    static_assert(type::k == 1U);
};

template <>
struct db15_inverse::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct db15_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xc2068ca3U, 0x4405656fU>;
    static_assert(type::k == 2U);
};

template <>
struct db15_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbb81c067U, 0x3cf389f3U>;
    static_assert(type::k == 2U);
};

template <>
struct db15_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xc25509e3U, 0x437c8b23U>;
    static_assert(type::k == 2U);
};

template <>
struct db15_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbbb5f6c1U, 0x3c99cdb4U>;
    static_assert(type::k == 2U);
};

template <>
struct db15_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xc28acfa4U, 0x4333f5bbU>;
    static_assert(type::k == 2U);
};

template <>
struct db15_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbbdf7b07U, 0x3c6b028bU>;
    static_assert(type::k == 2U);
};

template <>
struct db15_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xc2a0434aU, 0x430fc773U>;
    static_assert(type::k == 2U);
};

template <>
struct db15_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbbefaa91U, 0x3c403658U>;
    static_assert(type::k == 2U);
};

template <>
struct db15_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xc29d2fb4U, 0x42eaebd4U>;
    static_assert(type::k == 2U);
};

template <>
struct db15_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbbd35730U, 0x3c18938bU>;
    static_assert(type::k == 2U);
};

template <>
struct db15_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xc2736caeU, 0x42afd218U>;
    static_assert(type::k == 2U);
};

template <>
struct db15_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbb89a246U, 0x3bcf871aU>;
    static_assert(type::k == 2U);
};

template <>
struct db15_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xc1ee65beU, 0x424cfc4fU>;
    static_assert(type::k == 2U);
};

template <>
struct db15_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3dc508a8U, 0x3b36dcbaU>;
    static_assert(type::k == 2U);
};

template <>
struct db15_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, 0, 0xc124c99bU>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
