// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct db17_inverse;

struct db17 {
    static constexpr const char* name = "db17";
    static constexpr uint32_t tap_size = 34U;
    static constexpr int32_t delay_even = 8;
    static constexpr int32_t delay_odd = 9;
    static constexpr uint32_t num_steps = 21U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db17.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db17";
    using inverse = db17_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct db17::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0x413975daU>;
    static_assert(type::k == 1U);
};

template <>
struct db17::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbdaa0352U, 0xbaf424f1U>;
    static_assert(type::k == 2U);
};

template <>
struct db17::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0x425dd5d8U, 0xc2587367U>;
    static_assert(type::k == 2U);
};

template <>
struct db17::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3a6f9ae5U, 0xbb916e16U>;
    static_assert(type::k == 2U);
};

template <>
struct db17::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x420250a5U, 0xc2eec72dU>;
    static_assert(type::k == 2U);
};

template <>
struct db17::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3b777363U, 0xbbbe4b98U>;
    static_assert(type::k == 2U);
};

template <>
struct db17::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x42dadf42U, 0xc31a08e5U>;
    static_assert(type::k == 2U);
};

template <>
struct db17::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ba4ccdaU, 0xbbefec3cU>;
    static_assert(type::k == 2U);
};

template <>
struct db17::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x42f03079U, 0xc33aea89U>;
    static_assert(type::k == 2U);
};

template <>
struct db17::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ba5e898U, 0xbc0ebac0U>;
    static_assert(type::k == 2U);
};

template <>
struct db17::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x42e14512U, 0xc35e6872U>;
    static_assert(type::k == 2U);
};

template <>
struct db17::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3b929cf9U, 0xbc2cd8ddU>;
    static_assert(type::k == 2U);
};

template <>
struct db17::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x42bd6923U, 0xc38b084aU>;
    static_assert(type::k == 2U);
};

template <>
struct db17::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3b6ba977U, 0xbc62345bU>;
    static_assert(type::k == 2U);
};

template <>
struct db17::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x4290dbebU, 0xc3c2b3a7U>;
    static_assert(type::k == 2U);
};

template <>
struct db17::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3b284c62U, 0xbcb26771U>;
    static_assert(type::k == 2U);
};

template <>
struct db17::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x4237ac46U, 0xc44cb794U>;
    static_assert(type::k == 2U);
};

template <>
struct db17::step<17> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct db17::step<18> {
    using type = StaticStep<StepType::kPredict, 0, 0x3aa01096U>;
    static_assert(type::k == 1U);
};

template <>
struct db17::step<19> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3883f3b8U>;
    static_assert(type::k == 1U);
};

template <>
struct db17::step<20> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xc678552bU>;
    static_assert(type::k == 1U);
};

struct db17_inverse {
    static constexpr const char* name = "db17-inverse";
    static constexpr uint32_t tap_size = 34U;
    static constexpr uint32_t num_steps = 21U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db17.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db17_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct db17_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xb883f3b8U>;
    static_assert(type::k == 1U);
};

template <>
struct db17_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x4678552aU>;
    static_assert(type::k == 1U);
};

template <>
struct db17_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0xbaa01096U>;
    static_assert(type::k == 1U);
};

template <>
struct db17_inverse::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct db17_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xc237ac46U, 0x444cb794U>;
    static_assert(type::k == 2U);
};

template <>
struct db17_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbb284c62U, 0x3cb26771U>;
    static_assert(type::k == 2U);
};

template <>
struct db17_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xc290dbebU, 0x43c2b3a7U>;
    static_assert(type::k == 2U);
};

template <>
struct db17_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbb6ba977U, 0x3c62345bU>;
    static_assert(type::k == 2U);
};

template <>
struct db17_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xc2bd6923U, 0x438b084aU>;
    static_assert(type::k == 2U);
};

template <>
struct db17_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbb929cf9U, 0x3c2cd8ddU>;
    static_assert(type::k == 2U);
};

template <>
struct db17_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xc2e14512U, 0x435e6872U>;
    static_assert(type::k == 2U);
};

template <>
struct db17_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbba5e898U, 0x3c0ebac0U>;
    static_assert(type::k == 2U);
};

template <>
struct db17_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xc2f03079U, 0x433aea89U>;
    static_assert(type::k == 2U);
};

template <>
struct db17_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbba4ccdaU, 0x3befec3cU>;
    static_assert(type::k == 2U);
};

template <>
struct db17_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xc2dadf42U, 0x431a08e5U>;
    static_assert(type::k == 2U);
};

template <>
struct db17_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbb777363U, 0x3bbe4b98U>;
    static_assert(type::k == 2U);
};

template <>
struct db17_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xc20250a5U, 0x42eec72dU>;
    static_assert(type::k == 2U);
};

template <>
struct db17_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xba6f9ae5U, 0x3b916e16U>;
    static_assert(type::k == 2U);
};

template <>
struct db17_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xc25dd5d8U, 0x42587367U>;
    static_assert(type::k == 2U);
};

template <>
struct db17_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3daa0352U, 0x3af424f1U>;
    static_assert(type::k == 2U);
};

template <>
struct db17_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, 0, 0xc13975daU>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
