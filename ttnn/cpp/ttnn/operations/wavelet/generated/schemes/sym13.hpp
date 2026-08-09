// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct sym13_inverse;

struct sym13 {
    static constexpr const char* name = "sym13";
    static constexpr uint32_t tap_size = 26U;
    static constexpr int32_t delay_even = 6;
    static constexpr int32_t delay_odd = 7;
    static constexpr uint32_t num_steps = 17U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/sym13.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::sym13";
    using inverse = sym13_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct sym13::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0x3f0624faU>;
    static_assert(type::k == 1U);
};

template <>
struct sym13::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbed27e0bU, 0x3db64ae9U>;
    static_assert(type::k == 2U);
};

template <>
struct sym13::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe1230a3U, 0x44075315U>;
    static_assert(type::k == 2U);
};

template <>
struct sym13::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbaf2247bU, 0xb60a3bfeU>;
    static_assert(type::k == 2U);
};

template <>
struct sym13::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x47b153a5U, 0x48a7bbc9U>;
    static_assert(type::k == 2U);
};

template <>
struct sym13::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xb615da57U, 0xb4896427U>;
    static_assert(type::k == 2U);
};

template <>
struct sym13::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x481e3639U, 0xc989a83fU>;
    static_assert(type::k == 2U);
};

template <>
struct sym13::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x35212530U, 0x3432fe7aU>;
    static_assert(type::k == 2U);
};

template <>
struct sym13::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xc94bbe77U, 0xca27bbaeU>;
    static_assert(type::k == 2U);
};

template <>
struct sym13::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3478c29aU, 0x33c5f49bU>;
    static_assert(type::k == 2U);
};

template <>
struct sym13::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xca0fcbfbU, 0x49dca552U>;
    static_assert(type::k == 2U);
};

template <>
struct sym13::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xb3a6c627U, 0xb380bdc3U>;
    static_assert(type::k == 2U);
};

template <>
struct sym13::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x49b491f8U, 0xca95702aU>;
    static_assert(type::k == 2U);
};

template <>
struct sym13::step<13> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct sym13::step<14> {
    using type = StaticStep<StepType::kPredict, 0, 0x33e25105U>;
    static_assert(type::k == 1U);
};

template <>
struct sym13::step<15> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x39a77d46U>;
    static_assert(type::k == 1U);
};

template <>
struct sym13::step<16> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xc543a46dU>;
    static_assert(type::k == 1U);
};

struct sym13_inverse {
    static constexpr const char* name = "sym13-inverse";
    static constexpr uint32_t tap_size = 26U;
    static constexpr uint32_t num_steps = 17U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/sym13.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::sym13_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct sym13_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xb9a77d46U>;
    static_assert(type::k == 1U);
};

template <>
struct sym13_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x4543a46dU>;
    static_assert(type::k == 1U);
};

template <>
struct sym13_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0xb3e25105U>;
    static_assert(type::k == 1U);
};

template <>
struct sym13_inverse::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct sym13_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xc9b491f8U, 0x4a95702aU>;
    static_assert(type::k == 2U);
};

template <>
struct sym13_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x33a6c627U, 0x3380bdc3U>;
    static_assert(type::k == 2U);
};

template <>
struct sym13_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x4a0fcbfbU, 0xc9dca552U>;
    static_assert(type::k == 2U);
};

template <>
struct sym13_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xb478c29aU, 0xb3c5f49bU>;
    static_assert(type::k == 2U);
};

template <>
struct sym13_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x494bbe77U, 0x4a27bbaeU>;
    static_assert(type::k == 2U);
};

template <>
struct sym13_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xb5212530U, 0xb432fe7aU>;
    static_assert(type::k == 2U);
};

template <>
struct sym13_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xc81e3639U, 0x4989a83fU>;
    static_assert(type::k == 2U);
};

template <>
struct sym13_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3615da57U, 0x34896427U>;
    static_assert(type::k == 2U);
};

template <>
struct sym13_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xc7b153a5U, 0xc8a7bbc9U>;
    static_assert(type::k == 2U);
};

template <>
struct sym13_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3af2247bU, 0x360a3bfeU>;
    static_assert(type::k == 2U);
};

template <>
struct sym13_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e1230a3U, 0xc4075315U>;
    static_assert(type::k == 2U);
};

template <>
struct sym13_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ed27e0bU, 0xbdb64ae9U>;
    static_assert(type::k == 2U);
};

template <>
struct sym13_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, 0, 0xbf0624faU>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
