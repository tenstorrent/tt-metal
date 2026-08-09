// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct db19_inverse;

struct db19 {
    static constexpr const char* name = "db19";
    static constexpr uint32_t tap_size = 38U;
    static constexpr int32_t delay_even = 9;
    static constexpr int32_t delay_odd = 10;
    static constexpr uint32_t num_steps = 23U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db19.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db19";
    using inverse = db19_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct db19::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0x414e1930U>;
    static_assert(type::k == 1U);
};

template <>
struct db19::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd99ed2dU, 0xbab301c0U>;
    static_assert(type::k == 2U);
};

template <>
struct db19::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0x42bd6a3cU, 0xc2727f5dU>;
    static_assert(type::k == 2U);
};

template <>
struct db19::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3b8bb852U, 0xbb41ed5dU>;
    static_assert(type::k == 2U);
};

template <>
struct db19::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x439e08b3U, 0xc2c5573bU>;
    static_assert(type::k == 2U);
};

template <>
struct db19::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xb817222aU, 0xbb1ba604U>;
    static_assert(type::k == 2U);
};

template <>
struct db19::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xc3986842U, 0xc4a33735U>;
    static_assert(type::k == 2U);
};

template <>
struct db19::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3a0f7aafU, 0xba2f6275U>;
    static_assert(type::k == 2U);
};

template <>
struct db19::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x4496e2f5U, 0xc4d6789bU>;
    static_assert(type::k == 2U);
};

template <>
struct db19::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3a08b589U, 0xba4f88deU>;
    static_assert(type::k == 2U);
};

template <>
struct db19::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x44963a09U, 0xc4f9b78bU>;
    static_assert(type::k == 2U);
};

template <>
struct db19::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3a00d768U, 0xba71d59cU>;
    static_assert(type::k == 2U);
};

template <>
struct db19::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x4486ca8cU, 0xc5138879U>;
    static_assert(type::k == 2U);
};

template <>
struct db19::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x39dddd09U, 0xba925e42U>;
    static_assert(type::k == 2U);
};

template <>
struct db19::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x445fd6b0U, 0xc5388995U>;
    static_assert(type::k == 2U);
};

template <>
struct db19::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x39b190d5U, 0xbabf565bU>;
    static_assert(type::k == 2U);
};

template <>
struct db19::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x442b41f3U, 0xc580db30U>;
    static_assert(type::k == 2U);
};

template <>
struct db19::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x397e4c8aU, 0xbb16577cU>;
    static_assert(type::k == 2U);
};

template <>
struct db19::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0x43d9f4efU, 0xc606ecb5U>;
    static_assert(type::k == 2U);
};

template <>
struct db19::step<19> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct db19::step<20> {
    using type = StaticStep<StepType::kPredict, 0, 0x38f2dc8bU>;
    static_assert(type::k == 1U);
};

template <>
struct db19::step<21> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x371fa136U>;
    static_assert(type::k == 1U);
};

template <>
struct db19::step<22> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xc7cd4669U>;
    static_assert(type::k == 1U);
};

struct db19_inverse {
    static constexpr const char* name = "db19-inverse";
    static constexpr uint32_t tap_size = 38U;
    static constexpr uint32_t num_steps = 23U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db19.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db19_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct db19_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xb71fa136U>;
    static_assert(type::k == 1U);
};

template <>
struct db19_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x47cd4669U>;
    static_assert(type::k == 1U);
};

template <>
struct db19_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0xb8f2dc8bU>;
    static_assert(type::k == 1U);
};

template <>
struct db19_inverse::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct db19_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xc3d9f4efU, 0x4606ecb5U>;
    static_assert(type::k == 2U);
};

template <>
struct db19_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xb97e4c8aU, 0x3b16577cU>;
    static_assert(type::k == 2U);
};

template <>
struct db19_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xc42b41f3U, 0x4580db30U>;
    static_assert(type::k == 2U);
};

template <>
struct db19_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xb9b190d5U, 0x3abf565bU>;
    static_assert(type::k == 2U);
};

template <>
struct db19_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xc45fd6b0U, 0x45388995U>;
    static_assert(type::k == 2U);
};

template <>
struct db19_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xb9dddd09U, 0x3a925e42U>;
    static_assert(type::k == 2U);
};

template <>
struct db19_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xc486ca8cU, 0x45138879U>;
    static_assert(type::k == 2U);
};

template <>
struct db19_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xba00d768U, 0x3a71d59cU>;
    static_assert(type::k == 2U);
};

template <>
struct db19_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xc4963a09U, 0x44f9b78bU>;
    static_assert(type::k == 2U);
};

template <>
struct db19_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xba08b589U, 0x3a4f88deU>;
    static_assert(type::k == 2U);
};

template <>
struct db19_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xc496e2f5U, 0x44d6789bU>;
    static_assert(type::k == 2U);
};

template <>
struct db19_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0xba0f7aafU, 0x3a2f6275U>;
    static_assert(type::k == 2U);
};

template <>
struct db19_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x43986842U, 0x44a33735U>;
    static_assert(type::k == 2U);
};

template <>
struct db19_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3817222aU, 0x3b1ba604U>;
    static_assert(type::k == 2U);
};

template <>
struct db19_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xc39e08b3U, 0x42c5573bU>;
    static_assert(type::k == 2U);
};

template <>
struct db19_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbb8bb852U, 0x3b41ed5dU>;
    static_assert(type::k == 2U);
};

template <>
struct db19_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0xc2bd6a3cU, 0x42727f5dU>;
    static_assert(type::k == 2U);
};

template <>
struct db19_inverse::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d99ed2dU, 0x3ab301c0U>;
    static_assert(type::k == 2U);
};

template <>
struct db19_inverse::step<22> {
    using type = StaticStep<StepType::kPredict, 0, 0xc14e1930U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
