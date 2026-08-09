// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct db27_inverse;

struct db27 {
    static constexpr const char* name = "db27";
    static constexpr uint32_t tap_size = 54U;
    static constexpr int32_t delay_even = 13;
    static constexpr int32_t delay_odd = 14;
    static constexpr uint32_t num_steps = 31U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db27.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db27";
    using inverse = db27_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct db27::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0x40ba16b6U>;
    static_assert(type::k == 1U);
};

template <>
struct db27::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe282e4eU, 0xbc4d9295U>;
    static_assert(type::k == 2U);
};

template <>
struct db27::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0x4157b438U, 0xc1945e5fU>;
    static_assert(type::k == 2U);
};

template <>
struct db27::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c709151U, 0xbc989bccU>;
    static_assert(type::k == 2U);
};

template <>
struct db27::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x41998d82U, 0xc1c086d9U>;
    static_assert(type::k == 2U);
};

template <>
struct db27::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c99dd46U, 0xbcc0916bU>;
    static_assert(type::k == 2U);
};

template <>
struct db27::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x41bc1182U, 0xc1ed90e8U>;
    static_assert(type::k == 2U);
};

template <>
struct db27::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3cb46cfeU, 0xbd199faaU>;
    static_assert(type::k == 2U);
};

template <>
struct db27::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x41aa3283U, 0x40278a50U>;
    static_assert(type::k == 2U);
};

template <>
struct db27::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f006af2U, 0x3c22f4e6U>;
    static_assert(type::k == 2U);
};

template <>
struct db27::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f5ed735U, 0xbfec8238U>;
    static_assert(type::k == 2U);
};

template <>
struct db27::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3faadca1U, 0xbf824d5cU>;
    static_assert(type::k == 2U);
};

template <>
struct db27::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f7edee6U, 0xbf32c87bU>;
    static_assert(type::k == 2U);
};

template <>
struct db27::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc12ec602U, 0xbf783519U>;
    static_assert(type::k == 2U);
};

template <>
struct db27::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xb8adef79U, 0x3dbbe269U>;
    static_assert(type::k == 2U);
};

template <>
struct db27::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc2b4876bU, 0xc6d522b2U>;
    static_assert(type::k == 2U);
};

template <>
struct db27::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x381978d4U, 0xb7a789b9U>;
    static_assert(type::k == 2U);
};

template <>
struct db27::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x47433cfcU, 0xc7c55335U>;
    static_assert(type::k == 2U);
};

template <>
struct db27::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0x3725fd4eU, 0xb7c189b1U>;
    static_assert(type::k == 2U);
};

template <>
struct db27::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0x47294c10U, 0xc7e55daeU>;
    static_assert(type::k == 2U);
};

template <>
struct db27::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0x370edca4U, 0xb7e4fac2U>;
    static_assert(type::k == 2U);
};

template <>
struct db27::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0x470f1ab6U, 0xc80aa97bU>;
    static_assert(type::k == 2U);
};

template <>
struct db27::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0x36ec50c9U, 0xb80e6f2aU>;
    static_assert(type::k == 2U);
};

template <>
struct db27::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0x46e60e9bU, 0xc833abd5U>;
    static_assert(type::k == 2U);
};

template <>
struct db27::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0x36b660a9U, 0xb844c735U>;
    static_assert(type::k == 2U);
};

template <>
struct db27::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0x46a685c6U, 0xc88b8a09U>;
    static_assert(type::k == 2U);
};

template <>
struct db27::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0x366ad480U, 0xb8cb9e8cU>;
    static_assert(type::k == 2U);
};

template <>
struct db27::step<27> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct db27::step<28> {
    using type = StaticStep<StepType::kPredict, 0, 0x4620ed81U>;
    static_assert(type::k == 1U);
};

template <>
struct db27::step<29> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xbbade0e7U>;
    static_assert(type::k == 1U);
};

template <>
struct db27::step<30> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x433c7412U>;
    static_assert(type::k == 1U);
};

struct db27_inverse {
    static constexpr const char* name = "db27-inverse";
    static constexpr uint32_t tap_size = 54U;
    static constexpr uint32_t num_steps = 31U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db27.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db27_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct db27_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3bade0e7U>;
    static_assert(type::k == 1U);
};

template <>
struct db27_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xc33c7412U>;
    static_assert(type::k == 1U);
};

template <>
struct db27_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0xc620ed81U>;
    static_assert(type::k == 1U);
};

template <>
struct db27_inverse::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct db27_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xb66ad480U, 0x38cb9e8cU>;
    static_assert(type::k == 2U);
};

template <>
struct db27_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc6a685c6U, 0x488b8a09U>;
    static_assert(type::k == 2U);
};

template <>
struct db27_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xb6b660a9U, 0x3844c735U>;
    static_assert(type::k == 2U);
};

template <>
struct db27_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc6e60e9bU, 0x4833abd5U>;
    static_assert(type::k == 2U);
};

template <>
struct db27_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xb6ec50c9U, 0x380e6f2aU>;
    static_assert(type::k == 2U);
};

template <>
struct db27_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc70f1ab6U, 0x480aa97bU>;
    static_assert(type::k == 2U);
};

template <>
struct db27_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xb70edca4U, 0x37e4fac2U>;
    static_assert(type::k == 2U);
};

template <>
struct db27_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc7294c10U, 0x47e55daeU>;
    static_assert(type::k == 2U);
};

template <>
struct db27_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xb725fd4eU, 0x37c189b1U>;
    static_assert(type::k == 2U);
};

template <>
struct db27_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc7433cfcU, 0x47c55335U>;
    static_assert(type::k == 2U);
};

template <>
struct db27_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xb81978d4U, 0x37a789b9U>;
    static_assert(type::k == 2U);
};

template <>
struct db27_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x42b4876bU, 0x46d522b2U>;
    static_assert(type::k == 2U);
};

template <>
struct db27_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x38adef79U, 0xbdbbe269U>;
    static_assert(type::k == 2U);
};

template <>
struct db27_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x412ec602U, 0x3f783519U>;
    static_assert(type::k == 2U);
};

template <>
struct db27_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf7edee6U, 0x3f32c87bU>;
    static_assert(type::k == 2U);
};

template <>
struct db27_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfaadca1U, 0x3f824d5cU>;
    static_assert(type::k == 2U);
};

template <>
struct db27_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf5ed735U, 0x3fec8238U>;
    static_assert(type::k == 2U);
};

template <>
struct db27_inverse::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf006af2U, 0xbc22f4e6U>;
    static_assert(type::k == 2U);
};

template <>
struct db27_inverse::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0xc1aa3283U, 0xc0278a50U>;
    static_assert(type::k == 2U);
};

template <>
struct db27_inverse::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbcb46cfeU, 0x3d199faaU>;
    static_assert(type::k == 2U);
};

template <>
struct db27_inverse::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0xc1bc1182U, 0x41ed90e8U>;
    static_assert(type::k == 2U);
};

template <>
struct db27_inverse::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc99dd46U, 0x3cc0916bU>;
    static_assert(type::k == 2U);
};

template <>
struct db27_inverse::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0xc1998d82U, 0x41c086d9U>;
    static_assert(type::k == 2U);
};

template <>
struct db27_inverse::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc709151U, 0x3c989bccU>;
    static_assert(type::k == 2U);
};

template <>
struct db27_inverse::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0xc157b438U, 0x41945e5fU>;
    static_assert(type::k == 2U);
};

template <>
struct db27_inverse::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e282e4eU, 0x3c4d9295U>;
    static_assert(type::k == 2U);
};

template <>
struct db27_inverse::step<30> {
    using type = StaticStep<StepType::kPredict, 0, 0xc0ba16b6U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
