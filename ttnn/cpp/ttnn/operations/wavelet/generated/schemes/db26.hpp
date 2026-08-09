// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct db26_inverse;

struct db26 {
    static constexpr const char* name = "db26";
    static constexpr uint32_t tap_size = 52U;
    static constexpr int32_t delay_even = 13;
    static constexpr int32_t delay_odd = 13;
    static constexpr uint32_t num_steps = 29U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db26.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db26";
    using inverse = db26_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct db26::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe9da76cU>;
    static_assert(type::k == 1U);
};

template <>
struct db26::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbee2ba4dU, 0x3e2d800eU>;
    static_assert(type::k == 2U);
};

template <>
struct db26::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf042539U, 0x3ecc71caU>;
    static_assert(type::k == 2U);
};

template <>
struct db26::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf2129dfU, 0x3f0b23e3U>;
    static_assert(type::k == 2U);
};

template <>
struct db26::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf345e6aU, 0x3f1865a8U>;
    static_assert(type::k == 2U);
};

template <>
struct db26::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf50a865U, 0x3f2d3091U>;
    static_assert(type::k == 2U);
};

template <>
struct db26::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf5f89a3U, 0x3f351d6cU>;
    static_assert(type::k == 2U);
};

template <>
struct db26::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf7af02dU, 0x3f3026b7U>;
    static_assert(type::k == 2U);
};

template <>
struct db26::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf8f251aU, 0x3e8dd4a1U>;
    static_assert(type::k == 2U);
};

template <>
struct db26::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc01c68c6U, 0x3e30fdc4U>;
    static_assert(type::k == 2U);
};

template <>
struct db26::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xc064ef12U, 0x3e8b063cU>;
    static_assert(type::k == 2U);
};

template <>
struct db26::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ba82c8aU, 0x3e880aceU>;
    static_assert(type::k == 2U);
};

template <>
struct db26::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x3fc921a4U, 0x42894e03U>;
    static_assert(type::k == 2U);
};

template <>
struct db26::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc6f8322U, 0x3bee5336U>;
    static_assert(type::k == 2U);
};

template <>
struct db26::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xc310c41eU, 0x42b5caa4U>;
    static_assert(type::k == 2U);
};

template <>
struct db26::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc44486bU, 0x3be06c3fU>;
    static_assert(type::k == 2U);
};

template <>
struct db26::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xc3230e52U, 0x42a6662fU>;
    static_assert(type::k == 2U);
};

template <>
struct db26::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc606f7fU, 0x3bc8ca64U>;
    static_assert(type::k == 2U);
};

template <>
struct db26::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xc33d7472U, 0x4291fa19U>;
    static_assert(type::k == 2U);
};

template <>
struct db26::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc84ce3bU, 0x3bacf47cU>;
    static_assert(type::k == 2U);
};

template <>
struct db26::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0xc3653b0cU, 0x4276bc66U>;
    static_assert(type::k == 2U);
};

template <>
struct db26::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbca56179U, 0x3b8ef29cU>;
    static_assert(type::k == 2U);
};

template <>
struct db26::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0xc394ad50U, 0x42462300U>;
    static_assert(type::k == 2U);
};

template <>
struct db26::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbce4bee7U, 0x3b5c65b1U>;
    static_assert(type::k == 2U);
};

template <>
struct db26::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0xc3e7351eU, 0x420f4033U>;
    static_assert(type::k == 2U);
};

template <>
struct db26::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0x2202a316U, 0x3b0db9bbU>;
    static_assert(type::k == 2U);
};

template <>
struct db26::step<26> {
    using type = StaticStep<StepType::kPredict, 0, 0xc18a470eU>;
    static_assert(type::k == 1U);
};

template <>
struct db26::step<27> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x475a59aeU>;
    static_assert(type::k == 1U);
};

template <>
struct db26::step<28> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3796121fU>;
    static_assert(type::k == 1U);
};

struct db26_inverse {
    static constexpr const char* name = "db26-inverse";
    static constexpr uint32_t tap_size = 52U;
    static constexpr uint32_t num_steps = 29U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db26.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db26_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct db26_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x475a59adU>;
    static_assert(type::k == 1U);
};

template <>
struct db26_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3796121eU>;
    static_assert(type::k == 1U);
};

template <>
struct db26_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0x418a470eU>;
    static_assert(type::k == 1U);
};

template <>
struct db26_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xa202a316U, 0xbb0db9bbU>;
    static_assert(type::k == 2U);
};

template <>
struct db26_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x43e7351eU, 0xc20f4033U>;
    static_assert(type::k == 2U);
};

template <>
struct db26_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ce4bee7U, 0xbb5c65b1U>;
    static_assert(type::k == 2U);
};

template <>
struct db26_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x4394ad50U, 0xc2462300U>;
    static_assert(type::k == 2U);
};

template <>
struct db26_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ca56179U, 0xbb8ef29cU>;
    static_assert(type::k == 2U);
};

template <>
struct db26_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x43653b0cU, 0xc276bc66U>;
    static_assert(type::k == 2U);
};

template <>
struct db26_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c84ce3bU, 0xbbacf47cU>;
    static_assert(type::k == 2U);
};

template <>
struct db26_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x433d7472U, 0xc291fa19U>;
    static_assert(type::k == 2U);
};

template <>
struct db26_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c606f7fU, 0xbbc8ca64U>;
    static_assert(type::k == 2U);
};

template <>
struct db26_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x43230e52U, 0xc2a6662fU>;
    static_assert(type::k == 2U);
};

template <>
struct db26_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c44486bU, 0xbbe06c3fU>;
    static_assert(type::k == 2U);
};

template <>
struct db26_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x4310c41eU, 0xc2b5caa4U>;
    static_assert(type::k == 2U);
};

template <>
struct db26_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c6f8322U, 0xbbee5336U>;
    static_assert(type::k == 2U);
};

template <>
struct db26_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xbfc921a4U, 0xc2894e03U>;
    static_assert(type::k == 2U);
};

template <>
struct db26_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbba82c8aU, 0xbe880aceU>;
    static_assert(type::k == 2U);
};

template <>
struct db26_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0x4064ef12U, 0xbe8b063cU>;
    static_assert(type::k == 2U);
};

template <>
struct db26_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0x401c68c6U, 0xbe30fdc4U>;
    static_assert(type::k == 2U);
};

template <>
struct db26_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f8f251aU, 0xbe8dd4a1U>;
    static_assert(type::k == 2U);
};

template <>
struct db26_inverse::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f7af02dU, 0xbf3026b7U>;
    static_assert(type::k == 2U);
};

template <>
struct db26_inverse::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f5f89a3U, 0xbf351d6cU>;
    static_assert(type::k == 2U);
};

template <>
struct db26_inverse::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f50a865U, 0xbf2d3091U>;
    static_assert(type::k == 2U);
};

template <>
struct db26_inverse::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f345e6aU, 0xbf1865a8U>;
    static_assert(type::k == 2U);
};

template <>
struct db26_inverse::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f2129dfU, 0xbf0b23e3U>;
    static_assert(type::k == 2U);
};

template <>
struct db26_inverse::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f042539U, 0xbecc71caU>;
    static_assert(type::k == 2U);
};

template <>
struct db26_inverse::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ee2ba4dU, 0xbe2d800eU>;
    static_assert(type::k == 2U);
};

template <>
struct db26_inverse::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e9da76cU>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
