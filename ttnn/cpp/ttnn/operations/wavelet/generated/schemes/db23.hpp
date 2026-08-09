// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct db23_inverse;

struct db23 {
    static constexpr const char* name = "db23";
    static constexpr uint32_t tap_size = 46U;
    static constexpr int32_t delay_even = 11;
    static constexpr int32_t delay_odd = 12;
    static constexpr uint32_t num_steps = 27U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db23.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db23";
    using inverse = db23_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct db23::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0x40b9e151U>;
    static_assert(type::k == 1U);
};

template <>
struct db23::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe2646f8U, 0xbc2fdaf5U>;
    static_assert(type::k == 2U);
};

template <>
struct db23::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0x417cdd33U, 0xc129ab31U>;
    static_assert(type::k == 2U);
};

template <>
struct db23::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c93f928U, 0xbc0a6708U>;
    static_assert(type::k == 2U);
};

template <>
struct db23::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x41bd38f5U, 0xc1640ec1U>;
    static_assert(type::k == 2U);
};

template <>
struct db23::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3cc1e348U, 0xbc8f67beU>;
    static_assert(type::k == 2U);
};

template <>
struct db23::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x41ef1f3eU, 0xc1b891b9U>;
    static_assert(type::k == 2U);
};

template <>
struct db23::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ced177bU, 0xbcb6104aU>;
    static_assert(type::k == 2U);
};

template <>
struct db23::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x420eb580U, 0xc1d47ae2U>;
    static_assert(type::k == 2U);
};

template <>
struct db23::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f3e93e2U, 0xbcc46566U>;
    static_assert(type::k == 2U);
};

template <>
struct db23::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xba2b53d3U, 0xbfab435dU>;
    static_assert(type::k == 2U);
};

template <>
struct db23::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc169a160U, 0xc39856b9U>;
    static_assert(type::k == 2U);
};

template <>
struct db23::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x3b4ffb3aU, 0xbb84022aU>;
    static_assert(type::k == 2U);
};

template <>
struct db23::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x43740cd2U, 0xc3d36567U>;
    static_assert(type::k == 2U);
};

template <>
struct db23::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x3b1a21aeU, 0xbb9732edU>;
    static_assert(type::k == 2U);
};

template <>
struct db23::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x435866d0U, 0xc3f52fb2U>;
    static_assert(type::k == 2U);
};

template <>
struct db23::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x3b059b5dU, 0xbbb2dfdeU>;
    static_assert(type::k == 2U);
};

template <>
struct db23::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x43372ebdU, 0xc4148a64U>;
    static_assert(type::k == 2U);
};

template <>
struct db23::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0x3adc994bU, 0xbbdf5b2aU>;
    static_assert(type::k == 2U);
};

template <>
struct db23::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0x4312b518U, 0xc4415b43U>;
    static_assert(type::k == 2U);
};

template <>
struct db23::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0x3aa97827U, 0xbc1b1028U>;
    static_assert(type::k == 2U);
};

template <>
struct db23::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0x42d35206U, 0xc496f187U>;
    static_assert(type::k == 2U);
};

template <>
struct db23::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0x3a591680U, 0xbca14723U>;
    static_assert(type::k == 2U);
};

template <>
struct db23::step<23> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct db23::step<24> {
    using type = StaticStep<StepType::kPredict, 0, 0x424b2d62U>;
    static_assert(type::k == 1U);
};

template <>
struct db23::step<25> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3ac84f09U>;
    static_assert(type::k == 1U);
};

template <>
struct db23::step<26> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xc4239665U>;
    static_assert(type::k == 1U);
};

struct db23_inverse {
    static constexpr const char* name = "db23-inverse";
    static constexpr uint32_t tap_size = 46U;
    static constexpr uint32_t num_steps = 27U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db23.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db23_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct db23_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xbac84f09U>;
    static_assert(type::k == 1U);
};

template <>
struct db23_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x44239665U>;
    static_assert(type::k == 1U);
};

template <>
struct db23_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0xc24b2d62U>;
    static_assert(type::k == 1U);
};

template <>
struct db23_inverse::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct db23_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xba591680U, 0x3ca14723U>;
    static_assert(type::k == 2U);
};

template <>
struct db23_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc2d35206U, 0x4496f187U>;
    static_assert(type::k == 2U);
};

template <>
struct db23_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xbaa97827U, 0x3c1b1028U>;
    static_assert(type::k == 2U);
};

template <>
struct db23_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc312b518U, 0x44415b43U>;
    static_assert(type::k == 2U);
};

template <>
struct db23_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xbadc994bU, 0x3bdf5b2aU>;
    static_assert(type::k == 2U);
};

template <>
struct db23_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc3372ebdU, 0x44148a64U>;
    static_assert(type::k == 2U);
};

template <>
struct db23_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xbb059b5dU, 0x3bb2dfdeU>;
    static_assert(type::k == 2U);
};

template <>
struct db23_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc35866d0U, 0x43f52fb2U>;
    static_assert(type::k == 2U);
};

template <>
struct db23_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xbb1a21aeU, 0x3b9732edU>;
    static_assert(type::k == 2U);
};

template <>
struct db23_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc3740cd2U, 0x43d36567U>;
    static_assert(type::k == 2U);
};

template <>
struct db23_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xbb4ffb3aU, 0x3b84022aU>;
    static_assert(type::k == 2U);
};

template <>
struct db23_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x4169a160U, 0x439856b9U>;
    static_assert(type::k == 2U);
};

template <>
struct db23_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x3a2b53d3U, 0x3fab435dU>;
    static_assert(type::k == 2U);
};

template <>
struct db23_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf3e93e2U, 0x3cc46566U>;
    static_assert(type::k == 2U);
};

template <>
struct db23_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xc20eb580U, 0x41d47ae2U>;
    static_assert(type::k == 2U);
};

template <>
struct db23_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbced177bU, 0x3cb6104aU>;
    static_assert(type::k == 2U);
};

template <>
struct db23_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0xc1ef1f3eU, 0x41b891b9U>;
    static_assert(type::k == 2U);
};

template <>
struct db23_inverse::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbcc1e348U, 0x3c8f67beU>;
    static_assert(type::k == 2U);
};

template <>
struct db23_inverse::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0xc1bd38f5U, 0x41640ec1U>;
    static_assert(type::k == 2U);
};

template <>
struct db23_inverse::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc93f928U, 0x3c0a6708U>;
    static_assert(type::k == 2U);
};

template <>
struct db23_inverse::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0xc17cdd33U, 0x4129ab31U>;
    static_assert(type::k == 2U);
};

template <>
struct db23_inverse::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e2646f8U, 0x3c2fdaf5U>;
    static_assert(type::k == 2U);
};

template <>
struct db23_inverse::step<26> {
    using type = StaticStep<StepType::kPredict, 0, 0xc0b9e151U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
