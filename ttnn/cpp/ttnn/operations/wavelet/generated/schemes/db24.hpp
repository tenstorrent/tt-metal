// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct db24_inverse;

struct db24 {
    static constexpr const char* name = "db24";
    static constexpr uint32_t tap_size = 48U;
    static constexpr int32_t delay_even = 12;
    static constexpr int32_t delay_odd = 12;
    static constexpr uint32_t num_steps = 27U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db24.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db24";
    using inverse = db24_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct db24::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0xbeab8d4aU>;
    static_assert(type::k == 1U);
};

template <>
struct db24::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbef5d97eU, 0x3e3ac5f4U>;
    static_assert(type::k == 2U);
};

template <>
struct db24::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf0dc230U, 0x3ed954adU>;
    static_assert(type::k == 2U);
};

template <>
struct db24::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf2d1b2eU, 0x3f14e4c7U>;
    static_assert(type::k == 2U);
};

template <>
struct db24::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf408c76U, 0x3f2687deU>;
    static_assert(type::k == 2U);
};

template <>
struct db24::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf5b4993U, 0x3f90582aU>;
    static_assert(type::k == 2U);
};

template <>
struct db24::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf2ce2b5U, 0xbd8781f8U>;
    static_assert(type::k == 2U);
};

template <>
struct db24::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc1a0e2d9U, 0xbee16a13U>;
    static_assert(type::k == 2U);
};

template <>
struct db24::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xbd02ec64U, 0x3d392a7dU>;
    static_assert(type::k == 2U);
};

template <>
struct db24::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc0231bd7U, 0x41d3b973U>;
    static_assert(type::k == 2U);
};

template <>
struct db24::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xbb8153b7U, 0x3de5b675U>;
    static_assert(type::k == 2U);
};

template <>
struct db24::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc1021b58U, 0x4102b3bcU>;
    static_assert(type::k == 2U);
};

template <>
struct db24::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xbdfbf486U, 0x3da89fcbU>;
    static_assert(type::k == 2U);
};

template <>
struct db24::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc14dad3bU, 0x40fc9b90U>;
    static_assert(type::k == 2U);
};

template <>
struct db24::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe0e8e1aU, 0x3d9db84bU>;
    static_assert(type::k == 2U);
};

template <>
struct db24::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc16adcf8U, 0x40e52715U>;
    static_assert(type::k == 2U);
};

template <>
struct db24::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe2572d8U, 0x3d8b6ab6U>;
    static_assert(type::k == 2U);
};

template <>
struct db24::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc18af97aU, 0x40c60792U>;
    static_assert(type::k == 2U);
};

template <>
struct db24::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe4867c4U, 0x3d6bc7c2U>;
    static_assert(type::k == 2U);
};

template <>
struct db24::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc1ad6321U, 0x40a38226U>;
    static_assert(type::k == 2U);
};

template <>
struct db24::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe8247fcU, 0x3d3cfcc5U>;
    static_assert(type::k == 2U);
};

template <>
struct db24::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc1f06c13U, 0x407b847bU>;
    static_assert(type::k == 2U);
};

template <>
struct db24::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0xbecb1f31U, 0x3d084b29U>;
    static_assert(type::k == 2U);
};

template <>
struct db24::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0x2910aeeaU, 0x40215267U>;
    static_assert(type::k == 2U);
};

template <>
struct db24::step<24> {
    using type = StaticStep<StepType::kPredict, 0, 0xbc8339f6U>;
    static_assert(type::k == 1U);
};

template <>
struct db24::step<25> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x4452144bU>;
    static_assert(type::k == 1U);
};

template <>
struct db24::step<26> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3a9bfaaeU>;
    static_assert(type::k == 1U);
};

struct db24_inverse {
    static constexpr const char* name = "db24-inverse";
    static constexpr uint32_t tap_size = 48U;
    static constexpr uint32_t num_steps = 27U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db24.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db24_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct db24_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x4452144bU>;
    static_assert(type::k == 1U);
};

template <>
struct db24_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3a9bfaaeU>;
    static_assert(type::k == 1U);
};

template <>
struct db24_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0x3c8339f6U>;
    static_assert(type::k == 1U);
};

template <>
struct db24_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xa910aeeaU, 0xc0215267U>;
    static_assert(type::k == 2U);
};

template <>
struct db24_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ecb1f31U, 0xbd084b29U>;
    static_assert(type::k == 2U);
};

template <>
struct db24_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x41f06c13U, 0xc07b847bU>;
    static_assert(type::k == 2U);
};

template <>
struct db24_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e8247fcU, 0xbd3cfcc5U>;
    static_assert(type::k == 2U);
};

template <>
struct db24_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x41ad6321U, 0xc0a38226U>;
    static_assert(type::k == 2U);
};

template <>
struct db24_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e4867c4U, 0xbd6bc7c2U>;
    static_assert(type::k == 2U);
};

template <>
struct db24_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x418af97aU, 0xc0c60792U>;
    static_assert(type::k == 2U);
};

template <>
struct db24_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e2572d8U, 0xbd8b6ab6U>;
    static_assert(type::k == 2U);
};

template <>
struct db24_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x416adcf8U, 0xc0e52715U>;
    static_assert(type::k == 2U);
};

template <>
struct db24_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e0e8e1aU, 0xbd9db84bU>;
    static_assert(type::k == 2U);
};

template <>
struct db24_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x414dad3bU, 0xc0fc9b90U>;
    static_assert(type::k == 2U);
};

template <>
struct db24_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x3dfbf486U, 0xbda89fcbU>;
    static_assert(type::k == 2U);
};

template <>
struct db24_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x41021b58U, 0xc102b3bcU>;
    static_assert(type::k == 2U);
};

template <>
struct db24_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x3b8153b7U, 0xbde5b675U>;
    static_assert(type::k == 2U);
};

template <>
struct db24_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x40231bd7U, 0xc1d3b973U>;
    static_assert(type::k == 2U);
};

template <>
struct db24_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0x3d02ec64U, 0xbd392a7dU>;
    static_assert(type::k == 2U);
};

template <>
struct db24_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0x41a0e2d9U, 0x3ee16a13U>;
    static_assert(type::k == 2U);
};

template <>
struct db24_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f2ce2b5U, 0x3d8781f8U>;
    static_assert(type::k == 2U);
};

template <>
struct db24_inverse::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f5b4993U, 0xbf90582aU>;
    static_assert(type::k == 2U);
};

template <>
struct db24_inverse::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f408c76U, 0xbf2687deU>;
    static_assert(type::k == 2U);
};

template <>
struct db24_inverse::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f2d1b2eU, 0xbf14e4c7U>;
    static_assert(type::k == 2U);
};

template <>
struct db24_inverse::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f0dc230U, 0xbed954adU>;
    static_assert(type::k == 2U);
};

template <>
struct db24_inverse::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ef5d97eU, 0xbe3ac5f4U>;
    static_assert(type::k == 2U);
};

template <>
struct db24_inverse::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0x3eab8d4aU>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
