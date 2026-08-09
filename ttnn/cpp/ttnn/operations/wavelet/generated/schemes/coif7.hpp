// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct coif7_inverse;

struct coif7 {
    static constexpr const char* name = "coif7";
    static constexpr uint32_t tap_size = 42U;
    static constexpr int32_t delay_even = 10;
    static constexpr int32_t delay_odd = 11;
    static constexpr uint32_t num_steps = 25U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/coif7.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::coif7";
    using inverse = coif7_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct coif7::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0xbfc3f55eU>;
    static_assert(type::k == 1U);
};

template <>
struct coif7::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3eea6b37U, 0x3f238683U>;
    static_assert(type::k == 2U);
};

template <>
struct coif7::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xbfa45b70U, 0x3f89ea77U>;
    static_assert(type::k == 2U);
};

template <>
struct coif7::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf35302cU, 0x3ed165b5U>;
    static_assert(type::k == 2U);
};

template <>
struct coif7::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xbfa1f784U, 0x3f0d67cdU>;
    static_assert(type::k == 2U);
};

template <>
struct coif7::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe9d4c90U, 0x3e6aba8aU>;
    static_assert(type::k == 2U);
};

template <>
struct coif7::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xbee42d4aU, 0x3e0c711dU>;
    static_assert(type::k == 2U);
};

template <>
struct coif7::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd9f3281U, 0xbd85299fU>;
    static_assert(type::k == 2U);
};

template <>
struct coif7::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x3debb413U, 0xbeae1527U>;
    static_assert(type::k == 2U);
};

template <>
struct coif7::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e39f22aU, 0xbecb8e2fU>;
    static_assert(type::k == 2U);
};

template <>
struct coif7::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f1a67aaU, 0xbf17b5bbU>;
    static_assert(type::k == 2U);
};

template <>
struct coif7::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ec9ad09U, 0xbf20d752U>;
    static_assert(type::k == 2U);
};

template <>
struct coif7::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f31deabU, 0xbfb6c7f4U>;
    static_assert(type::k == 2U);
};

template <>
struct coif7::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f095055U, 0xbff334f5U>;
    static_assert(type::k == 2U);
};

template <>
struct coif7::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f0387aaU, 0xc03ec994U>;
    static_assert(type::k == 2U);
};

template <>
struct coif7::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3eaba091U, 0xbdc6717eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif7::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x4123b53aU, 0xc1c4f837U>;
    static_assert(type::k == 2U);
};

template <>
struct coif7::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d261cc3U, 0xbe0665d6U>;
    static_assert(type::k == 2U);
};

template <>
struct coif7::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0x40f3c75bU, 0xc209f5b7U>;
    static_assert(type::k == 2U);
};

template <>
struct coif7::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ced844aU, 0xbe543b98U>;
    static_assert(type::k == 2U);
};

template <>
struct coif7::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0x409a6580U, 0xc2919877U>;
    static_assert(type::k == 2U);
};

template <>
struct coif7::step<21> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct coif7::step<22> {
    using type = StaticStep<StepType::kPredict, 0, 0x3c610fd2U>;
    static_assert(type::k == 1U);
};

template <>
struct coif7::step<23> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xba17fa7bU>;
    static_assert(type::k == 1U);
};

template <>
struct coif7::step<24> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x44d79c0bU>;
    static_assert(type::k == 1U);
};

struct coif7_inverse {
    static constexpr const char* name = "coif7-inverse";
    static constexpr uint32_t tap_size = 42U;
    static constexpr uint32_t num_steps = 25U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/coif7.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::coif7_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct coif7_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3a17fa7aU>;
    static_assert(type::k == 1U);
};

template <>
struct coif7_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xc4d79c0aU>;
    static_assert(type::k == 1U);
};

template <>
struct coif7_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0xbc610fd2U>;
    static_assert(type::k == 1U);
};

template <>
struct coif7_inverse::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct coif7_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xc09a6580U, 0x42919877U>;
    static_assert(type::k == 2U);
};

template <>
struct coif7_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbced844aU, 0x3e543b98U>;
    static_assert(type::k == 2U);
};

template <>
struct coif7_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0f3c75bU, 0x4209f5b7U>;
    static_assert(type::k == 2U);
};

template <>
struct coif7_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd261cc3U, 0x3e0665d6U>;
    static_assert(type::k == 2U);
};

template <>
struct coif7_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xc123b53aU, 0x41c4f837U>;
    static_assert(type::k == 2U);
};

template <>
struct coif7_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbeaba091U, 0x3dc6717eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif7_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf0387aaU, 0x403ec994U>;
    static_assert(type::k == 2U);
};

template <>
struct coif7_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf095055U, 0x3ff334f5U>;
    static_assert(type::k == 2U);
};

template <>
struct coif7_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf31deabU, 0x3fb6c7f4U>;
    static_assert(type::k == 2U);
};

template <>
struct coif7_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbec9ad09U, 0x3f20d752U>;
    static_assert(type::k == 2U);
};

template <>
struct coif7_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf1a67aaU, 0x3f17b5bbU>;
    static_assert(type::k == 2U);
};

template <>
struct coif7_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe39f22aU, 0x3ecb8e2fU>;
    static_assert(type::k == 2U);
};

template <>
struct coif7_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xbdebb413U, 0x3eae1527U>;
    static_assert(type::k == 2U);
};

template <>
struct coif7_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d9f3281U, 0x3d85299fU>;
    static_assert(type::k == 2U);
};

template <>
struct coif7_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ee42d4aU, 0xbe0c711dU>;
    static_assert(type::k == 2U);
};

template <>
struct coif7_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e9d4c90U, 0xbe6aba8aU>;
    static_assert(type::k == 2U);
};

template <>
struct coif7_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0x3fa1f784U, 0xbf0d67cdU>;
    static_assert(type::k == 2U);
};

template <>
struct coif7_inverse::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f35302cU, 0xbed165b5U>;
    static_assert(type::k == 2U);
};

template <>
struct coif7_inverse::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0x3fa45b70U, 0xbf89ea77U>;
    static_assert(type::k == 2U);
};

template <>
struct coif7_inverse::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbeea6b37U, 0xbf238683U>;
    static_assert(type::k == 2U);
};

template <>
struct coif7_inverse::step<24> {
    using type = StaticStep<StepType::kPredict, 0, 0x3fc3f55eU>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
