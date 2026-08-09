// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct db20_inverse;

struct db20 {
    static constexpr const char* name = "db20";
    static constexpr uint32_t tap_size = 40U;
    static constexpr int32_t delay_even = 10;
    static constexpr int32_t delay_odd = 10;
    static constexpr uint32_t num_steps = 23U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db20.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db20";
    using inverse = db20_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct db20::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0xbece0a8eU>;
    static_assert(type::k == 1U);
};

template <>
struct db20::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf14d2daU, 0x3d92c36eU>;
    static_assert(type::k == 2U);
};

template <>
struct db20::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf376c5aU, 0x3e4f2d20U>;
    static_assert(type::k == 2U);
};

template <>
struct db20::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf6776f4U, 0x3ea685f2U>;
    static_assert(type::k == 2U);
};

template <>
struct db20::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x4006c007U, 0x3ed43cc0U>;
    static_assert(type::k == 2U);
};

template <>
struct db20::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3b094439U, 0xbf5d29f4U>;
    static_assert(type::k == 2U);
};

template <>
struct db20::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x3feaae52U, 0x3f0afd9bU>;
    static_assert(type::k == 2U);
};

template <>
struct db20::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf9c5d18U, 0x3f64422fU>;
    static_assert(type::k == 2U);
};

template <>
struct db20::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf63b8e1U, 0x3f24cb7bU>;
    static_assert(type::k == 2U);
};

template <>
struct db20::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfb4afcbU, 0x3f78d965U>;
    static_assert(type::k == 2U);
};

template <>
struct db20::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf83e6b6U, 0x3f28cd53U>;
    static_assert(type::k == 2U);
};

template <>
struct db20::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfd045acU, 0x3f7116b2U>;
    static_assert(type::k == 2U);
};

template <>
struct db20::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf9920f0U, 0x3f1bbe88U>;
    static_assert(type::k == 2U);
};

template <>
struct db20::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbff6026bU, 0x3f556d55U>;
    static_assert(type::k == 2U);
};

template <>
struct db20::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xbfb97305U, 0x3f0521cfU>;
    static_assert(type::k == 2U);
};

template <>
struct db20::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc019d845U, 0x3f30af14U>;
    static_assert(type::k == 2U);
};

template <>
struct db20::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xbff23512U, 0x3ed4fe25U>;
    static_assert(type::k == 2U);
};

template <>
struct db20::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc0568882U, 0x3f0749feU>;
    static_assert(type::k == 2U);
};

template <>
struct db20::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xc03dfa5cU, 0x3e98bdacU>;
    static_assert(type::k == 2U);
};

template <>
struct db20::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0x2b91c7d4U, 0x3eac7bb0U>;
    static_assert(type::k == 2U);
};

template <>
struct db20::step<20> {
    using type = StaticStep<StepType::kPredict, 0, 0xbe12249dU>;
    static_assert(type::k == 1U);
};

template <>
struct db20::step<21> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xc4185023U>;
    static_assert(type::k == 1U);
};

template <>
struct db20::step<22> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xbad722caU>;
    static_assert(type::k == 1U);
};

struct db20_inverse {
    static constexpr const char* name = "db20-inverse";
    static constexpr uint32_t tap_size = 40U;
    static constexpr uint32_t num_steps = 23U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db20.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db20_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct db20_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xc4185023U>;
    static_assert(type::k == 1U);
};

template <>
struct db20_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xbad722caU>;
    static_assert(type::k == 1U);
};

template <>
struct db20_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0x3e12249dU>;
    static_assert(type::k == 1U);
};

template <>
struct db20_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xab91c7d4U, 0xbeac7bb0U>;
    static_assert(type::k == 2U);
};

template <>
struct db20_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x403dfa5cU, 0xbe98bdacU>;
    static_assert(type::k == 2U);
};

template <>
struct db20_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x40568882U, 0xbf0749feU>;
    static_assert(type::k == 2U);
};

template <>
struct db20_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ff23512U, 0xbed4fe25U>;
    static_assert(type::k == 2U);
};

template <>
struct db20_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x4019d845U, 0xbf30af14U>;
    static_assert(type::k == 2U);
};

template <>
struct db20_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x3fb97305U, 0xbf0521cfU>;
    static_assert(type::k == 2U);
};

template <>
struct db20_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ff6026bU, 0xbf556d55U>;
    static_assert(type::k == 2U);
};

template <>
struct db20_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f9920f0U, 0xbf1bbe88U>;
    static_assert(type::k == 2U);
};

template <>
struct db20_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fd045acU, 0xbf7116b2U>;
    static_assert(type::k == 2U);
};

template <>
struct db20_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f83e6b6U, 0xbf28cd53U>;
    static_assert(type::k == 2U);
};

template <>
struct db20_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fb4afcbU, 0xbf78d965U>;
    static_assert(type::k == 2U);
};

template <>
struct db20_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f63b8e1U, 0xbf24cb7bU>;
    static_assert(type::k == 2U);
};

template <>
struct db20_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f9c5d18U, 0xbf64422fU>;
    static_assert(type::k == 2U);
};

template <>
struct db20_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xbfeaae52U, 0xbf0afd9bU>;
    static_assert(type::k == 2U);
};

template <>
struct db20_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbb094439U, 0x3f5d29f4U>;
    static_assert(type::k == 2U);
};

template <>
struct db20_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xc006c007U, 0xbed43cc0U>;
    static_assert(type::k == 2U);
};

template <>
struct db20_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f6776f4U, 0xbea685f2U>;
    static_assert(type::k == 2U);
};

template <>
struct db20_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f376c5aU, 0xbe4f2d20U>;
    static_assert(type::k == 2U);
};

template <>
struct db20_inverse::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f14d2daU, 0xbd92c36eU>;
    static_assert(type::k == 2U);
};

template <>
struct db20_inverse::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ece0a8eU>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
