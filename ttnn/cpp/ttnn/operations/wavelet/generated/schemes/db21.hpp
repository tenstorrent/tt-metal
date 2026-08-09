// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct db21_inverse;

struct db21 {
    static constexpr const char* name = "db21";
    static constexpr uint32_t tap_size = 42U;
    static constexpr int32_t delay_even = 10;
    static constexpr int32_t delay_odd = 11;
    static constexpr uint32_t num_steps = 25U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db21.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db21";
    using inverse = db21_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct db21::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0x416ae5f1U>;
    static_assert(type::k == 1U);
};

template <>
struct db21::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd87f382U, 0xba795177U>;
    static_assert(type::k == 2U);
};

template <>
struct db21::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0x42dfe5e6U, 0xc2914ec9U>;
    static_assert(type::k == 2U);
};

template <>
struct db21::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3b41956bU, 0xbb0ae14fU>;
    static_assert(type::k == 2U);
};

template <>
struct db21::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x43219f47U, 0xc2f18e6aU>;
    static_assert(type::k == 2U);
};

template <>
struct db21::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3b82604bU, 0xbb45a169U>;
    static_assert(type::k == 2U);
};

template <>
struct db21::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x434ff104U, 0xc31b38b7U>;
    static_assert(type::k == 2U);
};

template <>
struct db21::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d61e89cU, 0xbb68c290U>;
    static_assert(type::k == 2U);
};

template <>
struct db21::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xbcb03573U, 0xc18e9dfaU>;
    static_assert(type::k == 2U);
};

template <>
struct db21::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbef8d439U, 0xc08809b4U>;
    static_assert(type::k == 2U);
};

template <>
struct db21::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e5cfdfbU, 0xbe905974U>;
    static_assert(type::k == 2U);
};

template <>
struct db21::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x4058f49bU, 0xc0aec372U>;
    static_assert(type::k == 2U);
};

template <>
struct db21::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e383822U, 0xbea5ce68U>;
    static_assert(type::k == 2U);
};

template <>
struct db21::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x40448c5fU, 0xc0ca9ce6U>;
    static_assert(type::k == 2U);
};

template <>
struct db21::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e2184b3U, 0xbec3ed7aU>;
    static_assert(type::k == 2U);
};

template <>
struct db21::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x4027356dU, 0xc0f57996U>;
    static_assert(type::k == 2U);
};

template <>
struct db21::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e057c1aU, 0xbef4f7a6U>;
    static_assert(type::k == 2U);
};

template <>
struct db21::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x4005c3b6U, 0xc120229eU>;
    static_assert(type::k == 2U);
};

template <>
struct db21::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0x3dcca086U, 0xbf2a8cccU>;
    static_assert(type::k == 2U);
};

template <>
struct db21::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fc021a0U, 0xc17ac9c7U>;
    static_assert(type::k == 2U);
};

template <>
struct db21::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0x3d82a8faU, 0xbfb1f0d9U>;
    static_assert(type::k == 2U);
};

template <>
struct db21::step<21> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct db21::step<22> {
    using type = StaticStep<StepType::kPredict, 0, 0x3f3826b0U>;
    static_assert(type::k == 1U);
};

template <>
struct db21::step<23> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x39c1756eU>;
    static_assert(type::k == 1U);
};

template <>
struct db21::step<24> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xc529613bU>;
    static_assert(type::k == 1U);
};

struct db21_inverse {
    static constexpr const char* name = "db21-inverse";
    static constexpr uint32_t tap_size = 42U;
    static constexpr uint32_t num_steps = 25U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db21.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db21_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct db21_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xb9c1756eU>;
    static_assert(type::k == 1U);
};

template <>
struct db21_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x4529613bU>;
    static_assert(type::k == 1U);
};

template <>
struct db21_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0xbf3826b0U>;
    static_assert(type::k == 1U);
};

template <>
struct db21_inverse::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct db21_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xbd82a8faU, 0x3fb1f0d9U>;
    static_assert(type::k == 2U);
};

template <>
struct db21_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfc021a0U, 0x417ac9c7U>;
    static_assert(type::k == 2U);
};

template <>
struct db21_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xbdcca086U, 0x3f2a8cccU>;
    static_assert(type::k == 2U);
};

template <>
struct db21_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc005c3b6U, 0x4120229eU>;
    static_assert(type::k == 2U);
};

template <>
struct db21_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe057c1aU, 0x3ef4f7a6U>;
    static_assert(type::k == 2U);
};

template <>
struct db21_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc027356dU, 0x40f57996U>;
    static_assert(type::k == 2U);
};

template <>
struct db21_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe2184b3U, 0x3ec3ed7aU>;
    static_assert(type::k == 2U);
};

template <>
struct db21_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc0448c5fU, 0x40ca9ce6U>;
    static_assert(type::k == 2U);
};

template <>
struct db21_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe383822U, 0x3ea5ce68U>;
    static_assert(type::k == 2U);
};

template <>
struct db21_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc058f49bU, 0x40aec372U>;
    static_assert(type::k == 2U);
};

template <>
struct db21_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe5cfdfbU, 0x3e905974U>;
    static_assert(type::k == 2U);
};

template <>
struct db21_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ef8d439U, 0x408809b4U>;
    static_assert(type::k == 2U);
};

template <>
struct db21_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x3cb03573U, 0x418e9dfaU>;
    static_assert(type::k == 2U);
};

template <>
struct db21_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd61e89cU, 0x3b68c290U>;
    static_assert(type::k == 2U);
};

template <>
struct db21_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xc34ff104U, 0x431b38b7U>;
    static_assert(type::k == 2U);
};

template <>
struct db21_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbb82604bU, 0x3b45a169U>;
    static_assert(type::k == 2U);
};

template <>
struct db21_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0xc3219f47U, 0x42f18e6aU>;
    static_assert(type::k == 2U);
};

template <>
struct db21_inverse::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbb41956bU, 0x3b0ae14fU>;
    static_assert(type::k == 2U);
};

template <>
struct db21_inverse::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0xc2dfe5e6U, 0x42914ec9U>;
    static_assert(type::k == 2U);
};

template <>
struct db21_inverse::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d87f382U, 0x3a795177U>;
    static_assert(type::k == 2U);
};

template <>
struct db21_inverse::step<24> {
    using type = StaticStep<StepType::kPredict, 0, 0xc16ae5f1U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
