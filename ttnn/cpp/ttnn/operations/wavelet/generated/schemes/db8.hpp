// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct db8_inverse;

struct db8 {
    static constexpr const char* name = "db8";
    static constexpr uint32_t tap_size = 16U;
    static constexpr int32_t delay_even = 4;
    static constexpr int32_t delay_odd = 4;
    static constexpr uint32_t num_steps = 11U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db8.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db8";
    using inverse = db8_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct db8::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe32191aU>;
    static_assert(type::k == 1U);
};

template <>
struct db8::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf0b94daU, 0x3e2cde6dU>;
    static_assert(type::k == 2U);
};

template <>
struct db8::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf5d423bU, 0x3ee13c51U>;
    static_assert(type::k == 2U);
};

template <>
struct db8::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf98e0d3U, 0x3f22a776U>;
    static_assert(type::k == 2U);
};

template <>
struct db8::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xbfb7a0a3U, 0x3f2dedd9U>;
    static_assert(type::k == 2U);
};

template <>
struct db8::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfe86c9dU, 0x3f299c06U>;
    static_assert(type::k == 2U);
};

template <>
struct db8::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xc014b555U, 0x3f0c031eU>;
    static_assert(type::k == 2U);
};

template <>
struct db8::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3798bf5cU, 0x3edc4421U>;
    static_assert(type::k == 2U);
};

template <>
struct db8::step<8> {
    using type = StaticStep<StepType::kPredict, 0, 0xbe830ccfU>;
    static_assert(type::k == 1U);
};

template <>
struct db8::step<9> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x412e3785U>;
    static_assert(type::k == 1U);
};

template <>
struct db8::step<10> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3dbc1660U>;
    static_assert(type::k == 1U);
};

struct db8_inverse {
    static constexpr const char* name = "db8-inverse";
    static constexpr uint32_t tap_size = 16U;
    static constexpr uint32_t num_steps = 11U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db8.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db8_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct db8_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x412e3785U>;
    static_assert(type::k == 1U);
};

template <>
struct db8_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3dbc1660U>;
    static_assert(type::k == 1U);
};

template <>
struct db8_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0x3e830ccfU>;
    static_assert(type::k == 1U);
};

template <>
struct db8_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xb798bf5cU, 0xbedc4421U>;
    static_assert(type::k == 2U);
};

template <>
struct db8_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x4014b555U, 0xbf0c031eU>;
    static_assert(type::k == 2U);
};

template <>
struct db8_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fe86c9dU, 0xbf299c06U>;
    static_assert(type::k == 2U);
};

template <>
struct db8_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x3fb7a0a3U, 0xbf2dedd9U>;
    static_assert(type::k == 2U);
};

template <>
struct db8_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f98e0d3U, 0xbf22a776U>;
    static_assert(type::k == 2U);
};

template <>
struct db8_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f5d423bU, 0xbee13c51U>;
    static_assert(type::k == 2U);
};

template <>
struct db8_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f0b94daU, 0xbe2cde6dU>;
    static_assert(type::k == 2U);
};

template <>
struct db8_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e32191aU>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
