// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct sym9_inverse;

struct sym9 {
    static constexpr const char* name = "sym9";
    static constexpr uint32_t tap_size = 18U;
    static constexpr int32_t delay_even = 4;
    static constexpr int32_t delay_odd = 5;
    static constexpr uint32_t num_steps = 13U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/sym9.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::sym9";
    using inverse = sym9_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct sym9::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0xbee283bdU>;
    static_assert(type::k == 1U);
};

template <>
struct sym9::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ebd6fc5U, 0x3e7d9b59U>;
    static_assert(type::k == 2U);
};

template <>
struct sym9::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xbea6ae79U, 0xbe9c4720U>;
    static_assert(type::k == 2U);
};

template <>
struct sym9::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e701e6fU, 0xbf09def3U>;
    static_assert(type::k == 2U);
};

template <>
struct sym9::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f096ac6U, 0xbfb7f1c4U>;
    static_assert(type::k == 2U);
};

template <>
struct sym9::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f04a224U, 0xbf45f8b4U>;
    static_assert(type::k == 2U);
};

template <>
struct sym9::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f8f6e1aU, 0x3cd45c6fU>;
    static_assert(type::k == 2U);
};

template <>
struct sym9::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe08ccf5U, 0x4218553cU>;
    static_assert(type::k == 2U);
};

template <>
struct sym9::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xbcd65948U, 0xba9bcd0cU>;
    static_assert(type::k == 2U);
};

template <>
struct sym9::step<9> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct sym9::step<10> {
    using type = StaticStep<StepType::kPredict, 0, 0x439ae296U>;
    static_assert(type::k == 1U);
};

template <>
struct sym9::step<11> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xc1a12620U>;
    static_assert(type::k == 1U);
};

template <>
struct sym9::step<12> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3d4b5701U>;
    static_assert(type::k == 1U);
};

struct sym9_inverse {
    static constexpr const char* name = "sym9-inverse";
    static constexpr uint32_t tap_size = 18U;
    static constexpr uint32_t num_steps = 13U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/sym9.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::sym9_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct sym9_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x41a12620U>;
    static_assert(type::k == 1U);
};

template <>
struct sym9_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xbd4b5701U>;
    static_assert(type::k == 1U);
};

template <>
struct sym9_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0xc39ae296U>;
    static_assert(type::k == 1U);
};

template <>
struct sym9_inverse::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct sym9_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x3cd65948U, 0x3a9bcd0cU>;
    static_assert(type::k == 2U);
};

template <>
struct sym9_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e08ccf5U, 0xc218553cU>;
    static_assert(type::k == 2U);
};

template <>
struct sym9_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf8f6e1aU, 0xbcd45c6fU>;
    static_assert(type::k == 2U);
};

template <>
struct sym9_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf04a224U, 0x3f45f8b4U>;
    static_assert(type::k == 2U);
};

template <>
struct sym9_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf096ac6U, 0x3fb7f1c4U>;
    static_assert(type::k == 2U);
};

template <>
struct sym9_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe701e6fU, 0x3f09def3U>;
    static_assert(type::k == 2U);
};

template <>
struct sym9_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ea6ae79U, 0x3e9c4720U>;
    static_assert(type::k == 2U);
};

template <>
struct sym9_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbebd6fc5U, 0xbe7d9b59U>;
    static_assert(type::k == 2U);
};

template <>
struct sym9_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, 0, 0x3ee283bdU>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
