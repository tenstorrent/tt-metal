// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct coif4_inverse;

struct coif4 {
    static constexpr const char* name = "coif4";
    static constexpr uint32_t tap_size = 24U;
    static constexpr int32_t delay_even = 6;
    static constexpr int32_t delay_odd = 6;
    static constexpr uint32_t num_steps = 15U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/coif4.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::coif4";
    using inverse = coif4_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct coif4::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f0c2fabU>;
    static_assert(type::k == 1U);
};

template <>
struct coif4::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f8f868cU, 0xbed7b164U>;
    static_assert(type::k == 2U);
};

template <>
struct coif4::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ea475e6U, 0xbf1b3c40U>;
    static_assert(type::k == 2U);
};

template <>
struct coif4::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e49e6b8U, 0xbf94d358U>;
    static_assert(type::k == 2U);
};

template <>
struct coif4::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xbde4addcU, 0xbdaf0dceU>;
    static_assert(type::k == 2U);
};

template <>
struct coif4::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbee15fe9U, 0x3e8252d9U>;
    static_assert(type::k == 2U);
};

template <>
struct coif4::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xbeff532aU, 0x3e3b2502U>;
    static_assert(type::k == 2U);
};

template <>
struct coif4::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc0441ae4U, 0x3f4a7d06U>;
    static_assert(type::k == 2U);
};

template <>
struct coif4::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf88ad1bU, 0x3e979639U>;
    static_assert(type::k == 2U);
};

template <>
struct coif4::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf8599ddU, 0x3f6de0d6U>;
    static_assert(type::k == 2U);
};

template <>
struct coif4::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xc081d1c9U, 0x3f73bf89U>;
    static_assert(type::k == 2U);
};

template <>
struct coif4::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x37193d6cU, 0x3e7c534fU>;
    static_assert(type::k == 2U);
};

template <>
struct coif4::step<12> {
    using type = StaticStep<StepType::kPredict, 0, 0xbee04fecU>;
    static_assert(type::k == 1U);
};

template <>
struct coif4::step<13> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x416cc8b1U>;
    static_assert(type::k == 1U);
};

template <>
struct coif4::step<14> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3d8a6344U>;
    static_assert(type::k == 1U);
};

struct coif4_inverse {
    static constexpr const char* name = "coif4-inverse";
    static constexpr uint32_t tap_size = 24U;
    static constexpr uint32_t num_steps = 15U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/coif4.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::coif4_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct coif4_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x416cc8b1U>;
    static_assert(type::k == 1U);
};

template <>
struct coif4_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3d8a6344U>;
    static_assert(type::k == 1U);
};

template <>
struct coif4_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0x3ee04fecU>;
    static_assert(type::k == 1U);
};

template <>
struct coif4_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xb7193d6cU, 0xbe7c534fU>;
    static_assert(type::k == 2U);
};

template <>
struct coif4_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x4081d1c9U, 0xbf73bf89U>;
    static_assert(type::k == 2U);
};

template <>
struct coif4_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f8599ddU, 0xbf6de0d6U>;
    static_assert(type::k == 2U);
};

template <>
struct coif4_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f88ad1bU, 0xbe979639U>;
    static_assert(type::k == 2U);
};

template <>
struct coif4_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x40441ae4U, 0xbf4a7d06U>;
    static_assert(type::k == 2U);
};

template <>
struct coif4_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x3eff532aU, 0xbe3b2502U>;
    static_assert(type::k == 2U);
};

template <>
struct coif4_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ee15fe9U, 0xbe8252d9U>;
    static_assert(type::k == 2U);
};

template <>
struct coif4_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x3de4addcU, 0x3daf0dceU>;
    static_assert(type::k == 2U);
};

template <>
struct coif4_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe49e6b8U, 0x3f94d358U>;
    static_assert(type::k == 2U);
};

template <>
struct coif4_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xbea475e6U, 0x3f1b3c40U>;
    static_assert(type::k == 2U);
};

template <>
struct coif4_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf8f868cU, 0x3ed7b164U>;
    static_assert(type::k == 2U);
};

template <>
struct coif4_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf0c2fabU>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
