// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct sym10_inverse;

struct sym10 {
    static constexpr const char* name = "sym10";
    static constexpr uint32_t tap_size = 20U;
    static constexpr int32_t delay_even = 5;
    static constexpr int32_t delay_odd = 5;
    static constexpr uint32_t num_steps = 13U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/sym10.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::sym10";
    using inverse = sym10_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct sym10::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0x4100da5eU>;
    static_assert(type::k == 1U);
};

template <>
struct sym10::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3cf7ed7aU, 0xbdfa7199U>;
    static_assert(type::k == 2U);
};

template <>
struct sym10::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0x410c2659U, 0xc1d32cdeU>;
    static_assert(type::k == 2U);
};

template <>
struct sym10::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c91e6a7U, 0xbd0e4c87U>;
    static_assert(type::k == 2U);
};

template <>
struct sym10::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x3fbd9f46U, 0xc0b94194U>;
    static_assert(type::k == 2U);
};

template <>
struct sym10::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc33740aU, 0xbba5437eU>;
    static_assert(type::k == 2U);
};

template <>
struct sym10::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xc115fd09U, 0x40485726U>;
    static_assert(type::k == 2U);
};

template <>
struct sym10::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd87c6b7U, 0x3cd30b98U>;
    static_assert(type::k == 2U);
};

template <>
struct sym10::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xc151faf2U, 0x4123b39cU>;
    static_assert(type::k == 2U);
};

template <>
struct sym10::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d5a17b0U, 0x3d724591U>;
    static_assert(type::k == 2U);
};

template <>
struct sym10::step<10> {
    using type = StaticStep<StepType::kPredict, 0, 0xc15da752U>;
    static_assert(type::k == 1U);
};

template <>
struct sym10::step<11> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x4037f660U>;
    static_assert(type::k == 1U);
};

template <>
struct sym10::step<12> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3eb21f94U>;
    static_assert(type::k == 1U);
};

struct sym10_inverse {
    static constexpr const char* name = "sym10-inverse";
    static constexpr uint32_t tap_size = 20U;
    static constexpr uint32_t num_steps = 13U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/sym10.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::sym10_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct sym10_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x4037f660U>;
    static_assert(type::k == 1U);
};

template <>
struct sym10_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3eb21f94U>;
    static_assert(type::k == 1U);
};

template <>
struct sym10_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0x415da752U>;
    static_assert(type::k == 1U);
};

template <>
struct sym10_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd5a17b0U, 0xbd724591U>;
    static_assert(type::k == 2U);
};

template <>
struct sym10_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x4151faf2U, 0xc123b39cU>;
    static_assert(type::k == 2U);
};

template <>
struct sym10_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d87c6b7U, 0xbcd30b98U>;
    static_assert(type::k == 2U);
};

template <>
struct sym10_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x4115fd09U, 0xc0485726U>;
    static_assert(type::k == 2U);
};

template <>
struct sym10_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c33740aU, 0x3ba5437eU>;
    static_assert(type::k == 2U);
};

template <>
struct sym10_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xbfbd9f46U, 0x40b94194U>;
    static_assert(type::k == 2U);
};

template <>
struct sym10_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc91e6a7U, 0x3d0e4c87U>;
    static_assert(type::k == 2U);
};

template <>
struct sym10_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xc10c2659U, 0x41d32cdeU>;
    static_assert(type::k == 2U);
};

template <>
struct sym10_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbcf7ed7aU, 0x3dfa7199U>;
    static_assert(type::k == 2U);
};

template <>
struct sym10_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xc100da5eU>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
