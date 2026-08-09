// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct sym18_inverse;

struct sym18 {
    static constexpr const char* name = "sym18";
    static constexpr uint32_t tap_size = 36U;
    static constexpr int32_t delay_even = 9;
    static constexpr int32_t delay_odd = 9;
    static constexpr uint32_t num_steps = 21U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/sym18.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::sym18";
    using inverse = sym18_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct sym18::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ff6d0ceU>;
    static_assert(type::k == 1U);
};

template <>
struct sym18::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd985febU, 0xbed13f96U>;
    static_assert(type::k == 2U);
};

template <>
struct sym18::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0x410d5fefU, 0x3fbcbdb4U>;
    static_assert(type::k == 2U);
};

template <>
struct sym18::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3be20605U, 0xbdbd0e26U>;
    static_assert(type::k == 2U);
};

template <>
struct sym18::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xc1b7e9acU, 0xc05fb30cU>;
    static_assert(type::k == 2U);
};

template <>
struct sym18::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d5d60aaU, 0x3cb816ecU>;
    static_assert(type::k == 2U);
};

template <>
struct sym18::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x4095c2fbU, 0xc17ed169U>;
    static_assert(type::k == 2U);
};

template <>
struct sym18::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d514854U, 0xbd984dc5U>;
    static_assert(type::k == 2U);
};

template <>
struct sym18::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f692132U, 0xc07c1d95U>;
    static_assert(type::k == 2U);
};

template <>
struct sym18::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd09be8aU, 0xbc6f11deU>;
    static_assert(type::k == 2U);
};

template <>
struct sym18::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0b84a4eU, 0x3ffe1626U>;
    static_assert(type::k == 2U);
};

template <>
struct sym18::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbddfd137U, 0x3d85a491U>;
    static_assert(type::k == 2U);
};

template <>
struct sym18::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0f3e54dU, 0x40b7e7c8U>;
    static_assert(type::k == 2U);
};

template <>
struct sym18::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c719bf6U, 0x3dc905b4U>;
    static_assert(type::k == 2U);
};

template <>
struct sym18::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xc16ce41fU, 0xc0885544U>;
    static_assert(type::k == 2U);
};

template <>
struct sym18::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc1cdde1U, 0x3cf78b8cU>;
    static_assert(type::k == 2U);
};

template <>
struct sym18::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x41e55398U, 0x40fb3b59U>;
    static_assert(type::k == 2U);
};

template <>
struct sym18::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3cf68319U, 0xbc949e47U>;
    static_assert(type::k == 2U);
};

template <>
struct sym18::step<18> {
    using type = StaticStep<StepType::kPredict, 0, 0xc1c7133eU>;
    static_assert(type::k == 1U);
};

template <>
struct sym18::step<19> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x4072f793U>;
    static_assert(type::k == 1U);
};

template <>
struct sym18::step<20> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3e86ddb2U>;
    static_assert(type::k == 1U);
};

struct sym18_inverse {
    static constexpr const char* name = "sym18-inverse";
    static constexpr uint32_t tap_size = 36U;
    static constexpr uint32_t num_steps = 21U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/sym18.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::sym18_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct sym18_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x4072f794U>;
    static_assert(type::k == 1U);
};

template <>
struct sym18_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3e86ddb2U>;
    static_assert(type::k == 1U);
};

template <>
struct sym18_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0x41c7133eU>;
    static_assert(type::k == 1U);
};

template <>
struct sym18_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbcf68319U, 0x3c949e47U>;
    static_assert(type::k == 2U);
};

template <>
struct sym18_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xc1e55398U, 0xc0fb3b59U>;
    static_assert(type::k == 2U);
};

template <>
struct sym18_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c1cdde1U, 0xbcf78b8cU>;
    static_assert(type::k == 2U);
};

template <>
struct sym18_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x416ce41fU, 0x40885544U>;
    static_assert(type::k == 2U);
};

template <>
struct sym18_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc719bf6U, 0xbdc905b4U>;
    static_assert(type::k == 2U);
};

template <>
struct sym18_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x40f3e54dU, 0xc0b7e7c8U>;
    static_assert(type::k == 2U);
};

template <>
struct sym18_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ddfd137U, 0xbd85a491U>;
    static_assert(type::k == 2U);
};

template <>
struct sym18_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x40b84a4eU, 0xbffe1626U>;
    static_assert(type::k == 2U);
};

template <>
struct sym18_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d09be8aU, 0x3c6f11deU>;
    static_assert(type::k == 2U);
};

template <>
struct sym18_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf692132U, 0x407c1d95U>;
    static_assert(type::k == 2U);
};

template <>
struct sym18_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd514854U, 0x3d984dc5U>;
    static_assert(type::k == 2U);
};

template <>
struct sym18_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xc095c2fbU, 0x417ed169U>;
    static_assert(type::k == 2U);
};

template <>
struct sym18_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd5d60aaU, 0xbcb816ecU>;
    static_assert(type::k == 2U);
};

template <>
struct sym18_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x41b7e9acU, 0x405fb30cU>;
    static_assert(type::k == 2U);
};

template <>
struct sym18_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbbe20605U, 0x3dbd0e26U>;
    static_assert(type::k == 2U);
};

template <>
struct sym18_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xc10d5fefU, 0xbfbcbdb4U>;
    static_assert(type::k == 2U);
};

template <>
struct sym18_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d985febU, 0x3ed13f96U>;
    static_assert(type::k == 2U);
};

template <>
struct sym18_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0xbff6d0ceU>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
