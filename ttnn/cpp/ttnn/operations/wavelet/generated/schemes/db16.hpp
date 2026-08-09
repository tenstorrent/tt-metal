// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct db16_inverse;

struct db16 {
    static constexpr const char* name = "db16";
    static constexpr uint32_t tap_size = 32U;
    static constexpr int32_t delay_even = 8;
    static constexpr int32_t delay_odd = 8;
    static constexpr uint32_t num_steps = 19U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db16.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db16";
    using inverse = db16_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct db16::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0xbddf8f77U>;
    static_assert(type::k == 1U);
};

template <>
struct db16::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe77d765U, 0x3db942bfU>;
    static_assert(type::k == 2U);
};

template <>
struct db16::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xbee0fd9eU, 0x3e865e6fU>;
    static_assert(type::k == 2U);
};

template <>
struct db16::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf20f50fU, 0x3ed72a8aU>;
    static_assert(type::k == 2U);
};

template <>
struct db16::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf49576dU, 0x3f0d79eaU>;
    static_assert(type::k == 2U);
};

template <>
struct db16::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf6ebd80U, 0x3f294d66U>;
    static_assert(type::k == 2U);
};

template <>
struct db16::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf87270bU, 0x3f3b875fU>;
    static_assert(type::k == 2U);
};

template <>
struct db16::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf968533U, 0x3f463487U>;
    static_assert(type::k == 2U);
};

template <>
struct db16::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xbfa463e8U, 0x3f467d74U>;
    static_assert(type::k == 2U);
};

template <>
struct db16::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfb4ead8U, 0x3f408a22U>;
    static_assert(type::k == 2U);
};

template <>
struct db16::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xbfc7a156U, 0x3f335496U>;
    static_assert(type::k == 2U);
};

template <>
struct db16::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfe23097U, 0x3f23d03dU>;
    static_assert(type::k == 2U);
};

template <>
struct db16::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xc002a5c9U, 0x3f10d4cbU>;
    static_assert(type::k == 2U);
};

template <>
struct db16::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc01e9054U, 0x3eface9aU>;
    static_assert(type::k == 2U);
};

template <>
struct db16::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xc04e7a57U, 0x3ecea7a4U>;
    static_assert(type::k == 2U);
};

template <>
struct db16::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x2f7b29e1U, 0x3e9eb326U>;
    static_assert(type::k == 2U);
};

template <>
struct db16::step<16> {
    using type = StaticStep<StepType::kPredict, 0, 0xbe44185dU>;
    static_assert(type::k == 1U);
};

template <>
struct db16::step<17> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x432a2877U>;
    static_assert(type::k == 1U);
};

template <>
struct db16::step<18> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3bc092ebU>;
    static_assert(type::k == 1U);
};

struct db16_inverse {
    static constexpr const char* name = "db16-inverse";
    static constexpr uint32_t tap_size = 32U;
    static constexpr uint32_t num_steps = 19U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db16.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db16_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct db16_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x432a2876U>;
    static_assert(type::k == 1U);
};

template <>
struct db16_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3bc092eaU>;
    static_assert(type::k == 1U);
};

template <>
struct db16_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0x3e44185dU>;
    static_assert(type::k == 1U);
};

template <>
struct db16_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xaf7b29e1U, 0xbe9eb326U>;
    static_assert(type::k == 2U);
};

template <>
struct db16_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x404e7a57U, 0xbecea7a4U>;
    static_assert(type::k == 2U);
};

template <>
struct db16_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x401e9054U, 0xbeface9aU>;
    static_assert(type::k == 2U);
};

template <>
struct db16_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x4002a5c9U, 0xbf10d4cbU>;
    static_assert(type::k == 2U);
};

template <>
struct db16_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fe23097U, 0xbf23d03dU>;
    static_assert(type::k == 2U);
};

template <>
struct db16_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x3fc7a156U, 0xbf335496U>;
    static_assert(type::k == 2U);
};

template <>
struct db16_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fb4ead8U, 0xbf408a22U>;
    static_assert(type::k == 2U);
};

template <>
struct db16_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x3fa463e8U, 0xbf467d74U>;
    static_assert(type::k == 2U);
};

template <>
struct db16_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f968533U, 0xbf463487U>;
    static_assert(type::k == 2U);
};

template <>
struct db16_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f87270bU, 0xbf3b875fU>;
    static_assert(type::k == 2U);
};

template <>
struct db16_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f6ebd80U, 0xbf294d66U>;
    static_assert(type::k == 2U);
};

template <>
struct db16_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f49576dU, 0xbf0d79eaU>;
    static_assert(type::k == 2U);
};

template <>
struct db16_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f20f50fU, 0xbed72a8aU>;
    static_assert(type::k == 2U);
};

template <>
struct db16_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ee0fd9eU, 0xbe865e6fU>;
    static_assert(type::k == 2U);
};

template <>
struct db16_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e77d765U, 0xbdb942bfU>;
    static_assert(type::k == 2U);
};

template <>
struct db16_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ddf8f77U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
