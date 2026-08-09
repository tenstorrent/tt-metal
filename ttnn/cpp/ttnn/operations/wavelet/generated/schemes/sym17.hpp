// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct sym17_inverse;

struct sym17 {
    static constexpr const char* name = "sym17";
    static constexpr uint32_t tap_size = 34U;
    static constexpr int32_t delay_even = 8;
    static constexpr int32_t delay_odd = 9;
    static constexpr uint32_t num_steps = 21U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/sym17.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::sym17";
    using inverse = sym17_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct sym17::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0xbf259de9U>;
    static_assert(type::k == 1U);
};

template <>
struct sym17::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ee98127U, 0xbe21b8d6U>;
    static_assert(type::k == 2U);
};

template <>
struct sym17::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e9aef62U, 0xbfae201bU>;
    static_assert(type::k == 2U);
};

template <>
struct sym17::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ebd8569U, 0x3da69f7dU>;
    static_assert(type::k == 2U);
};

template <>
struct sym17::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf12faa9U, 0xc08d29bfU>;
    static_assert(type::k == 2U);
};

template <>
struct sym17::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e2c7f4dU, 0x3d668c0cU>;
    static_assert(type::k == 2U);
};

template <>
struct sym17::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xc08ac954U, 0xc17ab6adU>;
    static_assert(type::k == 2U);
};

template <>
struct sym17::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d5347a0U, 0x3b871434U>;
    static_assert(type::k == 2U);
};

template <>
struct sym17::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0cb8641U, 0x427a7a67U>;
    static_assert(type::k == 2U);
};

template <>
struct sym17::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc3d2a2eU, 0xbb1c4517U>;
    static_assert(type::k == 2U);
};

template <>
struct sym17::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x42280964U, 0x433e8fabU>;
    static_assert(type::k == 2U);
};

template <>
struct sym17::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbb6f6638U, 0xbad64d4cU>;
    static_assert(type::k == 2U);
};

template <>
struct sym17::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x4340623eU, 0x4387611eU>;
    static_assert(type::k == 2U);
};

template <>
struct sym17::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbae67fe2U, 0xb986f651U>;
    static_assert(type::k == 2U);
};

template <>
struct sym17::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x42946d01U, 0xc428032cU>;
    static_assert(type::k == 2U);
};

template <>
struct sym17::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3a6fe024U, 0x398d3286U>;
    static_assert(type::k == 2U);
};

template <>
struct sym17::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xc3e1a217U, 0xc485cbd7U>;
    static_assert(type::k == 2U);
};

template <>
struct sym17::step<17> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct sym17::step<18> {
    using type = StaticStep<StepType::kPredict, 0, 0x39d661cdU>;
    static_assert(type::k == 1U);
};

template <>
struct sym17::step<19> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xbcb05cf6U>;
    static_assert(type::k == 1U);
};

template <>
struct sym17::step<20> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x4239cc68U>;
    static_assert(type::k == 1U);
};

struct sym17_inverse {
    static constexpr const char* name = "sym17-inverse";
    static constexpr uint32_t tap_size = 34U;
    static constexpr uint32_t num_steps = 21U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/sym17.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::sym17_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct sym17_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3cb05cf7U>;
    static_assert(type::k == 1U);
};

template <>
struct sym17_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xc239cc69U>;
    static_assert(type::k == 1U);
};

template <>
struct sym17_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0xb9d661cdU>;
    static_assert(type::k == 1U);
};

template <>
struct sym17_inverse::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct sym17_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x43e1a217U, 0x4485cbd7U>;
    static_assert(type::k == 2U);
};

template <>
struct sym17_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xba6fe024U, 0xb98d3286U>;
    static_assert(type::k == 2U);
};

template <>
struct sym17_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xc2946d01U, 0x4428032cU>;
    static_assert(type::k == 2U);
};

template <>
struct sym17_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ae67fe2U, 0x3986f651U>;
    static_assert(type::k == 2U);
};

template <>
struct sym17_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xc340623eU, 0xc387611eU>;
    static_assert(type::k == 2U);
};

template <>
struct sym17_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3b6f6638U, 0x3ad64d4cU>;
    static_assert(type::k == 2U);
};

template <>
struct sym17_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xc2280964U, 0xc33e8fabU>;
    static_assert(type::k == 2U);
};

template <>
struct sym17_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c3d2a2eU, 0x3b1c4517U>;
    static_assert(type::k == 2U);
};

template <>
struct sym17_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x40cb8641U, 0xc27a7a67U>;
    static_assert(type::k == 2U);
};

template <>
struct sym17_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd5347a0U, 0xbb871434U>;
    static_assert(type::k == 2U);
};

template <>
struct sym17_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x408ac954U, 0x417ab6adU>;
    static_assert(type::k == 2U);
};

template <>
struct sym17_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe2c7f4dU, 0xbd668c0cU>;
    static_assert(type::k == 2U);
};

template <>
struct sym17_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f12faa9U, 0x408d29bfU>;
    static_assert(type::k == 2U);
};

template <>
struct sym17_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbebd8569U, 0xbda69f7dU>;
    static_assert(type::k == 2U);
};

template <>
struct sym17_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe9aef62U, 0x3fae201bU>;
    static_assert(type::k == 2U);
};

template <>
struct sym17_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbee98127U, 0x3e21b8d6U>;
    static_assert(type::k == 2U);
};

template <>
struct sym17_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, 0, 0x3f259de9U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
