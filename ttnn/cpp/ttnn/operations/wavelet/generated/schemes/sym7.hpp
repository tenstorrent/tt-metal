// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct sym7_inverse;

struct sym7 {
    static constexpr const char* name = "sym7";
    static constexpr uint32_t tap_size = 14U;
    static constexpr int32_t delay_even = 3;
    static constexpr int32_t delay_odd = 4;
    static constexpr uint32_t num_steps = 11U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/sym7.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::sym7";
    using inverse = sym7_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct sym7::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0x3ec7f647U>;
    static_assert(type::k == 1U);
};

template <>
struct sym7::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbead7f93U, 0xbdd6b5a8U>;
    static_assert(type::k == 2U);
};

template <>
struct sym7::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e0c8cd3U, 0x3fad571eU>;
    static_assert(type::k == 2U);
};

template <>
struct sym7::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbedded1eU, 0xc2c039d6U>;
    static_assert(type::k == 2U);
};

template <>
struct sym7::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x3c2a76b1U, 0xb7113a57U>;
    static_assert(type::k == 2U);
};

template <>
struct sym7::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x45a69178U, 0x4685ce71U>;
    static_assert(type::k == 2U);
};

template <>
struct sym7::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xb7a3231eU, 0xb8dcf677U>;
    static_assert(type::k == 2U);
};

template <>
struct sym7::step<7> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct sym7::step<8> {
    using type = StaticStep<StepType::kPredict, 0, 0x460ad39eU>;
    static_assert(type::k == 1U);
};

template <>
struct sym7::step<9> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x4240b00fU>;
    static_assert(type::k == 1U);
};

template <>
struct sym7::step<10> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xbcaa0ebaU>;
    static_assert(type::k == 1U);
};

struct sym7_inverse {
    static constexpr const char* name = "sym7-inverse";
    static constexpr uint32_t tap_size = 14U;
    static constexpr uint32_t num_steps = 11U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/sym7.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::sym7_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct sym7_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xc240b010U>;
    static_assert(type::k == 1U);
};

template <>
struct sym7_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3caa0ebbU>;
    static_assert(type::k == 1U);
};

template <>
struct sym7_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0xc60ad39eU>;
    static_assert(type::k == 1U);
};

template <>
struct sym7_inverse::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct sym7_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x37a3231eU, 0x38dcf677U>;
    static_assert(type::k == 2U);
};

template <>
struct sym7_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc5a69178U, 0xc685ce71U>;
    static_assert(type::k == 2U);
};

template <>
struct sym7_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xbc2a76b1U, 0x37113a57U>;
    static_assert(type::k == 2U);
};

template <>
struct sym7_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3edded1eU, 0x42c039d6U>;
    static_assert(type::k == 2U);
};

template <>
struct sym7_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe0c8cd3U, 0xbfad571eU>;
    static_assert(type::k == 2U);
};

template <>
struct sym7_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ead7f93U, 0x3dd6b5a8U>;
    static_assert(type::k == 2U);
};

template <>
struct sym7_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, 0, 0xbec7f647U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
