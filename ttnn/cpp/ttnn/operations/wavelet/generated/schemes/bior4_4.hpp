// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct bior4_4_inverse;

struct bior4_4 {
    static constexpr const char* name = "bior4.4";
    static constexpr uint32_t tap_size = 10U;
    static constexpr int32_t delay_even = 2;
    static constexpr int32_t delay_odd = 3;
    static constexpr uint32_t num_steps = 7U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/bior4_4.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::bior4_4";
    using inverse = bior4_4_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct bior4_4::step<0> {
    using type = StaticStep<StepType::kUpdate, -1, 0xbfcb0673U, 0xbfcb0673U>;
    static_assert(type::k == 2U);
};

template <>
struct bior4_4::step<1> {
    using type = StaticStep<StepType::kPredict, 0, 0xbd5901aeU, 0xbd5901aeU>;
    static_assert(type::k == 2U);
};

template <>
struct bior4_4::step<2> {
    using type = StaticStep<StepType::kUpdate, -1, 0x3f620676U, 0x3f620676U>;
    static_assert(type::k == 2U);
};

template <>
struct bior4_4::step<3> {
    using type = StaticStep<StepType::kPredict, 0, 0x3ee31355U, 0x3ee31355U>;
    static_assert(type::k == 2U);
};

template <>
struct bior4_4::step<4> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct bior4_4::step<5> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3f93263dU>;
    static_assert(type::k == 1U);
};

template <>
struct bior4_4::step<6> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xbf5eaf70U>;
    static_assert(type::k == 1U);
};

struct bior4_4_inverse {
    static constexpr const char* name = "bior4.4-inverse";
    static constexpr uint32_t tap_size = 10U;
    static constexpr uint32_t num_steps = 7U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/bior4_4.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::bior4_4_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct bior4_4_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xbf93263dU>;
    static_assert(type::k == 1U);
};

template <>
struct bior4_4_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3f5eaf6fU>;
    static_assert(type::k == 1U);
};

template <>
struct bior4_4_inverse::step<2> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct bior4_4_inverse::step<3> {
    using type = StaticStep<StepType::kPredict, 0, 0xbee31355U, 0xbee31355U>;
    static_assert(type::k == 2U);
};

template <>
struct bior4_4_inverse::step<4> {
    using type = StaticStep<StepType::kUpdate, -1, 0xbf620676U, 0xbf620676U>;
    static_assert(type::k == 2U);
};

template <>
struct bior4_4_inverse::step<5> {
    using type = StaticStep<StepType::kPredict, 0, 0x3d5901aeU, 0x3d5901aeU>;
    static_assert(type::k == 2U);
};

template <>
struct bior4_4_inverse::step<6> {
    using type = StaticStep<StepType::kUpdate, -1, 0x3fcb0673U, 0x3fcb0673U>;
    static_assert(type::k == 2U);
};

}  // namespace ttnn::operations::wavelet::schemes
