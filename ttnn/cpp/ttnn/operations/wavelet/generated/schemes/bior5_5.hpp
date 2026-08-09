// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct bior5_5_inverse;

struct bior5_5 {
    static constexpr const char* name = "bior5.5";
    static constexpr uint32_t tap_size = 12U;
    static constexpr int32_t delay_even = 3;
    static constexpr int32_t delay_odd = 3;
    static constexpr uint32_t num_steps = 7U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/bior5_5.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::bior5_5";
    using inverse = bior5_5_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct bior5_5::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0x409fc8e8U, 0x409fc8e8U>;
    static_assert(type::k == 2U);
};

template <>
struct bior5_5::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbb8f1ccaU, 0xbb8f1ccaU>;
    static_assert(type::k == 2U);
};

template <>
struct bior5_5::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0b2bec3U, 0xc0b2bec3U>;
    static_assert(type::k == 2U);
};

template <>
struct bior5_5::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3eb457aeU, 0x3eb457aeU>;
    static_assert(type::k == 2U);
};

template <>
struct bior5_5::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe948714U, 0xbe948714U>;
    static_assert(type::k == 2U);
};

template <>
struct bior5_5::step<5> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3f6cca4eU>;
    static_assert(type::k == 1U);
};

template <>
struct bior5_5::step<6> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3f8a6253U>;
    static_assert(type::k == 1U);
};

struct bior5_5_inverse {
    static constexpr const char* name = "bior5.5-inverse";
    static constexpr uint32_t tap_size = 12U;
    static constexpr uint32_t num_steps = 7U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/bior5_5.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::bior5_5_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct bior5_5_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3f6cca4eU>;
    static_assert(type::k == 1U);
};

template <>
struct bior5_5_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3f8a6253U>;
    static_assert(type::k == 1U);
};

template <>
struct bior5_5_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e948714U, 0x3e948714U>;
    static_assert(type::k == 2U);
};

template <>
struct bior5_5_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbeb457aeU, 0xbeb457aeU>;
    static_assert(type::k == 2U);
};

template <>
struct bior5_5_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x40b2bec3U, 0x40b2bec3U>;
    static_assert(type::k == 2U);
};

template <>
struct bior5_5_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3b8f1ccaU, 0x3b8f1ccaU>;
    static_assert(type::k == 2U);
};

template <>
struct bior5_5_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xc09fc8e8U, 0xc09fc8e8U>;
    static_assert(type::k == 2U);
};

}  // namespace ttnn::operations::wavelet::schemes
