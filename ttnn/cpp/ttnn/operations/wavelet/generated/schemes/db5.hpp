// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct db5_inverse;

struct db5 {
    static constexpr const char* name = "db5";
    static constexpr uint32_t tap_size = 10U;
    static constexpr int32_t delay_even = 2;
    static constexpr int32_t delay_odd = 3;
    static constexpr uint32_t num_steps = 9U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db5.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db5";
    using inverse = db5_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct db5::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0x40716092U>;
    static_assert(type::k == 1U);
};

template <>
struct db5::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe7dacbfU, 0xbd7cdf7cU>;
    static_assert(type::k == 2U);
};

template <>
struct db5::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0x40f31f60U, 0xc198381cU>;
    static_assert(type::k == 2U);
};

template <>
struct db5::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d365b58U, 0xbe116b63U>;
    static_assert(type::k == 2U);
};

template <>
struct db5::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x40dd682fU, 0xc230ea09U>;
    static_assert(type::k == 2U);
};

template <>
struct db5::step<5> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct db5::step<6> {
    using type = StaticStep<StepType::kPredict, 0, 0x3cb923adU>;
    static_assert(type::k == 1U);
};

template <>
struct db5::step<7> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3cb1c33eU>;
    static_assert(type::k == 1U);
};

template <>
struct db5::step<8> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xc23855eeU>;
    static_assert(type::k == 1U);
};

struct db5_inverse {
    static constexpr const char* name = "db5-inverse";
    static constexpr uint32_t tap_size = 10U;
    static constexpr uint32_t num_steps = 9U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db5.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db5_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct db5_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xbcb1c33eU>;
    static_assert(type::k == 1U);
};

template <>
struct db5_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x423855efU>;
    static_assert(type::k == 1U);
};

template <>
struct db5_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0xbcb923adU>;
    static_assert(type::k == 1U);
};

template <>
struct db5_inverse::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct db5_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0dd682fU, 0x4230ea09U>;
    static_assert(type::k == 2U);
};

template <>
struct db5_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd365b58U, 0x3e116b63U>;
    static_assert(type::k == 2U);
};

template <>
struct db5_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0f31f60U, 0x4198381cU>;
    static_assert(type::k == 2U);
};

template <>
struct db5_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e7dacbfU, 0x3d7cdf7cU>;
    static_assert(type::k == 2U);
};

template <>
struct db5_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, 0, 0xc0716092U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
