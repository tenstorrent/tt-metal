// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct db6_inverse;

struct db6 {
    static constexpr const char* name = "db6";
    static constexpr uint32_t tap_size = 12U;
    static constexpr int32_t delay_even = 3;
    static constexpr int32_t delay_odd = 3;
    static constexpr uint32_t num_steps = 9U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db6.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db6";
    using inverse = db6_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct db6::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe66eb17U>;
    static_assert(type::k == 1U);
};

template <>
struct db6::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf3a3317U, 0x3e5bbe63U>;
    static_assert(type::k == 2U);
};

template <>
struct db6::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf9000bdU, 0x3f01cb1eU>;
    static_assert(type::k == 2U);
};

template <>
struct db6::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfccd67fU, 0x3f28d9acU>;
    static_assert(type::k == 2U);
};

template <>
struct db6::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0031911U, 0x3f170cccU>;
    static_assert(type::k == 2U);
};

template <>
struct db6::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x39ac3318U, 0x3ef8bb57U>;
    static_assert(type::k == 2U);
};

template <>
struct db6::step<6> {
    using type = StaticStep<StepType::kPredict, 0, 0xbe916742U>;
    static_assert(type::k == 1U);
};

template <>
struct db6::step<7> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x40ad8536U>;
    static_assert(type::k == 1U);
};

template <>
struct db6::step<8> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3e3cd7a8U>;
    static_assert(type::k == 1U);
};

struct db6_inverse {
    static constexpr const char* name = "db6-inverse";
    static constexpr uint32_t tap_size = 12U;
    static constexpr uint32_t num_steps = 9U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db6.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db6_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct db6_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x40ad8535U>;
    static_assert(type::k == 1U);
};

template <>
struct db6_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3e3cd7a7U>;
    static_assert(type::k == 1U);
};

template <>
struct db6_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0x3e916742U>;
    static_assert(type::k == 1U);
};

template <>
struct db6_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xb9ac3318U, 0xbef8bb57U>;
    static_assert(type::k == 2U);
};

template <>
struct db6_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x40031911U, 0xbf170cccU>;
    static_assert(type::k == 2U);
};

template <>
struct db6_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fccd67fU, 0xbf28d9acU>;
    static_assert(type::k == 2U);
};

template <>
struct db6_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f9000bdU, 0xbf01cb1eU>;
    static_assert(type::k == 2U);
};

template <>
struct db6_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f3a3317U, 0xbe5bbe63U>;
    static_assert(type::k == 2U);
};

template <>
struct db6_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e66eb17U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
