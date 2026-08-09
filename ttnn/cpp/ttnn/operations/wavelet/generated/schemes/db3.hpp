// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct db3_inverse;

struct db3 {
    static constexpr const char* name = "db3";
    static constexpr uint32_t tap_size = 6U;
    static constexpr int32_t delay_even = 1;
    static constexpr int32_t delay_odd = 2;
    static constexpr uint32_t num_steps = 7U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db3.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db3";
    using inverse = db3_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct db3::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0x401b3b59U>;
    static_assert(type::k == 1U);
};

template <>
struct db3::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbeb46c28U, 0xbe8836b0U>;
    static_assert(type::k == 2U);
};

template <>
struct db3::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0x40394d5fU, 0xc16ee965U>;
    static_assert(type::k == 2U);
};

template <>
struct db3::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct db3::step<4> {
    using type = StaticStep<StepType::kPredict, 0, 0x3d87a26cU>;
    static_assert(type::k == 1U);
};

template <>
struct db3::step<5> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3dab813bU>;
    static_assert(type::k == 1U);
};

template <>
struct db3::step<6> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xc13f0fccU>;
    static_assert(type::k == 1U);
};

struct db3_inverse {
    static constexpr const char* name = "db3-inverse";
    static constexpr uint32_t tap_size = 6U;
    static constexpr uint32_t num_steps = 7U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db3.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db3_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct db3_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xbdab813bU>;
    static_assert(type::k == 1U);
};

template <>
struct db3_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x413f0fccU>;
    static_assert(type::k == 1U);
};

template <>
struct db3_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0xbd87a26cU>;
    static_assert(type::k == 1U);
};

template <>
struct db3_inverse::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct db3_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0394d5fU, 0x416ee965U>;
    static_assert(type::k == 2U);
};

template <>
struct db3_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3eb46c28U, 0x3e8836b0U>;
    static_assert(type::k == 2U);
};

template <>
struct db3_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, 0, 0xc01b3b59U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
