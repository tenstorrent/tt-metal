// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct db4_inverse;

struct db4 {
    static constexpr const char* name = "db4";
    static constexpr uint32_t tap_size = 8U;
    static constexpr int32_t delay_even = 2;
    static constexpr int32_t delay_odd = 2;
    static constexpr uint32_t num_steps = 7U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db4.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db4";
    using inverse = db4_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct db4::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0xbea50158U>;
    static_assert(type::k == 1U);
};

template <>
struct db4::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf8efde8U, 0x3e957ae1U>;
    static_assert(type::k == 2U);
};

template <>
struct db4::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xbfd82e6fU, 0x3f0a3f4bU>;
    static_assert(type::k == 2U);
};

template <>
struct db4::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3bd8d641U, 0x3f0e0706U>;
    static_assert(type::k == 2U);
};

template <>
struct db4::step<4> {
    using type = StaticStep<StepType::kPredict, 0, 0xbea3600dU>;
    static_assert(type::k == 1U);
};

template <>
struct db4::step<5> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x40288fc6U>;
    static_assert(type::k == 1U);
};

template <>
struct db4::step<6> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3ec265d3U>;
    static_assert(type::k == 1U);
};

struct db4_inverse {
    static constexpr const char* name = "db4-inverse";
    static constexpr uint32_t tap_size = 8U;
    static constexpr uint32_t num_steps = 7U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db4.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db4_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct db4_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x40288fc6U>;
    static_assert(type::k == 1U);
};

template <>
struct db4_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3ec265d3U>;
    static_assert(type::k == 1U);
};

template <>
struct db4_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0x3ea3600dU>;
    static_assert(type::k == 1U);
};

template <>
struct db4_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbbd8d641U, 0xbf0e0706U>;
    static_assert(type::k == 2U);
};

template <>
struct db4_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x3fd82e6fU, 0xbf0a3f4bU>;
    static_assert(type::k == 2U);
};

template <>
struct db4_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f8efde8U, 0xbe957ae1U>;
    static_assert(type::k == 2U);
};

template <>
struct db4_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ea50158U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
