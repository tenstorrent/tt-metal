// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct db9_inverse;

struct db9 {
    static constexpr const char* name = "db9";
    static constexpr uint32_t tap_size = 18U;
    static constexpr int32_t delay_even = 4;
    static constexpr int32_t delay_odd = 5;
    static constexpr uint32_t num_steps = 13U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db9.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db9";
    using inverse = db9_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct db9::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0x40ccea05U>;
    static_assert(type::k == 1U);
};

template <>
struct db9::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe1c1a9fU, 0xbc420148U>;
    static_assert(type::k == 2U);
};

template <>
struct db9::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0x4185ff09U, 0xc1fea509U>;
    static_assert(type::k == 2U);
};

template <>
struct db9::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c72e273U, 0xbcd5cf06U>;
    static_assert(type::k == 2U);
};

template <>
struct db9::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x41e13389U, 0xc252b315U>;
    static_assert(type::k == 2U);
};

template <>
struct db9::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c8cfb44U, 0xbd1c8988U>;
    static_assert(type::k == 2U);
};

template <>
struct db9::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x41cd0aeaU, 0xc298b48dU>;
    static_assert(type::k == 2U);
};

template <>
struct db9::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c56142fU, 0xbd7ab33bU>;
    static_assert(type::k == 2U);
};

template <>
struct db9::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x4182b122U, 0xc323d51fU>;
    static_assert(type::k == 2U);
};

template <>
struct db9::step<9> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct db9::step<10> {
    using type = StaticStep<StepType::kPredict, 0, 0x3bc8024aU>;
    static_assert(type::k == 1U);
};

template <>
struct db9::step<11> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3b2496c1U>;
    static_assert(type::k == 1U);
};

template <>
struct db9::step<12> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xc3c7170aU>;
    static_assert(type::k == 1U);
};

struct db9_inverse {
    static constexpr const char* name = "db9-inverse";
    static constexpr uint32_t tap_size = 18U;
    static constexpr uint32_t num_steps = 13U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db9.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db9_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct db9_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xbb2496c1U>;
    static_assert(type::k == 1U);
};

template <>
struct db9_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x43c7170aU>;
    static_assert(type::k == 1U);
};

template <>
struct db9_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0xbbc8024aU>;
    static_assert(type::k == 1U);
};

template <>
struct db9_inverse::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct db9_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xc182b122U, 0x4323d51fU>;
    static_assert(type::k == 2U);
};

template <>
struct db9_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc56142fU, 0x3d7ab33bU>;
    static_assert(type::k == 2U);
};

template <>
struct db9_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xc1cd0aeaU, 0x4298b48dU>;
    static_assert(type::k == 2U);
};

template <>
struct db9_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc8cfb44U, 0x3d1c8988U>;
    static_assert(type::k == 2U);
};

template <>
struct db9_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xc1e13389U, 0x4252b315U>;
    static_assert(type::k == 2U);
};

template <>
struct db9_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc72e273U, 0x3cd5cf06U>;
    static_assert(type::k == 2U);
};

template <>
struct db9_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xc185ff09U, 0x41fea509U>;
    static_assert(type::k == 2U);
};

template <>
struct db9_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e1c1a9fU, 0x3c420148U>;
    static_assert(type::k == 2U);
};

template <>
struct db9_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, 0, 0xc0ccea05U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
