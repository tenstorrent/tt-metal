// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct db14_inverse;

struct db14 {
    static constexpr const char* name = "db14";
    static constexpr uint32_t tap_size = 28U;
    static constexpr int32_t delay_even = 7;
    static constexpr int32_t delay_odd = 7;
    static constexpr uint32_t num_steps = 17U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db14.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db14";
    using inverse = db14_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct db14::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0xbdd443ceU>;
    static_assert(type::k == 1U);
};

template <>
struct db14::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbea1af73U, 0x3dd1ec8fU>;
    static_assert(type::k == 2U);
};

template <>
struct db14::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf042ffbU, 0x3e95f1b0U>;
    static_assert(type::k == 2U);
};

template <>
struct db14::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf3706f9U, 0x3eeea0a7U>;
    static_assert(type::k == 2U);
};

template <>
struct db14::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf616371U, 0x3f18f8f9U>;
    static_assert(type::k == 2U);
};

template <>
struct db14::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf85c08dU, 0x3f33c135U>;
    static_assert(type::k == 2U);
};

template <>
struct db14::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf96024eU, 0x3f3f917cU>;
    static_assert(type::k == 2U);
};

template <>
struct db14::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfa8df46U, 0x3f4423c1U>;
    static_assert(type::k == 2U);
};

template <>
struct db14::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xbfb9fad4U, 0x3f3adfe7U>;
    static_assert(type::k == 2U);
};

template <>
struct db14::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfd42f87U, 0x3f2e8093U>;
    static_assert(type::k == 2U);
};

template <>
struct db14::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xbff382afU, 0x3f1a2db1U>;
    static_assert(type::k == 2U);
};

template <>
struct db14::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc014fa64U, 0x3f068b01U>;
    static_assert(type::k == 2U);
};

template <>
struct db14::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0412486U, 0x3edbf332U>;
    static_assert(type::k == 2U);
};

template <>
struct db14::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x31819ec5U, 0x3ea9a82dU>;
    static_assert(type::k == 2U);
};

template <>
struct db14::step<14> {
    using type = StaticStep<StepType::kPredict, 0, 0xbe4fab92U>;
    static_assert(type::k == 1U);
};

template <>
struct db14::step<15> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x42ab413dU>;
    static_assert(type::k == 1U);
};

template <>
struct db14::step<16> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3c3f5730U>;
    static_assert(type::k == 1U);
};

struct db14_inverse {
    static constexpr const char* name = "db14-inverse";
    static constexpr uint32_t tap_size = 28U;
    static constexpr uint32_t num_steps = 17U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db14.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db14_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct db14_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x42ab413dU>;
    static_assert(type::k == 1U);
};

template <>
struct db14_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3c3f5730U>;
    static_assert(type::k == 1U);
};

template <>
struct db14_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0x3e4fab92U>;
    static_assert(type::k == 1U);
};

template <>
struct db14_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xb1819ec5U, 0xbea9a82dU>;
    static_assert(type::k == 2U);
};

template <>
struct db14_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x40412486U, 0xbedbf332U>;
    static_assert(type::k == 2U);
};

template <>
struct db14_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x4014fa64U, 0xbf068b01U>;
    static_assert(type::k == 2U);
};

template <>
struct db14_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ff382afU, 0xbf1a2db1U>;
    static_assert(type::k == 2U);
};

template <>
struct db14_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fd42f87U, 0xbf2e8093U>;
    static_assert(type::k == 2U);
};

template <>
struct db14_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x3fb9fad4U, 0xbf3adfe7U>;
    static_assert(type::k == 2U);
};

template <>
struct db14_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fa8df46U, 0xbf4423c1U>;
    static_assert(type::k == 2U);
};

template <>
struct db14_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f96024eU, 0xbf3f917cU>;
    static_assert(type::k == 2U);
};

template <>
struct db14_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f85c08dU, 0xbf33c135U>;
    static_assert(type::k == 2U);
};

template <>
struct db14_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f616371U, 0xbf18f8f9U>;
    static_assert(type::k == 2U);
};

template <>
struct db14_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f3706f9U, 0xbeeea0a7U>;
    static_assert(type::k == 2U);
};

template <>
struct db14_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f042ffbU, 0xbe95f1b0U>;
    static_assert(type::k == 2U);
};

template <>
struct db14_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ea1af73U, 0xbdd1ec8fU>;
    static_assert(type::k == 2U);
};

template <>
struct db14_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x3dd443ceU>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
