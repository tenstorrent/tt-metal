// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct db10_inverse;

struct db10 {
    static constexpr const char* name = "db10";
    static constexpr uint32_t tap_size = 20U;
    static constexpr int32_t delay_even = 5;
    static constexpr int32_t delay_odd = 5;
    static constexpr uint32_t num_steps = 13U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db10.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db10";
    using inverse = db10_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct db10::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe112156U>;
    static_assert(type::k == 1U);
};

template <>
struct db10::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbee0419fU, 0x3e0e45bbU>;
    static_assert(type::k == 2U);
};

template <>
struct db10::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf347442U, 0x3ec28606U>;
    static_assert(type::k == 2U);
};

template <>
struct db10::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf78c6beU, 0x3f13eb41U>;
    static_assert(type::k == 2U);
};

template <>
struct db10::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf95e6e8U, 0x3f2df1dfU>;
    static_assert(type::k == 2U);
};

template <>
struct db10::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfb3b948U, 0x3f3a53d6U>;
    static_assert(type::k == 2U);
};

template <>
struct db10::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xbfce6c35U, 0x3f2e3788U>;
    static_assert(type::k == 2U);
};

template <>
struct db10::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfff52e4U, 0x3f1d7005U>;
    static_assert(type::k == 2U);
};

template <>
struct db10::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xc024aa8eU, 0x3f003cedU>;
    static_assert(type::k == 2U);
};

template <>
struct db10::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x358dc38aU, 0x3ec6fd94U>;
    static_assert(type::k == 2U);
};

template <>
struct db10::step<10> {
    using type = StaticStep<StepType::kPredict, 0, 0xbe6fce26U>;
    static_assert(type::k == 1U);
};

template <>
struct db10::step<11> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x41ad98afU>;
    static_assert(type::k == 1U);
};

template <>
struct db10::step<12> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3d3cc279U>;
    static_assert(type::k == 1U);
};

struct db10_inverse {
    static constexpr const char* name = "db10-inverse";
    static constexpr uint32_t tap_size = 20U;
    static constexpr uint32_t num_steps = 13U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db10.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db10_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct db10_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x41ad98afU>;
    static_assert(type::k == 1U);
};

template <>
struct db10_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3d3cc279U>;
    static_assert(type::k == 1U);
};

template <>
struct db10_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0x3e6fce26U>;
    static_assert(type::k == 1U);
};

template <>
struct db10_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xb58dc38aU, 0xbec6fd94U>;
    static_assert(type::k == 2U);
};

template <>
struct db10_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x4024aa8eU, 0xbf003cedU>;
    static_assert(type::k == 2U);
};

template <>
struct db10_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fff52e4U, 0xbf1d7005U>;
    static_assert(type::k == 2U);
};

template <>
struct db10_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x3fce6c35U, 0xbf2e3788U>;
    static_assert(type::k == 2U);
};

template <>
struct db10_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fb3b948U, 0xbf3a53d6U>;
    static_assert(type::k == 2U);
};

template <>
struct db10_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f95e6e8U, 0xbf2df1dfU>;
    static_assert(type::k == 2U);
};

template <>
struct db10_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f78c6beU, 0xbf13eb41U>;
    static_assert(type::k == 2U);
};

template <>
struct db10_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f347442U, 0xbec28606U>;
    static_assert(type::k == 2U);
};

template <>
struct db10_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ee0419fU, 0xbe0e45bbU>;
    static_assert(type::k == 2U);
};

template <>
struct db10_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e112156U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
