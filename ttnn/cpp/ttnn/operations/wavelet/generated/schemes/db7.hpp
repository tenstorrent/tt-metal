// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct db7_inverse;

struct db7 {
    static constexpr const char* name = "db7";
    static constexpr uint32_t tap_size = 14U;
    static constexpr int32_t delay_even = 3;
    static constexpr int32_t delay_odd = 4;
    static constexpr uint32_t num_steps = 11U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db7.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db7";
    using inverse = db7_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct db7::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0x40a2fdf1U>;
    static_assert(type::k == 1U);
};

template <>
struct db7::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe419440U, 0xbcc4984fU>;
    static_assert(type::k == 2U);
};

template <>
struct db7::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0x4144912fU, 0xc1ca9ed3U>;
    static_assert(type::k == 2U);
};

template <>
struct db7::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ccef850U, 0xbd56a186U>;
    static_assert(type::k == 2U);
};

template <>
struct db7::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x41870250U, 0xc22bc5caU>;
    static_assert(type::k == 2U);
};

template <>
struct db7::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3cbafc65U, 0xbdb36c6fU>;
    static_assert(type::k == 2U);
};

template <>
struct db7::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x41366111U, 0xc2b9d21dU>;
    static_assert(type::k == 2U);
};

template <>
struct db7::step<7> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct db7::step<8> {
    using type = StaticStep<StepType::kPredict, 0, 0x3c30569dU>;
    static_assert(type::k == 1U);
};

template <>
struct db7::step<9> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3be5244cU>;
    static_assert(type::k == 1U);
};

template <>
struct db7::step<10> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xc30f00cfU>;
    static_assert(type::k == 1U);
};

struct db7_inverse {
    static constexpr const char* name = "db7-inverse";
    static constexpr uint32_t tap_size = 14U;
    static constexpr uint32_t num_steps = 11U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db7.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db7_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct db7_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xbbe5244cU>;
    static_assert(type::k == 1U);
};

template <>
struct db7_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x430f00cfU>;
    static_assert(type::k == 1U);
};

template <>
struct db7_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0xbc30569dU>;
    static_assert(type::k == 1U);
};

template <>
struct db7_inverse::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct db7_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xc1366111U, 0x42b9d21dU>;
    static_assert(type::k == 2U);
};

template <>
struct db7_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbcbafc65U, 0x3db36c6fU>;
    static_assert(type::k == 2U);
};

template <>
struct db7_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xc1870250U, 0x422bc5caU>;
    static_assert(type::k == 2U);
};

template <>
struct db7_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbccef850U, 0x3d56a186U>;
    static_assert(type::k == 2U);
};

template <>
struct db7_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xc144912fU, 0x41ca9ed3U>;
    static_assert(type::k == 2U);
};

template <>
struct db7_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e419440U, 0x3cc4984fU>;
    static_assert(type::k == 2U);
};

template <>
struct db7_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, 0, 0xc0a2fdf1U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
