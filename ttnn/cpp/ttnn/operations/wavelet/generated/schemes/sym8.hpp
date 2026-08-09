// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct sym8_inverse;

struct sym8 {
    static constexpr const char* name = "sym8";
    static constexpr uint32_t tap_size = 16U;
    static constexpr int32_t delay_even = 4;
    static constexpr int32_t delay_odd = 4;
    static constexpr uint32_t num_steps = 11U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/sym8.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::sym8";
    using inverse = sym8_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct sym8::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0x40c7a6aeU>;
    static_assert(type::k == 1U);
};

template <>
struct sym8::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d0cd233U, 0xbe2003feU>;
    static_assert(type::k == 2U);
};

template <>
struct sym8::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0x40a42f9eU, 0xc19803d5U>;
    static_assert(type::k == 2U);
};

template <>
struct sym8::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3bd79d29U, 0xbcc0d8ddU>;
    static_assert(type::k == 2U);
};

template <>
struct sym8::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xc064e7f0U, 0xbfce91faU>;
    static_assert(type::k == 2U);
};

template <>
struct sym8::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd7e1f01U, 0x3c656605U>;
    static_assert(type::k == 2U);
};

template <>
struct sym8::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xc14f82efU, 0x41032b89U>;
    static_assert(type::k == 2U);
};

template <>
struct sym8::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d7d5dc5U, 0x3d63dc5fU>;
    static_assert(type::k == 2U);
};

template <>
struct sym8::step<8> {
    using type = StaticStep<StepType::kPredict, 0, 0xc1451e62U>;
    static_assert(type::k == 1U);
};

template <>
struct sym8::step<9> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x4027ead7U>;
    static_assert(type::k == 1U);
};

template <>
struct sym8::step<10> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3ec324c5U>;
    static_assert(type::k == 1U);
};

struct sym8_inverse {
    static constexpr const char* name = "sym8-inverse";
    static constexpr uint32_t tap_size = 16U;
    static constexpr uint32_t num_steps = 11U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/sym8.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::sym8_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct sym8_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x4027ead7U>;
    static_assert(type::k == 1U);
};

template <>
struct sym8_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3ec324c5U>;
    static_assert(type::k == 1U);
};

template <>
struct sym8_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0x41451e62U>;
    static_assert(type::k == 1U);
};

template <>
struct sym8_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd7d5dc5U, 0xbd63dc5fU>;
    static_assert(type::k == 2U);
};

template <>
struct sym8_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x414f82efU, 0xc1032b89U>;
    static_assert(type::k == 2U);
};

template <>
struct sym8_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d7e1f01U, 0xbc656605U>;
    static_assert(type::k == 2U);
};

template <>
struct sym8_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x4064e7f0U, 0x3fce91faU>;
    static_assert(type::k == 2U);
};

template <>
struct sym8_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbbd79d29U, 0x3cc0d8ddU>;
    static_assert(type::k == 2U);
};

template <>
struct sym8_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0a42f9eU, 0x419803d5U>;
    static_assert(type::k == 2U);
};

template <>
struct sym8_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd0cd233U, 0x3e2003feU>;
    static_assert(type::k == 2U);
};

template <>
struct sym8_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0c7a6aeU>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
