// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct bior6_8_inverse;

struct bior6_8 {
    static constexpr const char* name = "bior6.8";
    static constexpr uint32_t tap_size = 18U;
    static constexpr int32_t delay_even = 4;
    static constexpr int32_t delay_odd = 5;
    static constexpr uint32_t num_steps = 9U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/bior6_8.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::bior6_8";
    using inverse = bior6_8_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct bior6_8::step<0> {
    using type = StaticStep<StepType::kUpdate, -1, 0xbf7f4545U, 0xbf7f4545U>;
    static_assert(type::k == 2U);
};

template <>
struct bior6_8::step<1> {
    using type = StaticStep<StepType::kPredict, 0, 0x3e8c09c3U, 0x3e8c09c3U>;
    static_assert(type::k == 2U);
};

template <>
struct bior6_8::step<2> {
    using type = StaticStep<StepType::kUpdate, -1, 0xbec66137U, 0xbec66137U>;
    static_assert(type::k == 2U);
};

template <>
struct bior6_8::step<3> {
    using type = StaticStep<StepType::kPredict, 0, 0xbe92b08eU, 0xbe92b08eU>;
    static_assert(type::k == 2U);
};

template <>
struct bior6_8::step<4> {
    using type = StaticStep<StepType::kUpdate, -1, 0x3f0c70abU, 0x3f0c70abU>;
    static_assert(type::k == 2U);
};

template <>
struct bior6_8::step<5> {
    using type = StaticStep<StepType::kPredict, -1, 0xbdcc701dU, 0x3eb0084bU, 0x3eb0084bU, 0xbdcc701dU>;
    static_assert(type::k == 4U);
};

template <>
struct bior6_8::step<6> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct bior6_8::step<7> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3f935e00U>;
    static_assert(type::k == 1U);
};

template <>
struct bior6_8::step<8> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xbf5e5b2cU>;
    static_assert(type::k == 1U);
};

struct bior6_8_inverse {
    static constexpr const char* name = "bior6.8-inverse";
    static constexpr uint32_t tap_size = 18U;
    static constexpr uint32_t num_steps = 9U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/bior6_8.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::bior6_8_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct bior6_8_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xbf935e00U>;
    static_assert(type::k == 1U);
};

template <>
struct bior6_8_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3f5e5b2cU>;
    static_assert(type::k == 1U);
};

template <>
struct bior6_8_inverse::step<2> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct bior6_8_inverse::step<3> {
    using type = StaticStep<StepType::kPredict, -1, 0x3dcc701dU, 0xbeb0084bU, 0xbeb0084bU, 0x3dcc701dU>;
    static_assert(type::k == 4U);
};

template <>
struct bior6_8_inverse::step<4> {
    using type = StaticStep<StepType::kUpdate, -1, 0xbf0c70abU, 0xbf0c70abU>;
    static_assert(type::k == 2U);
};

template <>
struct bior6_8_inverse::step<5> {
    using type = StaticStep<StepType::kPredict, 0, 0x3e92b08eU, 0x3e92b08eU>;
    static_assert(type::k == 2U);
};

template <>
struct bior6_8_inverse::step<6> {
    using type = StaticStep<StepType::kUpdate, -1, 0x3ec66137U, 0x3ec66137U>;
    static_assert(type::k == 2U);
};

template <>
struct bior6_8_inverse::step<7> {
    using type = StaticStep<StepType::kPredict, 0, 0xbe8c09c3U, 0xbe8c09c3U>;
    static_assert(type::k == 2U);
};

template <>
struct bior6_8_inverse::step<8> {
    using type = StaticStep<StepType::kUpdate, -1, 0x3f7f4545U, 0x3f7f4545U>;
    static_assert(type::k == 2U);
};

}  // namespace ttnn::operations::wavelet::schemes
