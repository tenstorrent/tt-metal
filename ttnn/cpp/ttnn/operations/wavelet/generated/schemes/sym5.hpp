// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct sym5_inverse;

struct sym5 {
    static constexpr const char* name = "sym5";
    static constexpr uint32_t tap_size = 10U;
    static constexpr int32_t delay_even = 2;
    static constexpr int32_t delay_odd = 3;
    static constexpr uint32_t num_steps = 9U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/sym5.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::sym5";
    using inverse = sym5_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct sym5::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0xbf8a3d2cU>;
    static_assert(type::k == 1U);
};

template <>
struct sym5::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3eff3e6eU, 0xbde7a34dU>;
    static_assert(type::k == 2U);
};

template <>
struct sym5::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f0031b4U, 0xc01dd216U>;
    static_assert(type::k == 2U);
};

template <>
struct sym5::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e762c54U, 0x3d64bb07U>;
    static_assert(type::k == 2U);
};

template <>
struct sym5::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xbfa6f391U, 0xc054e6a5U>;
    static_assert(type::k == 2U);
};

template <>
struct sym5::step<5> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct sym5::step<6> {
    using type = StaticStep<StepType::kPredict, 0, 0x3dd03454U>;
    static_assert(type::k == 1U);
};

template <>
struct sym5::step<7> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xbec11547U>;
    static_assert(type::k == 1U);
};

template <>
struct sym5::step<8> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x4029b595U>;
    static_assert(type::k == 1U);
};

struct sym5_inverse {
    static constexpr const char* name = "sym5-inverse";
    static constexpr uint32_t tap_size = 10U;
    static constexpr uint32_t num_steps = 9U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/sym5.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::sym5_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct sym5_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3ec11547U>;
    static_assert(type::k == 1U);
};

template <>
struct sym5_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xc029b595U>;
    static_assert(type::k == 1U);
};

template <>
struct sym5_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0xbdd03454U>;
    static_assert(type::k == 1U);
};

template <>
struct sym5_inverse::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct sym5_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x3fa6f391U, 0x4054e6a5U>;
    static_assert(type::k == 2U);
};

template <>
struct sym5_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe762c54U, 0xbd64bb07U>;
    static_assert(type::k == 2U);
};

template <>
struct sym5_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf0031b4U, 0x401dd216U>;
    static_assert(type::k == 2U);
};

template <>
struct sym5_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbeff3e6eU, 0x3de7a34dU>;
    static_assert(type::k == 2U);
};

template <>
struct sym5_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, 0, 0x3f8a3d2cU>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
