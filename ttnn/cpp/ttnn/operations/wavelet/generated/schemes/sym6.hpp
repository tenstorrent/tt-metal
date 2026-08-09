// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct sym6_inverse;

struct sym6 {
    static constexpr const char* name = "sym6";
    static constexpr uint32_t tap_size = 12U;
    static constexpr int32_t delay_even = 3;
    static constexpr int32_t delay_odd = 3;
    static constexpr uint32_t num_steps = 9U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/sym6.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::sym6";
    using inverse = sym6_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct sym6::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0x408d365aU>;
    static_assert(type::k == 1U);
};

template <>
struct sym6::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d162ea9U, 0xbe5cb6b8U>;
    static_assert(type::k == 2U);
};

template <>
struct sym6::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f9541eeU, 0xc11d46c2U>;
    static_assert(type::k == 2U);
};

template <>
struct sym6::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd1000d8U, 0xbbdd162cU>;
    static_assert(type::k == 2U);
};

template <>
struct sym6::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xc15fd19cU, 0x40a141ceU>;
    static_assert(type::k == 2U);
};

template <>
struct sym6::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d8c0a91U, 0x3d36b1d7U>;
    static_assert(type::k == 2U);
};

template <>
struct sym6::step<6> {
    using type = StaticStep<StepType::kPredict, 0, 0xc13a3af3U>;
    static_assert(type::k == 1U);
};

template <>
struct sym6::step<7> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x401b612aU>;
    static_assert(type::k == 1U);
};

template <>
struct sym6::step<8> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3ed2e3daU>;
    static_assert(type::k == 1U);
};

struct sym6_inverse {
    static constexpr const char* name = "sym6-inverse";
    static constexpr uint32_t tap_size = 12U;
    static constexpr uint32_t num_steps = 9U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/sym6.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::sym6_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct sym6_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x401b612aU>;
    static_assert(type::k == 1U);
};

template <>
struct sym6_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3ed2e3daU>;
    static_assert(type::k == 1U);
};

template <>
struct sym6_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0x413a3af3U>;
    static_assert(type::k == 1U);
};

template <>
struct sym6_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd8c0a91U, 0xbd36b1d7U>;
    static_assert(type::k == 2U);
};

template <>
struct sym6_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x415fd19cU, 0xc0a141ceU>;
    static_assert(type::k == 2U);
};

template <>
struct sym6_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d1000d8U, 0x3bdd162cU>;
    static_assert(type::k == 2U);
};

template <>
struct sym6_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf9541eeU, 0x411d46c2U>;
    static_assert(type::k == 2U);
};

template <>
struct sym6_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd162ea9U, 0x3e5cb6b8U>;
    static_assert(type::k == 2U);
};

template <>
struct sym6_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xc08d365aU>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
