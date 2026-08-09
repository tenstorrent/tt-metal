// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct sym12_inverse;

struct sym12 {
    static constexpr const char* name = "sym12";
    static constexpr uint32_t tap_size = 24U;
    static constexpr int32_t delay_even = 6;
    static constexpr int32_t delay_odd = 6;
    static constexpr uint32_t num_steps = 15U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/sym12.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::sym12";
    using inverse = sym12_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct sym12::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0xc11dc8dcU>;
    static_assert(type::k == 1U);
};

template <>
struct sym12::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbcd99adbU, 0x3dcd8fd0U>;
    static_assert(type::k == 2U);
};

template <>
struct sym12::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xc144914cU, 0x42034e85U>;
    static_assert(type::k == 2U);
};

template <>
struct sym12::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbca2f88fU, 0x3d22c6f9U>;
    static_assert(type::k == 2U);
};

template <>
struct sym12::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0f101eeU, 0x411b2d73U>;
    static_assert(type::k == 2U);
};

template <>
struct sym12::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbb1c07d1U, 0x3c890d86U>;
    static_assert(type::k == 2U);
};

template <>
struct sym12::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x409bde90U, 0x3f9c82e7U>;
    static_assert(type::k == 2U);
};

template <>
struct sym12::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c7d9ebeU, 0xbc14f181U>;
    static_assert(type::k == 2U);
};

template <>
struct sym12::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x419a2d9bU, 0xc0f4d248U>;
    static_assert(type::k == 2U);
};

template <>
struct sym12::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d3e286eU, 0xbcc39b1bU>;
    static_assert(type::k == 2U);
};

template <>
struct sym12::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x419841f4U, 0xc18382a8U>;
    static_assert(type::k == 2U);
};

template <>
struct sym12::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c524e64U, 0xbd2ec1beU>;
    static_assert(type::k == 2U);
};

template <>
struct sym12::step<12> {
    using type = StaticStep<StepType::kPredict, 0, 0xc1af2ea5U>;
    static_assert(type::k == 1U);
};

template <>
struct sym12::step<13> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x40bd5eccU>;
    static_assert(type::k == 1U);
};

template <>
struct sym12::step<14> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3e2d0961U>;
    static_assert(type::k == 1U);
};

struct sym12_inverse {
    static constexpr const char* name = "sym12-inverse";
    static constexpr uint32_t tap_size = 24U;
    static constexpr uint32_t num_steps = 15U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/sym12.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::sym12_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct sym12_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x40bd5eccU>;
    static_assert(type::k == 1U);
};

template <>
struct sym12_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3e2d0961U>;
    static_assert(type::k == 1U);
};

template <>
struct sym12_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0x41af2ea5U>;
    static_assert(type::k == 1U);
};

template <>
struct sym12_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc524e64U, 0x3d2ec1beU>;
    static_assert(type::k == 2U);
};

template <>
struct sym12_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xc19841f4U, 0x418382a8U>;
    static_assert(type::k == 2U);
};

template <>
struct sym12_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd3e286eU, 0x3cc39b1bU>;
    static_assert(type::k == 2U);
};

template <>
struct sym12_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xc19a2d9bU, 0x40f4d248U>;
    static_assert(type::k == 2U);
};

template <>
struct sym12_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc7d9ebeU, 0x3c14f181U>;
    static_assert(type::k == 2U);
};

template <>
struct sym12_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xc09bde90U, 0xbf9c82e7U>;
    static_assert(type::k == 2U);
};

template <>
struct sym12_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3b1c07d1U, 0xbc890d86U>;
    static_assert(type::k == 2U);
};

template <>
struct sym12_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x40f101eeU, 0xc11b2d73U>;
    static_assert(type::k == 2U);
};

template <>
struct sym12_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ca2f88fU, 0xbd22c6f9U>;
    static_assert(type::k == 2U);
};

template <>
struct sym12_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x4144914cU, 0xc2034e85U>;
    static_assert(type::k == 2U);
};

template <>
struct sym12_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3cd99adbU, 0xbdcd8fd0U>;
    static_assert(type::k == 2U);
};

template <>
struct sym12_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x411dc8dcU>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
