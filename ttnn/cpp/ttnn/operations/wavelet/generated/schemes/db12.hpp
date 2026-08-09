// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct db12_inverse;

struct db12 {
    static constexpr const char* name = "db12";
    static constexpr uint32_t tap_size = 24U;
    static constexpr int32_t delay_even = 6;
    static constexpr int32_t delay_odd = 6;
    static constexpr uint32_t num_steps = 15U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db12.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db12";
    using inverse = db12_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct db12::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0xbdf517c2U>;
    static_assert(type::k == 1U);
};

template <>
struct db12::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbebbcdb5U, 0x3df1a1d5U>;
    static_assert(type::k == 2U);
};

template <>
struct db12::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf188e7fU, 0x3ea9ba0fU>;
    static_assert(type::k == 2U);
};

template <>
struct db12::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf52b989U, 0x3f04c688U>;
    static_assert(type::k == 2U);
};

template <>
struct db12::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf806a75U, 0x3f24c53cU>;
    static_assert(type::k == 2U);
};

template <>
struct db12::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf98539cU, 0x3f3ae670U>;
    static_assert(type::k == 2U);
};

template <>
struct db12::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xbfab01e3U, 0x3f3d1e04U>;
    static_assert(type::k == 2U);
};

template <>
struct db12::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfc5172bU, 0x3f37e349U>;
    static_assert(type::k == 2U);
};

template <>
struct db12::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xbfe184a0U, 0x3f24c064U>;
    static_assert(type::k == 2U);
};

template <>
struct db12::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc00a8e93U, 0x3f111fb5U>;
    static_assert(type::k == 2U);
};

template <>
struct db12::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0336879U, 0x3eec799dU>;
    static_assert(type::k == 2U);
};

template <>
struct db12::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3386a5dbU, 0x3eb6a506U>;
    static_assert(type::k == 2U);
};

template <>
struct db12::step<12> {
    using type = StaticStep<StepType::kPredict, 0, 0xbe5e1751U>;
    static_assert(type::k == 1U);
};

template <>
struct db12::step<13> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x422c8125U>;
    static_assert(type::k == 1U);
};

template <>
struct db12::step<14> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3cbdf459U>;
    static_assert(type::k == 1U);
};

struct db12_inverse {
    static constexpr const char* name = "db12-inverse";
    static constexpr uint32_t tap_size = 24U;
    static constexpr uint32_t num_steps = 15U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db12.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db12_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct db12_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x422c8126U>;
    static_assert(type::k == 1U);
};

template <>
struct db12_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3cbdf45aU>;
    static_assert(type::k == 1U);
};

template <>
struct db12_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0x3e5e1751U>;
    static_assert(type::k == 1U);
};

template <>
struct db12_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xb386a5dbU, 0xbeb6a506U>;
    static_assert(type::k == 2U);
};

template <>
struct db12_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x40336879U, 0xbeec799dU>;
    static_assert(type::k == 2U);
};

template <>
struct db12_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x400a8e93U, 0xbf111fb5U>;
    static_assert(type::k == 2U);
};

template <>
struct db12_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x3fe184a0U, 0xbf24c064U>;
    static_assert(type::k == 2U);
};

template <>
struct db12_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fc5172bU, 0xbf37e349U>;
    static_assert(type::k == 2U);
};

template <>
struct db12_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x3fab01e3U, 0xbf3d1e04U>;
    static_assert(type::k == 2U);
};

template <>
struct db12_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f98539cU, 0xbf3ae670U>;
    static_assert(type::k == 2U);
};

template <>
struct db12_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f806a75U, 0xbf24c53cU>;
    static_assert(type::k == 2U);
};

template <>
struct db12_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f52b989U, 0xbf04c688U>;
    static_assert(type::k == 2U);
};

template <>
struct db12_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f188e7fU, 0xbea9ba0fU>;
    static_assert(type::k == 2U);
};

template <>
struct db12_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ebbcdb5U, 0xbdf1a1d5U>;
    static_assert(type::k == 2U);
};

template <>
struct db12_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x3df517c2U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
