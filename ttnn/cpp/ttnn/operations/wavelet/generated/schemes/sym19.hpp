// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct sym19_inverse;

struct sym19 {
    static constexpr const char* name = "sym19";
    static constexpr uint32_t tap_size = 38U;
    static constexpr int32_t delay_even = 9;
    static constexpr int32_t delay_odd = 10;
    static constexpr uint32_t num_steps = 23U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/sym19.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::sym19";
    using inverse = sym19_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct sym19::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0x3f96c358U>;
    static_assert(type::k == 1U);
};

template <>
struct sym19::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbefc9bd0U, 0x3ddb8272U>;
    static_assert(type::k == 2U);
};

template <>
struct sym19::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf12c4e2U, 0x418e0f72U>;
    static_assert(type::k == 2U);
};

template <>
struct sym19::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd630c3cU, 0xbaa39e47U>;
    static_assert(type::k == 2U);
};

template <>
struct sym19::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x41c5d73eU, 0xc4234e71U>;
    static_assert(type::k == 2U);
};

template <>
struct sym19::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ac00390U, 0xb956243cU>;
    static_assert(type::k == 2U);
};

template <>
struct sym19::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x44b89fe0U, 0x44e1c563U>;
    static_assert(type::k == 2U);
};

template <>
struct sym19::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xb963d2a8U, 0xb98b06adU>;
    static_assert(type::k == 2U);
};

template <>
struct sym19::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x44e72460U, 0x438d9a71U>;
    static_assert(type::k == 2U);
};

template <>
struct sym19::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xb8a36c93U, 0x3a0abfaeU>;
    static_assert(type::k == 2U);
};

template <>
struct sym19::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xc470ec0aU, 0xc3528c61U>;
    static_assert(type::k == 2U);
};

template <>
struct sym19::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x396bc716U, 0x3b40f744U>;
    static_assert(type::k == 2U);
};

template <>
struct sym19::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xc397fa4aU, 0xc2177c3aU>;
    static_assert(type::k == 2U);
};

template <>
struct sym19::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3b4a2ce1U, 0x3d38cb4bU>;
    static_assert(type::k == 2U);
};

template <>
struct sym19::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xc1ab44abU, 0xc01b273fU>;
    static_assert(type::k == 2U);
};

template <>
struct sym19::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3de0e54bU, 0x3dc327c8U>;
    static_assert(type::k == 2U);
};

template <>
struct sym19::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0101c91U, 0x3fac4c02U>;
    static_assert(type::k == 2U);
};

template <>
struct sym19::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd875266U, 0xbe0831c6U>;
    static_assert(type::k == 2U);
};

template <>
struct sym19::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0x40086367U, 0xc12cfaa4U>;
    static_assert(type::k == 2U);
};

template <>
struct sym19::step<19> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct sym19::step<20> {
    using type = StaticStep<StepType::kPredict, 0, 0x3dac7d59U>;
    static_assert(type::k == 1U);
};

template <>
struct sym19::step<21> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xbe265efcU>;
    static_assert(type::k == 1U);
};

template <>
struct sym19::step<22> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x40c4f516U>;
    static_assert(type::k == 1U);
};

struct sym19_inverse {
    static constexpr const char* name = "sym19-inverse";
    static constexpr uint32_t tap_size = 38U;
    static constexpr uint32_t num_steps = 23U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/sym19.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::sym19_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct sym19_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3e265efcU>;
    static_assert(type::k == 1U);
};

template <>
struct sym19_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xc0c4f516U>;
    static_assert(type::k == 1U);
};

template <>
struct sym19_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0xbdac7d59U>;
    static_assert(type::k == 1U);
};

template <>
struct sym19_inverse::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct sym19_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0086367U, 0x412cfaa4U>;
    static_assert(type::k == 2U);
};

template <>
struct sym19_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d875266U, 0x3e0831c6U>;
    static_assert(type::k == 2U);
};

template <>
struct sym19_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x40101c91U, 0xbfac4c02U>;
    static_assert(type::k == 2U);
};

template <>
struct sym19_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbde0e54bU, 0xbdc327c8U>;
    static_assert(type::k == 2U);
};

template <>
struct sym19_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x41ab44abU, 0x401b273fU>;
    static_assert(type::k == 2U);
};

template <>
struct sym19_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbb4a2ce1U, 0xbd38cb4bU>;
    static_assert(type::k == 2U);
};

template <>
struct sym19_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x4397fa4aU, 0x42177c3aU>;
    static_assert(type::k == 2U);
};

template <>
struct sym19_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xb96bc716U, 0xbb40f744U>;
    static_assert(type::k == 2U);
};

template <>
struct sym19_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x4470ec0aU, 0x43528c61U>;
    static_assert(type::k == 2U);
};

template <>
struct sym19_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x38a36c93U, 0xba0abfaeU>;
    static_assert(type::k == 2U);
};

template <>
struct sym19_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xc4e72460U, 0xc38d9a71U>;
    static_assert(type::k == 2U);
};

template <>
struct sym19_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3963d2a8U, 0x398b06adU>;
    static_assert(type::k == 2U);
};

template <>
struct sym19_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xc4b89fe0U, 0xc4e1c563U>;
    static_assert(type::k == 2U);
};

template <>
struct sym19_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbac00390U, 0x3956243cU>;
    static_assert(type::k == 2U);
};

template <>
struct sym19_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xc1c5d73eU, 0x44234e71U>;
    static_assert(type::k == 2U);
};

template <>
struct sym19_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d630c3cU, 0x3aa39e47U>;
    static_assert(type::k == 2U);
};

template <>
struct sym19_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f12c4e2U, 0xc18e0f72U>;
    static_assert(type::k == 2U);
};

template <>
struct sym19_inverse::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3efc9bd0U, 0xbddb8272U>;
    static_assert(type::k == 2U);
};

template <>
struct sym19_inverse::step<22> {
    using type = StaticStep<StepType::kPredict, 0, 0xbf96c358U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
