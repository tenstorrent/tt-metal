// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct sym11_inverse;

struct sym11 {
    static constexpr const char* name = "sym11";
    static constexpr uint32_t tap_size = 22U;
    static constexpr int32_t delay_even = 5;
    static constexpr int32_t delay_odd = 6;
    static constexpr uint32_t num_steps = 15U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/sym11.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::sym11";
    using inverse = sym11_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct sym11::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0x3e6757edU>;
    static_assert(type::k == 1U);
};

template <>
struct sym11::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe5c1be8U, 0xbf5fb198U>;
    static_assert(type::k == 2U);
};

template <>
struct sym11::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f060c1bU, 0xbd10fe6eU>;
    static_assert(type::k == 2U);
};

template <>
struct sym11::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3dde2ab3U, 0x3f9425b9U>;
    static_assert(type::k == 2U);
};

template <>
struct sym11::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe86ea4dU, 0x3f4a1970U>;
    static_assert(type::k == 2U);
};

template <>
struct sym11::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf814ebfU, 0x411ce1cdU>;
    static_assert(type::k == 2U);
};

template <>
struct sym11::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xbdd04f14U, 0xbba7457dU>;
    static_assert(type::k == 2U);
};

template <>
struct sym11::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x42bd45a1U, 0xc136af6cU>;
    static_assert(type::k == 2U);
};

template <>
struct sym11::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x3a9a129fU, 0x3b9fceccU>;
    static_assert(type::k == 2U);
};

template <>
struct sym11::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc21b9f09U, 0x42adda63U>;
    static_assert(type::k == 2U);
};

template <>
struct sym11::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xbbcb2aedU, 0xbc85d853U>;
    static_assert(type::k == 2U);
};

template <>
struct sym11::step<11> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct sym11::step<12> {
    using type = StaticStep<StepType::kPredict, 0, 0x4259f82cU>;
    static_assert(type::k == 1U);
};

template <>
struct sym11::step<13> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xc08bf212U>;
    static_assert(type::k == 1U);
};

template <>
struct sym11::step<14> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3e6a25edU>;
    static_assert(type::k == 1U);
};

struct sym11_inverse {
    static constexpr const char* name = "sym11-inverse";
    static constexpr uint32_t tap_size = 22U;
    static constexpr uint32_t num_steps = 15U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/sym11.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::sym11_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct sym11_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x408bf212U>;
    static_assert(type::k == 1U);
};

template <>
struct sym11_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xbe6a25edU>;
    static_assert(type::k == 1U);
};

template <>
struct sym11_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0xc259f82cU>;
    static_assert(type::k == 1U);
};

template <>
struct sym11_inverse::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct sym11_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x3bcb2aedU, 0x3c85d853U>;
    static_assert(type::k == 2U);
};

template <>
struct sym11_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x421b9f09U, 0xc2adda63U>;
    static_assert(type::k == 2U);
};

template <>
struct sym11_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xba9a129fU, 0xbb9fceccU>;
    static_assert(type::k == 2U);
};

template <>
struct sym11_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc2bd45a1U, 0x4136af6cU>;
    static_assert(type::k == 2U);
};

template <>
struct sym11_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x3dd04f14U, 0x3ba7457dU>;
    static_assert(type::k == 2U);
};

template <>
struct sym11_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f814ebfU, 0xc11ce1cdU>;
    static_assert(type::k == 2U);
};

template <>
struct sym11_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e86ea4dU, 0xbf4a1970U>;
    static_assert(type::k == 2U);
};

template <>
struct sym11_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbdde2ab3U, 0xbf9425b9U>;
    static_assert(type::k == 2U);
};

template <>
struct sym11_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf060c1bU, 0x3d10fe6eU>;
    static_assert(type::k == 2U);
};

template <>
struct sym11_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e5c1be8U, 0x3f5fb198U>;
    static_assert(type::k == 2U);
};

template <>
struct sym11_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, 0, 0xbe6757edU>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
