// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct db11_inverse;

struct db11 {
    static constexpr const char* name = "db11";
    static constexpr uint32_t tap_size = 22U;
    static constexpr int32_t delay_even = 5;
    static constexpr int32_t delay_odd = 6;
    static constexpr uint32_t num_steps = 15U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db11.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db11";
    using inverse = db11_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct db11::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0x40f69b64U>;
    static_assert(type::k == 1U);
};

template <>
struct db11::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe02acd2U, 0xbbdc3c52U>;
    static_assert(type::k == 2U);
};

template <>
struct db11::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0x41a85aa4U, 0xc21968e6U>;
    static_assert(type::k == 2U);
};

template <>
struct db11::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c1714adU, 0xbc75cd50U>;
    static_assert(type::k == 2U);
};

template <>
struct db11::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x421db8f9U, 0xc2803fd0U>;
    static_assert(type::k == 2U);
};

template <>
struct db11::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c4a9bd3U, 0xbcb12d12U>;
    static_assert(type::k == 2U);
};

template <>
struct db11::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x422a670bU, 0xc2ac72e3U>;
    static_assert(type::k == 2U);
};

template <>
struct db11::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c3a234fU, 0xbcecd4ddU>;
    static_assert(type::k == 2U);
};

template <>
struct db11::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x4209ea8fU, 0xc2f31944U>;
    static_assert(type::k == 2U);
};

template <>
struct db11::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c06c213U, 0xbd3ca165U>;
    static_assert(type::k == 2U);
};

template <>
struct db11::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x41adb6c2U, 0xc381c6e4U>;
    static_assert(type::k == 2U);
};

template <>
struct db11::step<11> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct db11::step<12> {
    using type = StaticStep<StepType::kPredict, 0, 0x3b7c7eabU>;
    static_assert(type::k == 1U);
};

template <>
struct db11::step<13> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3a7c4a71U>;
    static_assert(type::k == 1U);
};

template <>
struct db11::step<14> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xc481e1c3U>;
    static_assert(type::k == 1U);
};

struct db11_inverse {
    static constexpr const char* name = "db11-inverse";
    static constexpr uint32_t tap_size = 22U;
    static constexpr uint32_t num_steps = 15U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db11.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db11_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct db11_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xba7c4a70U>;
    static_assert(type::k == 1U);
};

template <>
struct db11_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x4481e1c2U>;
    static_assert(type::k == 1U);
};

template <>
struct db11_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0xbb7c7eabU>;
    static_assert(type::k == 1U);
};

template <>
struct db11_inverse::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct db11_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xc1adb6c2U, 0x4381c6e4U>;
    static_assert(type::k == 2U);
};

template <>
struct db11_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc06c213U, 0x3d3ca165U>;
    static_assert(type::k == 2U);
};

template <>
struct db11_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xc209ea8fU, 0x42f31944U>;
    static_assert(type::k == 2U);
};

template <>
struct db11_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc3a234fU, 0x3cecd4ddU>;
    static_assert(type::k == 2U);
};

template <>
struct db11_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xc22a670bU, 0x42ac72e3U>;
    static_assert(type::k == 2U);
};

template <>
struct db11_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc4a9bd3U, 0x3cb12d12U>;
    static_assert(type::k == 2U);
};

template <>
struct db11_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xc21db8f9U, 0x42803fd0U>;
    static_assert(type::k == 2U);
};

template <>
struct db11_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc1714adU, 0x3c75cd50U>;
    static_assert(type::k == 2U);
};

template <>
struct db11_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xc1a85aa4U, 0x421968e6U>;
    static_assert(type::k == 2U);
};

template <>
struct db11_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e02acd2U, 0x3bdc3c52U>;
    static_assert(type::k == 2U);
};

template <>
struct db11_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, 0, 0xc0f69b64U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
