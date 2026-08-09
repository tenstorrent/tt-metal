// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct db18_inverse;

struct db18 {
    static constexpr const char* name = "db18";
    static constexpr uint32_t tap_size = 36U;
    static constexpr int32_t delay_even = 9;
    static constexpr int32_t delay_odd = 9;
    static constexpr uint32_t num_steps = 21U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db18.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db18";
    using inverse = db18_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct db18::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0xbee561d3U>;
    static_assert(type::k == 1U);
};

template <>
struct db18::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf27866dU, 0x3da174e2U>;
    static_assert(type::k == 2U);
};

template <>
struct db18::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf7325cfU, 0x3e603123U>;
    static_assert(type::k == 2U);
};

template <>
struct db18::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3dddbf02U, 0x3ea9d8deU>;
    static_assert(type::k == 2U);
};

template <>
struct db18::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x3dfa9d9eU, 0x3f721e88U>;
    static_assert(type::k == 2U);
};

template <>
struct db18::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf060844U, 0x3e980e22U>;
    static_assert(type::k == 2U);
};

template <>
struct db18::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xc000a94cU, 0x3fbb03f0U>;
    static_assert(type::k == 2U);
};

template <>
struct db18::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf03d537U, 0x3ebb4f92U>;
    static_assert(type::k == 2U);
};

template <>
struct db18::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xc01b7729U, 0x3fd1cec0U>;
    static_assert(type::k == 2U);
};

template <>
struct db18::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf1bfe91U, 0x3ec26aa4U>;
    static_assert(type::k == 2U);
};

template <>
struct db18::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xc036861eU, 0x3fcb69c1U>;
    static_assert(type::k == 2U);
};

template <>
struct db18::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf38b064U, 0x3eb1b96bU>;
    static_assert(type::k == 2U);
};

template <>
struct db18::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xc05ce3b8U, 0x3fb101c3U>;
    static_assert(type::k == 2U);
};

template <>
struct db18::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf66f909U, 0x3e944a2cU>;
    static_assert(type::k == 2U);
};

template <>
struct db18::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0907a22U, 0x3f8ddd30U>;
    static_assert(type::k == 2U);
};

template <>
struct db18::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfa18300U, 0x3e62cdb9U>;
    static_assert(type::k == 2U);
};

template <>
struct db18::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0e3760cU, 0x3f4ae212U>;
    static_assert(type::k == 2U);
};

template <>
struct db18::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x2cebd15fU, 0x3e100f4eU>;
    static_assert(type::k == 2U);
};

template <>
struct db18::step<18> {
    using type = StaticStep<StepType::kPredict, 0, 0xbec15f0fU>;
    static_assert(type::k == 1U);
};

template <>
struct db18::step<19> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x43f39bf9U>;
    static_assert(type::k == 1U);
};

template <>
struct db18::step<20> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3b0682afU>;
    static_assert(type::k == 1U);
};

struct db18_inverse {
    static constexpr const char* name = "db18-inverse";
    static constexpr uint32_t tap_size = 36U;
    static constexpr uint32_t num_steps = 21U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db18.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db18_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct db18_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x43f39bf9U>;
    static_assert(type::k == 1U);
};

template <>
struct db18_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3b0682afU>;
    static_assert(type::k == 1U);
};

template <>
struct db18_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0x3ec15f0fU>;
    static_assert(type::k == 1U);
};

template <>
struct db18_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xacebd15fU, 0xbe100f4eU>;
    static_assert(type::k == 2U);
};

template <>
struct db18_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x40e3760cU, 0xbf4ae212U>;
    static_assert(type::k == 2U);
};

template <>
struct db18_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fa18300U, 0xbe62cdb9U>;
    static_assert(type::k == 2U);
};

template <>
struct db18_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x40907a22U, 0xbf8ddd30U>;
    static_assert(type::k == 2U);
};

template <>
struct db18_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f66f909U, 0xbe944a2cU>;
    static_assert(type::k == 2U);
};

template <>
struct db18_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x405ce3b8U, 0xbfb101c3U>;
    static_assert(type::k == 2U);
};

template <>
struct db18_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f38b064U, 0xbeb1b96bU>;
    static_assert(type::k == 2U);
};

template <>
struct db18_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x4036861eU, 0xbfcb69c1U>;
    static_assert(type::k == 2U);
};

template <>
struct db18_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f1bfe91U, 0xbec26aa4U>;
    static_assert(type::k == 2U);
};

template <>
struct db18_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x401b7729U, 0xbfd1cec0U>;
    static_assert(type::k == 2U);
};

template <>
struct db18_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f03d537U, 0xbebb4f92U>;
    static_assert(type::k == 2U);
};

template <>
struct db18_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x4000a94cU, 0xbfbb03f0U>;
    static_assert(type::k == 2U);
};

template <>
struct db18_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f060844U, 0xbe980e22U>;
    static_assert(type::k == 2U);
};

template <>
struct db18_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xbdfa9d9eU, 0xbf721e88U>;
    static_assert(type::k == 2U);
};

template <>
struct db18_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbdddbf02U, 0xbea9d8deU>;
    static_assert(type::k == 2U);
};

template <>
struct db18_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f7325cfU, 0xbe603123U>;
    static_assert(type::k == 2U);
};

template <>
struct db18_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f27866dU, 0xbda174e2U>;
    static_assert(type::k == 2U);
};

template <>
struct db18_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ee561d3U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
