// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct db22_inverse;

struct db22 {
    static constexpr const char* name = "db22";
    static constexpr uint32_t tap_size = 44U;
    static constexpr int32_t delay_even = 11;
    static constexpr int32_t delay_odd = 11;
    static constexpr uint32_t num_steps = 25U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db22.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db22";
    using inverse = db22_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct db22::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0xbebb2d2bU>;
    static_assert(type::k == 1U);
};

template <>
struct db22::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf0679b0U, 0x3e9f59c0U>;
    static_assert(type::k == 2U);
};

template <>
struct db22::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf0fd18eU, 0xc0869cf4U>;
    static_assert(type::k == 2U);
};

template <>
struct db22::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e96c737U, 0xbaa4653dU>;
    static_assert(type::k == 2U);
};

template <>
struct db22::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xc12c6556U, 0xc13e4b38U>;
    static_assert(type::k == 2U);
};

template <>
struct db22::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbda56cacU, 0x3d4c2559U>;
    static_assert(type::k == 2U);
};

template <>
struct db22::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xc18e187bU, 0x40ec8281U>;
    static_assert(type::k == 2U);
};

template <>
struct db22::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3bd512dbU, 0x3d28e0e2U>;
    static_assert(type::k == 2U);
};

template <>
struct db22::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x3fcc35f9U, 0x428eb3e4U>;
    static_assert(type::k == 2U);
};

template <>
struct db22::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc4a989aU, 0x3bbb8471U>;
    static_assert(type::k == 2U);
};

template <>
struct db22::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xc3217b2bU, 0x42e180adU>;
    static_assert(type::k == 2U);
};

template <>
struct db22::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc11d62fU, 0x3bbee3eaU>;
    static_assert(type::k == 2U);
};

template <>
struct db22::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xc336cd17U, 0x42da775eU>;
    static_assert(type::k == 2U);
};

template <>
struct db22::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc26acccU, 0x3bb172c1U>;
    static_assert(type::k == 2U);
};

template <>
struct db22::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xc353e63eU, 0x42c4085aU>;
    static_assert(type::k == 2U);
};

template <>
struct db22::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc45134bU, 0x3b9a8b46U>;
    static_assert(type::k == 2U);
};

template <>
struct db22::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xc3806283U, 0x42a64171U>;
    static_assert(type::k == 2U);
};

template <>
struct db22::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc764365U, 0x3b7f3aebU>;
    static_assert(type::k == 2U);
};

template <>
struct db22::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xc3a75094U, 0x42850f8eU>;
    static_assert(type::k == 2U);
};

template <>
struct db22::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbcab348aU, 0x3b43d8b0U>;
    static_assert(type::k == 2U);
};

template <>
struct db22::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0xc402cf1eU, 0x423f6562U>;
    static_assert(type::k == 2U);
};

template <>
struct db22::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0x25da456dU, 0x3afa80a7U>;
    static_assert(type::k == 2U);
};

template <>
struct db22::step<22> {
    using type = StaticStep<StepType::kPredict, 0, 0xc1b7be44U>;
    static_assert(type::k == 1U);
};

template <>
struct db22::step<23> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xc675368fU>;
    static_assert(type::k == 1U);
};

template <>
struct db22::step<24> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xb885a175U>;
    static_assert(type::k == 1U);
};

struct db22_inverse {
    static constexpr const char* name = "db22-inverse";
    static constexpr uint32_t tap_size = 44U;
    static constexpr uint32_t num_steps = 25U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db22.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db22_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct db22_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xc6753690U>;
    static_assert(type::k == 1U);
};

template <>
struct db22_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xb885a175U>;
    static_assert(type::k == 1U);
};

template <>
struct db22_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0x41b7be44U>;
    static_assert(type::k == 1U);
};

template <>
struct db22_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xa5da456dU, 0xbafa80a7U>;
    static_assert(type::k == 2U);
};

template <>
struct db22_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x4402cf1eU, 0xc23f6562U>;
    static_assert(type::k == 2U);
};

template <>
struct db22_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3cab348aU, 0xbb43d8b0U>;
    static_assert(type::k == 2U);
};

template <>
struct db22_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x43a75094U, 0xc2850f8eU>;
    static_assert(type::k == 2U);
};

template <>
struct db22_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c764365U, 0xbb7f3aebU>;
    static_assert(type::k == 2U);
};

template <>
struct db22_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x43806283U, 0xc2a64171U>;
    static_assert(type::k == 2U);
};

template <>
struct db22_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c45134bU, 0xbb9a8b46U>;
    static_assert(type::k == 2U);
};

template <>
struct db22_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x4353e63eU, 0xc2c4085aU>;
    static_assert(type::k == 2U);
};

template <>
struct db22_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c26acccU, 0xbbb172c1U>;
    static_assert(type::k == 2U);
};

template <>
struct db22_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x4336cd17U, 0xc2da775eU>;
    static_assert(type::k == 2U);
};

template <>
struct db22_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c11d62fU, 0xbbbee3eaU>;
    static_assert(type::k == 2U);
};

template <>
struct db22_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x43217b2bU, 0xc2e180adU>;
    static_assert(type::k == 2U);
};

template <>
struct db22_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c4a989aU, 0xbbbb8471U>;
    static_assert(type::k == 2U);
};

template <>
struct db22_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xbfcc35f9U, 0xc28eb3e4U>;
    static_assert(type::k == 2U);
};

template <>
struct db22_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbbd512dbU, 0xbd28e0e2U>;
    static_assert(type::k == 2U);
};

template <>
struct db22_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0x418e187bU, 0xc0ec8281U>;
    static_assert(type::k == 2U);
};

template <>
struct db22_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3da56cacU, 0xbd4c2559U>;
    static_assert(type::k == 2U);
};

template <>
struct db22_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0x412c6556U, 0x413e4b38U>;
    static_assert(type::k == 2U);
};

template <>
struct db22_inverse::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe96c737U, 0x3aa4653dU>;
    static_assert(type::k == 2U);
};

template <>
struct db22_inverse::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f0fd18eU, 0x40869cf4U>;
    static_assert(type::k == 2U);
};

template <>
struct db22_inverse::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f0679b0U, 0xbe9f59c0U>;
    static_assert(type::k == 2U);
};

template <>
struct db22_inverse::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ebb2d2bU>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
