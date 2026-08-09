// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct db31_inverse;

struct db31 {
    static constexpr const char* name = "db31";
    static constexpr uint32_t tap_size = 62U;
    static constexpr int32_t delay_even = 15;
    static constexpr int32_t delay_odd = 16;
    static constexpr uint32_t num_steps = 35U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db31.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db31";
    using inverse = db31_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct db31::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0x40d9992dU>;
    static_assert(type::k == 1U);
};

template <>
struct db31::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe0bfab0U, 0xbbf990deU>;
    static_assert(type::k == 2U);
};

template <>
struct db31::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0x41e0f07fU, 0xc1ab0325U>;
    static_assert(type::k == 2U);
};

template <>
struct db31::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c6d5fdaU, 0xbc3b639eU>;
    static_assert(type::k == 2U);
};

template <>
struct db31::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x4204e8c1U, 0xc1db1603U>;
    static_assert(type::k == 2U);
};

template <>
struct db31::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c69bd9bU, 0xbc68e67aU>;
    static_assert(type::k == 2U);
};

template <>
struct db31::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x4174c323U, 0xc20f2817U>;
    static_assert(type::k == 2U);
};

template <>
struct db31::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3bbac730U, 0xbcc43a64U>;
    static_assert(type::k == 2U);
};

template <>
struct db31::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x41a72e8aU, 0xc2257539U>;
    static_assert(type::k == 2U);
};

template <>
struct db31::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c8a991aU, 0xbcb66e8fU>;
    static_assert(type::k == 2U);
};

template <>
struct db31::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x420dadf5U, 0xc2327f23U>;
    static_assert(type::k == 2U);
};

template <>
struct db31::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c9cc5e1U, 0xbcc61e42U>;
    static_assert(type::k == 2U);
};

template <>
struct db31::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x42158e21U, 0xc1dbedd6U>;
    static_assert(type::k == 2U);
};

template <>
struct db31::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d0664f8U, 0xbaf53fe3U>;
    static_assert(type::k == 2U);
};

template <>
struct db31::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x43675f0dU, 0xc15ade52U>;
    static_assert(type::k == 2U);
};

template <>
struct db31::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c21e12dU, 0xbb866d6cU>;
    static_assert(type::k == 2U);
};

template <>
struct db31::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x43215ac1U, 0xc2c7665dU>;
    static_assert(type::k == 2U);
};

template <>
struct db31::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd5c1e4bU, 0xbbc9e18aU>;
    static_assert(type::k == 2U);
};

template <>
struct db31::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xbc92897eU, 0x4194ecadU>;
    static_assert(type::k == 2U);
};

template <>
struct db31::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbea6afb2U, 0x428f1b59U>;
    static_assert(type::k == 2U);
};

template <>
struct db31::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0xbc656f8eU, 0xbddf7180U>;
    static_assert(type::k == 2U);
};

template <>
struct db31::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0x4112a1dcU, 0xc1a58f05U>;
    static_assert(type::k == 2U);
};

template <>
struct db31::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0x3d45eb59U, 0xbe074877U>;
    static_assert(type::k == 2U);
};

template <>
struct db31::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0x40f237a2U, 0xc1c069ebU>;
    static_assert(type::k == 2U);
};

template <>
struct db31::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0x3d2a4cb5U, 0xbe1fb36aU>;
    static_assert(type::k == 2U);
};

template <>
struct db31::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0x40cd2f03U, 0xc1e7e6dfU>;
    static_assert(type::k == 2U);
};

template <>
struct db31::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0x3d0d4d19U, 0xbe45effbU>;
    static_assert(type::k == 2U);
};

template <>
struct db31::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0x40a58c1aU, 0xc215a46cU>;
    static_assert(type::k == 2U);
};

template <>
struct db31::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0x3cdaf9beU, 0xbe882b94U>;
    static_assert(type::k == 2U);
};

template <>
struct db31::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0x4070a3d5U, 0xc2677d86U>;
    static_assert(type::k == 2U);
};

template <>
struct db31::step<30> {
    using type = StaticStep<StepType::kPredict, -1, 0x3c8d8d67U, 0xbf0c57eaU>;
    static_assert(type::k == 2U);
};

template <>
struct db31::step<31> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct db31::step<32> {
    using type = StaticStep<StepType::kPredict, 0, 0x3fe97c02U>;
    static_assert(type::k == 1U);
};

template <>
struct db31::step<33> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3790ffceU>;
    static_assert(type::k == 1U);
};

template <>
struct db31::step<34> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xc761fcc5U>;
    static_assert(type::k == 1U);
};

struct db31_inverse {
    static constexpr const char* name = "db31-inverse";
    static constexpr uint32_t tap_size = 62U;
    static constexpr uint32_t num_steps = 35U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db31.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db31_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct db31_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xb790ffcfU>;
    static_assert(type::k == 1U);
};

template <>
struct db31_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x4761fcc6U>;
    static_assert(type::k == 1U);
};

template <>
struct db31_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0xbfe97c02U>;
    static_assert(type::k == 1U);
};

template <>
struct db31_inverse::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct db31_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xbc8d8d67U, 0x3f0c57eaU>;
    static_assert(type::k == 2U);
};

template <>
struct db31_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc070a3d5U, 0x42677d86U>;
    static_assert(type::k == 2U);
};

template <>
struct db31_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xbcdaf9beU, 0x3e882b94U>;
    static_assert(type::k == 2U);
};

template <>
struct db31_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc0a58c1aU, 0x4215a46cU>;
    static_assert(type::k == 2U);
};

template <>
struct db31_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xbd0d4d19U, 0x3e45effbU>;
    static_assert(type::k == 2U);
};

template <>
struct db31_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc0cd2f03U, 0x41e7e6dfU>;
    static_assert(type::k == 2U);
};

template <>
struct db31_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xbd2a4cb5U, 0x3e1fb36aU>;
    static_assert(type::k == 2U);
};

template <>
struct db31_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc0f237a2U, 0x41c069ebU>;
    static_assert(type::k == 2U);
};

template <>
struct db31_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xbd45eb59U, 0x3e074877U>;
    static_assert(type::k == 2U);
};

template <>
struct db31_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc112a1dcU, 0x41a58f05U>;
    static_assert(type::k == 2U);
};

template <>
struct db31_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x3c656f8eU, 0x3ddf7180U>;
    static_assert(type::k == 2U);
};

template <>
struct db31_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ea6afb2U, 0xc28f1b59U>;
    static_assert(type::k == 2U);
};

template <>
struct db31_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x3c92897eU, 0xc194ecadU>;
    static_assert(type::k == 2U);
};

template <>
struct db31_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d5c1e4bU, 0x3bc9e18aU>;
    static_assert(type::k == 2U);
};

template <>
struct db31_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xc3215ac1U, 0x42c7665dU>;
    static_assert(type::k == 2U);
};

template <>
struct db31_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc21e12dU, 0x3b866d6cU>;
    static_assert(type::k == 2U);
};

template <>
struct db31_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0xc3675f0dU, 0x415ade52U>;
    static_assert(type::k == 2U);
};

template <>
struct db31_inverse::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd0664f8U, 0x3af53fe3U>;
    static_assert(type::k == 2U);
};

template <>
struct db31_inverse::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0xc2158e21U, 0x41dbedd6U>;
    static_assert(type::k == 2U);
};

template <>
struct db31_inverse::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc9cc5e1U, 0x3cc61e42U>;
    static_assert(type::k == 2U);
};

template <>
struct db31_inverse::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0xc20dadf5U, 0x42327f23U>;
    static_assert(type::k == 2U);
};

template <>
struct db31_inverse::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc8a991aU, 0x3cb66e8fU>;
    static_assert(type::k == 2U);
};

template <>
struct db31_inverse::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0xc1a72e8aU, 0x42257539U>;
    static_assert(type::k == 2U);
};

template <>
struct db31_inverse::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbbbac730U, 0x3cc43a64U>;
    static_assert(type::k == 2U);
};

template <>
struct db31_inverse::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0xc174c323U, 0x420f2817U>;
    static_assert(type::k == 2U);
};

template <>
struct db31_inverse::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc69bd9bU, 0x3c68e67aU>;
    static_assert(type::k == 2U);
};

template <>
struct db31_inverse::step<30> {
    using type = StaticStep<StepType::kPredict, -1, 0xc204e8c1U, 0x41db1603U>;
    static_assert(type::k == 2U);
};

template <>
struct db31_inverse::step<31> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc6d5fdaU, 0x3c3b639eU>;
    static_assert(type::k == 2U);
};

template <>
struct db31_inverse::step<32> {
    using type = StaticStep<StepType::kPredict, -1, 0xc1e0f07fU, 0x41ab0325U>;
    static_assert(type::k == 2U);
};

template <>
struct db31_inverse::step<33> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e0bfab0U, 0x3bf990deU>;
    static_assert(type::k == 2U);
};

template <>
struct db31_inverse::step<34> {
    using type = StaticStep<StepType::kPredict, 0, 0xc0d9992dU>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
