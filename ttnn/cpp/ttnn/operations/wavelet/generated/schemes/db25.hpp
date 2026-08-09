// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct db25_inverse;

struct db25 {
    static constexpr const char* name = "db25";
    static constexpr uint32_t tap_size = 50U;
    static constexpr int32_t delay_even = 12;
    static constexpr int32_t delay_odd = 13;
    static constexpr uint32_t num_steps = 29U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db25.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db25";
    using inverse = db25_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct db25::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0x40abe960U>;
    static_assert(type::k == 1U);
};

template <>
struct db25::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe33d698U, 0xbc8043f9U>;
    static_assert(type::k == 2U);
};

template <>
struct db25::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0x4141933aU, 0xc18870d6U>;
    static_assert(type::k == 2U);
};

template <>
struct db25::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c8fcb53U, 0xbcc053d6U>;
    static_assert(type::k == 2U);
};

template <>
struct db25::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x4187f907U, 0xc1b27147U>;
    static_assert(type::k == 2U);
};

template <>
struct db25::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3cbae63eU, 0xbd1b30d5U>;
    static_assert(type::k == 2U);
};

template <>
struct db25::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x418e7013U, 0x407683a9U>;
    static_assert(type::k == 2U);
};

template <>
struct db25::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e85be15U, 0x3beb6bf5U>;
    static_assert(type::k == 2U);
};

template <>
struct db25::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x3fcf8333U, 0xc053e963U>;
    static_assert(type::k == 2U);
};

template <>
struct db25::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f208548U, 0xbefd101cU>;
    static_assert(type::k == 2U);
};

template <>
struct db25::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ff2d980U, 0xbfb1a570U>;
    static_assert(type::k == 2U);
};

template <>
struct db25::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fa54d09U, 0xbef9dbe9U>;
    static_assert(type::k == 2U);
};

template <>
struct db25::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xbc0743d4U, 0xbf422576U>;
    static_assert(type::k == 2U);
};

template <>
struct db25::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf85a573U, 0xc3dcde1cU>;
    static_assert(type::k == 2U);
};

template <>
struct db25::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x3b13daaaU, 0xba50a387U>;
    static_assert(type::k == 2U);
};

template <>
struct db25::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x449c285bU, 0xc5113465U>;
    static_assert(type::k == 2U);
};

template <>
struct db25::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x39e14c1bU, 0xba7127dbU>;
    static_assert(type::k == 2U);
};

template <>
struct db25::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x4487d48fU, 0xc528aa69U>;
    static_assert(type::k == 2U);
};

template <>
struct db25::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0x39c2443cU, 0xba8eb701U>;
    static_assert(type::k == 2U);
};

template <>
struct db25::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0x44659a67U, 0xc54c33f8U>;
    static_assert(type::k == 2U);
};

template <>
struct db25::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0x39a077baU, 0xbab1e3f6U>;
    static_assert(type::k == 2U);
};

template <>
struct db25::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0x44383407U, 0xc584986aU>;
    static_assert(type::k == 2U);
};

template <>
struct db25::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0x397720b7U, 0xbaf6598fU>;
    static_assert(type::k == 2U);
};

template <>
struct db25::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0x4405039cU, 0xc5ce74cfU>;
    static_assert(type::k == 2U);
};

template <>
struct db25::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0x391eb766U, 0xbb7f8679U>;
    static_assert(type::k == 2U);
};

template <>
struct db25::step<25> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct db25::step<26> {
    using type = StaticStep<StepType::kPredict, 0, 0x43803ce0U>;
    static_assert(type::k == 1U);
};

template <>
struct db25::step<27> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3ade23ecU>;
    static_assert(type::k == 1U);
};

template <>
struct db25::step<28> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xc41382a7U>;
    static_assert(type::k == 1U);
};

struct db25_inverse {
    static constexpr const char* name = "db25-inverse";
    static constexpr uint32_t tap_size = 50U;
    static constexpr uint32_t num_steps = 29U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db25.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db25_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct db25_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xbade23edU>;
    static_assert(type::k == 1U);
};

template <>
struct db25_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x441382a7U>;
    static_assert(type::k == 1U);
};

template <>
struct db25_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0xc3803ce0U>;
    static_assert(type::k == 1U);
};

template <>
struct db25_inverse::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct db25_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xb91eb766U, 0x3b7f8679U>;
    static_assert(type::k == 2U);
};

template <>
struct db25_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc405039cU, 0x45ce74cfU>;
    static_assert(type::k == 2U);
};

template <>
struct db25_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xb97720b7U, 0x3af6598fU>;
    static_assert(type::k == 2U);
};

template <>
struct db25_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc4383407U, 0x4584986aU>;
    static_assert(type::k == 2U);
};

template <>
struct db25_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xb9a077baU, 0x3ab1e3f6U>;
    static_assert(type::k == 2U);
};

template <>
struct db25_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc4659a67U, 0x454c33f8U>;
    static_assert(type::k == 2U);
};

template <>
struct db25_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xb9c2443cU, 0x3a8eb701U>;
    static_assert(type::k == 2U);
};

template <>
struct db25_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc487d48fU, 0x4528aa69U>;
    static_assert(type::k == 2U);
};

template <>
struct db25_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xb9e14c1bU, 0x3a7127dbU>;
    static_assert(type::k == 2U);
};

template <>
struct db25_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc49c285bU, 0x45113465U>;
    static_assert(type::k == 2U);
};

template <>
struct db25_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xbb13daaaU, 0x3a50a387U>;
    static_assert(type::k == 2U);
};

template <>
struct db25_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f85a573U, 0x43dcde1cU>;
    static_assert(type::k == 2U);
};

template <>
struct db25_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x3c0743d4U, 0x3f422576U>;
    static_assert(type::k == 2U);
};

template <>
struct db25_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfa54d09U, 0x3ef9dbe9U>;
    static_assert(type::k == 2U);
};

template <>
struct db25_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xbff2d980U, 0x3fb1a570U>;
    static_assert(type::k == 2U);
};

template <>
struct db25_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf208548U, 0x3efd101cU>;
    static_assert(type::k == 2U);
};

template <>
struct db25_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0xbfcf8333U, 0x4053e963U>;
    static_assert(type::k == 2U);
};

template <>
struct db25_inverse::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe85be15U, 0xbbeb6bf5U>;
    static_assert(type::k == 2U);
};

template <>
struct db25_inverse::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0xc18e7013U, 0xc07683a9U>;
    static_assert(type::k == 2U);
};

template <>
struct db25_inverse::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbcbae63eU, 0x3d1b30d5U>;
    static_assert(type::k == 2U);
};

template <>
struct db25_inverse::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0xc187f907U, 0x41b27147U>;
    static_assert(type::k == 2U);
};

template <>
struct db25_inverse::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc8fcb53U, 0x3cc053d6U>;
    static_assert(type::k == 2U);
};

template <>
struct db25_inverse::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0xc141933aU, 0x418870d6U>;
    static_assert(type::k == 2U);
};

template <>
struct db25_inverse::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e33d698U, 0x3c8043f9U>;
    static_assert(type::k == 2U);
};

template <>
struct db25_inverse::step<28> {
    using type = StaticStep<StepType::kPredict, 0, 0xc0abe960U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
