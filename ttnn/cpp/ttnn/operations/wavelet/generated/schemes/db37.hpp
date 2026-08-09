// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct db37_inverse;

struct db37 {
    static constexpr const char* name = "db37";
    static constexpr uint32_t tap_size = 74U;
    static constexpr int32_t delay_even = 18;
    static constexpr int32_t delay_odd = 19;
    static constexpr uint32_t num_steps = 41U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db37.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db37";
    using inverse = db37_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct db37::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0x4093d72cU>;
    static_assert(type::k == 1U);
};

template <>
struct db37::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d9746fbU, 0xbca79bc9U>;
    static_assert(type::k == 2U);
};

template <>
struct db37::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xbd558556U, 0xc03f5191U>;
    static_assert(type::k == 2U);
};

template <>
struct db37::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf822602U, 0xbf91dd41U>;
    static_assert(type::k == 2U);
};

template <>
struct db37::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x3efeb75eU, 0xbeac89e3U>;
    static_assert(type::k == 2U);
};

template <>
struct db37::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fc48be6U, 0xbfc56267U>;
    static_assert(type::k == 2U);
};

template <>
struct db37::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x3eb6bf5dU, 0xbecbbad0U>;
    static_assert(type::k == 2U);
};

template <>
struct db37::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fc3106cU, 0xbfd1ae79U>;
    static_assert(type::k == 2U);
};

template <>
struct db37::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ecc8896U, 0xbe84b581U>;
    static_assert(type::k == 2U);
};

template <>
struct db37::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x4012c9bbU, 0xbf0a1445U>;
    static_assert(type::k == 2U);
};

template <>
struct db37::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f216ad6U, 0xbe43f136U>;
    static_assert(type::k == 2U);
};

template <>
struct db37::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fe7c019U, 0xbf9622d9U>;
    static_assert(type::k == 2U);
};

template <>
struct db37::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f1668bdU, 0xbeed5fa9U>;
    static_assert(type::k == 2U);
};

template <>
struct db37::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fe930a2U, 0xbfc1ec82U>;
    static_assert(type::k == 2U);
};

template <>
struct db37::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e849ecfU, 0xbf01e785U>;
    static_assert(type::k == 2U);
};

template <>
struct db37::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e70dc6aU, 0xc05c0bb7U>;
    static_assert(type::k == 2U);
};

template <>
struct db37::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e394a76U, 0xbff3d554U>;
    static_assert(type::k == 2U);
};

template <>
struct db37::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f00c879U, 0xbf925704U>;
    static_assert(type::k == 2U);
};

template <>
struct db37::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f5cee04U, 0xc0c5fd98U>;
    static_assert(type::k == 2U);
};

template <>
struct db37::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e254ba1U, 0x3a4586dcU>;
    static_assert(type::k == 2U);
};

template <>
struct db37::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0xc4c744c2U, 0x41af5d71U>;
    static_assert(type::k == 2U);
};

template <>
struct db37::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3bc8ed22U, 0x3a263ce1U>;
    static_assert(type::k == 2U);
};

template <>
struct db37::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0x438426c9U, 0xc322fae1U>;
    static_assert(type::k == 2U);
};

template <>
struct db37::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0xba85e2d2U, 0xbb77e7fcU>;
    static_assert(type::k == 2U);
};

template <>
struct db37::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0xc065a8feU, 0x4474d5edU>;
    static_assert(type::k == 2U);
};

template <>
struct db37::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0xba976cacU, 0x3e951d70U>;
    static_assert(type::k == 2U);
};

template <>
struct db37::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0xc05bcb1eU, 0xc3810555U>;
    static_assert(type::k == 2U);
};

template <>
struct db37::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3b7df967U, 0xbc0d928eU>;
    static_assert(type::k == 2U);
};

template <>
struct db37::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0x42e77517U, 0xc3c26364U>;
    static_assert(type::k == 2U);
};

template <>
struct db37::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3b2891e3U, 0xbc245603U>;
    static_assert(type::k == 2U);
};

template <>
struct db37::step<30> {
    using type = StaticStep<StepType::kPredict, -1, 0x42c7657aU, 0xc3e4884aU>;
    static_assert(type::k == 2U);
};

template <>
struct db37::step<31> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3b0f626eU, 0xbc452990U>;
    static_assert(type::k == 2U);
};

template <>
struct db37::step<32> {
    using type = StaticStep<StepType::kPredict, -1, 0x42a632b3U, 0xc40cf44dU>;
    static_assert(type::k == 2U);
};

template <>
struct db37::step<33> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ae878f6U, 0xbc7d3be1U>;
    static_assert(type::k == 2U);
};

template <>
struct db37::step<34> {
    using type = StaticStep<StepType::kPredict, -1, 0x428165eeU, 0xc4410481U>;
    static_assert(type::k == 2U);
};

template <>
struct db37::step<35> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3aa9c454U, 0xbcc2f5c2U>;
    static_assert(type::k == 2U);
};

template <>
struct db37::step<36> {
    using type = StaticStep<StepType::kPredict, -1, 0x42281355U, 0xc4c6086bU>;
    static_assert(type::k == 2U);
};

template <>
struct db37::step<37> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct db37::step<38> {
    using type = StaticStep<StepType::kPredict, 0, 0x3a2577acU>;
    static_assert(type::k == 1U);
};

template <>
struct db37::step<39> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xb32816ddU>;
    static_assert(type::k == 1U);
};

template <>
struct db37::step<40> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x4bc2f1a9U>;
    static_assert(type::k == 1U);
};

struct db37_inverse {
    static constexpr const char* name = "db37-inverse";
    static constexpr uint32_t tap_size = 74U;
    static constexpr uint32_t num_steps = 41U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db37.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db37_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct db37_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x332816ddU>;
    static_assert(type::k == 1U);
};

template <>
struct db37_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xcbc2f1a9U>;
    static_assert(type::k == 1U);
};

template <>
struct db37_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0xba2577acU>;
    static_assert(type::k == 1U);
};

template <>
struct db37_inverse::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct db37_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xc2281355U, 0x44c6086bU>;
    static_assert(type::k == 2U);
};

template <>
struct db37_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbaa9c454U, 0x3cc2f5c2U>;
    static_assert(type::k == 2U);
};

template <>
struct db37_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xc28165eeU, 0x44410481U>;
    static_assert(type::k == 2U);
};

template <>
struct db37_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbae878f6U, 0x3c7d3be1U>;
    static_assert(type::k == 2U);
};

template <>
struct db37_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xc2a632b3U, 0x440cf44dU>;
    static_assert(type::k == 2U);
};

template <>
struct db37_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbb0f626eU, 0x3c452990U>;
    static_assert(type::k == 2U);
};

template <>
struct db37_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xc2c7657aU, 0x43e4884aU>;
    static_assert(type::k == 2U);
};

template <>
struct db37_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbb2891e3U, 0x3c245603U>;
    static_assert(type::k == 2U);
};

template <>
struct db37_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xc2e77517U, 0x43c26364U>;
    static_assert(type::k == 2U);
};

template <>
struct db37_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbb7df967U, 0x3c0d928eU>;
    static_assert(type::k == 2U);
};

template <>
struct db37_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x405bcb1eU, 0x43810555U>;
    static_assert(type::k == 2U);
};

template <>
struct db37_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3a976cacU, 0xbe951d70U>;
    static_assert(type::k == 2U);
};

template <>
struct db37_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x4065a8feU, 0xc474d5edU>;
    static_assert(type::k == 2U);
};

template <>
struct db37_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3a85e2d2U, 0x3b77e7fcU>;
    static_assert(type::k == 2U);
};

template <>
struct db37_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xc38426c9U, 0x4322fae1U>;
    static_assert(type::k == 2U);
};

template <>
struct db37_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbbc8ed22U, 0xba263ce1U>;
    static_assert(type::k == 2U);
};

template <>
struct db37_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0x44c744c2U, 0xc1af5d71U>;
    static_assert(type::k == 2U);
};

template <>
struct db37_inverse::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe254ba1U, 0xba4586dcU>;
    static_assert(type::k == 2U);
};

template <>
struct db37_inverse::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf5cee04U, 0x40c5fd98U>;
    static_assert(type::k == 2U);
};

template <>
struct db37_inverse::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf00c879U, 0x3f925704U>;
    static_assert(type::k == 2U);
};

template <>
struct db37_inverse::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe394a76U, 0x3ff3d554U>;
    static_assert(type::k == 2U);
};

template <>
struct db37_inverse::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe70dc6aU, 0x405c0bb7U>;
    static_assert(type::k == 2U);
};

template <>
struct db37_inverse::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe849ecfU, 0x3f01e785U>;
    static_assert(type::k == 2U);
};

template <>
struct db37_inverse::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfe930a2U, 0x3fc1ec82U>;
    static_assert(type::k == 2U);
};

template <>
struct db37_inverse::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf1668bdU, 0x3eed5fa9U>;
    static_assert(type::k == 2U);
};

template <>
struct db37_inverse::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfe7c019U, 0x3f9622d9U>;
    static_assert(type::k == 2U);
};

template <>
struct db37_inverse::step<30> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf216ad6U, 0x3e43f136U>;
    static_assert(type::k == 2U);
};

template <>
struct db37_inverse::step<31> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc012c9bbU, 0x3f0a1445U>;
    static_assert(type::k == 2U);
};

template <>
struct db37_inverse::step<32> {
    using type = StaticStep<StepType::kPredict, -1, 0xbecc8896U, 0x3e84b581U>;
    static_assert(type::k == 2U);
};

template <>
struct db37_inverse::step<33> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfc3106cU, 0x3fd1ae79U>;
    static_assert(type::k == 2U);
};

template <>
struct db37_inverse::step<34> {
    using type = StaticStep<StepType::kPredict, -1, 0xbeb6bf5dU, 0x3ecbbad0U>;
    static_assert(type::k == 2U);
};

template <>
struct db37_inverse::step<35> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfc48be6U, 0x3fc56267U>;
    static_assert(type::k == 2U);
};

template <>
struct db37_inverse::step<36> {
    using type = StaticStep<StepType::kPredict, -1, 0xbefeb75eU, 0x3eac89e3U>;
    static_assert(type::k == 2U);
};

template <>
struct db37_inverse::step<37> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f822602U, 0x3f91dd41U>;
    static_assert(type::k == 2U);
};

template <>
struct db37_inverse::step<38> {
    using type = StaticStep<StepType::kPredict, -1, 0x3d558556U, 0x403f5191U>;
    static_assert(type::k == 2U);
};

template <>
struct db37_inverse::step<39> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd9746fbU, 0x3ca79bc9U>;
    static_assert(type::k == 2U);
};

template <>
struct db37_inverse::step<40> {
    using type = StaticStep<StepType::kPredict, 0, 0xc093d72cU>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
