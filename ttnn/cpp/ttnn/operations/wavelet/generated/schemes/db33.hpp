// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct db33_inverse;

struct db33 {
    static constexpr const char* name = "db33";
    static constexpr uint32_t tap_size = 66U;
    static constexpr int32_t delay_even = 16;
    static constexpr int32_t delay_odd = 17;
    static constexpr uint32_t num_steps = 37U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db33.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db33";
    using inverse = db33_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct db33::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0x40878b71U>;
    static_assert(type::k == 1U);
};

template <>
struct db33::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe590adcU, 0xbcee2df0U>;
    static_assert(type::k == 2U);
};

template <>
struct db33::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0x411e4a94U, 0xc12d4df0U>;
    static_assert(type::k == 2U);
};

template <>
struct db33::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d0db327U, 0xbced5eedU>;
    static_assert(type::k == 2U);
};

template <>
struct db33::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x414e8436U, 0xc0f1b997U>;
    static_assert(type::k == 2U);
};

template <>
struct db33::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d379f11U, 0xbcd01449U>;
    static_assert(type::k == 2U);
};

template <>
struct db33::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x416e66e1U, 0xc126d93aU>;
    static_assert(type::k == 2U);
};

template <>
struct db33::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d523afcU, 0xbd211067U>;
    static_assert(type::k == 2U);
};

template <>
struct db33::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x421ddba9U, 0xc154b6f2U>;
    static_assert(type::k == 2U);
};

template <>
struct db33::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xba6c8838U, 0xbcb70c75U>;
    static_assert(type::k == 2U);
};

template <>
struct db33::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xc22ae6c0U, 0xc3f4c10aU>;
    static_assert(type::k == 2U);
};

template <>
struct db33::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3aec04f2U, 0xbaab2153U>;
    static_assert(type::k == 2U);
};

template <>
struct db33::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x442699feU, 0xc44e35bdU>;
    static_assert(type::k == 2U);
};

template <>
struct db33::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3a911de5U, 0xbabfc3ecU>;
    static_assert(type::k == 2U);
};

template <>
struct db33::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x4422020cU, 0xc4a866afU>;
    static_assert(type::k == 2U);
};

template <>
struct db33::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3a3eebbcU, 0x3803c226U>;
    static_assert(type::k == 2U);
};

template <>
struct db33::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xc7341902U, 0x441f0ed3U>;
    static_assert(type::k == 2U);
};

template <>
struct db33::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x38a14604U, 0x37bb579bU>;
    static_assert(type::k == 2U);
};

template <>
struct db33::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0x469ae930U, 0xc64a1744U>;
    static_assert(type::k == 2U);
};

template <>
struct db33::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0xb7dda3feU, 0xb8531577U>;
    static_assert(type::k == 2U);
};

template <>
struct db33::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0xc31805e2U, 0x47142aabU>;
    static_assert(type::k == 2U);
};

template <>
struct db33::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0xb80e8318U, 0x3c136909U>;
    static_assert(type::k == 2U);
};

template <>
struct db33::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0xc2de6244U, 0xc42f9c2dU>;
    static_assert(type::k == 2U);
};

template <>
struct db33::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3aba96fdU, 0xbb65b94cU>;
    static_assert(type::k == 2U);
};

template <>
struct db33::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0x438ea3e3U, 0xc451e6b9U>;
    static_assert(type::k == 2U);
};

template <>
struct db33::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3a9c1c86U, 0xbb855544U>;
    static_assert(type::k == 2U);
};

template <>
struct db33::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0x4375c2aeU, 0xc4777182U>;
    static_assert(type::k == 2U);
};

template <>
struct db33::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3a846d1fU, 0xbba07079U>;
    static_assert(type::k == 2U);
};

template <>
struct db33::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0x434c3d3aU, 0xc4991500U>;
    static_assert(type::k == 2U);
};

template <>
struct db33::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3a560e20U, 0xbbceb209U>;
    static_assert(type::k == 2U);
};

template <>
struct db33::step<30> {
    using type = StaticStep<StepType::kPredict, -1, 0x431e8863U, 0xc4d241a4U>;
    static_assert(type::k == 2U);
};

template <>
struct db33::step<31> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3a1bd90aU, 0xbc1f9994U>;
    static_assert(type::k == 2U);
};

template <>
struct db33::step<32> {
    using type = StaticStep<StepType::kPredict, -1, 0x42cd503bU, 0xc5585663U>;
    static_assert(type::k == 2U);
};

template <>
struct db33::step<33> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct db33::step<34> {
    using type = StaticStep<StepType::kPredict, 0, 0x39977792U>;
    static_assert(type::k == 1U);
};

template <>
struct db33::step<35> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xb3e75fd7U>;
    static_assert(type::k == 1U);
};

template <>
struct db33::step<36> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x4b0d9f90U>;
    static_assert(type::k == 1U);
};

struct db33_inverse {
    static constexpr const char* name = "db33-inverse";
    static constexpr uint32_t tap_size = 66U;
    static constexpr uint32_t num_steps = 37U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db33.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db33_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct db33_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x33e75fd7U>;
    static_assert(type::k == 1U);
};

template <>
struct db33_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xcb0d9f90U>;
    static_assert(type::k == 1U);
};

template <>
struct db33_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0xb9977792U>;
    static_assert(type::k == 1U);
};

template <>
struct db33_inverse::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct db33_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xc2cd503bU, 0x45585663U>;
    static_assert(type::k == 2U);
};

template <>
struct db33_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xba1bd90aU, 0x3c1f9994U>;
    static_assert(type::k == 2U);
};

template <>
struct db33_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xc31e8863U, 0x44d241a4U>;
    static_assert(type::k == 2U);
};

template <>
struct db33_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xba560e20U, 0x3bceb209U>;
    static_assert(type::k == 2U);
};

template <>
struct db33_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xc34c3d3aU, 0x44991500U>;
    static_assert(type::k == 2U);
};

template <>
struct db33_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xba846d1fU, 0x3ba07079U>;
    static_assert(type::k == 2U);
};

template <>
struct db33_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xc375c2aeU, 0x44777182U>;
    static_assert(type::k == 2U);
};

template <>
struct db33_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xba9c1c86U, 0x3b855544U>;
    static_assert(type::k == 2U);
};

template <>
struct db33_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xc38ea3e3U, 0x4451e6b9U>;
    static_assert(type::k == 2U);
};

template <>
struct db33_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbaba96fdU, 0x3b65b94cU>;
    static_assert(type::k == 2U);
};

template <>
struct db33_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x42de6244U, 0x442f9c2dU>;
    static_assert(type::k == 2U);
};

template <>
struct db33_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x380e8318U, 0xbc136909U>;
    static_assert(type::k == 2U);
};

template <>
struct db33_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x431805e2U, 0xc7142aabU>;
    static_assert(type::k == 2U);
};

template <>
struct db33_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x37dda3feU, 0x38531577U>;
    static_assert(type::k == 2U);
};

template <>
struct db33_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xc69ae930U, 0x464a1744U>;
    static_assert(type::k == 2U);
};

template <>
struct db33_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0xb8a14604U, 0xb7bb579bU>;
    static_assert(type::k == 2U);
};

template <>
struct db33_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0x47341902U, 0xc41f0ed3U>;
    static_assert(type::k == 2U);
};

template <>
struct db33_inverse::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0xba3eebbcU, 0xb803c226U>;
    static_assert(type::k == 2U);
};

template <>
struct db33_inverse::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0xc422020cU, 0x44a866afU>;
    static_assert(type::k == 2U);
};

template <>
struct db33_inverse::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0xba911de5U, 0x3abfc3ecU>;
    static_assert(type::k == 2U);
};

template <>
struct db33_inverse::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0xc42699feU, 0x444e35bdU>;
    static_assert(type::k == 2U);
};

template <>
struct db33_inverse::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbaec04f2U, 0x3aab2153U>;
    static_assert(type::k == 2U);
};

template <>
struct db33_inverse::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0x422ae6c0U, 0x43f4c10aU>;
    static_assert(type::k == 2U);
};

template <>
struct db33_inverse::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3a6c8838U, 0x3cb70c75U>;
    static_assert(type::k == 2U);
};

template <>
struct db33_inverse::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0xc21ddba9U, 0x4154b6f2U>;
    static_assert(type::k == 2U);
};

template <>
struct db33_inverse::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd523afcU, 0x3d211067U>;
    static_assert(type::k == 2U);
};

template <>
struct db33_inverse::step<30> {
    using type = StaticStep<StepType::kPredict, -1, 0xc16e66e1U, 0x4126d93aU>;
    static_assert(type::k == 2U);
};

template <>
struct db33_inverse::step<31> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd379f11U, 0x3cd01449U>;
    static_assert(type::k == 2U);
};

template <>
struct db33_inverse::step<32> {
    using type = StaticStep<StepType::kPredict, -1, 0xc14e8436U, 0x40f1b997U>;
    static_assert(type::k == 2U);
};

template <>
struct db33_inverse::step<33> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd0db327U, 0x3ced5eedU>;
    static_assert(type::k == 2U);
};

template <>
struct db33_inverse::step<34> {
    using type = StaticStep<StepType::kPredict, -1, 0xc11e4a94U, 0x412d4df0U>;
    static_assert(type::k == 2U);
};

template <>
struct db33_inverse::step<35> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e590adcU, 0x3cee2df0U>;
    static_assert(type::k == 2U);
};

template <>
struct db33_inverse::step<36> {
    using type = StaticStep<StepType::kPredict, 0, 0xc0878b71U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
