// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct db35_inverse;

struct db35 {
    static constexpr const char* name = "db35";
    static constexpr uint32_t tap_size = 70U;
    static constexpr int32_t delay_even = 17;
    static constexpr int32_t delay_odd = 18;
    static constexpr uint32_t num_steps = 39U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db35.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db35";
    using inverse = db35_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct db35::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0x408b275cU>;
    static_assert(type::k == 1U);
};

template <>
struct db35::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe553d22U, 0xbce27b28U>;
    static_assert(type::k == 2U);
};

template <>
struct db35::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0x411d56efU, 0xc145e8a3U>;
    static_assert(type::k == 2U);
};

template <>
struct db35::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3cf77aa6U, 0xbd18c670U>;
    static_assert(type::k == 2U);
};

template <>
struct db35::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x413c0362U, 0xc161d36fU>;
    static_assert(type::k == 2U);
};

template <>
struct db35::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d11fd80U, 0xbd202bc7U>;
    static_assert(type::k == 2U);
};

template <>
struct db35::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x41627239U, 0xc122f81eU>;
    static_assert(type::k == 2U);
};

template <>
struct db35::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d4ed0c3U, 0xbc898f08U>;
    static_assert(type::k == 2U);
};

template <>
struct db35::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x41a4d7d0U, 0xc0f58707U>;
    static_assert(type::k == 2U);
};

template <>
struct db35::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d42ecbeU, 0xbcffe559U>;
    static_assert(type::k == 2U);
};

template <>
struct db35::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x41a3a5c1U, 0xc1804cacU>;
    static_assert(type::k == 2U);
};

template <>
struct db35::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d8503f0U, 0xbd268664U>;
    static_assert(type::k == 2U);
};

template <>
struct db35::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xc02f65a8U, 0xc15f4647U>;
    static_assert(type::k == 2U);
};

template <>
struct db35::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc53e74dU, 0x3f2398bfU>;
    static_assert(type::k == 2U);
};

template <>
struct db35::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xbfdb7e53U, 0xc0218b56U>;
    static_assert(type::k == 2U);
};

template <>
struct db35::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ec20e6dU, 0xbeec7b65U>;
    static_assert(type::k == 2U);
};

template <>
struct db35::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x4006e189U, 0xc0eee19fU>;
    static_assert(type::k == 2U);
};

template <>
struct db35::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e087040U, 0x3adefd41U>;
    static_assert(type::k == 2U);
};

template <>
struct db35::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xc44ba2d4U, 0x412b7f97U>;
    static_assert(type::k == 2U);
};

template <>
struct db35::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3bb2ef70U, 0x3aa34413U>;
    static_assert(type::k == 2U);
};

template <>
struct db35::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0x4391cab7U, 0xc336c2e7U>;
    static_assert(type::k == 2U);
};

template <>
struct db35::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3b29183dU, 0xbb6099c1U>;
    static_assert(type::k == 2U);
};

template <>
struct db35::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0x4039e451U, 0xc3c1b14dU>;
    static_assert(type::k == 2U);
};

template <>
struct db35::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ad0eda8U, 0xbeab471aU>;
    static_assert(type::k == 2U);
};

template <>
struct db35::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0x403f0e28U, 0xc336adc8U>;
    static_assert(type::k == 2U);
};

template <>
struct db35::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3bb35f8fU, 0xbca9b9e8U>;
    static_assert(type::k == 2U);
};

template <>
struct db35::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0x42411050U, 0xc3190e06U>;
    static_assert(type::k == 2U);
};

template <>
struct db35::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3bd617e0U, 0xbcc3c333U>;
    static_assert(type::k == 2U);
};

template <>
struct db35::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0x422762f2U, 0xc33429e3U>;
    static_assert(type::k == 2U);
};

template <>
struct db35::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3bb5e10eU, 0xbceb339cU>;
    static_assert(type::k == 2U);
};

template <>
struct db35::step<30> {
    using type = StaticStep<StepType::kPredict, -1, 0x420b519cU, 0xc35e90efU>;
    static_assert(type::k == 2U);
};

template <>
struct db35::step<31> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3b933a68U, 0xbd1743d3U>;
    static_assert(type::k == 2U);
};

template <>
struct db35::step<32> {
    using type = StaticStep<StepType::kPredict, -1, 0x41d8a065U, 0xc3989adcU>;
    static_assert(type::k == 2U);
};

template <>
struct db35::step<33> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3b56b973U, 0xbd693cceU>;
    static_assert(type::k == 2U);
};

template <>
struct db35::step<34> {
    using type = StaticStep<StepType::kPredict, -1, 0x418c7df2U, 0xc41cc85cU>;
    static_assert(type::k == 2U);
};

template <>
struct db35::step<35> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct db35::step<36> {
    using type = StaticStep<StepType::kPredict, 0, 0x3ad100b2U>;
    static_assert(type::k == 1U);
};

template <>
struct db35::step<37> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3406b254U>;
    static_assert(type::k == 1U);
};

template <>
struct db35::step<38> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xcaf345ceU>;
    static_assert(type::k == 1U);
};

struct db35_inverse {
    static constexpr const char* name = "db35-inverse";
    static constexpr uint32_t tap_size = 70U;
    static constexpr uint32_t num_steps = 39U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db35.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db35_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct db35_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xb406b254U>;
    static_assert(type::k == 1U);
};

template <>
struct db35_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x4af345ceU>;
    static_assert(type::k == 1U);
};

template <>
struct db35_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0xbad100b2U>;
    static_assert(type::k == 1U);
};

template <>
struct db35_inverse::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct db35_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xc18c7df2U, 0x441cc85cU>;
    static_assert(type::k == 2U);
};

template <>
struct db35_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbb56b973U, 0x3d693cceU>;
    static_assert(type::k == 2U);
};

template <>
struct db35_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xc1d8a065U, 0x43989adcU>;
    static_assert(type::k == 2U);
};

template <>
struct db35_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbb933a68U, 0x3d1743d3U>;
    static_assert(type::k == 2U);
};

template <>
struct db35_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xc20b519cU, 0x435e90efU>;
    static_assert(type::k == 2U);
};

template <>
struct db35_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbbb5e10eU, 0x3ceb339cU>;
    static_assert(type::k == 2U);
};

template <>
struct db35_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xc22762f2U, 0x433429e3U>;
    static_assert(type::k == 2U);
};

template <>
struct db35_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbbd617e0U, 0x3cc3c333U>;
    static_assert(type::k == 2U);
};

template <>
struct db35_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xc2411050U, 0x43190e06U>;
    static_assert(type::k == 2U);
};

template <>
struct db35_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbbb35f8fU, 0x3ca9b9e8U>;
    static_assert(type::k == 2U);
};

template <>
struct db35_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xc03f0e28U, 0x4336adc8U>;
    static_assert(type::k == 2U);
};

template <>
struct db35_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbad0eda8U, 0x3eab471aU>;
    static_assert(type::k == 2U);
};

template <>
struct db35_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xc039e451U, 0x43c1b14dU>;
    static_assert(type::k == 2U);
};

template <>
struct db35_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbb29183dU, 0x3b6099c1U>;
    static_assert(type::k == 2U);
};

template <>
struct db35_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xc391cab7U, 0x4336c2e7U>;
    static_assert(type::k == 2U);
};

template <>
struct db35_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbbb2ef70U, 0xbaa34413U>;
    static_assert(type::k == 2U);
};

template <>
struct db35_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0x444ba2d4U, 0xc12b7f97U>;
    static_assert(type::k == 2U);
};

template <>
struct db35_inverse::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe087040U, 0xbadefd41U>;
    static_assert(type::k == 2U);
};

template <>
struct db35_inverse::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0xc006e189U, 0x40eee19fU>;
    static_assert(type::k == 2U);
};

template <>
struct db35_inverse::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbec20e6dU, 0x3eec7b65U>;
    static_assert(type::k == 2U);
};

template <>
struct db35_inverse::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0x3fdb7e53U, 0x40218b56U>;
    static_assert(type::k == 2U);
};

template <>
struct db35_inverse::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c53e74dU, 0xbf2398bfU>;
    static_assert(type::k == 2U);
};

template <>
struct db35_inverse::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0x402f65a8U, 0x415f4647U>;
    static_assert(type::k == 2U);
};

template <>
struct db35_inverse::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd8503f0U, 0x3d268664U>;
    static_assert(type::k == 2U);
};

template <>
struct db35_inverse::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0xc1a3a5c1U, 0x41804cacU>;
    static_assert(type::k == 2U);
};

template <>
struct db35_inverse::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd42ecbeU, 0x3cffe559U>;
    static_assert(type::k == 2U);
};

template <>
struct db35_inverse::step<30> {
    using type = StaticStep<StepType::kPredict, -1, 0xc1a4d7d0U, 0x40f58707U>;
    static_assert(type::k == 2U);
};

template <>
struct db35_inverse::step<31> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd4ed0c3U, 0x3c898f08U>;
    static_assert(type::k == 2U);
};

template <>
struct db35_inverse::step<32> {
    using type = StaticStep<StepType::kPredict, -1, 0xc1627239U, 0x4122f81eU>;
    static_assert(type::k == 2U);
};

template <>
struct db35_inverse::step<33> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd11fd80U, 0x3d202bc7U>;
    static_assert(type::k == 2U);
};

template <>
struct db35_inverse::step<34> {
    using type = StaticStep<StepType::kPredict, -1, 0xc13c0362U, 0x4161d36fU>;
    static_assert(type::k == 2U);
};

template <>
struct db35_inverse::step<35> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbcf77aa6U, 0x3d18c670U>;
    static_assert(type::k == 2U);
};

template <>
struct db35_inverse::step<36> {
    using type = StaticStep<StepType::kPredict, -1, 0xc11d56efU, 0x4145e8a3U>;
    static_assert(type::k == 2U);
};

template <>
struct db35_inverse::step<37> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e553d22U, 0x3ce27b28U>;
    static_assert(type::k == 2U);
};

template <>
struct db35_inverse::step<38> {
    using type = StaticStep<StepType::kPredict, 0, 0xc08b275cU>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
