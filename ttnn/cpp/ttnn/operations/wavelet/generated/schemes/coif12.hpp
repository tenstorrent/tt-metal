// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct coif12_inverse;

struct coif12 {
    static constexpr const char* name = "coif12";
    static constexpr uint32_t tap_size = 72U;
    static constexpr int32_t delay_even = 18;
    static constexpr int32_t delay_odd = 18;
    static constexpr uint32_t num_steps = 39U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/coif12.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::coif12";
    using inverse = coif12_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct coif12::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f3df1fbU>;
    static_assert(type::k == 1U);
};

template <>
struct coif12::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ff359a5U, 0xbef501f7U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ed81474U, 0xbef18395U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fe7f6ffU, 0xc0049615U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e9da792U, 0xbeedd75eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif12::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fa894eaU, 0xc01197b6U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e86de2dU, 0xbeaad398U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fa2de51U, 0xbf9f7f13U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e2a069fU, 0xbe87d076U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f281772U, 0xbf8047d2U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x3db07defU, 0xbdf62c1fU>;
    static_assert(type::k == 2U);
};

template <>
struct coif12::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e00f171U, 0xbefaa382U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xbce236f2U, 0xbcbd0aeaU>;
    static_assert(type::k == 2U);
};

template <>
struct coif12::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbeb55271U, 0x3e1a1aabU>;
    static_assert(type::k == 2U);
};

template <>
struct coif12::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe07dc56U, 0x3d829f08U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf689e5aU, 0x3f2f696eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif12::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe440bebU, 0x3e288b87U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfd99bcaU, 0x3f8080f1U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe8a3c90U, 0x3e744b14U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfcef3aeU, 0x3fdd17ddU>;
    static_assert(type::k == 2U);
};

template <>
struct coif12::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0xbeba48d3U, 0x3e895346U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc02ee850U, 0x3fcdc467U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0xbedef8adU, 0x3e9668cdU>;
    static_assert(type::k == 2U);
};

template <>
struct coif12::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc146d331U, 0x4004307aU>;
    static_assert(type::k == 2U);
};

template <>
struct coif12::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e12cf97U, 0x3da44d12U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc0128dafU, 0xc0defbb0U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf412ee7U, 0x3edd9b72U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc031a68fU, 0x3fa91f94U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf60924dU, 0x3eb85436U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc0515a16U, 0x3f91e5a5U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12::step<30> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf8776feU, 0x3e9c84deU>;
    static_assert(type::k == 2U);
};

template <>
struct coif12::step<31> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc0822fb0U, 0x3f71e48eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif12::step<32> {
    using type = StaticStep<StepType::kPredict, -1, 0xbfafb51fU, 0x3e7bb36bU>;
    static_assert(type::k == 2U);
};

template <>
struct coif12::step<33> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc0b42ff7U, 0x3f3a7de4U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12::step<34> {
    using type = StaticStep<StepType::kPredict, -1, 0xc008c6b5U, 0x3e35daebU>;
    static_assert(type::k == 2U);
};

template <>
struct coif12::step<35> {
    using type = StaticStep<StepType::kUpdate, 0, 0x2799a8fcU, 0x3eef92e7U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12::step<36> {
    using type = StaticStep<StepType::kPredict, 0, 0xbdaf4b8bU>;
    static_assert(type::k == 1U);
};

template <>
struct coif12::step<37> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xc5044914U>;
    static_assert(type::k == 1U);
};

template <>
struct coif12::step<38> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xb9f7b4ecU>;
    static_assert(type::k == 1U);
};

struct coif12_inverse {
    static constexpr const char* name = "coif12-inverse";
    static constexpr uint32_t tap_size = 72U;
    static constexpr uint32_t num_steps = 39U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/coif12.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::coif12_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct coif12_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xc5044914U>;
    static_assert(type::k == 1U);
};

template <>
struct coif12_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xb9f7b4edU>;
    static_assert(type::k == 1U);
};

template <>
struct coif12_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0x3daf4b8bU>;
    static_assert(type::k == 1U);
};

template <>
struct coif12_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xa799a8fcU, 0xbeef92e7U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x4008c6b5U, 0xbe35daebU>;
    static_assert(type::k == 2U);
};

template <>
struct coif12_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x40b42ff7U, 0xbf3a7de4U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x3fafb51fU, 0xbe7bb36bU>;
    static_assert(type::k == 2U);
};

template <>
struct coif12_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x40822fb0U, 0xbf71e48eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif12_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f8776feU, 0xbe9c84deU>;
    static_assert(type::k == 2U);
};

template <>
struct coif12_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x40515a16U, 0xbf91e5a5U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f60924dU, 0xbeb85436U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x4031a68fU, 0xbfa91f94U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f412ee7U, 0xbedd9b72U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x40128dafU, 0x40defbb0U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe12cf97U, 0xbda44d12U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x4146d331U, 0xc004307aU>;
    static_assert(type::k == 2U);
};

template <>
struct coif12_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x3edef8adU, 0xbe9668cdU>;
    static_assert(type::k == 2U);
};

template <>
struct coif12_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x402ee850U, 0xbfcdc467U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0x3eba48d3U, 0xbe895346U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fcef3aeU, 0xbfdd17ddU>;
    static_assert(type::k == 2U);
};

template <>
struct coif12_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e8a3c90U, 0xbe744b14U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12_inverse::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fd99bcaU, 0xbf8080f1U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12_inverse::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e440bebU, 0xbe288b87U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12_inverse::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f689e5aU, 0xbf2f696eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif12_inverse::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e07dc56U, 0xbd829f08U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12_inverse::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3eb55271U, 0xbe1a1aabU>;
    static_assert(type::k == 2U);
};

template <>
struct coif12_inverse::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ce236f2U, 0x3cbd0aeaU>;
    static_assert(type::k == 2U);
};

template <>
struct coif12_inverse::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe00f171U, 0x3efaa382U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12_inverse::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0xbdb07defU, 0x3df62c1fU>;
    static_assert(type::k == 2U);
};

template <>
struct coif12_inverse::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf281772U, 0x3f8047d2U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12_inverse::step<30> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe2a069fU, 0x3e87d076U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12_inverse::step<31> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfa2de51U, 0x3f9f7f13U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12_inverse::step<32> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe86de2dU, 0x3eaad398U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12_inverse::step<33> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfa894eaU, 0x401197b6U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12_inverse::step<34> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe9da792U, 0x3eedd75eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif12_inverse::step<35> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfe7f6ffU, 0x40049615U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12_inverse::step<36> {
    using type = StaticStep<StepType::kPredict, -1, 0xbed81474U, 0x3ef18395U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12_inverse::step<37> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbff359a5U, 0x3ef501f7U>;
    static_assert(type::k == 2U);
};

template <>
struct coif12_inverse::step<38> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf3df1fbU>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
