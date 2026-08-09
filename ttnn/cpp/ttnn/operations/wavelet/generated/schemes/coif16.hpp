// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct coif16_inverse;

struct coif16 {
    static constexpr const char* name = "coif16";
    static constexpr uint32_t tap_size = 96U;
    static constexpr int32_t delay_even = 24;
    static constexpr int32_t delay_odd = 24;
    static constexpr uint32_t num_steps = 51U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/coif16.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::coif16";
    using inverse = coif16_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct coif16::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f48dd25U>;
    static_assert(type::k == 1U);
};

template <>
struct coif16::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0x400aa3f5U, 0xbef8a664U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ec01138U, 0xbeda836aU>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x4013e52bU, 0xc01a4d59U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e9fa1f4U, 0xbec6b4f2U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fe5c614U, 0xc02e389fU>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e8490e0U, 0xbeca4e6eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fddc5e7U, 0xbffceea9U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e8ba8a8U, 0xbe83d6a2U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fa97ae1U, 0xbfdebfa4U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e35218eU, 0xbe84ddb6U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f8a9a2fU, 0xbf8d7573U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x3dfc4c61U, 0xbe32ffc0U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f0be2adU, 0xbf57f8f4U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x3d772d07U, 0xbdad78e6U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3dd69a74U, 0xbeccf7b2U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xbca058d5U, 0xbc83ce03U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe9a6c44U, 0x3e026d70U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xbdc05845U, 0x3d3c6fdaU>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf424878U, 0x3f164a9cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe157152U, 0x3df06925U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfb01257U, 0x3f6642a2U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe4170acU, 0x3e42435aU>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfcfb753U, 0x3fafd70eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe9bea14U, 0x3e4f2793U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbff51b9eU, 0x3fdceb64U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe8ca564U, 0x3e9a3f53U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc00e8f27U, 0x3ff79445U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0xbecf17dbU, 0x3e8a0ac4U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc03d2b60U, 0x3ff4453cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<30> {
    using type = StaticStep<StepType::kPredict, -1, 0xbec54105U, 0x3e9a38e5U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<31> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc2363f98U, 0x401a7cd6U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<32> {
    using type = StaticStep<StepType::kPredict, -1, 0x3babfd20U, 0x3cb3c2b3U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<33> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc2fc34b9U, 0xc33dd2a3U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<34> {
    using type = StaticStep<StepType::kPredict, -1, 0xbc457600U, 0x3c00d8d4U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<35> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc315c72fU, 0x42a55b7cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<36> {
    using type = StaticStep<StepType::kPredict, -1, 0xbc5bff37U, 0x3bda8a28U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<37> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc3271ee7U, 0x4294e85bU>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<38> {
    using type = StaticStep<StepType::kPredict, -1, 0xbc780763U, 0x3bc4103cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<39> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc33ec7e2U, 0x42841cd1U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<40> {
    using type = StaticStep<StepType::kPredict, -1, 0xbc8f955cU, 0x3babc1d6U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<41> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc3609b8cU, 0x4264373aU>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<42> {
    using type = StaticStep<StepType::kPredict, -1, 0xbcaca334U, 0x3b91e3d6U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<43> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc38ae52aU, 0x423dcee1U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<44> {
    using type = StaticStep<StepType::kPredict, -1, 0xbcde5efdU, 0x3b6beb3fU>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<45> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc3bec53dU, 0x42135b79U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<46> {
    using type = StaticStep<StepType::kPredict, -1, 0xbd2bb973U, 0x3b2bc442U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<47> {
    using type = StaticStep<StepType::kUpdate, 0, 0x22803075U, 0x41bed13fU>;
    static_assert(type::k == 2U);
};

template <>
struct coif16::step<48> {
    using type = StaticStep<StepType::kPredict, 0, 0xbaa6e778U>;
    static_assert(type::k == 1U);
};

template <>
struct coif16::step<49> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xc588bae2U>;
    static_assert(type::k == 1U);
};

template <>
struct coif16::step<50> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xb96fa7a0U>;
    static_assert(type::k == 1U);
};

struct coif16_inverse {
    static constexpr const char* name = "coif16-inverse";
    static constexpr uint32_t tap_size = 96U;
    static constexpr uint32_t num_steps = 51U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/coif16.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::coif16_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct coif16_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xc588bae2U>;
    static_assert(type::k == 1U);
};

template <>
struct coif16_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xb96fa79fU>;
    static_assert(type::k == 1U);
};

template <>
struct coif16_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0x3aa6e778U>;
    static_assert(type::k == 1U);
};

template <>
struct coif16_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xa2803075U, 0xc1bed13fU>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x3d2bb973U, 0xbb2bc442U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x43bec53dU, 0xc2135b79U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x3cde5efdU, 0xbb6beb3fU>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x438ae52aU, 0xc23dcee1U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x3caca334U, 0xbb91e3d6U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x43609b8cU, 0xc264373aU>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x3c8f955cU, 0xbbabc1d6U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x433ec7e2U, 0xc2841cd1U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x3c780763U, 0xbbc4103cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x43271ee7U, 0xc294e85bU>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x3c5bff37U, 0xbbda8a28U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x4315c72fU, 0xc2a55b7cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x3c457600U, 0xbc00d8d4U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x42fc34b9U, 0x433dd2a3U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xbbabfd20U, 0xbcb3c2b3U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0x42363f98U, 0xc01a7cd6U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ec54105U, 0xbe9a38e5U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0x403d2b60U, 0xbff4453cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ecf17dbU, 0xbe8a0ac4U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0x400e8f27U, 0xbff79445U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e8ca564U, 0xbe9a3f53U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ff51b9eU, 0xbfdceb64U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e9bea14U, 0xbe4f2793U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fcfb753U, 0xbfafd70eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e4170acU, 0xbe42435aU>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fb01257U, 0xbf6642a2U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<30> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e157152U, 0xbdf06925U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<31> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f424878U, 0xbf164a9cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<32> {
    using type = StaticStep<StepType::kPredict, -1, 0x3dc05845U, 0xbd3c6fdaU>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<33> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e9a6c44U, 0xbe026d70U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<34> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ca058d5U, 0x3c83ce03U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<35> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbdd69a74U, 0x3eccf7b2U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<36> {
    using type = StaticStep<StepType::kPredict, -1, 0xbd772d07U, 0x3dad78e6U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<37> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf0be2adU, 0x3f57f8f4U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<38> {
    using type = StaticStep<StepType::kPredict, -1, 0xbdfc4c61U, 0x3e32ffc0U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<39> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf8a9a2fU, 0x3f8d7573U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<40> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe35218eU, 0x3e84ddb6U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<41> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfa97ae1U, 0x3fdebfa4U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<42> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe8ba8a8U, 0x3e83d6a2U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<43> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfddc5e7U, 0x3ffceea9U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<44> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe8490e0U, 0x3eca4e6eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<45> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfe5c614U, 0x402e389fU>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<46> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe9fa1f4U, 0x3ec6b4f2U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<47> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc013e52bU, 0x401a4d59U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<48> {
    using type = StaticStep<StepType::kPredict, -1, 0xbec01138U, 0x3eda836aU>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<49> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc00aa3f5U, 0x3ef8a664U>;
    static_assert(type::k == 2U);
};

template <>
struct coif16_inverse::step<50> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf48dd25U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
