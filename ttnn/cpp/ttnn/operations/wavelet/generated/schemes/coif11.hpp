// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct coif11_inverse;

struct coif11 {
    static constexpr const char* name = "coif11";
    static constexpr uint32_t tap_size = 66U;
    static constexpr int32_t delay_even = 16;
    static constexpr int32_t delay_odd = 17;
    static constexpr uint32_t num_steps = 37U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/coif11.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::coif11";
    using inverse = coif11_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct coif11::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0xbfafb79aU>;
    static_assert(type::k == 1U);
};

template <>
struct coif11::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ef3aa97U, 0x3f787e0cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif11::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf69e645U, 0x3f515163U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf8709dfU, 0x3f62a41cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif11::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf6bd32bU, 0x3f11c4b9U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf8d34d0U, 0x3f21fe2aU>;
    static_assert(type::k == 2U);
};

template <>
struct coif11::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf108791U, 0x3efb4f26U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf1a4b62U, 0x3f0c4e91U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xbef0dc3fU, 0x3e78feadU>;
    static_assert(type::k == 2U);
};

template <>
struct coif11::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbeb2d5eeU, 0x3e7da62cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif11::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe385cd8U, 0x3d5df01eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif11::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd9f2071U, 0xbd9994f9U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x3d564429U, 0xbe14e371U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e4ff440U, 0xbed34948U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e8a7a9cU, 0xbe96db4aU>;
    static_assert(type::k == 2U);
};

template <>
struct coif11::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ee178c8U, 0xbf406a82U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ece4690U, 0xbf0a8e27U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f4fd7f9U, 0xbf3eb9f7U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f08a03bU, 0xbf2e4708U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f42bb30U, 0xbfa89d98U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f144ba0U, 0xbf66e40cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif11::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f7b9d7eU, 0xc0a22709U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e491cc5U, 0x3f1f3b28U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfcdb0aeU, 0xbebdb50fU>;
    static_assert(type::k == 2U);
};

template <>
struct coif11::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0x402b2eb7U, 0xc09be187U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e51a3fcU, 0xbee8503aU>;
    static_assert(type::k == 2U);
};

template <>
struct coif11::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0x400cf900U, 0xc0b87433U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e31a264U, 0xbf0c23e2U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0x3fe9d236U, 0xc0e5937cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif11::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e0ebb88U, 0xbf361304U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11::step<30> {
    using type = StaticStep<StepType::kPredict, -1, 0x3fb3f873U, 0xc11f3bb0U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11::step<31> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3dcdc94aU, 0xbf8e1841U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11::step<32> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f669b51U, 0xc1a5a268U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11::step<33> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct coif11::step<34> {
    using type = StaticStep<StepType::kPredict, 0, 0x3d45d553U>;
    static_assert(type::k == 1U);
};

template <>
struct coif11::step<35> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3881b000U>;
    static_assert(type::k == 1U);
};

template <>
struct coif11::step<36> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xc67cab3dU>;
    static_assert(type::k == 1U);
};

struct coif11_inverse {
    static constexpr const char* name = "coif11-inverse";
    static constexpr uint32_t tap_size = 66U;
    static constexpr uint32_t num_steps = 37U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/coif11.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::coif11_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct coif11_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xb881b001U>;
    static_assert(type::k == 1U);
};

template <>
struct coif11_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x467cab3eU>;
    static_assert(type::k == 1U);
};

template <>
struct coif11_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0xbd45d553U>;
    static_assert(type::k == 1U);
};

template <>
struct coif11_inverse::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct coif11_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf669b51U, 0x41a5a268U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbdcdc94aU, 0x3f8e1841U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xbfb3f873U, 0x411f3bb0U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe0ebb88U, 0x3f361304U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xbfe9d236U, 0x40e5937cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif11_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe31a264U, 0x3f0c23e2U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xc00cf900U, 0x40b87433U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe51a3fcU, 0x3ee8503aU>;
    static_assert(type::k == 2U);
};

template <>
struct coif11_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xc02b2eb7U, 0x409be187U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fcdb0aeU, 0x3ebdb50fU>;
    static_assert(type::k == 2U);
};

template <>
struct coif11_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe491cc5U, 0xbf1f3b28U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf7b9d7eU, 0x40a22709U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf144ba0U, 0x3f66e40cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif11_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf42bb30U, 0x3fa89d98U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf08a03bU, 0x3f2e4708U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf4fd7f9U, 0x3f3eb9f7U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0xbece4690U, 0x3f0a8e27U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11_inverse::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbee178c8U, 0x3f406a82U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11_inverse::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe8a7a9cU, 0x3e96db4aU>;
    static_assert(type::k == 2U);
};

template <>
struct coif11_inverse::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe4ff440U, 0x3ed34948U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11_inverse::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0xbd564429U, 0x3e14e371U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11_inverse::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d9f2071U, 0x3d9994f9U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11_inverse::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e385cd8U, 0xbd5df01eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif11_inverse::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3eb2d5eeU, 0xbe7da62cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif11_inverse::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ef0dc3fU, 0xbe78feadU>;
    static_assert(type::k == 2U);
};

template <>
struct coif11_inverse::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f1a4b62U, 0xbf0c4e91U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11_inverse::step<30> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f108791U, 0xbefb4f26U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11_inverse::step<31> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f8d34d0U, 0xbf21fe2aU>;
    static_assert(type::k == 2U);
};

template <>
struct coif11_inverse::step<32> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f6bd32bU, 0xbf11c4b9U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11_inverse::step<33> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f8709dfU, 0xbf62a41cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif11_inverse::step<34> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f69e645U, 0xbf515163U>;
    static_assert(type::k == 2U);
};

template <>
struct coif11_inverse::step<35> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbef3aa97U, 0xbf787e0cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif11_inverse::step<36> {
    using type = StaticStep<StepType::kPredict, 0, 0x3fafb79aU>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
