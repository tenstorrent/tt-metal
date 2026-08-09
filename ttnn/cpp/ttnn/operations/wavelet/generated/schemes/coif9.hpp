// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct coif9_inverse;

struct coif9 {
    static constexpr const char* name = "coif9";
    static constexpr uint32_t tap_size = 54U;
    static constexpr int32_t delay_even = 13;
    static constexpr int32_t delay_odd = 14;
    static constexpr uint32_t num_steps = 31U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/coif9.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::coif9";
    using inverse = coif9_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct coif9::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0xbfb7e2dbU>;
    static_assert(type::k == 1U);
};

template <>
struct coif9::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ef0126fU, 0x3f4fb93dU>;
    static_assert(type::k == 2U);
};

template <>
struct coif9::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf87c77eU, 0x3f70104fU>;
    static_assert(type::k == 2U);
};

template <>
struct coif9::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf622b40U, 0x3f26d935U>;
    static_assert(type::k == 2U);
};

template <>
struct coif9::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf8e4543U, 0x3f146413U>;
    static_assert(type::k == 2U);
};

template <>
struct coif9::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf381d06U, 0x3ee47d9cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif9::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xbef7a54dU, 0x3eeaba00U>;
    static_assert(type::k == 2U);
};

template <>
struct coif9::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbedd6d73U, 0x3e4e093cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif9::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe814a84U, 0x3db659cfU>;
    static_assert(type::k == 2U);
};

template <>
struct coif9::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd980fa7U, 0xbd85ab61U>;
    static_assert(type::k == 2U);
};

template <>
struct coif9::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x3da0896cU, 0xbe8231e3U>;
    static_assert(type::k == 2U);
};

template <>
struct coif9::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e4ebcecU, 0xbe9121beU>;
    static_assert(type::k == 2U);
};

template <>
struct coif9::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x3eae1d28U, 0xbf302404U>;
    static_assert(type::k == 2U);
};

template <>
struct coif9::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ee2307aU, 0xbef3fd2bU>;
    static_assert(type::k == 2U);
};

template <>
struct coif9::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f350081U, 0xbf42528eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif9::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ef92299U, 0xbf598825U>;
    static_assert(type::k == 2U);
};

template <>
struct coif9::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f4122f7U, 0xbfa98163U>;
    static_assert(type::k == 2U);
};

template <>
struct coif9::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f239696U, 0xc026d76fU>;
    static_assert(type::k == 2U);
};

template <>
struct coif9::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ec247d6U, 0x42ab099eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif9::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc3f9567U, 0xb86f1775U>;
    static_assert(type::k == 2U);
};

template <>
struct coif9::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0x4687d2ceU, 0xc70a41efU>;
    static_assert(type::k == 2U);
};

template <>
struct coif9::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0x37ec7da6U, 0xb8974b78U>;
    static_assert(type::k == 2U);
};

template <>
struct coif9::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0x465882f9U, 0xc72cbfeeU>;
    static_assert(type::k == 2U);
};

template <>
struct coif9::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0x37bdadb9U, 0xb8c504dcU>;
    static_assert(type::k == 2U);
};

template <>
struct coif9::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0x46265195U, 0xc770ab5eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif9::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0x37882750U, 0xb91aa481U>;
    static_assert(type::k == 2U);
};

template <>
struct coif9::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0x45d3e522U, 0xc7fbf63bU>;
    static_assert(type::k == 2U);
};

template <>
struct coif9::step<27> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct coif9::step<28> {
    using type = StaticStep<StepType::kPredict, 0, 0x37020d2bU>;
    static_assert(type::k == 1U);
};

template <>
struct coif9::step<29> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x365b2ad8U>;
    static_assert(type::k == 1U);
};

template <>
struct coif9::step<30> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xc89582e5U>;
    static_assert(type::k == 1U);
};

struct coif9_inverse {
    static constexpr const char* name = "coif9-inverse";
    static constexpr uint32_t tap_size = 54U;
    static constexpr uint32_t num_steps = 31U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/coif9.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::coif9_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct coif9_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xb65b2ad9U>;
    static_assert(type::k == 1U);
};

template <>
struct coif9_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x489582e6U>;
    static_assert(type::k == 1U);
};

template <>
struct coif9_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0xb7020d2bU>;
    static_assert(type::k == 1U);
};

template <>
struct coif9_inverse::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct coif9_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xc5d3e522U, 0x47fbf63bU>;
    static_assert(type::k == 2U);
};

template <>
struct coif9_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xb7882750U, 0x391aa481U>;
    static_assert(type::k == 2U);
};

template <>
struct coif9_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xc6265195U, 0x4770ab5eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif9_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xb7bdadb9U, 0x38c504dcU>;
    static_assert(type::k == 2U);
};

template <>
struct coif9_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xc65882f9U, 0x472cbfeeU>;
    static_assert(type::k == 2U);
};

template <>
struct coif9_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xb7ec7da6U, 0x38974b78U>;
    static_assert(type::k == 2U);
};

template <>
struct coif9_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xc687d2ceU, 0x470a41efU>;
    static_assert(type::k == 2U);
};

template <>
struct coif9_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c3f9567U, 0x386f1775U>;
    static_assert(type::k == 2U);
};

template <>
struct coif9_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xbec247d6U, 0xc2ab099eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif9_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf239696U, 0x4026d76fU>;
    static_assert(type::k == 2U);
};

template <>
struct coif9_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf4122f7U, 0x3fa98163U>;
    static_assert(type::k == 2U);
};

template <>
struct coif9_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbef92299U, 0x3f598825U>;
    static_assert(type::k == 2U);
};

template <>
struct coif9_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf350081U, 0x3f42528eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif9_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbee2307aU, 0x3ef3fd2bU>;
    static_assert(type::k == 2U);
};

template <>
struct coif9_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xbeae1d28U, 0x3f302404U>;
    static_assert(type::k == 2U);
};

template <>
struct coif9_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe4ebcecU, 0x3e9121beU>;
    static_assert(type::k == 2U);
};

template <>
struct coif9_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0xbda0896cU, 0x3e8231e3U>;
    static_assert(type::k == 2U);
};

template <>
struct coif9_inverse::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d980fa7U, 0x3d85ab61U>;
    static_assert(type::k == 2U);
};

template <>
struct coif9_inverse::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e814a84U, 0xbdb659cfU>;
    static_assert(type::k == 2U);
};

template <>
struct coif9_inverse::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3edd6d73U, 0xbe4e093cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif9_inverse::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ef7a54dU, 0xbeeaba00U>;
    static_assert(type::k == 2U);
};

template <>
struct coif9_inverse::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f381d06U, 0xbee47d9cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif9_inverse::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f8e4543U, 0xbf146413U>;
    static_assert(type::k == 2U);
};

template <>
struct coif9_inverse::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f622b40U, 0xbf26d935U>;
    static_assert(type::k == 2U);
};

template <>
struct coif9_inverse::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f87c77eU, 0xbf70104fU>;
    static_assert(type::k == 2U);
};

template <>
struct coif9_inverse::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbef0126fU, 0xbf4fb93dU>;
    static_assert(type::k == 2U);
};

template <>
struct coif9_inverse::step<30> {
    using type = StaticStep<StepType::kPredict, 0, 0x3fb7e2dbU>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
