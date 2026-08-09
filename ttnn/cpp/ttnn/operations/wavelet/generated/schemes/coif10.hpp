// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct coif10_inverse;

struct coif10 {
    static constexpr const char* name = "coif10";
    static constexpr uint32_t tap_size = 60U;
    static constexpr int32_t delay_even = 15;
    static constexpr int32_t delay_odd = 15;
    static constexpr uint32_t num_steps = 33U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/coif10.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::coif10";
    using inverse = coif10_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct coif10::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f36994fU>;
    static_assert(type::k == 1U);
};

template <>
struct coif10::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fe085b0U, 0xbef20d0cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif10::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ee3d29dU, 0xbeff6c25U>;
    static_assert(type::k == 2U);
};

template <>
struct coif10::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fc1b9aaU, 0xbff3e454U>;
    static_assert(type::k == 2U);
};

template <>
struct coif10::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e963c05U, 0xbf03b02aU>;
    static_assert(type::k == 2U);
};

template <>
struct coif10::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f87fb76U, 0xbfe8c01aU>;
    static_assert(type::k == 2U);
};

template <>
struct coif10::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e80189eU, 0xbe875580U>;
    static_assert(type::k == 2U);
};

template <>
struct coif10::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f3a76f0U, 0xbf8495ebU>;
    static_assert(type::k == 2U);
};

template <>
struct coif10::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x3dc0638fU, 0xbe4f4f66U>;
    static_assert(type::k == 2U);
};

template <>
struct coif10::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e2c13f3U, 0xbec3afddU>;
    static_assert(type::k == 2U);
};

template <>
struct coif10::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xbcea7347U, 0xbd2e394eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif10::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbeebf181U, 0x3de878d3U>;
    static_assert(type::k == 2U);
};

template <>
struct coif10::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe17ab51U, 0x3de28e90U>;
    static_assert(type::k == 2U);
};

template <>
struct coif10::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf6c6a82U, 0x3f17d176U>;
    static_assert(type::k == 2U);
};

template <>
struct coif10::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xbeaddb2aU, 0x3e51e90dU>;
    static_assert(type::k == 2U);
};

template <>
struct coif10::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf8bc1cbU, 0x3f93359cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif10::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xbeca25ecU, 0x3eababb7U>;
    static_assert(type::k == 2U);
};

template <>
struct coif10::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbffc701cU, 0x3f902a5bU>;
    static_assert(type::k == 2U);
};

template <>
struct coif10::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf17dd80U, 0x3eb87a17U>;
    static_assert(type::k == 2U);
};

template <>
struct coif10::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc0d49ffdU, 0x3fbb95f5U>;
    static_assert(type::k == 2U);
};

template <>
struct coif10::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f8ad245U, 0x3e18fe55U>;
    static_assert(type::k == 2U);
};

template <>
struct coif10::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbde97058U, 0xbf6c02faU>;
    static_assert(type::k == 2U);
};

template <>
struct coif10::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0xc18548d2U, 0x410b1bc6U>;
    static_assert(type::k == 2U);
};

template <>
struct coif10::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe10ec00U, 0x3d753e7cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif10::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0xc1a1676eU, 0x40e20211U>;
    static_assert(type::k == 2U);
};

template <>
struct coif10::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe347306U, 0x3d4b01eaU>;
    static_assert(type::k == 2U);
};

template <>
struct coif10::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0xc1d1fa2bU, 0x40b59724U>;
    static_assert(type::k == 2U);
};

template <>
struct coif10::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe7ae17aU, 0x3d1c0e14U>;
    static_assert(type::k == 2U);
};

template <>
struct coif10::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0xc22453d5U, 0x40829ca2U>;
    static_assert(type::k == 2U);
};

template <>
struct coif10::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0x2979b3f2U, 0x3cc7681eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif10::step<30> {
    using type = StaticStep<StepType::kPredict, 0, 0xbffa5ea7U>;
    static_assert(type::k == 1U);
};

template <>
struct coif10::step<31> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xc51851eeU>;
    static_assert(type::k == 1U);
};

template <>
struct coif10::step<32> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xb9d72042U>;
    static_assert(type::k == 1U);
};

struct coif10_inverse {
    static constexpr const char* name = "coif10-inverse";
    static constexpr uint32_t tap_size = 60U;
    static constexpr uint32_t num_steps = 33U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/coif10.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::coif10_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct coif10_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xc51851eeU>;
    static_assert(type::k == 1U);
};

template <>
struct coif10_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xb9d72041U>;
    static_assert(type::k == 1U);
};

template <>
struct coif10_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0x3ffa5ea7U>;
    static_assert(type::k == 1U);
};

template <>
struct coif10_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xa979b3f2U, 0xbcc7681eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif10_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x422453d5U, 0xc0829ca2U>;
    static_assert(type::k == 2U);
};

template <>
struct coif10_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e7ae17aU, 0xbd1c0e14U>;
    static_assert(type::k == 2U);
};

template <>
struct coif10_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x41d1fa2bU, 0xc0b59724U>;
    static_assert(type::k == 2U);
};

template <>
struct coif10_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e347306U, 0xbd4b01eaU>;
    static_assert(type::k == 2U);
};

template <>
struct coif10_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x41a1676eU, 0xc0e20211U>;
    static_assert(type::k == 2U);
};

template <>
struct coif10_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e10ec00U, 0xbd753e7cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif10_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x418548d2U, 0xc10b1bc6U>;
    static_assert(type::k == 2U);
};

template <>
struct coif10_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3de97058U, 0x3f6c02faU>;
    static_assert(type::k == 2U);
};

template <>
struct coif10_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf8ad245U, 0xbe18fe55U>;
    static_assert(type::k == 2U);
};

template <>
struct coif10_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x40d49ffdU, 0xbfbb95f5U>;
    static_assert(type::k == 2U);
};

template <>
struct coif10_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f17dd80U, 0xbeb87a17U>;
    static_assert(type::k == 2U);
};

template <>
struct coif10_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ffc701cU, 0xbf902a5bU>;
    static_assert(type::k == 2U);
};

template <>
struct coif10_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x3eca25ecU, 0xbeababb7U>;
    static_assert(type::k == 2U);
};

template <>
struct coif10_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f8bc1cbU, 0xbf93359cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif10_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0x3eaddb2aU, 0xbe51e90dU>;
    static_assert(type::k == 2U);
};

template <>
struct coif10_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f6c6a82U, 0xbf17d176U>;
    static_assert(type::k == 2U);
};

template <>
struct coif10_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e17ab51U, 0xbde28e90U>;
    static_assert(type::k == 2U);
};

template <>
struct coif10_inverse::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3eebf181U, 0xbde878d3U>;
    static_assert(type::k == 2U);
};

template <>
struct coif10_inverse::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0x3cea7347U, 0x3d2e394eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif10_inverse::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe2c13f3U, 0x3ec3afddU>;
    static_assert(type::k == 2U);
};

template <>
struct coif10_inverse::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0xbdc0638fU, 0x3e4f4f66U>;
    static_assert(type::k == 2U);
};

template <>
struct coif10_inverse::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf3a76f0U, 0x3f8495ebU>;
    static_assert(type::k == 2U);
};

template <>
struct coif10_inverse::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe80189eU, 0x3e875580U>;
    static_assert(type::k == 2U);
};

template <>
struct coif10_inverse::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf87fb76U, 0x3fe8c01aU>;
    static_assert(type::k == 2U);
};

template <>
struct coif10_inverse::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe963c05U, 0x3f03b02aU>;
    static_assert(type::k == 2U);
};

template <>
struct coif10_inverse::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfc1b9aaU, 0x3ff3e454U>;
    static_assert(type::k == 2U);
};

template <>
struct coif10_inverse::step<30> {
    using type = StaticStep<StepType::kPredict, -1, 0xbee3d29dU, 0x3eff6c25U>;
    static_assert(type::k == 2U);
};

template <>
struct coif10_inverse::step<31> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfe085b0U, 0x3ef20d0cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif10_inverse::step<32> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf36994fU>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
