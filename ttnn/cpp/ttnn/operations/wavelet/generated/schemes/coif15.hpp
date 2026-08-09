// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct coif15_inverse;

struct coif15 {
    static constexpr const char* name = "coif15";
    static constexpr uint32_t tap_size = 90U;
    static constexpr int32_t delay_even = 22;
    static constexpr int32_t delay_odd = 23;
    static constexpr uint32_t num_steps = 49U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/coif15.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::coif15";
    using inverse = coif15_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct coif15::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0xbfa51752U>;
    static_assert(type::k == 1U);
};

template <>
struct coif15::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ef7edf0U, 0x3fa1c91dU>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf3a1e34U, 0x3f248ec2U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfb2eea0U, 0x3fa8e3a4U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf2c3441U, 0x3f0545ecU>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfcaf1e5U, 0x3f810d8fU>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf26b590U, 0x3ede258cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf876e8cU, 0x3f7e1359U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xbedc88b3U, 0x3ede7a50U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf7e677dU, 0x3f2ac3b2U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xbec3c34bU, 0x3e91fe42U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf0f9f1bU, 0x3f06c91eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe8bb10fU, 0x3e1c3023U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbea71b98U, 0x3e698fb1U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xbde041a2U, 0x3d052d02U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd8dcc3cU, 0xbd918834U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x3d08ac22U, 0xbdb8feaaU>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e4240deU, 0xbec077d6U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e2e5a2aU, 0xbe46d806U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ed77fddU, 0xbf31377dU>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e8f7ffdU, 0xbea4b7e3U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f3f98c7U, 0xbf4b4e51U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ea9a875U, 0xbf01d8d5U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f65ebcfU, 0xbf8d8812U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f0173ccU, 0xbeec442eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f8e7a1eU, 0xbf9c0383U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0x3eeab79cU, 0xbf2e63f7U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f8a6749U, 0xbfdb7352U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f02b922U, 0xbf2c7b7aU>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fafa0ebU, 0xc1877086U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<30> {
    using type = StaticStep<StepType::kPredict, -1, 0x3d71cef5U, 0x3cceac9cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<31> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc21e159eU, 0xc1b98055U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<32> {
    using type = StaticStep<StepType::kPredict, -1, 0x3d2f2683U, 0xbd89e8ddU>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<33> {
    using type = StaticStep<StepType::kUpdate, 0, 0x416cca2cU, 0xc1dcfe53U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<34> {
    using type = StaticStep<StepType::kPredict, -1, 0x3d142129U, 0xbd9ad981U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<35> {
    using type = StaticStep<StepType::kUpdate, 0, 0x4153905dU, 0xc1f90ab5U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<36> {
    using type = StaticStep<StepType::kPredict, -1, 0x3d039228U, 0xbdb0aba9U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<37> {
    using type = StaticStep<StepType::kUpdate, 0, 0x41397960U, 0xc21031a2U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<38> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ce33fdcU, 0xbdd029f2U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<39> {
    using type = StaticStep<StepType::kUpdate, 0, 0x411d6a1aU, 0xc22d95edU>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<40> {
    using type = StaticStep<StepType::kPredict, -1, 0x3cbcc578U, 0xbe00eb0eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<41> {
    using type = StaticStep<StepType::kUpdate, 0, 0x40fe2d3eU, 0xc25ff4afU>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<42> {
    using type = StaticStep<StepType::kPredict, -1, 0x3c925089U, 0xbe315da4U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<43> {
    using type = StaticStep<StepType::kUpdate, 0, 0x40b8bf87U, 0xc2ad3df3U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<44> {
    using type = StaticStep<StepType::kPredict, -1, 0x3c3d2555U, 0xbeb6d6fdU>;
    static_assert(type::k == 2U);
};

template <>
struct coif15::step<45> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct coif15::step<46> {
    using type = StaticStep<StepType::kPredict, 0, 0x4033378eU>;
    static_assert(type::k == 1U);
};

template <>
struct coif15::step<47> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x37e7ec39U>;
    static_assert(type::k == 1U);
};

template <>
struct coif15::step<48> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xc70d49d6U>;
    static_assert(type::k == 1U);
};

struct coif15_inverse {
    static constexpr const char* name = "coif15-inverse";
    static constexpr uint32_t tap_size = 90U;
    static constexpr uint32_t num_steps = 49U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/coif15.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::coif15_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct coif15_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xb7e7ec3aU>;
    static_assert(type::k == 1U);
};

template <>
struct coif15_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x470d49d6U>;
    static_assert(type::k == 1U);
};

template <>
struct coif15_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0xc033378eU>;
    static_assert(type::k == 1U);
};

template <>
struct coif15_inverse::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct coif15_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xbc3d2555U, 0x3eb6d6fdU>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc0b8bf87U, 0x42ad3df3U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xbc925089U, 0x3e315da4U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc0fe2d3eU, 0x425ff4afU>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xbcbcc578U, 0x3e00eb0eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc11d6a1aU, 0x422d95edU>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xbce33fdcU, 0x3dd029f2U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc1397960U, 0x421031a2U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xbd039228U, 0x3db0aba9U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc153905dU, 0x41f90ab5U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xbd142129U, 0x3d9ad981U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc16cca2cU, 0x41dcfe53U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xbd2f2683U, 0x3d89e8ddU>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x421e159eU, 0x41b98055U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xbd71cef5U, 0xbcceac9cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfafa0ebU, 0x41877086U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf02b922U, 0x3f2c7b7aU>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf8a6749U, 0x3fdb7352U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0xbeeab79cU, 0x3f2e63f7U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf8e7a1eU, 0x3f9c0383U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf0173ccU, 0x3eec442eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf65ebcfU, 0x3f8d8812U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0xbea9a875U, 0x3f01d8d5U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf3f98c7U, 0x3f4b4e51U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe8f7ffdU, 0x3ea4b7e3U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbed77fddU, 0x3f31377dU>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<30> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe2e5a2aU, 0x3e46d806U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<31> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe4240deU, 0x3ec077d6U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<32> {
    using type = StaticStep<StepType::kPredict, -1, 0xbd08ac22U, 0x3db8feaaU>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<33> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d8dcc3cU, 0x3d918834U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<34> {
    using type = StaticStep<StepType::kPredict, -1, 0x3de041a2U, 0xbd052d02U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<35> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ea71b98U, 0xbe698fb1U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<36> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e8bb10fU, 0xbe1c3023U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<37> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f0f9f1bU, 0xbf06c91eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<38> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ec3c34bU, 0xbe91fe42U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<39> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f7e677dU, 0xbf2ac3b2U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<40> {
    using type = StaticStep<StepType::kPredict, -1, 0x3edc88b3U, 0xbede7a50U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<41> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f876e8cU, 0xbf7e1359U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<42> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f26b590U, 0xbede258cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<43> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fcaf1e5U, 0xbf810d8fU>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<44> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f2c3441U, 0xbf0545ecU>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<45> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fb2eea0U, 0xbfa8e3a4U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<46> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f3a1e34U, 0xbf248ec2U>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<47> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbef7edf0U, 0xbfa1c91dU>;
    static_assert(type::k == 2U);
};

template <>
struct coif15_inverse::step<48> {
    using type = StaticStep<StepType::kPredict, 0, 0x3fa51752U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
