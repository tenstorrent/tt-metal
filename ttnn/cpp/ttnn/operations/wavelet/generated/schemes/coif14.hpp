// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct coif14_inverse;

struct coif14 {
    static constexpr const char* name = "coif14";
    static constexpr uint32_t tap_size = 84U;
    static constexpr int32_t delay_even = 21;
    static constexpr int32_t delay_odd = 21;
    static constexpr uint32_t num_steps = 45U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/coif14.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::coif14";
    using inverse = coif14_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct coif14::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f43e572U>;
    static_assert(type::k == 1U);
};

template <>
struct coif14::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0x400264e6U, 0xbef719a7U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ecbcf10U, 0xbee5531dU>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x4004bbb3U, 0xc00f6204U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ea035c0U, 0xbed84189U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fc77a53U, 0xc0229789U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e865fd4U, 0xbec2ec26U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fc773f7U, 0xbfc7af77U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e74172cU, 0xbe8658b1U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f6e6b75U, 0xbfc32134U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e23b79dU, 0xbe411b94U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f1ddca9U, 0xbf549f9dU>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x3d8091a6U, 0xbe014454U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e091f26U, 0xbea6c31bU>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xbca04e2dU, 0xbcd71b1cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbec01d02U, 0x3dccb228U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xbdcf1c9bU, 0x3d92aedcU>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf4bbf3fU, 0x3f041d9dU>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe576ba8U, 0x3e14f8c2U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf81d4bbU, 0x3f82cc8cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe9df046U, 0x3e567fe9U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfd46fa7U, 0x3f992a19U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe996dadU, 0x3ea35baaU>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfdc7be3U, 0x3fd4ce7bU>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0xbedfd09dU, 0x3e9959fdU>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc0250d32U, 0x3fcb4655U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0xbee8596aU, 0x3eaa1e02U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc193e0dbU, 0x40017075U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0x3d19c6eaU, 0x3d5d59adU>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc156f02dU, 0xc1d49d96U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<30> {
    using type = StaticStep<StepType::kPredict, -1, 0xbdf54743U, 0x3d9723abU>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<31> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc18090a2U, 0x410527d9U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<32> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe0afe76U, 0x3d7ea64eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<33> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc1929e6eU, 0x40ebb51bU>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<34> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe20e560U, 0x3d5f7bfdU>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<35> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc1acd50aU, 0x40cba8a6U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<36> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe41e751U, 0x3d3d9821U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<37> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc1d66511U, 0x40a8fdc0U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<38> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe7a9a31U, 0x3d18d6ecU>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<39> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc213c116U, 0x4082c1c9U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<40> {
    using type = StaticStep<StepType::kPredict, -1, 0xbec23838U, 0x3cddc60fU>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<41> {
    using type = StaticStep<StepType::kUpdate, 0, 0x24dd9bc6U, 0x4028b75bU>;
    static_assert(type::k == 2U);
};

template <>
struct coif14::step<42> {
    using type = StaticStep<StepType::kPredict, 0, 0xbc56bb29U>;
    static_assert(type::k == 1U);
};

template <>
struct coif14::step<43> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xc555945bU>;
    static_assert(type::k == 1U);
};

template <>
struct coif14::step<44> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xb9996c47U>;
    static_assert(type::k == 1U);
};

struct coif14_inverse {
    static constexpr const char* name = "coif14-inverse";
    static constexpr uint32_t tap_size = 84U;
    static constexpr uint32_t num_steps = 45U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/coif14.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::coif14_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct coif14_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xc555945bU>;
    static_assert(type::k == 1U);
};

template <>
struct coif14_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xb9996c47U>;
    static_assert(type::k == 1U);
};

template <>
struct coif14_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0x3c56bb29U>;
    static_assert(type::k == 1U);
};

template <>
struct coif14_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xa4dd9bc6U, 0xc028b75bU>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ec23838U, 0xbcddc60fU>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x4213c116U, 0xc082c1c9U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e7a9a31U, 0xbd18d6ecU>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x41d66511U, 0xc0a8fdc0U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e41e751U, 0xbd3d9821U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x41acd50aU, 0xc0cba8a6U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e20e560U, 0xbd5f7bfdU>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x41929e6eU, 0xc0ebb51bU>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e0afe76U, 0xbd7ea64eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x418090a2U, 0xc10527d9U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x3df54743U, 0xbd9723abU>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x4156f02dU, 0x41d49d96U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xbd19c6eaU, 0xbd5d59adU>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x4193e0dbU, 0xc0017075U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ee8596aU, 0xbeaa1e02U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0x40250d32U, 0xbfcb4655U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0x3edfd09dU, 0xbe9959fdU>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fdc7be3U, 0xbfd4ce7bU>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e996dadU, 0xbea35baaU>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fd46fa7U, 0xbf992a19U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e9df046U, 0xbe567fe9U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f81d4bbU, 0xbf82cc8cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e576ba8U, 0xbe14f8c2U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f4bbf3fU, 0xbf041d9dU>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0x3dcf1c9bU, 0xbd92aedcU>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ec01d02U, 0xbdccb228U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<30> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ca04e2dU, 0x3cd71b1cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<31> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe091f26U, 0x3ea6c31bU>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<32> {
    using type = StaticStep<StepType::kPredict, -1, 0xbd8091a6U, 0x3e014454U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<33> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf1ddca9U, 0x3f549f9dU>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<34> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe23b79dU, 0x3e411b94U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<35> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf6e6b75U, 0x3fc32134U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<36> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe74172cU, 0x3e8658b1U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<37> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfc773f7U, 0x3fc7af77U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<38> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe865fd4U, 0x3ec2ec26U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<39> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfc77a53U, 0x40229789U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<40> {
    using type = StaticStep<StepType::kPredict, -1, 0xbea035c0U, 0x3ed84189U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<41> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc004bbb3U, 0x400f6204U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<42> {
    using type = StaticStep<StepType::kPredict, -1, 0xbecbcf10U, 0x3ee5531dU>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<43> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc00264e6U, 0x3ef719a7U>;
    static_assert(type::k == 2U);
};

template <>
struct coif14_inverse::step<44> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf43e572U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
