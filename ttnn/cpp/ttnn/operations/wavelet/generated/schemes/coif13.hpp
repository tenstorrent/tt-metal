// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct coif13_inverse;

struct coif13 {
    static constexpr const char* name = "coif13";
    static constexpr uint32_t tap_size = 78U;
    static constexpr int32_t delay_even = 19;
    static constexpr int32_t delay_odd = 20;
    static constexpr uint32_t num_steps = 43U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/coif13.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::coif13";
    using inverse = coif13_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct coif13::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0xbfa9b9bfU>;
    static_assert(type::k == 1U);
};

template <>
struct coif13::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ef62328U, 0x3f8f720dU>;
    static_assert(type::k == 2U);
};

template <>
struct coif13::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf4ecac3U, 0x3f388a73U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf9cf08fU, 0x3f8dbb57U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf471cccU, 0x3f0c248dU>;
    static_assert(type::k == 2U);
};

template <>
struct coif13::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfb06e09U, 0x3f51758eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif13::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf22b1ccU, 0x3eed3972U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf49e31eU, 0x3f51a762U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xbeef7cfdU, 0x3eb85a8aU>;
    static_assert(type::k == 2U);
};

template <>
struct coif13::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf3f3c1dU, 0x3ee62b05U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe85eb1bU, 0x3e6f90edU>;
    static_assert(type::k == 2U);
};

template <>
struct coif13::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbed2d4afU, 0x3e508285U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xbdff8676U, 0x3d3b1857U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd9c1aeeU, 0xbd86c1e5U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x3d21a71aU, 0xbe06dbd4U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e5b1373U, 0xbe9c5f5dU>;
    static_assert(type::k == 2U);
};

template <>
struct coif13::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e3b0cb9U, 0xbeaaee06U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3efcc9bdU, 0xbf0404d8U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0x3eafe612U, 0xbeea1357U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f1a9284U, 0xbf70e7f9U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f02cea5U, 0xbf038888U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f70f421U, 0xbf6c2e73U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f036bb9U, 0xbf3a45d0U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f6283e0U, 0xbfbc52d6U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f10fce3U, 0xbf4dd0a6U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f90d470U, 0xc103494fU>;
    static_assert(type::k == 2U);
};

template <>
struct coif13::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0x3df91f16U, 0x3e0930a0U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc0ee7aebU, 0xc0483777U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ea23acfU, 0xbf082457U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3feff006U, 0xc070df0eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif13::step<30> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e87ef08U, 0xbf1c0ea0U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13::step<31> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fd1f184U, 0xc08b5d05U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13::step<32> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e6b1f16U, 0xbf37f19eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif13::step<33> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fb22417U, 0xc0a815a9U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13::step<34> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e42f30cU, 0xbf64809cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif13::step<35> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f8f6740U, 0xc0d9a015U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13::step<36> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e16921aU, 0xbf9dcaddU>;
    static_assert(type::k == 2U);
};

template <>
struct coif13::step<37> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f4faa4cU, 0xc12903beU>;
    static_assert(type::k == 2U);
};

template <>
struct coif13::step<38> {
    using type = StaticStep<StepType::kPredict, -1, 0x3dc1e071U, 0xc0234f7dU>;
    static_assert(type::k == 2U);
};

template <>
struct coif13::step<39> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct coif13::step<40> {
    using type = StaticStep<StepType::kPredict, 0, 0x3ec8a601U>;
    static_assert(type::k == 1U);
};

template <>
struct coif13::step<41> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x38328f71U>;
    static_assert(type::k == 1U);
};

template <>
struct coif13::step<42> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xc6b78321U>;
    static_assert(type::k == 1U);
};

struct coif13_inverse {
    static constexpr const char* name = "coif13-inverse";
    static constexpr uint32_t tap_size = 78U;
    static constexpr uint32_t num_steps = 43U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/coif13.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::coif13_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct coif13_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xb8328f71U>;
    static_assert(type::k == 1U);
};

template <>
struct coif13_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x46b78321U>;
    static_assert(type::k == 1U);
};

template <>
struct coif13_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0xbec8a601U>;
    static_assert(type::k == 1U);
};

template <>
struct coif13_inverse::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct coif13_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xbdc1e071U, 0x40234f7dU>;
    static_assert(type::k == 2U);
};

template <>
struct coif13_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf4faa4cU, 0x412903beU>;
    static_assert(type::k == 2U);
};

template <>
struct coif13_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe16921aU, 0x3f9dcaddU>;
    static_assert(type::k == 2U);
};

template <>
struct coif13_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf8f6740U, 0x40d9a015U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe42f30cU, 0x3f64809cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif13_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfb22417U, 0x40a815a9U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe6b1f16U, 0x3f37f19eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif13_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfd1f184U, 0x408b5d05U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe87ef08U, 0x3f1c0ea0U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfeff006U, 0x4070df0eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif13_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xbea23acfU, 0x3f082457U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x40ee7aebU, 0x40483777U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xbdf91f16U, 0xbe0930a0U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf90d470U, 0x4103494fU>;
    static_assert(type::k == 2U);
};

template <>
struct coif13_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf10fce3U, 0x3f4dd0a6U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf6283e0U, 0x3fbc52d6U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf036bb9U, 0x3f3a45d0U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13_inverse::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf70f421U, 0x3f6c2e73U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13_inverse::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf02cea5U, 0x3f038888U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13_inverse::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf1a9284U, 0x3f70e7f9U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13_inverse::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0xbeafe612U, 0x3eea1357U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13_inverse::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbefcc9bdU, 0x3f0404d8U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13_inverse::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe3b0cb9U, 0x3eaaee06U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13_inverse::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe5b1373U, 0x3e9c5f5dU>;
    static_assert(type::k == 2U);
};

template <>
struct coif13_inverse::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0xbd21a71aU, 0x3e06dbd4U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13_inverse::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d9c1aeeU, 0x3d86c1e5U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13_inverse::step<30> {
    using type = StaticStep<StepType::kPredict, -1, 0x3dff8676U, 0xbd3b1857U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13_inverse::step<31> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ed2d4afU, 0xbe508285U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13_inverse::step<32> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e85eb1bU, 0xbe6f90edU>;
    static_assert(type::k == 2U);
};

template <>
struct coif13_inverse::step<33> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f3f3c1dU, 0xbee62b05U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13_inverse::step<34> {
    using type = StaticStep<StepType::kPredict, -1, 0x3eef7cfdU, 0xbeb85a8aU>;
    static_assert(type::k == 2U);
};

template <>
struct coif13_inverse::step<35> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f49e31eU, 0xbf51a762U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13_inverse::step<36> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f22b1ccU, 0xbeed3972U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13_inverse::step<37> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fb06e09U, 0xbf51758eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif13_inverse::step<38> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f471cccU, 0xbf0c248dU>;
    static_assert(type::k == 2U);
};

template <>
struct coif13_inverse::step<39> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f9cf08fU, 0xbf8dbb57U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13_inverse::step<40> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f4ecac3U, 0xbf388a73U>;
    static_assert(type::k == 2U);
};

template <>
struct coif13_inverse::step<41> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbef62328U, 0xbf8f720dU>;
    static_assert(type::k == 2U);
};

template <>
struct coif13_inverse::step<42> {
    using type = StaticStep<StepType::kPredict, 0, 0x3fa9b9bfU>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
