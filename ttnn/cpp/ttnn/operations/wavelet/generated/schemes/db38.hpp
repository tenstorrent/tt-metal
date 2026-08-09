// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct db38_inverse;

struct db38 {
    static constexpr const char* name = "db38";
    static constexpr uint32_t tap_size = 76U;
    static constexpr int32_t delay_even = 19;
    static constexpr int32_t delay_odd = 19;
    static constexpr uint32_t num_steps = 41U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db38.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db38";
    using inverse = db38_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct db38::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf164067U>;
    static_assert(type::k == 1U);
};

template <>
struct db38::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf295db5U, 0x3e3faf5dU>;
    static_assert(type::k == 2U);
};

template <>
struct db38::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf04b257U, 0x3ed9593cU>;
    static_assert(type::k == 2U);
};

template <>
struct db38::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf01a69aU, 0x3f1d5f42U>;
    static_assert(type::k == 2U);
};

template <>
struct db38::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf0b0ce9U, 0x3f1fe7ebU>;
    static_assert(type::k == 2U);
};

template <>
struct db38::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf31467cU, 0x3f305c30U>;
    static_assert(type::k == 2U);
};

template <>
struct db38::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf3fcd49U, 0x3f300d2aU>;
    static_assert(type::k == 2U);
};

template <>
struct db38::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf5b8229U, 0x3f43051fU>;
    static_assert(type::k == 2U);
};

template <>
struct db38::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf5dbadaU, 0x3f41a368U>;
    static_assert(type::k == 2U);
};

template <>
struct db38::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf763cd8U, 0x3f544831U>;
    static_assert(type::k == 2U);
};

template <>
struct db38::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf747238U, 0x3f538c2fU>;
    static_assert(type::k == 2U);
};

template <>
struct db38::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf83c796U, 0x3f8f003dU>;
    static_assert(type::k == 2U);
};

template <>
struct db38::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf46331fU, 0xbe92b228U>;
    static_assert(type::k == 2U);
};

template <>
struct db38::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf67c0f0U, 0xbe1a3a7eU>;
    static_assert(type::k == 2U);
};

template <>
struct db38::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xbd110d59U, 0x3f00c720U>;
    static_assert(type::k == 2U);
};

template <>
struct db38::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc05bb029U, 0x4074cf3cU>;
    static_assert(type::k == 2U);
};

template <>
struct db38::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe86bb60U, 0x3e4fae6eU>;
    static_assert(type::k == 2U);
};

template <>
struct db38::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc09fa087U, 0x40687827U>;
    static_assert(type::k == 2U);
};

template <>
struct db38::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe91eca8U, 0x3e497a6eU>;
    static_assert(type::k == 2U);
};

template <>
struct db38::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc0aba4efU, 0x405e91e4U>;
    static_assert(type::k == 2U);
};

template <>
struct db38::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe9db77aU, 0x3e3db735U>;
    static_assert(type::k == 2U);
};

template <>
struct db38::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc0bb3f17U, 0x403d631dU>;
    static_assert(type::k == 2U);
};

template <>
struct db38::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0xbeb8b610U, 0x3d3a448fU>;
    static_assert(type::k == 2U);
};

template <>
struct db38::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbea3123dU, 0x3ee20a0dU>;
    static_assert(type::k == 2U);
};

template <>
struct db38::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0xbc524196U, 0x3e2be467U>;
    static_assert(type::k == 2U);
};

template <>
struct db38::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc0a5647cU, 0x40211346U>;
    static_assert(type::k == 2U);
};

template <>
struct db38::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0xbee045adU, 0x3e3abbc2U>;
    static_assert(type::k == 2U);
};

template <>
struct db38::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc0c498f5U, 0x401217ccU>;
    static_assert(type::k == 2U);
};

template <>
struct db38::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0xbefd9cc1U, 0x3e26acebU>;
    static_assert(type::k == 2U);
};

template <>
struct db38::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc0e0991eU, 0x4001347fU>;
    static_assert(type::k == 2U);
};

template <>
struct db38::step<30> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf12ad01U, 0x3e11e56aU>;
    static_assert(type::k == 2U);
};

template <>
struct db38::step<31> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc103eeb9U, 0x3fdf677fU>;
    static_assert(type::k == 2U);
};

template <>
struct db38::step<32> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf2fdafdU, 0x3df85e92U>;
    static_assert(type::k == 2U);
};

template <>
struct db38::step<33> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc122a2c8U, 0x3fba55bbU>;
    static_assert(type::k == 2U);
};

template <>
struct db38::step<34> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf61b6f3U, 0x3dc97b15U>;
    static_assert(type::k == 2U);
};

template <>
struct db38::step<35> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc15e8f3eU, 0x3f912ca9U>;
    static_assert(type::k == 2U);
};

template <>
struct db38::step<36> {
    using type = StaticStep<StepType::kPredict, -1, 0xbfada92aU, 0x3d933b86U>;
    static_assert(type::k == 2U);
};

template <>
struct db38::step<37> {
    using type = StaticStep<StepType::kUpdate, 0, 0x1a47d820U, 0x3f3cb08fU>;
    static_assert(type::k == 2U);
};

template <>
struct db38::step<38> {
    using type = StaticStep<StepType::kPredict, 0, 0xbd0f9819U>;
    static_assert(type::k == 1U);
};

template <>
struct db38::step<39> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x4826a970U>;
    static_assert(type::k == 1U);
};

template <>
struct db38::step<40> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x36c49d19U>;
    static_assert(type::k == 1U);
};

struct db38_inverse {
    static constexpr const char* name = "db38-inverse";
    static constexpr uint32_t tap_size = 76U;
    static constexpr uint32_t num_steps = 41U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db38.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db38_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct db38_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x4826a970U>;
    static_assert(type::k == 1U);
};

template <>
struct db38_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x36c49d19U>;
    static_assert(type::k == 1U);
};

template <>
struct db38_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0x3d0f9819U>;
    static_assert(type::k == 1U);
};

template <>
struct db38_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x9a47d820U, 0xbf3cb08fU>;
    static_assert(type::k == 2U);
};

template <>
struct db38_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x3fada92aU, 0xbd933b86U>;
    static_assert(type::k == 2U);
};

template <>
struct db38_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x415e8f3eU, 0xbf912ca9U>;
    static_assert(type::k == 2U);
};

template <>
struct db38_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f61b6f3U, 0xbdc97b15U>;
    static_assert(type::k == 2U);
};

template <>
struct db38_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x4122a2c8U, 0xbfba55bbU>;
    static_assert(type::k == 2U);
};

template <>
struct db38_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f2fdafdU, 0xbdf85e92U>;
    static_assert(type::k == 2U);
};

template <>
struct db38_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x4103eeb9U, 0xbfdf677fU>;
    static_assert(type::k == 2U);
};

template <>
struct db38_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f12ad01U, 0xbe11e56aU>;
    static_assert(type::k == 2U);
};

template <>
struct db38_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x40e0991eU, 0xc001347fU>;
    static_assert(type::k == 2U);
};

template <>
struct db38_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x3efd9cc1U, 0xbe26acebU>;
    static_assert(type::k == 2U);
};

template <>
struct db38_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x40c498f5U, 0xc01217ccU>;
    static_assert(type::k == 2U);
};

template <>
struct db38_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ee045adU, 0xbe3abbc2U>;
    static_assert(type::k == 2U);
};

template <>
struct db38_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x40a5647cU, 0xc0211346U>;
    static_assert(type::k == 2U);
};

template <>
struct db38_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x3c524196U, 0xbe2be467U>;
    static_assert(type::k == 2U);
};

template <>
struct db38_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ea3123dU, 0xbee20a0dU>;
    static_assert(type::k == 2U);
};

template <>
struct db38_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0x3eb8b610U, 0xbd3a448fU>;
    static_assert(type::k == 2U);
};

template <>
struct db38_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0x40bb3f17U, 0xc03d631dU>;
    static_assert(type::k == 2U);
};

template <>
struct db38_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e9db77aU, 0xbe3db735U>;
    static_assert(type::k == 2U);
};

template <>
struct db38_inverse::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0x40aba4efU, 0xc05e91e4U>;
    static_assert(type::k == 2U);
};

template <>
struct db38_inverse::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e91eca8U, 0xbe497a6eU>;
    static_assert(type::k == 2U);
};

template <>
struct db38_inverse::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0x409fa087U, 0xc0687827U>;
    static_assert(type::k == 2U);
};

template <>
struct db38_inverse::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e86bb60U, 0xbe4fae6eU>;
    static_assert(type::k == 2U);
};

template <>
struct db38_inverse::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0x405bb029U, 0xc074cf3cU>;
    static_assert(type::k == 2U);
};

template <>
struct db38_inverse::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0x3d110d59U, 0xbf00c720U>;
    static_assert(type::k == 2U);
};

template <>
struct db38_inverse::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f67c0f0U, 0x3e1a3a7eU>;
    static_assert(type::k == 2U);
};

template <>
struct db38_inverse::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f46331fU, 0x3e92b228U>;
    static_assert(type::k == 2U);
};

template <>
struct db38_inverse::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f83c796U, 0xbf8f003dU>;
    static_assert(type::k == 2U);
};

template <>
struct db38_inverse::step<30> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f747238U, 0xbf538c2fU>;
    static_assert(type::k == 2U);
};

template <>
struct db38_inverse::step<31> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f763cd8U, 0xbf544831U>;
    static_assert(type::k == 2U);
};

template <>
struct db38_inverse::step<32> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f5dbadaU, 0xbf41a368U>;
    static_assert(type::k == 2U);
};

template <>
struct db38_inverse::step<33> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f5b8229U, 0xbf43051fU>;
    static_assert(type::k == 2U);
};

template <>
struct db38_inverse::step<34> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f3fcd49U, 0xbf300d2aU>;
    static_assert(type::k == 2U);
};

template <>
struct db38_inverse::step<35> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f31467cU, 0xbf305c30U>;
    static_assert(type::k == 2U);
};

template <>
struct db38_inverse::step<36> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f0b0ce9U, 0xbf1fe7ebU>;
    static_assert(type::k == 2U);
};

template <>
struct db38_inverse::step<37> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f01a69aU, 0xbf1d5f42U>;
    static_assert(type::k == 2U);
};

template <>
struct db38_inverse::step<38> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f04b257U, 0xbed9593cU>;
    static_assert(type::k == 2U);
};

template <>
struct db38_inverse::step<39> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f295db5U, 0xbe3faf5dU>;
    static_assert(type::k == 2U);
};

template <>
struct db38_inverse::step<40> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f164067U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
