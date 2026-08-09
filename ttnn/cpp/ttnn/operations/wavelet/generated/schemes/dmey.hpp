// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct dmey_inverse;

struct dmey {
    static constexpr const char* name = "dmey";
    static constexpr uint32_t tap_size = 62U;
    static constexpr int32_t delay_even = 15;
    static constexpr int32_t delay_odd = 16;
    static constexpr uint32_t num_steps = 33U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/dmey.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::dmey";
    using inverse = dmey_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct dmey::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0x3fcc49cdU>;
    static_assert(type::k == 1U);
};

template <>
struct dmey::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfe394fcU, 0xbf9361afU>;
    static_assert(type::k == 2U);
};

template <>
struct dmey::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0x41e03961U, 0x41e03961U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3b27b177U, 0x3b27b177U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xc1552e7dU, 0xc1552e7dU>;
    static_assert(type::k == 2U);
};

template <>
struct dmey::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc60ca93U, 0xbc60ca93U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xc1a13cf1U, 0xc1a13cf1U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc48b1ceU, 0xbc48b1ceU>;
    static_assert(type::k == 2U);
};

template <>
struct dmey::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x41156be6U, 0x41156be6U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbefe8d82U, 0xbefe8d82U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x3cb13575U, 0x3cb13575U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f05f88fU, 0x3f05f88fU>;
    static_assert(type::k == 2U);
};

template <>
struct dmey::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0a15f08U, 0xc0a15f08U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f1938d8U, 0x3f1938d8U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x408f820eU, 0x408f820eU>;
    static_assert(type::k == 2U);
};

template <>
struct dmey::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbed61e09U, 0xbed61e09U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf4bfc57U, 0xbf4bfc57U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e312b9bU, 0x3e312b9bU>;
    static_assert(type::k == 2U);
};

template <>
struct dmey::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xc11c4900U, 0xc11c4900U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd1680c2U, 0xbd1680c2U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0x40b1f793U, 0x40b1f793U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c3d07d0U, 0x3c3d07d0U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf472515U, 0xbf472515U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc819d45U, 0xbc819d45U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0xc087249eU, 0xc087249eU>;
    static_assert(type::k == 2U);
};

template <>
struct dmey::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d325a0cU, 0x3d325a0cU>;
    static_assert(type::k == 2U);
};

template <>
struct dmey::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0x4110bd18U, 0x4110bd18U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe1756e3U, 0xbe1756e3U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f85b0b3U, 0x3f85b0b3U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e99cdc1U, 0x3e99cdc1U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey::step<30> {
    using type = StaticStep<StepType::kPredict, -1, 0x399e86a9U, 0x399e86a9U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey::step<31> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xbfcc6b30U>;
    static_assert(type::k == 1U);
};

template <>
struct dmey::step<32> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xbf204c67U>;
    static_assert(type::k == 1U);
};

struct dmey_inverse {
    static constexpr const char* name = "dmey-inverse";
    static constexpr uint32_t tap_size = 62U;
    static constexpr uint32_t num_steps = 33U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/dmey.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::dmey_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct dmey_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xbfcc6b30U>;
    static_assert(type::k == 1U);
};

template <>
struct dmey_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xbf204c67U>;
    static_assert(type::k == 1U);
};

template <>
struct dmey_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xb99e86a9U, 0xb99e86a9U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe99cdc1U, 0xbe99cdc1U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf85b0b3U, 0xbf85b0b3U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e1756e3U, 0x3e1756e3U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xc110bd18U, 0xc110bd18U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd325a0cU, 0xbd325a0cU>;
    static_assert(type::k == 2U);
};

template <>
struct dmey_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x4087249eU, 0x4087249eU>;
    static_assert(type::k == 2U);
};

template <>
struct dmey_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c819d45U, 0x3c819d45U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f472515U, 0x3f472515U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc3d07d0U, 0xbc3d07d0U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0b1f793U, 0xc0b1f793U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d1680c2U, 0x3d1680c2U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x411c4900U, 0x411c4900U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe312b9bU, 0xbe312b9bU>;
    static_assert(type::k == 2U);
};

template <>
struct dmey_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f4bfc57U, 0x3f4bfc57U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ed61e09U, 0x3ed61e09U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xc08f820eU, 0xc08f820eU>;
    static_assert(type::k == 2U);
};

template <>
struct dmey_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf1938d8U, 0xbf1938d8U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0x40a15f08U, 0x40a15f08U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey_inverse::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf05f88fU, 0xbf05f88fU>;
    static_assert(type::k == 2U);
};

template <>
struct dmey_inverse::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0xbcb13575U, 0xbcb13575U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey_inverse::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3efe8d82U, 0x3efe8d82U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey_inverse::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0xc1156be6U, 0xc1156be6U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey_inverse::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c48b1ceU, 0x3c48b1ceU>;
    static_assert(type::k == 2U);
};

template <>
struct dmey_inverse::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0x41a13cf1U, 0x41a13cf1U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey_inverse::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c60ca93U, 0x3c60ca93U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey_inverse::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0x41552e7dU, 0x41552e7dU>;
    static_assert(type::k == 2U);
};

template <>
struct dmey_inverse::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbb27b177U, 0xbb27b177U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey_inverse::step<30> {
    using type = StaticStep<StepType::kPredict, -1, 0xc1e03961U, 0xc1e03961U>;
    static_assert(type::k == 2U);
};

template <>
struct dmey_inverse::step<31> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fe394fcU, 0x3f9361afU>;
    static_assert(type::k == 2U);
};

template <>
struct dmey_inverse::step<32> {
    using type = StaticStep<StepType::kPredict, 0, 0xbfcc49cdU>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
