// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct coif6_inverse;

struct coif6 {
    static constexpr const char* name = "coif6";
    static constexpr uint32_t tap_size = 36U;
    static constexpr int32_t delay_even = 9;
    static constexpr int32_t delay_odd = 9;
    static constexpr uint32_t num_steps = 21U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/coif6.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::coif6";
    using inverse = coif6_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct coif6::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f2023b4U>;
    static_assert(type::k == 1U);
};

template <>
struct coif6::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fb25fb5U, 0xbee6333eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif6::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ee48b54U, 0xbf111a8fU>;
    static_assert(type::k == 2U);
};

template <>
struct coif6::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f3c5fdbU, 0xbfc8ad5cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif6::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e424778U, 0xbeec180cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif6::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e55a66aU, 0xbed7da9cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif6::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xbd61f797U, 0xbdcca7eeU>;
    static_assert(type::k == 2U);
};

template <>
struct coif6::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf0d1151U, 0x3def5e10U>;
    static_assert(type::k == 2U);
};

template <>
struct coif6::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe8f94faU, 0x3e6a4618U>;
    static_assert(type::k == 2U);
};

template <>
struct coif6::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf6835ccU, 0x3f229d28U>;
    static_assert(type::k == 2U);
};

template <>
struct coif6::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf43585cU, 0x3ead0207U>;
    static_assert(type::k == 2U);
};

template <>
struct coif6::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc04866f1U, 0x3f686231U>;
    static_assert(type::k == 2U);
};

template <>
struct coif6::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xbfad75c5U, 0x3e9da230U>;
    static_assert(type::k == 2U);
};

template <>
struct coif6::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbeb6c5faU, 0x3f3c8b91U>;
    static_assert(type::k == 2U);
};

template <>
struct coif6::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0f206ffU, 0x4031d34bU>;
    static_assert(type::k == 2U);
};

template <>
struct coif6::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf042b80U, 0x3e073d7aU>;
    static_assert(type::k == 2U);
};

template <>
struct coif6::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xc13ff33dU, 0x3ff7e7c5U>;
    static_assert(type::k == 2U);
};

template <>
struct coif6::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x324d9b59U, 0x3daab5eeU>;
    static_assert(type::k == 2U);
};

template <>
struct coif6::step<18> {
    using type = StaticStep<StepType::kPredict, 0, 0xbf69dbe3U>;
    static_assert(type::k == 1U);
};

template <>
struct coif6::step<19> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x42baf0d8U>;
    static_assert(type::k == 1U);
};

template <>
struct coif6::step<20> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3c2f4913U>;
    static_assert(type::k == 1U);
};

struct coif6_inverse {
    static constexpr const char* name = "coif6-inverse";
    static constexpr uint32_t tap_size = 36U;
    static constexpr uint32_t num_steps = 21U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/coif6.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::coif6_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct coif6_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x42baf0d8U>;
    static_assert(type::k == 1U);
};

template <>
struct coif6_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3c2f4913U>;
    static_assert(type::k == 1U);
};

template <>
struct coif6_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0x3f69dbe3U>;
    static_assert(type::k == 1U);
};

template <>
struct coif6_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xb24d9b59U, 0xbdaab5eeU>;
    static_assert(type::k == 2U);
};

template <>
struct coif6_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x413ff33dU, 0xbff7e7c5U>;
    static_assert(type::k == 2U);
};

template <>
struct coif6_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f042b80U, 0xbe073d7aU>;
    static_assert(type::k == 2U);
};

template <>
struct coif6_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x40f206ffU, 0xc031d34bU>;
    static_assert(type::k == 2U);
};

template <>
struct coif6_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3eb6c5faU, 0xbf3c8b91U>;
    static_assert(type::k == 2U);
};

template <>
struct coif6_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x3fad75c5U, 0xbe9da230U>;
    static_assert(type::k == 2U);
};

template <>
struct coif6_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x404866f1U, 0xbf686231U>;
    static_assert(type::k == 2U);
};

template <>
struct coif6_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f43585cU, 0xbead0207U>;
    static_assert(type::k == 2U);
};

template <>
struct coif6_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f6835ccU, 0xbf229d28U>;
    static_assert(type::k == 2U);
};

template <>
struct coif6_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e8f94faU, 0xbe6a4618U>;
    static_assert(type::k == 2U);
};

template <>
struct coif6_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f0d1151U, 0xbdef5e10U>;
    static_assert(type::k == 2U);
};

template <>
struct coif6_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x3d61f797U, 0x3dcca7eeU>;
    static_assert(type::k == 2U);
};

template <>
struct coif6_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe55a66aU, 0x3ed7da9cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif6_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe424778U, 0x3eec180cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif6_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf3c5fdbU, 0x3fc8ad5cU>;
    static_assert(type::k == 2U);
};

template <>
struct coif6_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xbee48b54U, 0x3f111a8fU>;
    static_assert(type::k == 2U);
};

template <>
struct coif6_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfb25fb5U, 0x3ee6333eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif6_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf2023b4U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
