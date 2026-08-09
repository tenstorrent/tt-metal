// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct db29_inverse;

struct db29 {
    static constexpr const char* name = "db29";
    static constexpr uint32_t tap_size = 58U;
    static constexpr int32_t delay_even = 14;
    static constexpr int32_t delay_odd = 15;
    static constexpr uint32_t num_steps = 33U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db29.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db29";
    using inverse = db29_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct db29::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0x40c82df0U>;
    static_assert(type::k == 1U);
};

template <>
struct db29::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe14e159U, 0xbc1cb1f1U>;
    static_assert(type::k == 2U);
};

template <>
struct db29::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0x423e0a15U, 0xc195f132U>;
    static_assert(type::k == 2U);
};

template <>
struct db29::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbae9e4b1U, 0xbc34985bU>;
    static_assert(type::k == 2U);
};

template <>
struct db29::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xc1a3d04dU, 0xc2a87f48U>;
    static_assert(type::k == 2U);
};

template <>
struct db29::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3bd6bfa2U, 0xbb982cd1U>;
    static_assert(type::k == 2U);
};

template <>
struct db29::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x42df7b99U, 0xc2ff8382U>;
    static_assert(type::k == 2U);
};

template <>
struct db29::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3b9910ddU, 0xbbbacfb7U>;
    static_assert(type::k == 2U);
};

template <>
struct db29::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x42f08ae6U, 0xc315844aU>;
    static_assert(type::k == 2U);
};

template <>
struct db29::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ba8736aU, 0xbbde0fc3U>;
    static_assert(type::k == 2U);
};

template <>
struct db29::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x42f9894cU, 0x4356d743U>;
    static_assert(type::k == 2U);
};

template <>
struct db29::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbba658aeU, 0x3860a3f5U>;
    static_assert(type::k == 2U);
};

template <>
struct db29::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x453ba13eU, 0x43cdd728U>;
    static_assert(type::k == 2U);
};

template <>
struct db29::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x39ba416cU, 0xb9a5df35U>;
    static_assert(type::k == 2U);
};

template <>
struct db29::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x457aa180U, 0xc52a3594U>;
    static_assert(type::k == 2U);
};

template <>
struct db29::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x395c1c0dU, 0xb980c4d0U>;
    static_assert(type::k == 2U);
};

template <>
struct db29::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x42867575U, 0xc5933aebU>;
    static_assert(type::k == 2U);
};

template <>
struct db29::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x38d1d0e4U, 0xbc292d13U>;
    static_assert(type::k == 2U);
};

template <>
struct db29::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0x42c07f95U, 0xc412edeeU>;
    static_assert(type::k == 2U);
};

template <>
struct db29::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3adee734U, 0xbb7d6da6U>;
    static_assert(type::k == 2U);
};

template <>
struct db29::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0x4381491bU, 0xc423b5aeU>;
    static_assert(type::k == 2U);
};

template <>
struct db29::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ac827efU, 0xbb93322cU>;
    static_assert(type::k == 2U);
};

template <>
struct db29::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0x435e9d47U, 0xc4417ff2U>;
    static_assert(type::k == 2U);
};

template <>
struct db29::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3aa95805U, 0xbbb1b1abU>;
    static_assert(type::k == 2U);
};

template <>
struct db29::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0x4338682aU, 0xc4704493U>;
    static_assert(type::k == 2U);
};

template <>
struct db29::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3a886191U, 0xbbe5c422U>;
    static_assert(type::k == 2U);
};

template <>
struct db29::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0x430e9d55U, 0xc4a59cbdU>;
    static_assert(type::k == 2U);
};

template <>
struct db29::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3a45dc18U, 0xbc320f98U>;
    static_assert(type::k == 2U);
};

template <>
struct db29::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0x42b806e4U, 0xc52b021cU>;
    static_assert(type::k == 2U);
};

template <>
struct db29::step<29> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct db29::step<30> {
    using type = StaticStep<StepType::kPredict, 0, 0x39bf9dd2U>;
    static_assert(type::k == 1U);
};

template <>
struct db29::step<31> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xb504b2ebU>;
    static_assert(type::k == 1U);
};

template <>
struct db29::step<32> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x49f6ef5bU>;
    static_assert(type::k == 1U);
};

struct db29_inverse {
    static constexpr const char* name = "db29-inverse";
    static constexpr uint32_t tap_size = 58U;
    static constexpr uint32_t num_steps = 33U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db29.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db29_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct db29_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3504b2ebU>;
    static_assert(type::k == 1U);
};

template <>
struct db29_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xc9f6ef5bU>;
    static_assert(type::k == 1U);
};

template <>
struct db29_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0xb9bf9dd2U>;
    static_assert(type::k == 1U);
};

template <>
struct db29_inverse::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct db29_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xc2b806e4U, 0x452b021cU>;
    static_assert(type::k == 2U);
};

template <>
struct db29_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xba45dc18U, 0x3c320f98U>;
    static_assert(type::k == 2U);
};

template <>
struct db29_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xc30e9d55U, 0x44a59cbdU>;
    static_assert(type::k == 2U);
};

template <>
struct db29_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xba886191U, 0x3be5c422U>;
    static_assert(type::k == 2U);
};

template <>
struct db29_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xc338682aU, 0x44704493U>;
    static_assert(type::k == 2U);
};

template <>
struct db29_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbaa95805U, 0x3bb1b1abU>;
    static_assert(type::k == 2U);
};

template <>
struct db29_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xc35e9d47U, 0x44417ff2U>;
    static_assert(type::k == 2U);
};

template <>
struct db29_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbac827efU, 0x3b93322cU>;
    static_assert(type::k == 2U);
};

template <>
struct db29_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xc381491bU, 0x4423b5aeU>;
    static_assert(type::k == 2U);
};

template <>
struct db29_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbadee734U, 0x3b7d6da6U>;
    static_assert(type::k == 2U);
};

template <>
struct db29_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xc2c07f95U, 0x4412edeeU>;
    static_assert(type::k == 2U);
};

template <>
struct db29_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0xb8d1d0e4U, 0x3c292d13U>;
    static_assert(type::k == 2U);
};

template <>
struct db29_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xc2867575U, 0x45933aebU>;
    static_assert(type::k == 2U);
};

template <>
struct db29_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xb95c1c0dU, 0x3980c4d0U>;
    static_assert(type::k == 2U);
};

template <>
struct db29_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xc57aa180U, 0x452a3594U>;
    static_assert(type::k == 2U);
};

template <>
struct db29_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0xb9ba416cU, 0x39a5df35U>;
    static_assert(type::k == 2U);
};

template <>
struct db29_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0xc53ba13eU, 0xc3cdd728U>;
    static_assert(type::k == 2U);
};

template <>
struct db29_inverse::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ba658aeU, 0xb860a3f5U>;
    static_assert(type::k == 2U);
};

template <>
struct db29_inverse::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0xc2f9894cU, 0xc356d743U>;
    static_assert(type::k == 2U);
};

template <>
struct db29_inverse::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbba8736aU, 0x3bde0fc3U>;
    static_assert(type::k == 2U);
};

template <>
struct db29_inverse::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0xc2f08ae6U, 0x4315844aU>;
    static_assert(type::k == 2U);
};

template <>
struct db29_inverse::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbb9910ddU, 0x3bbacfb7U>;
    static_assert(type::k == 2U);
};

template <>
struct db29_inverse::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0xc2df7b99U, 0x42ff8382U>;
    static_assert(type::k == 2U);
};

template <>
struct db29_inverse::step<27> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbbd6bfa2U, 0x3b982cd1U>;
    static_assert(type::k == 2U);
};

template <>
struct db29_inverse::step<28> {
    using type = StaticStep<StepType::kPredict, -1, 0x41a3d04dU, 0x42a87f48U>;
    static_assert(type::k == 2U);
};

template <>
struct db29_inverse::step<29> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ae9e4b1U, 0x3c34985bU>;
    static_assert(type::k == 2U);
};

template <>
struct db29_inverse::step<30> {
    using type = StaticStep<StepType::kPredict, -1, 0xc23e0a15U, 0x4195f132U>;
    static_assert(type::k == 2U);
};

template <>
struct db29_inverse::step<31> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e14e159U, 0x3c1cb1f1U>;
    static_assert(type::k == 2U);
};

template <>
struct db29_inverse::step<32> {
    using type = StaticStep<StepType::kPredict, 0, 0xc0c82df0U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
