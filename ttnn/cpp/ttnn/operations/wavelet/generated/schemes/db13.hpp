// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct db13_inverse;

struct db13 {
    static constexpr const char* name = "db13";
    static constexpr uint32_t tap_size = 26U;
    static constexpr int32_t delay_even = 6;
    static constexpr int32_t delay_odd = 7;
    static constexpr uint32_t num_steps = 17U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db13.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db13";
    using inverse = db13_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct db13::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0x411012b7U>;
    static_assert(type::k == 1U);
};

template <>
struct db13::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbde0ab53U, 0xbb892a55U>;
    static_assert(type::k == 2U);
};

template <>
struct db13::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0x41c9ccbdU, 0xc23372a8U>;
    static_assert(type::k == 2U);
};

template <>
struct db13::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3bc69708U, 0xbc1a9d38U>;
    static_assert(type::k == 2U);
};

template <>
struct db13::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x424968cbU, 0xc2980d7aU>;
    static_assert(type::k == 2U);
};

template <>
struct db13::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c112bc7U, 0xbc60aa38U>;
    static_assert(type::k == 2U);
};

template <>
struct db13::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x4272a015U, 0xc2ca0ac5U>;
    static_assert(type::k == 2U);
};

template <>
struct db13::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c1728feU, 0xbc8eea98U>;
    static_assert(type::k == 2U);
};

template <>
struct db13::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x4260b2adU, 0xc300230eU>;
    static_assert(type::k == 2U);
};

template <>
struct db13::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3bfeba59U, 0xbcbb7399U>;
    static_assert(type::k == 2U);
};

template <>
struct db13::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x422eba92U, 0xc3340449U>;
    static_assert(type::k == 2U);
};

template <>
struct db13::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3bb605ceU, 0xbd14ebd2U>;
    static_assert(type::k == 2U);
};

template <>
struct db13::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x41dc0925U, 0xc3bf2c3bU>;
    static_assert(type::k == 2U);
};

template <>
struct db13::step<13> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct db13::step<14> {
    using type = StaticStep<StepType::kPredict, 0, 0x3b2b67b8U>;
    static_assert(type::k == 1U);
};

template <>
struct db13::step<15> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x39c9f211U>;
    static_assert(type::k == 1U);
};

template <>
struct db13::step<16> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xc52242f4U>;
    static_assert(type::k == 1U);
};

struct db13_inverse {
    static constexpr const char* name = "db13-inverse";
    static constexpr uint32_t tap_size = 26U;
    static constexpr uint32_t num_steps = 17U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/db13.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::db13_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct db13_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0xb9c9f212U>;
    static_assert(type::k == 1U);
};

template <>
struct db13_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x452242f4U>;
    static_assert(type::k == 1U);
};

template <>
struct db13_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0xbb2b67b8U>;
    static_assert(type::k == 1U);
};

template <>
struct db13_inverse::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct db13_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xc1dc0925U, 0x43bf2c3bU>;
    static_assert(type::k == 2U);
};

template <>
struct db13_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbbb605ceU, 0x3d14ebd2U>;
    static_assert(type::k == 2U);
};

template <>
struct db13_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xc22eba92U, 0x43340449U>;
    static_assert(type::k == 2U);
};

template <>
struct db13_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbbfeba59U, 0x3cbb7399U>;
    static_assert(type::k == 2U);
};

template <>
struct db13_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xc260b2adU, 0x4300230eU>;
    static_assert(type::k == 2U);
};

template <>
struct db13_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc1728feU, 0x3c8eea98U>;
    static_assert(type::k == 2U);
};

template <>
struct db13_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xc272a015U, 0x42ca0ac5U>;
    static_assert(type::k == 2U);
};

template <>
struct db13_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc112bc7U, 0x3c60aa38U>;
    static_assert(type::k == 2U);
};

template <>
struct db13_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xc24968cbU, 0x42980d7aU>;
    static_assert(type::k == 2U);
};

template <>
struct db13_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbbc69708U, 0x3c1a9d38U>;
    static_assert(type::k == 2U);
};

template <>
struct db13_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xc1c9ccbdU, 0x423372a8U>;
    static_assert(type::k == 2U);
};

template <>
struct db13_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3de0ab53U, 0x3b892a55U>;
    static_assert(type::k == 2U);
};

template <>
struct db13_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, 0, 0xc11012b7U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
