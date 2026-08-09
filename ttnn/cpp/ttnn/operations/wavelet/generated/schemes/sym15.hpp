// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct sym15_inverse;

struct sym15 {
    static constexpr const char* name = "sym15";
    static constexpr uint32_t tap_size = 30U;
    static constexpr int32_t delay_even = 7;
    static constexpr int32_t delay_odd = 8;
    static constexpr uint32_t num_steps = 19U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/sym15.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::sym15";
    using inverse = sym15_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct sym15::step<0> {
    using type = StaticStep<StepType::kPredict, 0, 0x3f41fc73U>;
    static_assert(type::k == 1U);
};

template <>
struct sym15::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbef67507U, 0x3e15495cU>;
    static_assert(type::k == 2U);
};

template <>
struct sym15::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xbeafb7e7U, 0xbfacfa99U>;
    static_assert(type::k == 2U);
};

template <>
struct sym15::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3eaa5047U, 0xbd968d68U>;
    static_assert(type::k == 2U);
};

template <>
struct sym15::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f059025U, 0x40d940a9U>;
    static_assert(type::k == 2U);
};

template <>
struct sym15::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe03603bU, 0xbd1918d8U>;
    static_assert(type::k == 2U);
};

template <>
struct sym15::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x411bf7e5U, 0x409cbe3fU>;
    static_assert(type::k == 2U);
};

template <>
struct sym15::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbcd37265U, 0x3de22490U>;
    static_assert(type::k == 2U);
};

template <>
struct sym15::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0d25fcaU, 0xbf8a38baU>;
    static_assert(type::k == 2U);
};

template <>
struct sym15::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d7cfccbU, 0x3fba24d1U>;
    static_assert(type::k == 2U);
};

template <>
struct sym15::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf2bb3aeU, 0xbd2d96f9U>;
    static_assert(type::k == 2U);
};

template <>
struct sym15::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x404e3e69U, 0x419861ffU>;
    static_assert(type::k == 2U);
};

template <>
struct sym15::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xbd361675U, 0x3b08f4b2U>;
    static_assert(type::k == 2U);
};

template <>
struct sym15::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc0b8be1fU, 0xc288f453U>;
    static_assert(type::k == 2U);
};

template <>
struct sym15::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x3c173c86U, 0xbcaa3d86U>;
    static_assert(type::k == 2U);
};

template <>
struct sym15::step<15> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct sym15::step<16> {
    using type = StaticStep<StepType::kPredict, 0, 0x422ca761U>;
    static_assert(type::k == 1U);
};

template <>
struct sym15::step<17> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xc074c524U>;
    static_assert(type::k == 1U);
};

template <>
struct sym15::step<18> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3e85df61U>;
    static_assert(type::k == 1U);
};

struct sym15_inverse {
    static constexpr const char* name = "sym15-inverse";
    static constexpr uint32_t tap_size = 30U;
    static constexpr uint32_t num_steps = 19U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/sym15.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::sym15_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct sym15_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x4074c524U>;
    static_assert(type::k == 1U);
};

template <>
struct sym15_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0xbe85df61U>;
    static_assert(type::k == 1U);
};

template <>
struct sym15_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0xc22ca761U>;
    static_assert(type::k == 1U);
};

template <>
struct sym15_inverse::step<3> {
    using type = StaticStep<StepType::kSwap, 0>;
    static_assert(type::k == 0U);
};

template <>
struct sym15_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0xbc173c86U, 0x3caa3d86U>;
    static_assert(type::k == 2U);
};

template <>
struct sym15_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x40b8be1fU, 0x4288f453U>;
    static_assert(type::k == 2U);
};

template <>
struct sym15_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x3d361675U, 0xbb08f4b2U>;
    static_assert(type::k == 2U);
};

template <>
struct sym15_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc04e3e69U, 0xc19861ffU>;
    static_assert(type::k == 2U);
};

template <>
struct sym15_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f2bb3aeU, 0x3d2d96f9U>;
    static_assert(type::k == 2U);
};

template <>
struct sym15_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd7cfccbU, 0xbfba24d1U>;
    static_assert(type::k == 2U);
};

template <>
struct sym15_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x40d25fcaU, 0x3f8a38baU>;
    static_assert(type::k == 2U);
};

template <>
struct sym15_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3cd37265U, 0xbde22490U>;
    static_assert(type::k == 2U);
};

template <>
struct sym15_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xc11bf7e5U, 0xc09cbe3fU>;
    static_assert(type::k == 2U);
};

template <>
struct sym15_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e03603bU, 0x3d1918d8U>;
    static_assert(type::k == 2U);
};

template <>
struct sym15_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf059025U, 0xc0d940a9U>;
    static_assert(type::k == 2U);
};

template <>
struct sym15_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbeaa5047U, 0x3d968d68U>;
    static_assert(type::k == 2U);
};

template <>
struct sym15_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x3eafb7e7U, 0x3facfa99U>;
    static_assert(type::k == 2U);
};

template <>
struct sym15_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ef67507U, 0xbe15495cU>;
    static_assert(type::k == 2U);
};

template <>
struct sym15_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, 0, 0xbf41fc73U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
