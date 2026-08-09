// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct sym16_inverse;

struct sym16 {
    static constexpr const char* name = "sym16";
    static constexpr uint32_t tap_size = 32U;
    static constexpr int32_t delay_even = 8;
    static constexpr int32_t delay_odd = 8;
    static constexpr uint32_t num_steps = 19U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/sym16.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::sym16";
    using inverse = sym16_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct sym16::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0000f3aU>;
    static_assert(type::k == 1U);
};

template <>
struct sym16::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d6f80dcU, 0x3eccbe2fU>;
    static_assert(type::k == 2U);
};

template <>
struct sym16::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xc08b6e58U, 0xbfac9d38U>;
    static_assert(type::k == 2U);
};

template <>
struct sym16::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd240fb0U, 0x3dddd0d4U>;
    static_assert(type::k == 2U);
};

template <>
struct sym16::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x40944606U, 0x402e050eU>;
    static_assert(type::k == 2U);
};

template <>
struct sym16::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbdcc3739U, 0xbd678f8eU>;
    static_assert(type::k == 2U);
};

template <>
struct sym16::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0318f36U, 0x40a86708U>;
    static_assert(type::k == 2U);
};

template <>
struct sym16::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbce8c1d4U, 0x3dad5754U>;
    static_assert(type::k == 2U);
};

template <>
struct sym16::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x40232489U, 0x3f969630U>;
    static_assert(type::k == 2U);
};

template <>
struct sym16::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e34d109U, 0xbd6071c7U>;
    static_assert(type::k == 2U);
};

template <>
struct sym16::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x3fe53fcfU, 0xc0638180U>;
    static_assert(type::k == 2U);
};

template <>
struct sym16::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbdb4d3d2U, 0xbe2b5bbdU>;
    static_assert(type::k == 2U);
};

template <>
struct sym16::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x40f33b3eU, 0x3f9a5682U>;
    static_assert(type::k == 2U);
};

template <>
struct sym16::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c21177dU, 0xbdde7498U>;
    static_assert(type::k == 2U);
};

template <>
struct sym16::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xc19ef242U, 0xc073183dU>;
    static_assert(type::k == 2U);
};

template <>
struct sym16::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3c6dc8a5U, 0x3cd44e24U>;
    static_assert(type::k == 2U);
};

template <>
struct sym16::step<16> {
    using type = StaticStep<StepType::kPredict, 0, 0xc189aa8fU>;
    static_assert(type::k == 1U);
};

template <>
struct sym16::step<17> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x40aec2f2U>;
    static_assert(type::k == 1U);
};

template <>
struct sym16::step<18> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3e3b8052U>;
    static_assert(type::k == 1U);
};

struct sym16_inverse {
    static constexpr const char* name = "sym16-inverse";
    static constexpr uint32_t tap_size = 32U;
    static constexpr uint32_t num_steps = 19U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/sym16.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::sym16_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct sym16_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x40aec2f2U>;
    static_assert(type::k == 1U);
};

template <>
struct sym16_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3e3b8052U>;
    static_assert(type::k == 1U);
};

template <>
struct sym16_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0x4189aa8fU>;
    static_assert(type::k == 1U);
};

template <>
struct sym16_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc6dc8a5U, 0xbcd44e24U>;
    static_assert(type::k == 2U);
};

template <>
struct sym16_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x419ef242U, 0x4073183dU>;
    static_assert(type::k == 2U);
};

template <>
struct sym16_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbc21177dU, 0x3dde7498U>;
    static_assert(type::k == 2U);
};

template <>
struct sym16_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0f33b3eU, 0xbf9a5682U>;
    static_assert(type::k == 2U);
};

template <>
struct sym16_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3db4d3d2U, 0x3e2b5bbdU>;
    static_assert(type::k == 2U);
};

template <>
struct sym16_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xbfe53fcfU, 0x40638180U>;
    static_assert(type::k == 2U);
};

template <>
struct sym16_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe34d109U, 0x3d6071c7U>;
    static_assert(type::k == 2U);
};

template <>
struct sym16_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0232489U, 0xbf969630U>;
    static_assert(type::k == 2U);
};

template <>
struct sym16_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ce8c1d4U, 0xbdad5754U>;
    static_assert(type::k == 2U);
};

template <>
struct sym16_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x40318f36U, 0xc0a86708U>;
    static_assert(type::k == 2U);
};

template <>
struct sym16_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3dcc3739U, 0x3d678f8eU>;
    static_assert(type::k == 2U);
};

template <>
struct sym16_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xc0944606U, 0xc02e050eU>;
    static_assert(type::k == 2U);
};

template <>
struct sym16_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d240fb0U, 0xbdddd0d4U>;
    static_assert(type::k == 2U);
};

template <>
struct sym16_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x408b6e58U, 0x3fac9d38U>;
    static_assert(type::k == 2U);
};

template <>
struct sym16_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd6f80dcU, 0xbeccbe2fU>;
    static_assert(type::k == 2U);
};

template <>
struct sym16_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0x40000f3aU>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
