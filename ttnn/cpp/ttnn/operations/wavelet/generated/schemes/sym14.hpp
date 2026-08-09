// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct sym14_inverse;

struct sym14 {
    static constexpr const char* name = "sym14";
    static constexpr uint32_t tap_size = 28U;
    static constexpr int32_t delay_even = 7;
    static constexpr int32_t delay_odd = 7;
    static constexpr uint32_t num_steps = 17U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/sym14.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::sym14";
    using inverse = sym14_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct sym14::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0xc013bcb8U>;
    static_assert(type::k == 1U);
};

template <>
struct sym14::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d3545f7U, 0x3ebac0a9U>;
    static_assert(type::k == 2U);
};

template <>
struct sym14::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0xc04e676dU, 0xbfd2622bU>;
    static_assert(type::k == 2U);
};

template <>
struct sym14::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbdc11569U, 0x3d934e2fU>;
    static_assert(type::k == 2U);
};

template <>
struct sym14::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x4047cc3eU, 0x4067ea6dU>;
    static_assert(type::k == 2U);
};

template <>
struct sym14::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd5bcb4eU, 0xbdb668baU>;
    static_assert(type::k == 2U);
};

template <>
struct sym14::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xbffe8a86U, 0x401248baU>;
    static_assert(type::k == 2U);
};

template <>
struct sym14::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e53d350U, 0x3d451a1bU>;
    static_assert(type::k == 2U);
};

template <>
struct sym14::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f16ea02U, 0xc04cb47cU>;
    static_assert(type::k == 2U);
};

template <>
struct sym14::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe9e84f1U, 0xbdd8655eU>;
    static_assert(type::k == 2U);
};

template <>
struct sym14::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x42639438U, 0x3f962710U>;
    static_assert(type::k == 2U);
};

template <>
struct sym14::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x38890bafU, 0xbc8fe0c0U>;
    static_assert(type::k == 2U);
};

template <>
struct sym14::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xc5220730U, 0xc38b8481U>;
    static_assert(type::k == 2U);
};

template <>
struct sym14::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x38e435d5U, 0x3979191bU>;
    static_assert(type::k == 2U);
};

template <>
struct sym14::step<14> {
    using type = StaticStep<StepType::kPredict, 0, 0xc5109377U>;
    static_assert(type::k == 1U);
};

template <>
struct sym14::step<15> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x427c9c98U>;
    static_assert(type::k == 1U);
};

template <>
struct sym14::step<16> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3c81b785U>;
    static_assert(type::k == 1U);
};

struct sym14_inverse {
    static constexpr const char* name = "sym14-inverse";
    static constexpr uint32_t tap_size = 28U;
    static constexpr uint32_t num_steps = 17U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/sym14.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::sym14_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct sym14_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x427c9c98U>;
    static_assert(type::k == 1U);
};

template <>
struct sym14_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3c81b785U>;
    static_assert(type::k == 1U);
};

template <>
struct sym14_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0x45109377U>;
    static_assert(type::k == 1U);
};

template <>
struct sym14_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xb8e435d5U, 0xb979191bU>;
    static_assert(type::k == 2U);
};

template <>
struct sym14_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x45220730U, 0x438b8481U>;
    static_assert(type::k == 2U);
};

template <>
struct sym14_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0xb8890bafU, 0x3c8fe0c0U>;
    static_assert(type::k == 2U);
};

template <>
struct sym14_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0xc2639438U, 0xbf962710U>;
    static_assert(type::k == 2U);
};

template <>
struct sym14_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e9e84f1U, 0x3dd8655eU>;
    static_assert(type::k == 2U);
};

template <>
struct sym14_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf16ea02U, 0x404cb47cU>;
    static_assert(type::k == 2U);
};

template <>
struct sym14_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe53d350U, 0xbd451a1bU>;
    static_assert(type::k == 2U);
};

template <>
struct sym14_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ffe8a86U, 0xc01248baU>;
    static_assert(type::k == 2U);
};

template <>
struct sym14_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d5bcb4eU, 0x3db668baU>;
    static_assert(type::k == 2U);
};

template <>
struct sym14_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xc047cc3eU, 0xc067ea6dU>;
    static_assert(type::k == 2U);
};

template <>
struct sym14_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3dc11569U, 0xbd934e2fU>;
    static_assert(type::k == 2U);
};

template <>
struct sym14_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x404e676dU, 0x3fd2622bU>;
    static_assert(type::k == 2U);
};

template <>
struct sym14_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd3545f7U, 0xbebac0a9U>;
    static_assert(type::k == 2U);
};

template <>
struct sym14_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x4013bcb8U>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
