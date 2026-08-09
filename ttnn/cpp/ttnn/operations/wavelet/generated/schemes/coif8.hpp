// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/wavelet/planner/static_scheme.hpp"

namespace ttnn::operations::wavelet::schemes {

struct coif8_inverse;

struct coif8 {
    static constexpr const char* name = "coif8";
    static constexpr uint32_t tap_size = 48U;
    static constexpr int32_t delay_even = 12;
    static constexpr int32_t delay_odd = 12;
    static constexpr uint32_t num_steps = 27U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/coif8.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::coif8";
    using inverse = coif8_inverse;

    template <std::size_t I>
    struct step;
};

template <>
struct coif8::step<0> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f2d232fU>;
    static_assert(type::k == 1U);
};

template <>
struct coif8::step<1> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fcb7a75U, 0xbeed98c0U>;
    static_assert(type::k == 2U);
};

template <>
struct coif8::step<2> {
    using type = StaticStep<StepType::kPredict, -1, 0x3eebb334U, 0xbf07bfacU>;
    static_assert(type::k == 2U);
};

template <>
struct coif8::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f9498ffU, 0xbfdef081U>;
    static_assert(type::k == 2U);
};

template <>
struct coif8::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e868f8fU, 0xbf0daed8U>;
    static_assert(type::k == 2U);
};

template <>
struct coif8::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f418884U, 0xbf8c75f5U>;
    static_assert(type::k == 2U);
};

template <>
struct coif8::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e23725fU, 0xbe5a8363U>;
    static_assert(type::k == 2U);
};

template <>
struct coif8::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3e0feac7U, 0xbf1bd607U>;
    static_assert(type::k == 2U);
};

template <>
struct coif8::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0xbd501735U, 0xbd263ca7U>;
    static_assert(type::k == 2U);
};

template <>
struct coif8::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbec11604U, 0x3e339215U>;
    static_assert(type::k == 2U);
};

template <>
struct coif8::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe77f6efU, 0x3dd8c0c1U>;
    static_assert(type::k == 2U);
};

template <>
struct coif8::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf851d4fU, 0x3f3cf394U>;
    static_assert(type::k == 2U);
};

template <>
struct coif8::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe92fb7cU, 0x3e948888U>;
    static_assert(type::k == 2U);
};

template <>
struct coif8::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfe16b69U, 0x3f848e2aU>;
    static_assert(type::k == 2U);
};

template <>
struct coif8::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf171e90U, 0x3e9ff98eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif8::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0xc0a598c1U, 0x3fb06f34U>;
    static_assert(type::k == 2U);
};

template <>
struct coif8::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0xbffc36c8U, 0x3e42b708U>;
    static_assert(type::k == 2U);
};

template <>
struct coif8::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbd875627U, 0x3f01e6dbU>;
    static_assert(type::k == 2U);
};

template <>
struct coif8::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0xc2038278U, 0x416ffb44U>;
    static_assert(type::k == 2U);
};

template <>
struct coif8::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbdb00207U, 0x3cf8b4cfU>;
    static_assert(type::k == 2U);
};

template <>
struct coif8::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0xc22bd787U, 0x413a215dU>;
    static_assert(type::k == 2U);
};

template <>
struct coif8::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbdf5a59eU, 0x3cbeaf04U>;
    static_assert(type::k == 2U);
};

template <>
struct coif8::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0xc28748e9U, 0x41056505U>;
    static_assert(type::k == 2U);
};

template <>
struct coif8::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0x2d143c91U, 0x3c723705U>;
    static_assert(type::k == 2U);
};

template <>
struct coif8::step<24> {
    using type = StaticStep<StepType::kPredict, 0, 0xc07ddc41U>;
    static_assert(type::k == 1U);
};

template <>
struct coif8::step<25> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x444f1413U>;
    static_assert(type::k == 1U);
};

template <>
struct coif8::step<26> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x3a9e3d55U>;
    static_assert(type::k == 1U);
};

struct coif8_inverse {
    static constexpr const char* name = "coif8-inverse";
    static constexpr uint32_t tap_size = 48U;
    static constexpr uint32_t num_steps = 27U;
    static constexpr const char* compute_scheme_header = "\"ttnn/cpp/ttnn/operations/wavelet/generated/schemes/coif8.hpp\"";
    static constexpr const char* compute_scheme_type = "ttnn::operations::wavelet::schemes::coif8_inverse";

    template <std::size_t I>
    struct step;
};

template <>
struct coif8_inverse::step<0> {
    using type = StaticStep<StepType::kScaleOdd, 0, 0x444f1413U>;
    static_assert(type::k == 1U);
};

template <>
struct coif8_inverse::step<1> {
    using type = StaticStep<StepType::kScaleEven, 0, 0x3a9e3d55U>;
    static_assert(type::k == 1U);
};

template <>
struct coif8_inverse::step<2> {
    using type = StaticStep<StepType::kPredict, 0, 0x407ddc41U>;
    static_assert(type::k == 1U);
};

template <>
struct coif8_inverse::step<3> {
    using type = StaticStep<StepType::kUpdate, 0, 0xad143c91U, 0xbc723705U>;
    static_assert(type::k == 2U);
};

template <>
struct coif8_inverse::step<4> {
    using type = StaticStep<StepType::kPredict, -1, 0x428748e9U, 0xc1056505U>;
    static_assert(type::k == 2U);
};

template <>
struct coif8_inverse::step<5> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3df5a59eU, 0xbcbeaf04U>;
    static_assert(type::k == 2U);
};

template <>
struct coif8_inverse::step<6> {
    using type = StaticStep<StepType::kPredict, -1, 0x422bd787U, 0xc13a215dU>;
    static_assert(type::k == 2U);
};

template <>
struct coif8_inverse::step<7> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3db00207U, 0xbcf8b4cfU>;
    static_assert(type::k == 2U);
};

template <>
struct coif8_inverse::step<8> {
    using type = StaticStep<StepType::kPredict, -1, 0x42038278U, 0xc16ffb44U>;
    static_assert(type::k == 2U);
};

template <>
struct coif8_inverse::step<9> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3d875627U, 0xbf01e6dbU>;
    static_assert(type::k == 2U);
};

template <>
struct coif8_inverse::step<10> {
    using type = StaticStep<StepType::kPredict, -1, 0x3ffc36c8U, 0xbe42b708U>;
    static_assert(type::k == 2U);
};

template <>
struct coif8_inverse::step<11> {
    using type = StaticStep<StepType::kUpdate, 0, 0x40a598c1U, 0xbfb06f34U>;
    static_assert(type::k == 2U);
};

template <>
struct coif8_inverse::step<12> {
    using type = StaticStep<StepType::kPredict, -1, 0x3f171e90U, 0xbe9ff98eU>;
    static_assert(type::k == 2U);
};

template <>
struct coif8_inverse::step<13> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3fe16b69U, 0xbf848e2aU>;
    static_assert(type::k == 2U);
};

template <>
struct coif8_inverse::step<14> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e92fb7cU, 0xbe948888U>;
    static_assert(type::k == 2U);
};

template <>
struct coif8_inverse::step<15> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3f851d4fU, 0xbf3cf394U>;
    static_assert(type::k == 2U);
};

template <>
struct coif8_inverse::step<16> {
    using type = StaticStep<StepType::kPredict, -1, 0x3e77f6efU, 0xbdd8c0c1U>;
    static_assert(type::k == 2U);
};

template <>
struct coif8_inverse::step<17> {
    using type = StaticStep<StepType::kUpdate, 0, 0x3ec11604U, 0xbe339215U>;
    static_assert(type::k == 2U);
};

template <>
struct coif8_inverse::step<18> {
    using type = StaticStep<StepType::kPredict, -1, 0x3d501735U, 0x3d263ca7U>;
    static_assert(type::k == 2U);
};

template <>
struct coif8_inverse::step<19> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbe0feac7U, 0x3f1bd607U>;
    static_assert(type::k == 2U);
};

template <>
struct coif8_inverse::step<20> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe23725fU, 0x3e5a8363U>;
    static_assert(type::k == 2U);
};

template <>
struct coif8_inverse::step<21> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf418884U, 0x3f8c75f5U>;
    static_assert(type::k == 2U);
};

template <>
struct coif8_inverse::step<22> {
    using type = StaticStep<StepType::kPredict, -1, 0xbe868f8fU, 0x3f0daed8U>;
    static_assert(type::k == 2U);
};

template <>
struct coif8_inverse::step<23> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbf9498ffU, 0x3fdef081U>;
    static_assert(type::k == 2U);
};

template <>
struct coif8_inverse::step<24> {
    using type = StaticStep<StepType::kPredict, -1, 0xbeebb334U, 0x3f07bfacU>;
    static_assert(type::k == 2U);
};

template <>
struct coif8_inverse::step<25> {
    using type = StaticStep<StepType::kUpdate, 0, 0xbfcb7a75U, 0x3eed98c0U>;
    static_assert(type::k == 2U);
};

template <>
struct coif8_inverse::step<26> {
    using type = StaticStep<StepType::kPredict, -1, 0xbf2d232fU>;
    static_assert(type::k == 1U);
};

}  // namespace ttnn::operations::wavelet::schemes
