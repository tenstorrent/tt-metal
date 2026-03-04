// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"

namespace ttnn {
namespace operations::unary {

using operations::complex::ComplexTensor;

enum class SigmoidMode {
    FAST_APPROXIMATE,
    ACCURATE_FAST_EXP,
    ACCURATE,
};

namespace detail {

Tensor unary_impl(
    const Tensor& input_tensor,
    const std::vector<EltwiseUnaryWithParam>& op_chain,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

}  // namespace detail

}  // namespace operations::unary

Tensor xielu(
    const Tensor& input,
    float alpha_p = 0.8f,
    float alpha_n = 0.8f,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

inline Tensor acos(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::ACOS}}, m, o, s);
}
inline Tensor asin(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::ASIN}}, m, o, s);
}
inline Tensor asinh(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::ASINH}}, m, o, s);
}
inline Tensor atan(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::ATAN}}, m, o, s);
}
inline Tensor atanh(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::ATANH}}, m, o, s);
}
inline Tensor cos(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::COS}}, m, o, s);
}
inline Tensor acosh(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::ACOSH}}, m, o, s);
}
inline Tensor cosh(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::COSH}}, m, o, s);
}
inline Tensor sinh(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::SINH}}, m, o, s);
}
inline Tensor erfinv(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::ERFINV}}, m, o, s);
}
inline Tensor exp2(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::EXP2}}, m, o, s);
}
inline Tensor expm1(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::EXPM1}}, m, o, s);
}
inline Tensor gez(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::GEZ}}, m, o, s);
}
inline Tensor gtz(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::GTZ}}, m, o, s);
}
inline Tensor i0(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::I0}}, m, o, s);
}
inline Tensor i1(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::I1}}, m, o, s);
}
inline Tensor isfinite(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::ISFINITE}}, m, o, s);
}
inline Tensor isinf(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::ISINF}}, m, o, s);
}
inline Tensor isnan(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::ISNAN}}, m, o, s);
}
inline Tensor isneginf(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::ISNEGINF}}, m, o, s);
}
inline Tensor isposinf(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::ISPOSINF}}, m, o, s);
}
inline Tensor lez(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::LEZ}}, m, o, s);
}
inline Tensor logical_not(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::LOGICAL_NOT_UNARY}}, m, o, s);
}
inline Tensor ltz(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::LTZ}}, m, o, s);
}
inline Tensor neg(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::NEG}}, m, o, s);
}
inline Tensor nez(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::NEZ}}, m, o, s);
}
inline Tensor reciprocal(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::RECIP}}, m, o, s);
}
ComplexTensor reciprocal(const ComplexTensor& t, const tt::tt_metal::MemoryConfig& m);
inline Tensor relu(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::RELU}}, m, o, s);
}
inline Tensor relu6(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::RELU6}}, m, o, s);
}
inline Tensor sign(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::SIGN}}, m, o, s);
}
inline Tensor signbit(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::SIGNBIT}}, m, o, s);
}
inline Tensor silu(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::SILU}}, m, o, s);
}
inline Tensor sin(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::SIN}}, m, o, s);
}
inline Tensor square(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::SQUARE}}, m, o, s);
}
inline Tensor tan(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::TAN}}, m, o, s);
}
inline Tensor tiled_prod(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::TILED_PROD}}, m, o, s);
}
inline Tensor bitwise_not(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::BITWISE_NOT}}, m, o, s);
}
inline Tensor alt_complex_rotate90(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::ALT_COMPLEX_ROTATE90}}, m, o, s);
}
inline Tensor floor(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::FLOOR}}, m, o, s);
}
inline Tensor ceil(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::CEIL}}, m, o, s);
}
inline Tensor trunc(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::TRUNC}}, m, o, s);
}
inline Tensor frac(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::FRAC}}, m, o, s);
}
inline Tensor hardsigmoid(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::HARDSIGMOID}}, m, o, s);
}
inline Tensor hardswish(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::HARDSWISH}}, m, o, s);
}
inline Tensor softsign(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::SOFTSIGN}}, m, o, s);
}
inline Tensor cbrt(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::CBRT}}, m, o, s);
}

// Unaries with fast_and_approximate_mode
inline Tensor exp(
    const Tensor& t,
    bool p = false,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::EXP, static_cast<float>(p)}}, m, o, s);
}
inline Tensor erf(
    const Tensor& t,
    bool p = false,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::ERF, static_cast<float>(p)}}, m, o, s);
}
inline Tensor erfc(
    const Tensor& t,
    bool p = false,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::ERFC, static_cast<float>(p)}}, m, o, s);
}
inline Tensor gelu(
    const Tensor& t,
    bool p = false,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::GELU, static_cast<float>(p)}}, m, o, s);
}
inline Tensor log(
    const Tensor& t,
    bool p = false,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::LOG, static_cast<float>(p)}}, m, o, s);
}
inline Tensor log10(
    const Tensor& t,
    bool p = false,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::LOG10, static_cast<float>(p)}}, m, o, s);
}
inline Tensor log2(
    const Tensor& t,
    bool p = false,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::LOG2, static_cast<float>(p)}}, m, o, s);
}
inline Tensor log1p(
    const Tensor& t,
    bool p = false,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::LOG1P, static_cast<float>(p)}}, m, o, s);
}
inline Tensor rsqrt(
    const Tensor& t,
    bool p = false,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::RSQRT, static_cast<float>(p)}}, m, o, s);
}
inline Tensor sqrt(
    const Tensor& t,
    bool p = false,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::SQRT, static_cast<float>(p)}}, m, o, s);
}

// Unaries with float parameter
inline Tensor heaviside(
    const Tensor& t,
    float p,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t,
        {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::HEAVISIDE, static_cast<float>(p)}},
        m,
        o,
        s);
}
inline Tensor leaky_relu(
    const Tensor& t,
    float p,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t,
        {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::LEAKY_RELU, static_cast<float>(p)}},
        m,
        o,
        s);
}
inline Tensor relu_max(
    const Tensor& t,
    float p,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t,
        {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::RELU_MAX, static_cast<float>(p)}},
        m,
        o,
        s);
}
inline Tensor relu_min(
    const Tensor& t,
    float p,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t,
        {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::RELU_MIN, static_cast<float>(p)}},
        m,
        o,
        s);
}
inline Tensor unary_remainder(
    const Tensor& t,
    float p,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t,
        {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::REMAINDER, static_cast<float>(p)}},
        m,
        o,
        s);
}
inline Tensor celu(
    const Tensor& t,
    float p,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::CELU, static_cast<float>(p)}}, m, o, s);
}
inline Tensor rpow(
    const Tensor& t,
    float p,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    const std::optional<CoreRangeSet>& s = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::RPOW, static_cast<float>(p)}}, m, o, s);
}
inline Tensor unary_fmod(
    const Tensor& t,
    float p,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::FMOD, static_cast<float>(p)}}, m, o);
}

// Unaries with two float parameter
inline Tensor threshold(
    const Tensor& t,
    float pa,
    float pb,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t,
        {operations::unary::UnaryWithParam{
            operations::unary::UnaryOpType::THRESHOLD, {static_cast<float>(pa), static_cast<float>(pb)}}},
        m,
        o);
}

// Unaries with optional integer parameter
inline Tensor round(
    const Tensor& t,
    const std::optional<int32_t>& p = std::nullopt,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::ROUND, p.value_or(0)}}, m, o);
}

// Other unaries
inline Tensor identity(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::IDENTITY}}, m, o);
}
inline Tensor abs(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    auto op_type = t.dtype() == tt::tt_metal::DataType::INT32 ? operations::unary::UnaryOpType::ABS_INT32
                                                              : operations::unary::UnaryOpType::ABS;
    return operations::unary::detail::unary_impl(t, {operations::unary::UnaryWithParam{op_type}}, m, o);
}
Tensor abs(const operations::unary::ComplexTensor& t, const tt::tt_metal::MemoryConfig& m);
inline Tensor eqz(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::EQZ}}, m, o);
}
inline Tensor mish(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::MISH}}, m, o);
}
inline Tensor hardmish(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::HARDMISH}}, m, o);
}
inline Tensor hardshrink(
    const Tensor& t,
    float lambda = 0.5f,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t,
        {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::HARDSHRINK, static_cast<float>(lambda)}},
        m,
        o);
}
inline Tensor logit(
    const Tensor& t,
    std::optional<float> eps = std::nullopt,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::LOGIT, {eps.value_or(-1.0f)}}}, m, o);
}
inline Tensor elu(
    const Tensor& t,
    float alpha = 1.0f,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::ELU, static_cast<float>(alpha)}}, m, o);
}
inline Tensor hardtanh(
    const Tensor& t,
    float min_val = -1.0f,
    float max_val = 1.0f,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::HARDTANH, {min_val, max_val}}}, m, o);
}
inline Tensor softshrink(
    const Tensor& t,
    float lambda = 0.5f,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t,
        {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::SOFTSHRINK, static_cast<float>(lambda)}},
        m,
        o);
}
Tensor deg2rad(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt);
Tensor rad2deg(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt);
inline Tensor clamp_tss(
    const Tensor& t,
    float min_val,
    float max_val,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::CLAMP_TSS, {min_val, max_val}}}, m, o);
}
inline Tensor clamp_tss(
    const Tensor& t,
    int32_t min_val,
    int32_t max_val,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t,
        {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::CLAMP_TSS, {min_val, max_val}}},
        m,
        o);
}
inline Tensor softplus(
    const Tensor& t,
    float beta,
    float threshold,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::SOFTPLUS, {beta, threshold}}}, m, o);
}
inline Tensor tanh(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    bool approx = false) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::TANH, static_cast<float>(approx)}}, m, o);
}
inline Tensor tanhshrink(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    bool /*approx*/ = false) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::TANHSHRINK}}, m, o);
}
inline Tensor prelu_sfpu(
    const Tensor& t,
    float value,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::PRELU_SFPU, value}}, m, o);
}
Tensor where_tss(
    const Tensor& c,
    const operations::unary::ScalarVariant& vt,
    const operations::unary::ScalarVariant& vf,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt);
inline Tensor selu(
    const Tensor& t,
    float scale = 1.050700987f,
    float alpha = 1.673263242f,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::SELU, {scale, alpha}}}, m, o);
}
Tensor bitcast(
    const Tensor& t,
    const tt::tt_metal::DataType& dtype,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt);
Tensor rdiv(
    const Tensor& t,
    float value,
    const std::optional<std::string>& rounding_mode = std::nullopt,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt);
inline Tensor swish(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::SILU}}, m, o);
}

// TS variant unaries
inline Tensor minimum(
    const Tensor& t,
    operations::unary::ScalarVariant p,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return std::visit(
        [&](auto param) {
            return operations::unary::detail::unary_impl(
                t, {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::MINIMUM, (param)}}, m, o);
        },
        p);
}
inline Tensor maximum(
    const Tensor& t,
    operations::unary::ScalarVariant p,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return std::visit(
        [&](auto param) {
            return operations::unary::detail::unary_impl(
                t, {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::MAXIMUM, (param)}}, m, o);
        },
        p);
}
inline Tensor fill(
    const Tensor& t,
    operations::unary::ScalarVariant p,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return std::visit(
        [&](auto param) {
            return operations::unary::detail::unary_impl(
                t, {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::FILL, (param)}}, m, o);
        },
        p);
}
inline Tensor power(
    const Tensor& t,
    operations::unary::ScalarVariant p,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return std::visit(
        [&](auto param) {
            return operations::unary::detail::unary_impl(
                t, {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::POWER, (param)}}, m, o);
        },
        p);
}
inline Tensor power_iterative(
    const Tensor& t,
    uint32_t exponent,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::POWER_ITERATIVE, exponent}}, m, o);
}
inline Tensor gt_unary(
    const Tensor& t,
    operations::unary::ScalarVariant p,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return std::visit(
        [&](auto param) {
            return operations::unary::detail::unary_impl(
                t, {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::UNARY_GT, (param)}}, m, o);
        },
        p);
}
inline Tensor lt_unary(
    const Tensor& t,
    operations::unary::ScalarVariant p,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return std::visit(
        [&](auto param) {
            return operations::unary::detail::unary_impl(
                t, {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::UNARY_LT, (param)}}, m, o);
        },
        p);
}
inline Tensor ne_unary(
    const Tensor& t,
    operations::unary::ScalarVariant p,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return std::visit(
        [&](auto param) {
            return operations::unary::detail::unary_impl(
                t, {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::UNARY_NE, (param)}}, m, o);
        },
        p);
}
inline Tensor eq_unary(
    const Tensor& t,
    operations::unary::ScalarVariant p,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return std::visit(
        [&](auto param) {
            return operations::unary::detail::unary_impl(
                t, {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::UNARY_EQ, (param)}}, m, o);
        },
        p);
}
inline Tensor ge_unary(
    const Tensor& t,
    operations::unary::ScalarVariant p,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return std::visit(
        [&](auto param) {
            return operations::unary::detail::unary_impl(
                t, {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::UNARY_GE, (param)}}, m, o);
        },
        p);
}
inline Tensor le_unary(
    const Tensor& t,
    operations::unary::ScalarVariant p,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return std::visit(
        [&](auto param) {
            return operations::unary::detail::unary_impl(
                t, {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::UNARY_LE, (param)}}, m, o);
        },
        p);
}

// Sigmoid and related
inline Tensor sigmoid(
    const Tensor& t,
    int vector_mode = (int32_t)operations::unary::VecMode::RC,
    operations::unary::SigmoidMode mode = operations::unary::SigmoidMode::ACCURATE,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    std::vector<operations::unary::EltwiseUnaryWithParam> op_chain;
    switch (mode) {
        case operations::unary::SigmoidMode::FAST_APPROXIMATE:
            op_chain = {operations::unary::UnaryWithParam(
                operations::unary::UnaryOpType::SIGMOID, {static_cast<float>(vector_mode), 1.0f})};
            break;
        case operations::unary::SigmoidMode::ACCURATE_FAST_EXP:
            op_chain = {
                operations::unary::UnaryWithParam(operations::unary::UnaryOpType::NEG),
                operations::unary::UnaryWithParam(operations::unary::UnaryOpType::EXP, 1.0f),
                operations::unary::UnaryWithParam(operations::unary::UnaryOpType::ADD_UNARY_SFPU, 1.0f),
                operations::unary::UnaryWithParam(operations::unary::UnaryOpType::RECIP)};
            break;
        case operations::unary::SigmoidMode::ACCURATE: [[fallthrough]];
        default:
            op_chain = {operations::unary::UnaryWithParam(
                operations::unary::UnaryOpType::SIGMOID, {static_cast<float>(vector_mode), 0.0f})};
    }
    return operations::unary::detail::unary_impl(t, op_chain, m, o);
}
inline Tensor sigmoid_accurate(
    const Tensor& t,
    bool fast_and_approximate_mode = false,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    auto
        op_chain = fast_and_approximate_mode
                       ? std::vector<
                             operations::unary::EltwiseUnaryWithParam>{operations::unary::UnaryWithParam(operations::unary::UnaryOpType::NEG), operations::unary::UnaryWithParam(operations::unary::UnaryOpType::EXP, 1.0f), operations::unary::UnaryWithParam(operations::unary::UnaryOpType::ADD_UNARY_SFPU, 1.0f), operations::unary::UnaryWithParam(operations::unary::UnaryOpType::RECIP)}
                       : std::vector<operations::unary::EltwiseUnaryWithParam>{operations::unary::UnaryWithParam(
                             operations::unary::UnaryOpType::SIGMOID,
                             {static_cast<float>(operations::unary::VecMode::RC), 0.0f})};
    return operations::unary::detail::unary_impl(t, op_chain, m, o);
}
inline Tensor log_sigmoid(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::LOGSIGMOID}}, m, o);
}
inline Tensor unary_chain(
    const Tensor& t,
    const std::vector<operations::unary::EltwiseUnaryWithParam>& chain,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return operations::unary::detail::unary_impl(t, chain, m, o);
}

// Rsub and SFPU binops
template <typename T>
inline Tensor rsub_sfpu(
    const Tensor& t,
    T param,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::RSUB, {param}}}, m, o);
}
inline Tensor add_sfpu(
    const Tensor& t,
    float p,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::EltwiseUnaryWithParam(operations::unary::UnaryOpType::ADD_UNARY_SFPU, (p))}, m, o);
}
inline Tensor add_sfpu(
    float p,
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::EltwiseUnaryWithParam(operations::unary::UnaryOpType::ADD_UNARY_SFPU, (p))}, m, o);
}
inline Tensor mul_sfpu(
    const Tensor& t,
    float p,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::EltwiseUnaryWithParam(operations::unary::UnaryOpType::MUL_UNARY_SFPU, (p))}, m, o);
}
inline Tensor mul_sfpu(
    float p,
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::EltwiseUnaryWithParam(operations::unary::UnaryOpType::MUL_UNARY_SFPU, (p))}, m, o);
}
inline Tensor sub_sfpu(
    const Tensor& t,
    float p,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::SUB_UNARY_SFPU, (p)}}, m, o);
}
inline Tensor sub_sfpu(
    float p,
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::RSUB, (p)}}, m, o);
}
inline Tensor div_sfpu(
    const Tensor& t,
    float p,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::DIV_UNARY_SFPU, (p)}}, m, o);
}
inline Tensor div_sfpu(
    float p,
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::RDIV, (p)}}, m, o);
}
inline Tensor unary_with_int32_param(
    operations::unary::UnaryOpType op_type,
    const Tensor& t,
    int32_t param,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return operations::unary::detail::unary_impl(t, {operations::unary::EltwiseUnaryWithParam{op_type, param}}, m, o);
}
}  // namespace ttnn
