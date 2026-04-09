// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/eltwise/unary_ng/unary_ng.hpp"

namespace ttnn {
namespace operations::unary {

enum class SigmoidMode {
    FAST_APPROXIMATE,
    ACCURATE_FAST_EXP,
    ACCURATE,
};

}  // namespace operations::unary

namespace detail {

Tensor unary_impl(
    const Tensor& input_tensor,
    const std::vector<operations::unary::EltwiseUnaryWithParam>& op_chain,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

}  // namespace detail

#define REGISTER_UNARY_OPERATION(op_name, op_type)                                        \
    inline Tensor op_name(                                                                \
        const Tensor& input_tensor,                                                       \
        const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,    \
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,               \
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {               \
        return ttnn::detail::unary_impl(                                                  \
            input_tensor,                                                                 \
            {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::op_type}}, \
            memory_config,                                                                \
            optional_output_tensor,                                                       \
            sub_core_grids);                                                              \
    }

#define REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(op_name, op_type)                         \
    inline Tensor op_name(                                                                                \
        const Tensor& input_tensor,                                                                       \
        bool fast_and_approximate_mode = false,                                                           \
        const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,                    \
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,                               \
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {                               \
        return ttnn::detail::unary_impl(                                                                  \
            input_tensor,                                                                                 \
            {operations::unary::UnaryWithParam{                                                           \
                operations::unary::UnaryOpType::op_type, static_cast<float>(fast_and_approximate_mode)}}, \
            memory_config,                                                                                \
            optional_output_tensor,                                                                       \
            sub_core_grids);                                                                              \
    }

#define REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(op_name, op_type)                   \
    inline Tensor op_name(                                                                \
        const Tensor& input_tensor,                                                       \
        float parameter,                                                                  \
        const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,    \
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,               \
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {               \
        return ttnn::detail::unary_impl(                                                  \
            input_tensor,                                                                 \
            {operations::unary::UnaryWithParam{                                           \
                operations::unary::UnaryOpType::op_type, static_cast<float>(parameter)}}, \
            memory_config,                                                                \
            optional_output_tensor,                                                       \
            sub_core_grids);                                                              \
    }

#define UNARY_OP_SCALAR_VARIANT(op_name, op_type)                                                                 \
    inline Tensor op_name(                                                                                        \
        const Tensor& input_tensor,                                                                               \
        operations::unary::ScalarVariant parameter,                                                               \
        const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,                            \
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {                                     \
        return std::visit(                                                                                        \
            [&](auto param) {                                                                                     \
                return ttnn::detail::unary_impl(                                                                  \
                    input_tensor,                                                                                 \
                    {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::op_type, (param)}}, \
                    memory_config,                                                                                \
                    optional_output_tensor);                                                                      \
            },                                                                                                    \
            parameter);                                                                                           \
    }

// Unaries without parameters
REGISTER_UNARY_OPERATION(hardsigmoid, HARDSIGMOID)

// Stubs for nuked ops still referenced by other modules.
// These declarations use the old unary path which passes through UnaryOpType dispatch.
// The underlying SFPU kernels may not exist yet; these stubs allow the host-side code to compile.
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(exp, EXP)
REGISTER_UNARY_OPERATION(reciprocal, RECIP)
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(gelu, GELU)
REGISTER_UNARY_OPERATION(relu, RELU)
REGISTER_UNARY_OPERATION(sqrt_op, SQRT)
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(log, LOG)
REGISTER_UNARY_OPERATION(log1p, LOG1P)
REGISTER_UNARY_OPERATION(tanh, TANH)
REGISTER_UNARY_OPERATION(log2, LOG2)
REGISTER_UNARY_OPERATION(log10, LOG10)
REGISTER_UNARY_OPERATION(sin, SIN)
REGISTER_UNARY_OPERATION(cos, COS)
REGISTER_UNARY_OPERATION(cosh, COSH)
REGISTER_UNARY_OPERATION(sinh, SINH)
REGISTER_UNARY_OPERATION(abs, ABS)
REGISTER_UNARY_OPERATION(sign, SIGN)
REGISTER_UNARY_OPERATION(square, SQUARE)
REGISTER_UNARY_OPERATION(eqz, EQZ)
REGISTER_UNARY_OPERATION(nez, NEZ)
REGISTER_UNARY_OPERATION(gtz, GTZ)
REGISTER_UNARY_OPERATION(ltz, LTZ)
REGISTER_UNARY_OPERATION(gez, GEZ)
REGISTER_UNARY_OPERATION(lez, LEZ)
REGISTER_UNARY_OPERATION(exp2, EXP2)
REGISTER_UNARY_OPERATION(expm1, EXPM1)
REGISTER_UNARY_OPERATION(signbit, SIGNBIT)
REGISTER_UNARY_OPERATION(asin, ASIN)
REGISTER_UNARY_OPERATION(acos, ACOS)
REGISTER_UNARY_OPERATION(acosh, ACOSH)
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(rsqrt, RSQRT)
REGISTER_UNARY_OPERATION(relu6, RELU6)
REGISTER_UNARY_OPERATION(atan, ATAN)
REGISTER_UNARY_OPERATION(asinh, ASINH)
REGISTER_UNARY_OPERATION(atanh, ATANH)
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(erf, ERF)
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(erfc, ERFC)
REGISTER_UNARY_OPERATION(isinf, ISINF)
REGISTER_UNARY_OPERATION(isposinf, ISPOSINF)
REGISTER_UNARY_OPERATION(isneginf, ISNEGINF)
REGISTER_UNARY_OPERATION(isnan, ISNAN)
REGISTER_UNARY_OPERATION(logical_not_unary, LOGICAL_NOT_UNARY)
REGISTER_UNARY_OPERATION(isfinite, ISFINITE)
REGISTER_UNARY_OPERATION(erfinv, ERFINV)
REGISTER_UNARY_OPERATION(i0, I0)
REGISTER_UNARY_OPERATION(i1, I1)
REGISTER_UNARY_OPERATION(tan, TAN)
REGISTER_UNARY_OPERATION(silu, SILU)
REGISTER_UNARY_OPERATION(neg, NEG)
REGISTER_UNARY_OPERATION(floor, FLOOR)
REGISTER_UNARY_OPERATION(ceil, CEIL)
REGISTER_UNARY_OPERATION(trunc, TRUNC)
REGISTER_UNARY_OPERATION(frac, FRAC)
REGISTER_UNARY_OPERATION(round, ROUND)
REGISTER_UNARY_OPERATION(tiled_prod, TILED_PROD)
REGISTER_UNARY_OPERATION(hardswish, HARDSWISH)
REGISTER_UNARY_OPERATION(tanhshrink, TANHSHRINK)
REGISTER_UNARY_OPERATION(softsign, SOFTSIGN)
REGISTER_UNARY_OPERATION(cbrt, CBRT)
REGISTER_UNARY_OPERATION(logsigmoid, LOGSIGMOID)
REGISTER_UNARY_OPERATION(lgamma, LGAMMA)
REGISTER_UNARY_OPERATION(swish, SWISH)
REGISTER_UNARY_OPERATION(hardmish, HARDMISH)
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(softshrink, SOFTSHRINK)
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(hardshrink, HARDSHRINK)
// fill takes a fill_value parameter, not a simple unary op
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(fill, FILL)
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(rpow, RPOW)
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(unary_fmod, FMOD)
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(unary_remainder, REMAINDER)
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(power_iterative, POWER_ITERATIVE)

// These are binary-like SFPU ops (take Tensor + scalar_or_tensor)
UNARY_OP_SCALAR_VARIANT(rsub_sfpu, RSUB)
UNARY_OP_SCALAR_VARIANT(sub_sfpu, SUB_UNARY_SFPU)
UNARY_OP_SCALAR_VARIANT(gt_unary, UNARY_GT)
UNARY_OP_SCALAR_VARIANT(lt_unary, UNARY_LT)
UNARY_OP_SCALAR_VARIANT(ne_unary, UNARY_NE)
UNARY_OP_SCALAR_VARIANT(eq_unary, UNARY_EQ)
UNARY_OP_SCALAR_VARIANT(ge_unary, UNARY_GE)
UNARY_OP_SCALAR_VARIANT(le_unary, UNARY_LE)

// Unaries with float parameter
UNARY_OP_SCALAR_VARIANT(power, POWER)
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(heaviside, HEAVISIDE)

// Sigmoid with special signature
Tensor sigmoid(
    const Tensor& input_tensor,
    int vector_mode = 4,
    operations::unary::SigmoidMode mode = operations::unary::SigmoidMode::ACCURATE,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

// Deprecated alias
inline Tensor sigmoid_accurate(
    const Tensor& input_tensor,
    [[maybe_unused]] bool fast_and_approximate_mode = false,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return sigmoid(input_tensor, 4, operations::unary::SigmoidMode::ACCURATE, memory_config, optional_output_tensor);
}

// swish is now a first-class SFPU operation registered via REGISTER_UNARY_OPERATION(swish, SWISH)

// sqrt (avoid name clash with std::sqrt)
inline Tensor sqrt(
    const Tensor& input_tensor,
    [[maybe_unused]] bool fast_and_approximate_mode = false,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return sqrt_op(input_tensor, memory_config, optional_output_tensor, sub_core_grids);
}

// logical_not alias
inline Tensor logical_not(
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return logical_not_unary(input_tensor, memory_config, optional_output_tensor, sub_core_grids);
}

// Softplus with beta and threshold parameters
Tensor softplus(
    const Tensor& input_tensor,
    float beta = 1.0f,
    float threshold = 20.0f,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

// xielu
Tensor xielu(
    const Tensor& input_tensor,
    float alpha_p = 0.8f,
    float alpha_n = 0.8f,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

// clamp_tss
Tensor clamp_tss(
    const Tensor& input_tensor,
    float min_val,
    float max_val,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

// Binary-like SFPU overloads (Tensor + Tensor) - stubs that throw
// These were binary ops but are referenced through the unary path
Tensor sub_sfpu(
    const Tensor& input_a,
    const Tensor& input_b,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

// float-Tensor overload
Tensor sub_sfpu(
    float lhs,
    const Tensor& rhs,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

// where_tss (ternary op exposed as unary chain with two scalar params)
Tensor where_tss(
    const Tensor& condition,
    float t_true,
    float t_false,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

// ComplexTensor overload for reciprocal (used by complex_binary division)
operations::complex::ComplexTensor reciprocal(
    const operations::complex::ComplexTensor& input, const tt::tt_metal::MemoryConfig& mem_config);

// abs overload for ComplexTensor
Tensor abs(const operations::complex::ComplexTensor& input_tensor, const tt::tt_metal::MemoryConfig& output_mem_config);

// rrelu: Randomized Leaky ReLU
// Eval mode (training=false): f(x) = x if x >= 0, slope * x if x < 0 where slope = (lower+upper)/2
// Train mode (training=true): f(x) = x if x >= 0, a * x if x < 0 where a ~ Uniform(lower, upper)
inline Tensor rrelu(
    const Tensor& input_tensor,
    float lower = 0.125f,
    float upper = 1.0f / 3.0f,
    bool training = false,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    if (training) {
        return ttnn::detail::unary_impl(
            input_tensor,
            {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::RRELU, lower, upper}},
            memory_config,
            optional_output_tensor,
            sub_core_grids);
    } else {
        float slope = (lower + upper) / 2.0f;
        return ttnn::detail::unary_impl(
            input_tensor,
            {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::RRELU, slope}},
            memory_config,
            optional_output_tensor,
            sub_core_grids);
    }
}

// hardtanh: two float parameters with defaults
inline Tensor hardtanh(
    const Tensor& input_tensor,
    float min_val = -1.0f,
    float max_val = 1.0f,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return ttnn::detail::unary_impl(
        input_tensor,
        {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::HARDTANH, min_val, max_val}},
        memory_config,
        optional_output_tensor,
        sub_core_grids);
}

// -----------------------------------------------------------------------------
// Functions defined without macros (non-SFPU operations kept)
// -----------------------------------------------------------------------------

Tensor identity(
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor logit(
    const Tensor& input_tensor,
    std::optional<float> eps = std::nullopt,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor unary_chain(
    const Tensor& input_tensor,
    const std::vector<operations::unary::EltwiseUnaryWithParam>& ops_chain,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor unary_with_int32_param(
    operations::unary::UnaryOpType op_type,
    const Tensor& input_tensor,
    int32_t param,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

}  // namespace ttnn
