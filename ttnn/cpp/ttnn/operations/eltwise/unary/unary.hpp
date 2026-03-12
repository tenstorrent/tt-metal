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

template <UnaryOpType unary_op_type>
struct ExecuteUnaryTSVariant {
    static Tensor invoke(
        const Tensor& input_tensor,
        ScalarVariant parameter,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

template <UnaryOpType unary_op_type>
struct ExecuteUnaryWithFloatParameter {
    static Tensor invoke(
        const Tensor& input_tensor,
        float parameter,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
};

}  // namespace operations::unary

#define REGISTER_UNARY_OPERATION(op_name, op_type)                                                     \
    inline Tensor op_name(                                                                             \
        const Tensor& t,                                                                               \
        const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,                             \
        const std::optional<Tensor>& o = std::nullopt,                                                 \
        const std::optional<CoreRangeSet>& s = std::nullopt) {                                         \
        return operations::unary::detail::unary_impl(                                                  \
            t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::op_type}}, m, o, s); \
    }

#define REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(op_name, op_type)                                \
    inline Tensor op_name(                                                                                       \
        const Tensor& t,                                                                                         \
        bool p = false,                                                                                          \
        const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,                                       \
        const std::optional<Tensor>& o = std::nullopt,                                                           \
        const std::optional<CoreRangeSet>& s = std::nullopt) {                                                   \
        return operations::unary::detail::unary_impl(                                                            \
            t,                                                                                                   \
            {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::op_type, static_cast<float>(p)}}, \
            m,                                                                                                   \
            o,                                                                                                   \
            s);                                                                                                  \
    }

#define REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(op_name, op_type)                                          \
    inline Tensor op_name(                                                                                       \
        const Tensor& t,                                                                                         \
        float p,                                                                                                 \
        const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,                                       \
        const std::optional<Tensor>& o = std::nullopt,                                                           \
        const std::optional<CoreRangeSet>& s = std::nullopt) {                                                   \
        return operations::unary::detail::unary_impl(                                                            \
            t,                                                                                                   \
            {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::op_type, static_cast<float>(p)}}, \
            m,                                                                                                   \
            o,                                                                                                   \
            s);                                                                                                  \
    }

#define UNARY_OP_SCALAR_VARIANT(op_name, op_type)                                                                 \
    inline Tensor op_name(                                                                                        \
        const Tensor& t,                                                                                          \
        operations::unary::ScalarVariant p,                                                                       \
        const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,                                        \
        const std::optional<Tensor>& o = std::nullopt) {                                                          \
        return std::visit(                                                                                        \
            [&](auto param) {                                                                                     \
                return operations::unary::detail::unary_impl(                                                     \
                    t,                                                                                            \
                    {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::op_type, (param)}}, \
                    m,                                                                                            \
                    o);                                                                                           \
            },                                                                                                    \
            p);                                                                                                   \
    }

// -----------------------------------------------------------------------------
// Functions defined with macros
// -----------------------------------------------------------------------------

REGISTER_UNARY_OPERATION(acos, ACOS)
REGISTER_UNARY_OPERATION(asin, ASIN)
REGISTER_UNARY_OPERATION(asinh, ASINH)
REGISTER_UNARY_OPERATION(atan, ATAN)
REGISTER_UNARY_OPERATION(atanh, ATANH)
REGISTER_UNARY_OPERATION(cos, COS)
REGISTER_UNARY_OPERATION(acosh, ACOSH)
REGISTER_UNARY_OPERATION(cosh, COSH)
REGISTER_UNARY_OPERATION(sinh, SINH)
REGISTER_UNARY_OPERATION(erfinv, ERFINV)
REGISTER_UNARY_OPERATION(exp2, EXP2)
REGISTER_UNARY_OPERATION(expm1, EXPM1)
REGISTER_UNARY_OPERATION(gez, GEZ)
REGISTER_UNARY_OPERATION(gtz, GTZ)
REGISTER_UNARY_OPERATION(i0, I0)
REGISTER_UNARY_OPERATION(i1, I1)
REGISTER_UNARY_OPERATION(isfinite, ISFINITE)
REGISTER_UNARY_OPERATION(isinf, ISINF)
REGISTER_UNARY_OPERATION(isnan, ISNAN)
REGISTER_UNARY_OPERATION(isneginf, ISNEGINF)
REGISTER_UNARY_OPERATION(isposinf, ISPOSINF)
REGISTER_UNARY_OPERATION(lez, LEZ)
REGISTER_UNARY_OPERATION(logical_not, LOGICAL_NOT_UNARY)
REGISTER_UNARY_OPERATION(ltz, LTZ)
REGISTER_UNARY_OPERATION(neg, NEG)
REGISTER_UNARY_OPERATION(nez, NEZ)
REGISTER_UNARY_OPERATION(reciprocal, RECIP)
REGISTER_UNARY_OPERATION(relu, RELU)
REGISTER_UNARY_OPERATION(relu6, RELU6)
REGISTER_UNARY_OPERATION(sign, SIGN)
REGISTER_UNARY_OPERATION(signbit, SIGNBIT)
REGISTER_UNARY_OPERATION(silu, SILU)
REGISTER_UNARY_OPERATION(sin, SIN)
REGISTER_UNARY_OPERATION(square, SQUARE)
REGISTER_UNARY_OPERATION(tan, TAN)
REGISTER_UNARY_OPERATION(tiled_prod, TILED_PROD)
REGISTER_UNARY_OPERATION(bitwise_not, BITWISE_NOT)
REGISTER_UNARY_OPERATION(alt_complex_rotate90, ALT_COMPLEX_ROTATE90)
REGISTER_UNARY_OPERATION(floor, FLOOR)
REGISTER_UNARY_OPERATION(ceil, CEIL)
REGISTER_UNARY_OPERATION(trunc, TRUNC)
REGISTER_UNARY_OPERATION(frac, FRAC)
REGISTER_UNARY_OPERATION(hardsigmoid, HARDSIGMOID)
REGISTER_UNARY_OPERATION(hardswish, HARDSWISH)
REGISTER_UNARY_OPERATION(softsign, SOFTSIGN)
REGISTER_UNARY_OPERATION(cbrt, CBRT)

// Unaries with fast_and_approximate_mode
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(exp, EXP)
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(erf, ERF)
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(erfc, ERFC)
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(gelu, GELU)
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(log, LOG)
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(log10, LOG10)
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(log2, LOG2)
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(log1p, LOG1P)
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(rsqrt, RSQRT)
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(sqrt, SQRT)
REGISTER_UNARY_OPERATION_WITH_FAST_AND_APPROXIMATE_MODE(mish, MISH)

// Unaries with float parameter
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(heaviside, HEAVISIDE)
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(leaky_relu, LEAKY_RELU)
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(relu_max, RELU_MAX)
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(relu_min, RELU_MIN)
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(unary_remainder, REMAINDER)
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(celu, CELU)
REGISTER_UNARY_OPERATION_WITH_FLOAT_PARAMETER(rpow, RPOW)

UNARY_OP_SCALAR_VARIANT(fill, FILL)
UNARY_OP_SCALAR_VARIANT(power, POWER)
UNARY_OP_SCALAR_VARIANT(gt_unary, UNARY_GT)
UNARY_OP_SCALAR_VARIANT(lt_unary, UNARY_LT)
UNARY_OP_SCALAR_VARIANT(ne_unary, UNARY_NE)
UNARY_OP_SCALAR_VARIANT(eq_unary, UNARY_EQ)
UNARY_OP_SCALAR_VARIANT(ge_unary, UNARY_GE)
UNARY_OP_SCALAR_VARIANT(le_unary, UNARY_LE)

// -----------------------------------------------------------------------------
// Functions defined without macros
// -----------------------------------------------------------------------------

Tensor xielu(
    const Tensor& input,
    float alpha_p = 0.8f,
    float alpha_n = 0.8f,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

operations::unary::ComplexTensor reciprocal(
    const operations::unary::ComplexTensor& input, const tt::tt_metal::MemoryConfig& output_mem_config);

Tensor unary_fmod(
    const Tensor& t,
    float p,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt);

Tensor threshold(
    const Tensor& t,
    float pa,
    float pb,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt);

Tensor round(
    const Tensor& t,
    const std::optional<int32_t>& p = std::nullopt,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt);

Tensor identity(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt);

Tensor abs(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt);

Tensor abs(const operations::unary::ComplexTensor& input_tensor, const tt::tt_metal::MemoryConfig& output_mem_config);

Tensor eqz(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt);

Tensor hardmish(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt);

Tensor hardshrink(
    const Tensor& t,
    float lambda = 0.5f,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt);

Tensor logit(
    const Tensor& t,
    std::optional<float> eps = std::nullopt,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt);

Tensor elu(
    const Tensor& t,
    float alpha = 1.0f,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt);

Tensor hardtanh(
    const Tensor& t,
    float min_val = -1.0f,
    float max_val = 1.0f,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt);

Tensor softshrink(
    const Tensor& t,
    float lambda = 0.5f,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt);

Tensor deg2rad(
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor rad2deg(
    const Tensor& input_tensor,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor clamp_tss(
    const Tensor& t,
    float min_val,
    float max_val,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt);

Tensor clamp_tss(
    const Tensor& t,
    int32_t min_val,
    int32_t max_val,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt);

Tensor softplus(
    const Tensor& t,
    float beta,
    float threshold,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt);

Tensor tanh(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    bool approx = false);

Tensor tanhshrink(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt,
    bool /*approx*/ = false);

Tensor prelu_sfpu(
    const Tensor& t,
    float value,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt);

Tensor where_tss(
    const Tensor& condition,
    const operations::unary::ScalarVariant& value_true,
    const operations::unary::ScalarVariant& value_false,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor selu(
    const Tensor& t,
    float scale = 1.050700987f,
    float alpha = 1.673263242f,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt);

Tensor bitcast(
    const Tensor& input_tensor,
    const tt::tt_metal::DataType& output_dtype,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor rdiv(
    const Tensor& input_tensor,
    float value,
    const std::optional<std::string>& rounding_mode = std::nullopt,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor swish(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt);

Tensor power_iterative(
    const Tensor& t,
    uint32_t exponent,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt);

Tensor sigmoid(
    const Tensor& t,
    int vector_mode = (int32_t)operations::unary::VecMode::RC,
    operations::unary::SigmoidMode mode = operations::unary::SigmoidMode::ACCURATE,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt);

Tensor sigmoid_accurate(
    const Tensor& t,
    bool fast_and_approximate_mode = false,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt);

Tensor log_sigmoid(
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt);

Tensor unary_chain(
    const Tensor& t,
    const std::vector<operations::unary::EltwiseUnaryWithParam>& chain,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt);

template <typename T>
inline Tensor rsub_sfpu(
    const Tensor& t,
    T param,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::RSUB, {param}}}, m, o);
}

Tensor add_sfpu(
    const Tensor& t,
    float p,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt);

Tensor add_sfpu(
    float p,
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt);

Tensor mul_sfpu(
    const Tensor& t,
    float p,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt);

Tensor mul_sfpu(
    float p,
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt);

Tensor sub_sfpu(
    const Tensor& t,
    float p,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt);

Tensor sub_sfpu(
    float p,
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt);

Tensor div_sfpu(
    const Tensor& t,
    float p,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt);

Tensor div_sfpu(
    float p,
    const Tensor& t,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt);

Tensor unary_with_int32_param(
    operations::unary::UnaryOpType op_type,
    const Tensor& t,
    int32_t param,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt);

}  // namespace ttnn
