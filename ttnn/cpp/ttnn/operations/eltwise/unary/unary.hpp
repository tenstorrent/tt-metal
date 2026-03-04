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

#define UNARY_OP(op_name, op_type)                                                                     \
    inline Tensor op_name(                                                                             \
        const Tensor& t,                                                                               \
        const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,                             \
        const std::optional<Tensor>& o = std::nullopt,                                                 \
        const std::optional<CoreRangeSet>& s = std::nullopt) {                                         \
        return operations::unary::detail::unary_impl(                                                  \
            t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::op_type}}, m, o, s); \
    }

#define UNARY_OP_FAST_APPROX(op_name, op_type)                                                                   \
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

#define UNARY_OP_FLOAT_PARAM(op_name, op_type)                                                                   \
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

UNARY_OP(acos, ACOS)
UNARY_OP(asin, ASIN)
UNARY_OP(asinh, ASINH)
UNARY_OP(atan, ATAN)
UNARY_OP(atanh, ATANH)
UNARY_OP(cos, COS)
UNARY_OP(acosh, ACOSH)
UNARY_OP(cosh, COSH)
UNARY_OP(sinh, SINH)
UNARY_OP(erfinv, ERFINV)
UNARY_OP(exp2, EXP2)
UNARY_OP(expm1, EXPM1)
UNARY_OP(gez, GEZ)
UNARY_OP(gtz, GTZ)
UNARY_OP(i0, I0)
UNARY_OP(i1, I1)
UNARY_OP(isfinite, ISFINITE)
UNARY_OP(isinf, ISINF)
UNARY_OP(isnan, ISNAN)
UNARY_OP(isneginf, ISNEGINF)
UNARY_OP(isposinf, ISPOSINF)
UNARY_OP(lez, LEZ)
UNARY_OP(logical_not, LOGICAL_NOT_UNARY)
UNARY_OP(ltz, LTZ)
UNARY_OP(neg, NEG)
UNARY_OP(nez, NEZ)
UNARY_OP(reciprocal, RECIP)
UNARY_OP(relu, RELU)
UNARY_OP(relu6, RELU6)
UNARY_OP(sign, SIGN)
UNARY_OP(signbit, SIGNBIT)
UNARY_OP(silu, SILU)
UNARY_OP(sin, SIN)
UNARY_OP(square, SQUARE)
UNARY_OP(tan, TAN)
UNARY_OP(tiled_prod, TILED_PROD)
UNARY_OP(bitwise_not, BITWISE_NOT)
UNARY_OP(alt_complex_rotate90, ALT_COMPLEX_ROTATE90)
UNARY_OP(floor, FLOOR)
UNARY_OP(ceil, CEIL)
UNARY_OP(trunc, TRUNC)
UNARY_OP(frac, FRAC)
UNARY_OP(hardsigmoid, HARDSIGMOID)
UNARY_OP(hardswish, HARDSWISH)
UNARY_OP(softsign, SOFTSIGN)
UNARY_OP(cbrt, CBRT)

UNARY_OP_FAST_APPROX(exp, EXP)
UNARY_OP_FAST_APPROX(erf, ERF)
UNARY_OP_FAST_APPROX(erfc, ERFC)
UNARY_OP_FAST_APPROX(gelu, GELU)
UNARY_OP_FAST_APPROX(log, LOG)
UNARY_OP_FAST_APPROX(log10, LOG10)
UNARY_OP_FAST_APPROX(log2, LOG2)
UNARY_OP_FAST_APPROX(log1p, LOG1P)
UNARY_OP_FAST_APPROX(rsqrt, RSQRT)
UNARY_OP_FAST_APPROX(sqrt, SQRT)

UNARY_OP_FLOAT_PARAM(heaviside, HEAVISIDE)
UNARY_OP_FLOAT_PARAM(leaky_relu, LEAKY_RELU)
UNARY_OP_FLOAT_PARAM(relu_max, RELU_MAX)
UNARY_OP_FLOAT_PARAM(relu_min, RELU_MIN)
UNARY_OP_FLOAT_PARAM(unary_remainder, REMAINDER)
UNARY_OP_FLOAT_PARAM(celu, CELU)
UNARY_OP_FLOAT_PARAM(rpow, RPOW)

UNARY_OP_SCALAR_VARIANT(minimum, MINIMUM)
UNARY_OP_SCALAR_VARIANT(maximum, MAXIMUM)
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

inline Tensor xielu(
    const Tensor& input,
    float alpha_p = 0.8f,
    float alpha_n = 0.8f,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return operations::unary::detail::unary_impl(
        input,
        {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::XIELU, {alpha_p, alpha_n}}},
        memory_config,
        optional_output_tensor);
}

operations::unary::ComplexTensor reciprocal(
    const operations::unary::ComplexTensor& t, const tt::tt_metal::MemoryConfig& m);

inline Tensor unary_fmod(
    const Tensor& t,
    float p,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::UnaryWithParam{operations::unary::UnaryOpType::FMOD, static_cast<float>(p)}}, m, o);
}

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

inline Tensor round(
    const Tensor& t,
    const std::optional<int32_t>& p = std::nullopt,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::ROUND, p.value_or(0)}}, m, o);
}

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

inline Tensor power_iterative(
    const Tensor& t,
    uint32_t exponent,
    const std::optional<tt::tt_metal::MemoryConfig>& m = std::nullopt,
    const std::optional<Tensor>& o = std::nullopt) {
    return operations::unary::detail::unary_impl(
        t, {operations::unary::EltwiseUnaryWithParam{operations::unary::UnaryOpType::POWER_ITERATIVE, exponent}}, m, o);
}

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
