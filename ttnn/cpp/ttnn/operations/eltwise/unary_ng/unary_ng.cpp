// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_ng.hpp"
#include "device/unary_ng_device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"

namespace ttnn::operations::unary_ng::detail {

Tensor unary_ng_impl(
    const Tensor& input_tensor,
    const std::vector<unary::EltwiseUnaryWithParam>& op_chain,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    TT_FATAL(!op_chain.empty(), "Op chain cannot be empty");
    DataType input_dtype = input_tensor.dtype();
    DataType output_dtype = input_dtype;
    if (op_chain.back().type() == unary::UnaryOpType::TYPECAST ||
        op_chain.back().type() == unary::UnaryOpType::BITCAST) {
        output_dtype = static_cast<DataType>(*op_chain.back().get_param_if<float>(1));
    }
    bool preserve_fp32_precision = (input_dtype == DataType::FLOAT32);
    bool fp32_dest_acc_en = preserve_fp32_precision || output_dtype == DataType::UINT32 ||
                            output_dtype == DataType::INT32 || output_dtype == DataType::FLOAT32 ||
                            output_dtype == DataType::UINT8 || input_dtype == DataType::UINT8 ||
                            input_dtype == DataType::UINT32 || input_dtype == DataType::INT32;
    bool bfp8_pack_precise =
        (op_chain.back().type() == unary::UnaryOpType::TYPECAST && output_dtype == DataType::BFLOAT8_B);

    auto output_memory_config = optional_output_tensor.has_value()
                                    ? optional_output_tensor.value().memory_config()
                                    : memory_config.value_or(input_tensor.memory_config());

    return prim::unary_ng(
        input_tensor,
        op_chain,
        output_dtype,
        output_memory_config,
        fp32_dest_acc_en,
        preserve_fp32_precision,
        bfp8_pack_precise,
        optional_output_tensor,
        sub_core_grids);
}

}  // namespace ttnn::operations::unary_ng::detail

namespace ttnn {

Tensor abs(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using namespace operations::unary;
    UnaryOpType op_type = UnaryOpType::ABS;
    if (input_tensor.dtype() == DataType::INT32) {
        op_type = UnaryOpType::ABS_INT32;
    }
    return operations::unary_ng::detail::unary_ng_impl(
        input_tensor, {UnaryWithParam{op_type}}, memory_config, optional_output_tensor, sub_core_grids);
}

Tensor abs(const ComplexTensor& input_tensor, const MemoryConfig& output_mem_config) {
    return ttnn::hypot(input_tensor[0], input_tensor[1], output_mem_config);
}

ComplexTensor reciprocal(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    Tensor a_plus_b = ttnn::add(input[0], input[1], std::nullopt, output_mem_config);
    Tensor a_minus_b = ttnn::subtract(input[0], input[1], std::nullopt, output_mem_config);
    Tensor asqr_plus_bsqr = ttnn::add(
        ttnn::square(input[0], output_mem_config),
        ttnn::square(input[1], output_mem_config),
        std::nullopt,
        output_mem_config);
    Tensor inv_dr = ttnn::reciprocal(asqr_plus_bsqr, output_mem_config);
    Tensor conj_im = ttnn::multiply(ttnn::neg(input[1], output_mem_config), inv_dr, std::nullopt, output_mem_config);
    Tensor conj_re = ttnn::multiply(input[0], inv_dr, std::nullopt, output_mem_config);
    return operations::complex::ComplexTensor({conj_re, conj_im});
}

// Helper macro: most basic ops just forward a single UnaryOpType to unary_ng_impl.
#define DEFINE_UNARY_NG_OP(op_name, OP_TYPE)                 \
    Tensor op_name(                                          \
        const Tensor& input_tensor,                          \
        const std::optional<MemoryConfig>& memory_config,    \
        const std::optional<Tensor>& optional_output_tensor, \
        const std::optional<CoreRangeSet>& sub_core_grids) { \
        using namespace operations::unary;                   \
        return operations::unary_ng::detail::unary_ng_impl(  \
            input_tensor,                                    \
            {UnaryWithParam{UnaryOpType::OP_TYPE}},          \
            memory_config,                                   \
            optional_output_tensor,                          \
            sub_core_grids);                                 \
    }

DEFINE_UNARY_NG_OP(neg, NEG)
DEFINE_UNARY_NG_OP(acos, ACOS)
DEFINE_UNARY_NG_OP(asin, ASIN)
DEFINE_UNARY_NG_OP(asinh, ASINH)
DEFINE_UNARY_NG_OP(atan, ATAN)
DEFINE_UNARY_NG_OP(atanh, ATANH)
DEFINE_UNARY_NG_OP(cos, COS)
DEFINE_UNARY_NG_OP(acosh, ACOSH)
DEFINE_UNARY_NG_OP(cosh, COSH)
DEFINE_UNARY_NG_OP(sinh, SINH)
DEFINE_UNARY_NG_OP(erfinv, ERFINV)
DEFINE_UNARY_NG_OP(exp2, EXP2)
DEFINE_UNARY_NG_OP(expm1, EXPM1)
DEFINE_UNARY_NG_OP(gez, GEZ)
DEFINE_UNARY_NG_OP(gtz, GTZ)
DEFINE_UNARY_NG_OP(i0, I0)
DEFINE_UNARY_NG_OP(i1, I1)
DEFINE_UNARY_NG_OP(isfinite, ISFINITE)
DEFINE_UNARY_NG_OP(isinf, ISINF)
DEFINE_UNARY_NG_OP(isnan, ISNAN)
DEFINE_UNARY_NG_OP(isneginf, ISNEGINF)
DEFINE_UNARY_NG_OP(isposinf, ISPOSINF)
DEFINE_UNARY_NG_OP(lez, LEZ)
DEFINE_UNARY_NG_OP(logical_not, LOGICAL_NOT_UNARY)
DEFINE_UNARY_NG_OP(ltz, LTZ)
DEFINE_UNARY_NG_OP(nez, NEZ)
DEFINE_UNARY_NG_OP(reciprocal, RECIP)
DEFINE_UNARY_NG_OP(relu, RELU)
DEFINE_UNARY_NG_OP(relu6, RELU6)
DEFINE_UNARY_NG_OP(sign, SIGN)
DEFINE_UNARY_NG_OP(signbit, SIGNBIT)
DEFINE_UNARY_NG_OP(silu, SILU)
DEFINE_UNARY_NG_OP(sin, SIN)
DEFINE_UNARY_NG_OP(square, SQUARE)
DEFINE_UNARY_NG_OP(tan, TAN)
DEFINE_UNARY_NG_OP(tiled_prod, TILED_PROD)
DEFINE_UNARY_NG_OP(bitwise_not, BITWISE_NOT)
DEFINE_UNARY_NG_OP(alt_complex_rotate90, ALT_COMPLEX_ROTATE90)
DEFINE_UNARY_NG_OP(floor, FLOOR)
DEFINE_UNARY_NG_OP(ceil, CEIL)
DEFINE_UNARY_NG_OP(trunc, TRUNC)
DEFINE_UNARY_NG_OP(frac, FRAC)
DEFINE_UNARY_NG_OP(hardsigmoid, HARDSIGMOID)
DEFINE_UNARY_NG_OP(hardswish, HARDSWISH)
DEFINE_UNARY_NG_OP(softsign, SOFTSIGN)
DEFINE_UNARY_NG_OP(cbrt, CBRT)
DEFINE_UNARY_NG_OP(lgamma, LGAMMA)
DEFINE_UNARY_NG_OP(eqz, EQZ)
DEFINE_UNARY_NG_OP(hardmish, HARDMISH)
DEFINE_UNARY_NG_OP(identity, IDENTITY)
DEFINE_UNARY_NG_OP(log_sigmoid, LOGSIGMOID)
DEFINE_UNARY_NG_OP(swish, SILU)
DEFINE_UNARY_NG_OP(tanhshrink, TANHSHRINK)

#undef DEFINE_UNARY_NG_OP

#define DEFINE_UNARY_NG_OP_WITH_FAST_AND_APPROXIMATE_MODE(op_name, OP_TYPE)                        \
    Tensor op_name(                                                                                \
        const Tensor& input_tensor,                                                                \
        bool fast_and_approximate_mode,                                                            \
        const std::optional<MemoryConfig>& memory_config,                                          \
        const std::optional<Tensor>& optional_output_tensor,                                       \
        const std::optional<CoreRangeSet>& sub_core_grids) {                                       \
        using namespace operations::unary;                                                         \
        return operations::unary_ng::detail::unary_ng_impl(                                        \
            input_tensor,                                                                          \
            {UnaryWithParam{UnaryOpType::OP_TYPE, static_cast<float>(fast_and_approximate_mode)}}, \
            memory_config,                                                                         \
            optional_output_tensor,                                                                \
            sub_core_grids);                                                                       \
    }

DEFINE_UNARY_NG_OP_WITH_FAST_AND_APPROXIMATE_MODE(exp, EXP)
DEFINE_UNARY_NG_OP_WITH_FAST_AND_APPROXIMATE_MODE(erf, ERF)
DEFINE_UNARY_NG_OP_WITH_FAST_AND_APPROXIMATE_MODE(erfc, ERFC)
DEFINE_UNARY_NG_OP_WITH_FAST_AND_APPROXIMATE_MODE(gelu, GELU)
DEFINE_UNARY_NG_OP_WITH_FAST_AND_APPROXIMATE_MODE(log, LOG)
DEFINE_UNARY_NG_OP_WITH_FAST_AND_APPROXIMATE_MODE(log10, LOG10)
DEFINE_UNARY_NG_OP_WITH_FAST_AND_APPROXIMATE_MODE(log2, LOG2)
DEFINE_UNARY_NG_OP_WITH_FAST_AND_APPROXIMATE_MODE(log1p, LOG1P)
DEFINE_UNARY_NG_OP_WITH_FAST_AND_APPROXIMATE_MODE(rsqrt, RSQRT)
DEFINE_UNARY_NG_OP_WITH_FAST_AND_APPROXIMATE_MODE(sqrt, SQRT)
DEFINE_UNARY_NG_OP_WITH_FAST_AND_APPROXIMATE_MODE(mish, MISH)

#undef DEFINE_UNARY_NG_OP_WITH_FAST_AND_APPROXIMATE_MODE

#define DEFINE_UNARY_NG_OP_WITH_FLOAT_PARAM(op_name, OP_TYPE)                      \
    Tensor op_name(                                                                \
        const Tensor& input_tensor,                                                \
        float parameter,                                                           \
        const std::optional<MemoryConfig>& memory_config,                          \
        const std::optional<Tensor>& optional_output_tensor,                       \
        const std::optional<CoreRangeSet>& sub_core_grids) {                       \
        using namespace operations::unary;                                         \
        return operations::unary_ng::detail::unary_ng_impl(                        \
            input_tensor,                                                          \
            {UnaryWithParam{UnaryOpType::OP_TYPE, static_cast<float>(parameter)}}, \
            memory_config,                                                         \
            optional_output_tensor,                                                \
            sub_core_grids);                                                       \
    }

DEFINE_UNARY_NG_OP_WITH_FLOAT_PARAM(heaviside, HEAVISIDE)
DEFINE_UNARY_NG_OP_WITH_FLOAT_PARAM(leaky_relu, LEAKY_RELU)
DEFINE_UNARY_NG_OP_WITH_FLOAT_PARAM(relu_max, RELU_MAX)
DEFINE_UNARY_NG_OP_WITH_FLOAT_PARAM(relu_min, RELU_MIN)
DEFINE_UNARY_NG_OP_WITH_FLOAT_PARAM(unary_remainder, REMAINDER)
DEFINE_UNARY_NG_OP_WITH_FLOAT_PARAM(celu, CELU)
DEFINE_UNARY_NG_OP_WITH_FLOAT_PARAM(rpow, RPOW)
DEFINE_UNARY_NG_OP_WITH_FLOAT_PARAM(unary_fmod, FMOD)
DEFINE_UNARY_NG_OP_WITH_FLOAT_PARAM(prelu_sfpu, PRELU_SFPU)
DEFINE_UNARY_NG_OP_WITH_FLOAT_PARAM(hardshrink, HARDSHRINK)
DEFINE_UNARY_NG_OP_WITH_FLOAT_PARAM(elu, ELU)
DEFINE_UNARY_NG_OP_WITH_FLOAT_PARAM(softshrink, SOFTSHRINK)

#undef DEFINE_UNARY_NG_OP_WITH_FLOAT_PARAM

#define DEFINE_UNARY_NG_OP_WITH_TWO_FLOAT_PARAMS(op_name, OP_TYPE)              \
    Tensor op_name(                                                             \
        const Tensor& input_tensor,                                             \
        float parameter_a,                                                      \
        float parameter_b,                                                      \
        const std::optional<MemoryConfig>& memory_config,                       \
        const std::optional<Tensor>& optional_output_tensor,                    \
        const std::optional<CoreRangeSet>& sub_core_grids) {                    \
        using namespace operations::unary;                                      \
        return operations::unary_ng::detail::unary_ng_impl(                     \
            input_tensor,                                                       \
            {UnaryWithParam{UnaryOpType::OP_TYPE, {parameter_a, parameter_b}}}, \
            memory_config,                                                      \
            optional_output_tensor,                                             \
            sub_core_grids);                                                    \
    }

DEFINE_UNARY_NG_OP_WITH_TWO_FLOAT_PARAMS(threshold, THRESHOLD)
DEFINE_UNARY_NG_OP_WITH_TWO_FLOAT_PARAMS(softplus, SOFTPLUS)
DEFINE_UNARY_NG_OP_WITH_TWO_FLOAT_PARAMS(hardtanh, HARDTANH)
DEFINE_UNARY_NG_OP_WITH_TWO_FLOAT_PARAMS(selu, SELU)

#undef DEFINE_UNARY_NG_OP_WITH_TWO_FLOAT_PARAMS

#define DEFINE_UNARY_NG_OP_SCALAR_VARIANT(op_name, OP_TYPE)                 \
    Tensor op_name(                                                         \
        const Tensor& input_tensor,                                         \
        operations::unary::ScalarVariant parameter,                         \
        const std::optional<MemoryConfig>& memory_config,                   \
        const std::optional<Tensor>& optional_output_tensor,                \
        const std::optional<CoreRangeSet>& sub_core_grids) {                \
        return std::visit(                                                  \
            [&](auto param) {                                               \
                using namespace operations::unary;                          \
                return operations::unary_ng::detail::unary_ng_impl(         \
                    input_tensor,                                           \
                    {EltwiseUnaryWithParam{UnaryOpType::OP_TYPE, (param)}}, \
                    memory_config,                                          \
                    optional_output_tensor,                                 \
                    sub_core_grids);                                        \
            },                                                              \
            parameter);                                                     \
    }

DEFINE_UNARY_NG_OP_SCALAR_VARIANT(fill, FILL)
DEFINE_UNARY_NG_OP_SCALAR_VARIANT(power, POWER)
DEFINE_UNARY_NG_OP_SCALAR_VARIANT(gt_unary, UNARY_GT)
DEFINE_UNARY_NG_OP_SCALAR_VARIANT(lt_unary, UNARY_LT)
DEFINE_UNARY_NG_OP_SCALAR_VARIANT(ne_unary, UNARY_NE)
DEFINE_UNARY_NG_OP_SCALAR_VARIANT(eq_unary, UNARY_EQ)
DEFINE_UNARY_NG_OP_SCALAR_VARIANT(ge_unary, UNARY_GE)
DEFINE_UNARY_NG_OP_SCALAR_VARIANT(le_unary, UNARY_LE)

#undef DEFINE_UNARY_NG_OP_SCALAR_VARIANT

// -----------------------------------------------------------------------------
// Ops with unique parameter signatures
// -----------------------------------------------------------------------------

Tensor xielu(
    const Tensor& input,
    float alpha_p,
    float alpha_n,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using namespace operations::unary;
    return operations::unary_ng::detail::unary_ng_impl(
        input,
        {UnaryWithParam{UnaryOpType::XIELU, {alpha_p, alpha_n}}},
        memory_config,
        optional_output_tensor,
        sub_core_grids);
}

Tensor round(
    const Tensor& input_tensor,
    const std::optional<int32_t>& parameter,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using namespace operations::unary;
    return operations::unary_ng::detail::unary_ng_impl(
        input_tensor,
        {EltwiseUnaryWithParam{UnaryOpType::ROUND, parameter.value_or(0)}},
        memory_config,
        optional_output_tensor,
        sub_core_grids);
}

Tensor logit(
    const Tensor& input_tensor,
    std::optional<float> eps,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using namespace operations::unary;
    return operations::unary_ng::detail::unary_ng_impl(
        input_tensor,
        {UnaryWithParam{UnaryOpType::LOGIT, {eps.value_or(-1.0f)}}},
        memory_config,
        optional_output_tensor,
        sub_core_grids);
}

Tensor deg2rad(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    constexpr float DEG_TO_RAD = 0.017453292519943295f;
    return ttnn::multiply(
        input_tensor,
        DEG_TO_RAD,
        std::optional(input_tensor.dtype()),
        memory_config,
        optional_output_tensor,
        {},
        {},
        {},
        std::nullopt,
        std::nullopt,
        sub_core_grids);
}

Tensor rad2deg(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    constexpr float RAD_TO_DEG = 57.29577951308232f;
    return ttnn::multiply(
        input_tensor,
        RAD_TO_DEG,
        std::optional(input_tensor.dtype()),
        memory_config,
        optional_output_tensor,
        {},
        {},
        {},
        std::nullopt,
        std::nullopt,
        sub_core_grids);
}

Tensor clamp_tss(
    const Tensor& input_tensor,
    float min_val,
    float max_val,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using namespace operations::unary;
    return operations::unary_ng::detail::unary_ng_impl(
        input_tensor,
        {UnaryWithParam{UnaryOpType::CLAMP_TSS, {min_val, max_val}}},
        memory_config,
        optional_output_tensor,
        sub_core_grids);
}

Tensor clamp_tss(
    const Tensor& input_tensor,
    int32_t min_val,
    int32_t max_val,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using namespace operations::unary;
    return operations::unary_ng::detail::unary_ng_impl(
        input_tensor,
        {EltwiseUnaryWithParam{UnaryOpType::CLAMP_TSS, {min_val, max_val}}},
        memory_config,
        optional_output_tensor,
        sub_core_grids);
}

Tensor tanh(
    const Tensor& input,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    bool approx,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using namespace operations::unary;
    return operations::unary_ng::detail::unary_ng_impl(
        input,
        {UnaryWithParam{UnaryOpType::TANH, static_cast<float>(approx)}},
        memory_config,
        optional_output_tensor,
        sub_core_grids);
}

Tensor where_tss(
    const Tensor& condition,
    const operations::unary::ScalarVariant& value_true,
    const operations::unary::ScalarVariant& value_false,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using namespace operations::unary;
    Tensor input = condition;
    bool has_float_scalar = std::holds_alternative<float>(value_true) || std::holds_alternative<float>(value_false);

    if ((condition.dtype() == DataType::INT32 || condition.dtype() == DataType::UINT32) && has_float_scalar) {
        input = ttnn::typecast(condition, DataType::FLOAT32, std::nullopt, std::nullopt, sub_core_grids);
    }
    UnaryOpType op_type = UnaryOpType::WHERE_TSS;
    auto param = std::visit(
        [op_type](const auto& val_true, const auto& val_false) {
            using T = std::decay_t<decltype(val_true)>;
            return EltwiseUnaryWithParam{op_type, std::vector<T>{val_true, val_false}};
        },
        value_true,
        value_false);

    return operations::unary_ng::detail::unary_ng_impl(
        input, {param}, memory_config, optional_output_tensor, sub_core_grids);
}

Tensor bitcast(
    const Tensor& input_tensor,
    const DataType& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using namespace operations::unary;
    if (optional_output_tensor.has_value()) {
        TT_FATAL(
            output_dtype == optional_output_tensor.value().dtype(),
            "If both output dtype and output tensor provided dtype should match");
    }
    EltwiseUnaryWithParam bitcast_op(
        UnaryOpType::BITCAST, {static_cast<float>(input_tensor.dtype()), static_cast<float>(output_dtype)});
    return operations::unary_ng::detail::unary_ng_impl(
        input_tensor, {bitcast_op}, memory_config, optional_output_tensor, sub_core_grids);
}

Tensor rdiv(
    const Tensor& input_tensor,
    float value,
    const std::optional<std::string>& rounding_mode,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using namespace operations::unary;
    TT_FATAL(
        (rounding_mode == std::nullopt || rounding_mode == "trunc" || rounding_mode == "floor"),
        "Incorrect rounding mode (expected None, 'trunc', or 'floor')");
    uint32_t rounding_mode_value = !rounding_mode ? 0 : (*rounding_mode == "trunc" ? 1 : 2);
    return operations::unary_ng::detail::unary_ng_impl(
        input_tensor,
        {UnaryWithParam{UnaryOpType::RDIV, {value, rounding_mode_value}}},
        memory_config,
        optional_output_tensor,
        sub_core_grids);
}

Tensor power_iterative(
    const Tensor& input_tensor,
    uint32_t exponent,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using namespace operations::unary;
    return operations::unary_ng::detail::unary_ng_impl(
        input_tensor,
        {EltwiseUnaryWithParam{UnaryOpType::POWER_ITERATIVE, exponent}},
        memory_config,
        optional_output_tensor,
        sub_core_grids);
}

Tensor sigmoid(
    const Tensor& input,
    int vector_mode,
    operations::unary::SigmoidMode mode,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using namespace operations::unary;
    std::vector<EltwiseUnaryWithParam> op_chain;
    switch (mode) {
        case SigmoidMode::FAST_APPROXIMATE:
            op_chain = {UnaryWithParam(UnaryOpType::SIGMOID, {static_cast<float>(vector_mode), 1.0f})};
            break;
        case SigmoidMode::ACCURATE_FAST_EXP:
            op_chain = {
                UnaryWithParam(UnaryOpType::NEG),
                UnaryWithParam(UnaryOpType::EXP, 1.0f),
                UnaryWithParam(UnaryOpType::ADD_UNARY_SFPU, 1.0f),
                UnaryWithParam(UnaryOpType::RECIP)};
            break;
        case SigmoidMode::ACCURATE: [[fallthrough]];
        default: op_chain = {UnaryWithParam(UnaryOpType::SIGMOID, {static_cast<float>(vector_mode), 0.0f})};
    }
    return operations::unary_ng::detail::unary_ng_impl(
        input, op_chain, memory_config, optional_output_tensor, sub_core_grids);
}

Tensor sigmoid_accurate(
    const Tensor& input,
    bool fast_and_approximate_mode,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using namespace operations::unary;
    auto op_chain =
        fast_and_approximate_mode
            ? std::vector<
                  EltwiseUnaryWithParam>{UnaryWithParam(UnaryOpType::NEG), UnaryWithParam(UnaryOpType::EXP, 1.0f), UnaryWithParam(UnaryOpType::ADD_UNARY_SFPU, 1.0f), UnaryWithParam(UnaryOpType::RECIP)}
            : std::vector<EltwiseUnaryWithParam>{
                  UnaryWithParam(UnaryOpType::SIGMOID, {static_cast<float>(VecMode::RC), 0.0f})};
    return operations::unary_ng::detail::unary_ng_impl(
        input, op_chain, memory_config, optional_output_tensor, sub_core_grids);
}

Tensor unary_chain(
    const Tensor& input_tensor,
    const std::vector<operations::unary::EltwiseUnaryWithParam>& ops_chain,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return operations::unary_ng::detail::unary_ng_impl(
        input_tensor, ops_chain, memory_config, optional_output_tensor, sub_core_grids);
}

template <typename T>
Tensor rsub_sfpu(
    const Tensor& input_tensor,
    T param,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using namespace operations::unary;
    return operations::unary_ng::detail::unary_ng_impl(
        input_tensor,
        {EltwiseUnaryWithParam{UnaryOpType::RSUB, {param}}},
        memory_config,
        optional_output_tensor,
        sub_core_grids);
}

template Tensor ttnn::rsub_sfpu<float>(
    const Tensor&,
    float,
    const std::optional<MemoryConfig>&,
    const std::optional<Tensor>&,
    const std::optional<CoreRangeSet>&);

template Tensor ttnn::rsub_sfpu<int>(
    const Tensor&,
    int,
    const std::optional<MemoryConfig>&,
    const std::optional<Tensor>&,
    const std::optional<CoreRangeSet>&);

Tensor add_sfpu(
    const Tensor& input_tensor,
    float param,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using namespace operations::unary;
    return operations::unary_ng::detail::unary_ng_impl(
        input_tensor,
        {EltwiseUnaryWithParam(UnaryOpType::ADD_UNARY_SFPU, param)},
        memory_config,
        optional_output_tensor,
        sub_core_grids);
}

Tensor add_sfpu(
    float param,
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using namespace operations::unary;
    return operations::unary_ng::detail::unary_ng_impl(
        input_tensor,
        {EltwiseUnaryWithParam(UnaryOpType::ADD_UNARY_SFPU, param)},
        memory_config,
        optional_output_tensor,
        sub_core_grids);
}

Tensor mul_sfpu(
    const Tensor& input_tensor,
    float param,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using namespace operations::unary;
    return operations::unary_ng::detail::unary_ng_impl(
        input_tensor,
        {EltwiseUnaryWithParam(UnaryOpType::MUL_UNARY_SFPU, param)},
        memory_config,
        optional_output_tensor,
        sub_core_grids);
}

Tensor mul_sfpu(
    float param,
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using namespace operations::unary;
    return operations::unary_ng::detail::unary_ng_impl(
        input_tensor,
        {EltwiseUnaryWithParam(UnaryOpType::MUL_UNARY_SFPU, param)},
        memory_config,
        optional_output_tensor,
        sub_core_grids);
}

Tensor sub_sfpu(
    const Tensor& input_tensor,
    float param,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using namespace operations::unary;
    return operations::unary_ng::detail::unary_ng_impl(
        input_tensor,
        {EltwiseUnaryWithParam{UnaryOpType::SUB_UNARY_SFPU, param}},
        memory_config,
        optional_output_tensor,
        sub_core_grids);
}

Tensor sub_sfpu(
    float param,
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using namespace operations::unary;
    return operations::unary_ng::detail::unary_ng_impl(
        input_tensor,
        {EltwiseUnaryWithParam{UnaryOpType::RSUB, param}},
        memory_config,
        optional_output_tensor,
        sub_core_grids);
}

Tensor div_sfpu(
    const Tensor& input_tensor,
    float param,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using namespace operations::unary;
    return operations::unary_ng::detail::unary_ng_impl(
        input_tensor,
        {EltwiseUnaryWithParam{UnaryOpType::DIV_UNARY_SFPU, param}},
        memory_config,
        optional_output_tensor,
        sub_core_grids);
}

Tensor div_sfpu(
    float param,
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using namespace operations::unary;
    return operations::unary_ng::detail::unary_ng_impl(
        input_tensor,
        {EltwiseUnaryWithParam{UnaryOpType::RDIV, param}},
        memory_config,
        optional_output_tensor,
        sub_core_grids);
}

Tensor unary_with_int32_param(
    operations::unary::UnaryOpType op_type,
    const Tensor& input_tensor,
    int32_t param,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using namespace operations::unary;
    return operations::unary_ng::detail::unary_ng_impl(
        input_tensor, {EltwiseUnaryWithParam{op_type, param}}, memory_config, optional_output_tensor, sub_core_grids);
}

}  // namespace ttnn
