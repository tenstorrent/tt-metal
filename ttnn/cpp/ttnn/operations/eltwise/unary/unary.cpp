// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "unary.hpp"

#include "common/unary_op_types.hpp"
#include "device/unary_device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/ternary/ternary.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"

namespace ttnn::operations::unary {

namespace detail {

inline Tensor unary_impl(
    const Tensor& input_tensor,
    const std::vector<EltwiseUnaryWithParam>& op_chain,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    TT_FATAL(!op_chain.empty(), "Op chain cannot be empty");
    DataType input_dtype = input_tensor.dtype();
    DataType output_dtype = (op_chain[0].type() == UnaryOpType::TYPECAST || op_chain[0].type() == UnaryOpType::BITCAST)
                                ? static_cast<DataType>(*op_chain[0].get_param_if<float>(1))
                                : input_dtype;
    bool preserve_fp32_precision = input_dtype == DataType::FLOAT32;
    bool fp32_dest_acc_en = preserve_fp32_precision or output_dtype == DataType::UINT32 or
                            output_dtype == DataType::INT32 or output_dtype == DataType::FLOAT32 or
                            output_dtype == DataType::UINT8 or input_dtype == DataType::UINT8 or
                            input_dtype == DataType::UINT32 or input_dtype == DataType::INT32;
    bool bfp8_pack_precise = (op_chain[0].type() == UnaryOpType::TYPECAST && output_dtype == DataType::BFLOAT8_B);

    auto output_memory_config = optional_output_tensor.has_value()
                                    ? optional_output_tensor.value().memory_config()
                                    : memory_config.value_or(input_tensor.memory_config());
    return prim::unary(
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

}  // namespace detail

template <UnaryOpType... unary_op_types>
Tensor ExecuteUnary<unary_op_types...>::invoke(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return detail::unary_impl(
        input_tensor, {UnaryWithParam{unary_op_types}...}, memory_config, optional_output_tensor, sub_core_grids);
}

template <>
ComplexTensor ExecuteUnary<UnaryOpType::RECIP>::invoke(
    const ComplexTensor& input, const MemoryConfig& output_mem_config) {
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
    return ComplexTensor({conj_re, conj_im});
}
template struct ExecuteUnary<UnaryOpType::ACOS>;
template struct ExecuteUnary<UnaryOpType::ASIN>;
template struct ExecuteUnary<UnaryOpType::ASINH>;
template struct ExecuteUnary<UnaryOpType::ATAN>;
template struct ExecuteUnary<UnaryOpType::ATANH>;
template struct ExecuteUnary<UnaryOpType::COS>;
template struct ExecuteUnary<UnaryOpType::ACOSH>;
template struct ExecuteUnary<UnaryOpType::COSH>;
template struct ExecuteUnary<UnaryOpType::SINH>;
template struct ExecuteUnary<UnaryOpType::ERFINV>;
template struct ExecuteUnary<UnaryOpType::EXP2>;
template struct ExecuteUnary<UnaryOpType::EXPM1>;
template struct ExecuteUnary<UnaryOpType::GEZ>;
template struct ExecuteUnary<UnaryOpType::GTZ>;
template struct ExecuteUnary<UnaryOpType::I0>;
template struct ExecuteUnary<UnaryOpType::I1>;
template struct ExecuteUnary<UnaryOpType::ISFINITE>;
template struct ExecuteUnary<UnaryOpType::ISINF>;
template struct ExecuteUnary<UnaryOpType::ISNAN>;
template struct ExecuteUnary<UnaryOpType::ISNEGINF>;
template struct ExecuteUnary<UnaryOpType::ISPOSINF>;
template struct ExecuteUnary<UnaryOpType::LEZ>;
template struct ExecuteUnary<UnaryOpType::LOGICAL_NOT_UNARY>;
template struct ExecuteUnary<UnaryOpType::LTZ>;
template struct ExecuteUnary<UnaryOpType::NEG>;
template struct ExecuteUnary<UnaryOpType::NEZ>;
template struct ExecuteUnary<UnaryOpType::RECIP>;
template struct ExecuteUnary<UnaryOpType::RELU>;
template struct ExecuteUnary<UnaryOpType::RELU6>;
template struct ExecuteUnary<UnaryOpType::SIGN>;
template struct ExecuteUnary<UnaryOpType::SIGNBIT>;
template struct ExecuteUnary<UnaryOpType::SILU>;
template struct ExecuteUnary<UnaryOpType::SIN>;
template struct ExecuteUnary<UnaryOpType::SQUARE>;
template struct ExecuteUnary<UnaryOpType::TAN>;
template struct ExecuteUnary<UnaryOpType::TILED_PROD>;
template struct ExecuteUnary<UnaryOpType::BITWISE_NOT>;
template struct ExecuteUnary<UnaryOpType::ALT_COMPLEX_ROTATE90>;
template struct ExecuteUnary<UnaryOpType::CEIL>;
template struct ExecuteUnary<UnaryOpType::FLOOR>;
template struct ExecuteUnary<UnaryOpType::TRUNC>;
template struct ExecuteUnary<UnaryOpType::FRAC>;
template struct ExecuteUnary<UnaryOpType::HARDSIGMOID>;
template struct ExecuteUnary<UnaryOpType::HARDSWISH>;
template struct ExecuteUnary<UnaryOpType::SOFTSIGN>;
template struct ExecuteUnary<UnaryOpType::CBRT>;

template <UnaryOpType unary_op_type>
Tensor ExecuteUnaryWithFastAndApproximateMode<unary_op_type>::invoke(
    const Tensor& input_tensor,
    const bool parameter,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return detail::unary_impl(
        input_tensor,
        {UnaryWithParam{unary_op_type, static_cast<float>(parameter)}},
        memory_config,
        optional_output_tensor,
        sub_core_grids);
}

template struct ExecuteUnaryWithFastAndApproximateMode<UnaryOpType::EXP>;
template struct ExecuteUnaryWithFastAndApproximateMode<UnaryOpType::ERF>;
template struct ExecuteUnaryWithFastAndApproximateMode<UnaryOpType::ERFC>;
template struct ExecuteUnaryWithFastAndApproximateMode<UnaryOpType::GELU>;
template struct ExecuteUnaryWithFastAndApproximateMode<UnaryOpType::LOG>;
template struct ExecuteUnaryWithFastAndApproximateMode<UnaryOpType::LOG10>;
template struct ExecuteUnaryWithFastAndApproximateMode<UnaryOpType::LOG2>;
template struct ExecuteUnaryWithFastAndApproximateMode<UnaryOpType::LOG1P>;
template struct ExecuteUnaryWithFastAndApproximateMode<UnaryOpType::RSQRT>;
template struct ExecuteUnaryWithFastAndApproximateMode<UnaryOpType::SQRT>;

template <UnaryOpType unary_op_type>
Tensor ExecuteUnaryWithVectorAndFastAndApproximateMode<unary_op_type>::invoke(
    const Tensor& input_tensor,
    const int mode,
    const bool parameter,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return detail::unary_impl(
        input_tensor,
        {UnaryWithParam{unary_op_type, {static_cast<float>(mode), static_cast<float>(parameter)}}},
        memory_config,
        optional_output_tensor);
}

template struct ExecuteUnaryWithVectorAndFastAndApproximateMode<UnaryOpType::SIGMOID>;

template <UnaryOpType unary_op_type>
Tensor ExecuteUnaryWithFloatParameter<unary_op_type>::invoke(
    const Tensor& input_tensor,
    const float parameter,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return detail::unary_impl(
        input_tensor,
        {UnaryWithParam{unary_op_type, static_cast<float>(parameter)}},
        memory_config,
        optional_output_tensor,
        sub_core_grids);
}

template <UnaryOpType unary_op_type>
Tensor ExecuteUnaryWithTwoFloatParameter<unary_op_type>::invoke(
    const Tensor& input_tensor,
    const float parameter_a,
    const float parameter_b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return detail::unary_impl(
        input_tensor,
        {UnaryWithParam{unary_op_type, {static_cast<float>(parameter_a), static_cast<float>(parameter_b)}}},
        memory_config,
        optional_output_tensor);
}

template <UnaryOpType unary_op_type>
Tensor ExecuteUnaryTSVariant<unary_op_type>::invoke(
    const Tensor& input_tensor,
    ScalarVariant parameter,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return std::visit(
        [&](auto param) {
            return detail::unary_impl(
                input_tensor, {EltwiseUnaryWithParam{unary_op_type, (param)}}, memory_config, optional_output_tensor);
        },
        parameter);
}

template struct ExecuteUnaryWithFloatParameter<UnaryOpType::ELU>;
template struct ExecuteUnaryWithFloatParameter<UnaryOpType::RSUB>;
template struct ExecuteUnaryWithFloatParameter<UnaryOpType::HEAVISIDE>;
template struct ExecuteUnaryWithFloatParameter<UnaryOpType::LEAKY_RELU>;
template struct ExecuteUnaryWithFloatParameter<UnaryOpType::RELU_MAX>;
template struct ExecuteUnaryWithFloatParameter<UnaryOpType::RELU_MIN>;
template struct ExecuteUnaryWithFloatParameter<UnaryOpType::REMAINDER>;
template struct ExecuteUnaryWithFloatParameter<UnaryOpType::FMOD>;
template struct ExecuteUnaryWithFloatParameter<UnaryOpType::CELU>;
template struct ExecuteUnaryWithFloatParameter<UnaryOpType::RPOW>;

// threshold(a,t,v) = (a <= t ? v : a)
template struct ExecuteUnaryWithTwoFloatParameter<UnaryOpType::THRESHOLD>;

template struct ExecuteUnaryTSVariant<UnaryOpType::MINIMUM>;
template struct ExecuteUnaryTSVariant<UnaryOpType::MAXIMUM>;
template struct ExecuteUnaryTSVariant<UnaryOpType::FILL>;

template struct ExecuteUnaryTSVariant<UnaryOpType::UNARY_EQ>;
template struct ExecuteUnaryTSVariant<UnaryOpType::UNARY_GE>;
template struct ExecuteUnaryTSVariant<UnaryOpType::UNARY_LE>;
template struct ExecuteUnaryTSVariant<UnaryOpType::UNARY_LT>;
template struct ExecuteUnaryTSVariant<UnaryOpType::UNARY_NE>;
template struct ExecuteUnaryTSVariant<UnaryOpType::UNARY_GT>;

Tensor Sigmoid_accurate::invoke(
    const Tensor& input,
    bool fast_and_approximate_mode,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return detail::unary_impl(
        input,
        {UnaryWithParam(UnaryOpType::NEG),
         UnaryWithParam(UnaryOpType::EXP, fast_and_approximate_mode ? 1.0f : 0.0f),
         UnaryWithParam(UnaryOpType::ADD_UNARY_SFPU, 1.0f),
         UnaryWithParam(UnaryOpType::RECIP)},
        memory_config,
        optional_output_tensor);
}

template struct ExecuteUnary<UnaryOpType::SIGMOID, UnaryOpType::LOG>;

Tensor Eqz::invoke(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    UnaryOpType op_type = UnaryOpType::EQZ;
    return detail::unary_impl(input_tensor, {UnaryWithParam{op_type}}, memory_config, optional_output_tensor);
}

Tensor Unary_chain::invoke(
    const Tensor& input_tensor,
    const std::vector<EltwiseUnaryWithParam>& ops_chain,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    TT_FATAL(!ops_chain.empty(), "Op chain cannot be empty");
    return detail::unary_impl(input_tensor, ops_chain, memory_config, optional_output_tensor);
}

Tensor Bitcast::invoke(
    const Tensor& input_tensor,
    const DataType& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    if (optional_output_tensor.has_value()) {
        TT_FATAL(
            output_dtype == optional_output_tensor.value().dtype(),
            "If both output dtype and output tensor provided dtype should match");
    }
    // Use unary infrastructure with BITCAST op type
    // BITCAST uses identity kernel (copy_tile + pack_tile) with output format for both CBs
    EltwiseUnaryWithParam bitcast_op(
        UnaryOpType::BITCAST, {static_cast<float>(input_tensor.dtype()), static_cast<float>(output_dtype)});
    return Unary_chain::invoke(input_tensor, {bitcast_op}, memory_config, optional_output_tensor);
}

Tensor Selu::invoke(
    const Tensor& input_tensor,
    float scale,
    float alpha,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    UnaryOpType op_type = UnaryOpType::SELU;

    return detail::unary_impl(
        input_tensor, {UnaryWithParam{op_type, {scale, alpha}}}, memory_config, optional_output_tensor);
}

Tensor Softplus::invoke(
    const Tensor& input,
    const float beta,
    const float threshold,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return detail::unary_impl(
        input, {UnaryWithParam{UnaryOpType::SOFTPLUS, {beta, threshold}}}, memory_config, optional_output_tensor);
}

// tanh[x] = (exp[2x] - 1) / (exp[2x] + 1)
Tensor Tanh::invoke(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    bool approx) {
    UnaryOpType op_type = UnaryOpType::TANH;

    return detail::unary_impl(
        input_tensor, {UnaryWithParam{op_type, static_cast<float>(approx)}}, memory_config, optional_output_tensor);
}

Tensor Prelu::invoke(
    const Tensor& input,
    float value,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return detail::unary_impl(
        input, {UnaryWithParam{UnaryOpType::PRELU_SFPU, value}}, memory_config, optional_output_tensor);
}

Tensor Identity::invoke(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    UnaryOpType op_type = UnaryOpType::IDENTITY;

    return detail::unary_impl(input_tensor, {UnaryWithParam{op_type}}, memory_config, optional_output_tensor);
}

Tensor Abs::invoke(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    UnaryOpType op_type = UnaryOpType::ABS;
    if (input_tensor.dtype() == DataType::INT32) {
        op_type = UnaryOpType::ABS_INT32;
    }
    return detail::unary_impl(input_tensor, {UnaryWithParam{op_type}}, memory_config, optional_output_tensor);
}

Tensor Abs::invoke(const ComplexTensor& input_tensor, const MemoryConfig& output_mem_config) {
    return ttnn::hypot(input_tensor[0], input_tensor[1], output_mem_config);
}

Tensor Mish::invoke(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    UnaryOpType op_type = UnaryOpType::MISH;

    return detail::unary_impl(input_tensor, {UnaryWithParam{op_type}}, memory_config, optional_output_tensor);
}

Tensor LogSigmoid::invoke(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    UnaryOpType op_type = UnaryOpType::LOGSIGMOID;

    return detail::unary_impl(input_tensor, {UnaryWithParam{op_type}}, memory_config, optional_output_tensor);
}

Tensor Hardmish::invoke(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    UnaryOpType op_type = UnaryOpType::HARDMISH;

    return detail::unary_impl(input_tensor, {UnaryWithParam{op_type}}, memory_config, optional_output_tensor);
}

Tensor Tanhshrink::invoke(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    bool /*approx*/) {
    UnaryOpType op_type = UnaryOpType::TANHSHRINK;
    return detail::unary_impl(input_tensor, {UnaryWithParam{op_type}}, memory_config, optional_output_tensor);
}

Tensor Hardshrink::invoke(
    const Tensor& input_tensor,
    const float lambda,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    UnaryOpType op_type = UnaryOpType::HARDSHRINK;
    TT_ASSERT(lambda >= 0);
    return detail::unary_impl(
        input_tensor, {UnaryWithParam{op_type, static_cast<float>(lambda)}}, memory_config, optional_output_tensor);
}

Tensor Elu::invoke(
    const Tensor& input_tensor,
    const float alpha,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    UnaryOpType op_type = UnaryOpType::ELU;
    return detail::unary_impl(
        input_tensor, {UnaryWithParam{op_type, static_cast<float>(alpha)}}, memory_config, optional_output_tensor);
}

Tensor Hardtanh::invoke(
    const Tensor& input_tensor,
    const float min_val,
    const float max_val,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    UnaryOpType op_type = UnaryOpType::HARDTANH;
    return detail::unary_impl(
        input_tensor, {UnaryWithParam{op_type, {min_val, max_val}}}, memory_config, optional_output_tensor);
}

Tensor Clamp::invoke(
    const Tensor& input_tensor,
    const float min_val,
    const float max_val,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    UnaryOpType op_type = UnaryOpType::CLAMP_TSS;
    return detail::unary_impl(
        input_tensor, {UnaryWithParam{op_type, {min_val, max_val}}}, memory_config, optional_output_tensor);
}

Tensor Clamp::invoke(
    const Tensor& input_tensor,
    const int32_t min_val,
    const int32_t max_val,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    UnaryOpType op_type = UnaryOpType::CLAMP_TSS;
    return detail::unary_impl(
        input_tensor,
        {EltwiseUnaryWithParam{op_type, {min_val, max_val}}},
        memory_config,
        optional_output_tensor);
}

Tensor Rdiv::invoke(
    const Tensor& input_tensor,
    float value,
    const std::optional<std::string>& rounding_mode,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    TT_FATAL(
        (rounding_mode == std::nullopt || rounding_mode == "trunc" || rounding_mode == "floor"),
        "Incorrect rounding mode (expected None, 'trunc', or 'floor')");
    UnaryOpType op_type = UnaryOpType::RDIV;
    // Convert rounding_mode to numeric value: 0 = none, 1 = trunc, 2 = floor
    uint32_t rounding_mode_value = !rounding_mode ? 0 : (*rounding_mode == "trunc" ? 1 : 2);
    return detail::unary_impl(
        input_tensor, {UnaryWithParam{op_type, {value, rounding_mode_value}}}, memory_config, optional_output_tensor);
}

template <typename T>
Tensor Rsub::invoke(
    const Tensor& input_tensor,
    T param,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    UnaryOpType op_type = UnaryOpType::RSUB;
    return detail::unary_impl(
        input_tensor, {EltwiseUnaryWithParam{op_type, {param}}}, memory_config, optional_output_tensor);
}
template Tensor Rsub::invoke<float>(
    const Tensor&, float, const std::optional<MemoryConfig>&, const std::optional<Tensor>&);
template Tensor Rsub::invoke<int32_t>(
    const Tensor&, int32_t, const std::optional<MemoryConfig>&, const std::optional<Tensor>&);

Tensor Softshrink::invoke(
    const Tensor& input_tensor,
    const float lambda,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    UnaryOpType op_type = UnaryOpType::SOFTSHRINK;
    TT_ASSERT(lambda >= 0);
    return detail::unary_impl(
        input_tensor, {UnaryWithParam{op_type, static_cast<float>(lambda)}}, memory_config, optional_output_tensor);
}

Tensor Logit::invoke(
    const Tensor& input_tensor,
    const std::optional<float> eps,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return detail::unary_impl(
        input_tensor,
        {UnaryWithParam{UnaryOpType::LOGIT, {eps.value_or(-1.0f)}}},
        memory_config,
        optional_output_tensor);
}

Tensor Deg2Rad::invoke(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    constexpr float DEG_TO_RAD = 0.017453292519943295f;  // pi/180
    return binary::BinaryOperation<operations::binary::BinaryOpType::MUL>::invoke(
        input_tensor,
        DEG_TO_RAD,
        input_tensor.dtype(),
        memory_config,
        optional_output_tensor,
        {},
        {},
        {},
        std::nullopt);
}

Tensor Rad2Deg::invoke(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    constexpr float RAD_TO_DEG = 57.29577951308232f;  // 180/pi
    return binary::BinaryOperation<operations::binary::BinaryOpType::MUL>::invoke(
        input_tensor,
        RAD_TO_DEG,
        input_tensor.dtype(),
        memory_config,
        optional_output_tensor,
        {},
        {},
        {},
        std::nullopt);
}

Tensor Where::invoke(
    const Tensor& condition,
    const ScalarVariant& value_true,
    const ScalarVariant& value_false,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    Tensor input = condition;
    // Check if we have any float scalars
    bool has_float_scalar = std::holds_alternative<float>(value_true) || std::holds_alternative<float>(value_false);

    // Convert input tensor to float32 only if input is INT32/UINT32 and scalars are float
    if ((condition.dtype() == DataType::INT32 || condition.dtype() == DataType::UINT32) && has_float_scalar) {
        input = ttnn::typecast(condition, DataType::FLOAT32);
    }
    UnaryOpType op_type = UnaryOpType::WHERE_TSS;
    auto param = std::visit(
        [op_type](const auto& val_true, const auto& val_false) {
            using T = std::decay_t<decltype(val_true)>;
            return EltwiseUnaryWithParam{op_type, std::vector<T>{val_true, val_false}};
        },
        value_true,
        value_false);

    return detail::unary_impl(input, {param}, memory_config, optional_output_tensor);
}

template <UnaryOpType unary_op_type, typename T>
Tensor ExecuteUnaryWithIntegerParameter<unary_op_type, T>::invoke(
    const Tensor& input_tensor,
    T parameter,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return detail::unary_impl(
        input_tensor,
        {EltwiseUnaryWithParam{unary_op_type, parameter}},
        memory_config,
        optional_output_tensor);
}

Tensor Swish::invoke(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    UnaryOpType op_type = UnaryOpType::SILU;
    return detail::unary_impl(input_tensor, {EltwiseUnaryWithParam{op_type}}, memory_config, optional_output_tensor);
}

template struct ExecuteUnaryWithIntegerParameter<UnaryOpType::POWER, uint32_t>;
template struct ExecuteUnaryWithIntegerParameter<UnaryOpType::LEFT_SHIFT, int32_t>;
template struct ExecuteUnaryWithIntegerParameter<UnaryOpType::RIGHT_SHIFT, int32_t>;
template struct ExecuteUnaryWithIntegerParameter<UnaryOpType::BITWISE_AND, int32_t>;
template struct ExecuteUnaryWithIntegerParameter<UnaryOpType::BITWISE_OR, int32_t>;
template struct ExecuteUnaryWithIntegerParameter<UnaryOpType::BITWISE_XOR, int32_t>;

template <UnaryOpType unary_op_type, typename T>
Tensor ExecuteUnaryWithOptionalIntegerParameter<unary_op_type, T>::invoke(
    const Tensor& input_tensor,
    const std::optional<T>& parameter,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return detail::unary_impl(
        input_tensor,
        {EltwiseUnaryWithParam{unary_op_type, parameter.value_or(0)}},
        memory_config,
        optional_output_tensor);
}

template struct ExecuteUnaryWithOptionalIntegerParameter<UnaryOpType::ROUND, int32_t>;

template <UnaryOpType unary_op_type, typename T>
Tensor SymmetricBinop<unary_op_type, T>::invoke(
    const Tensor& input_tensor,
    T param,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return detail::unary_impl(
        input_tensor, {EltwiseUnaryWithParam(unary_op_type, (param))}, memory_config, optional_output_tensor);
}

template <UnaryOpType unary_op_type, typename T>
Tensor SymmetricBinop<unary_op_type, T>::invoke(
    T param,
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return detail::unary_impl(
        input_tensor, {EltwiseUnaryWithParam(unary_op_type, (param))}, memory_config, optional_output_tensor);
}

// Explicit template instantiation
template struct SymmetricBinop<UnaryOpType::ADD_UNARY_SFPU>;
template struct SymmetricBinop<UnaryOpType::MUL_UNARY_SFPU>;

template <UnaryOpType unary_op_type, UnaryOpType unary_op_rev_type>
Tensor AsymmetricBinop<unary_op_type, unary_op_rev_type>::invoke(
    const Tensor& input_tensor,
    float param,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return detail::unary_impl(
        input_tensor, {EltwiseUnaryWithParam{unary_op_type, (param)}}, memory_config, optional_output_tensor);
}

template <UnaryOpType unary_op_type, UnaryOpType unary_op_rev_type>
Tensor AsymmetricBinop<unary_op_type, unary_op_rev_type>::invoke(
    float param,
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return detail::unary_impl(
        input_tensor, {EltwiseUnaryWithParam{unary_op_rev_type, (param)}}, memory_config, optional_output_tensor);
}
template struct AsymmetricBinop<UnaryOpType::SUB_UNARY_SFPU, UnaryOpType::RSUB>;
template struct AsymmetricBinop<UnaryOpType::DIV_UNARY_SFPU, UnaryOpType::RDIV>;

}  // namespace ttnn::operations::unary
