// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unary.hpp"

#include "ttnn/common/queue_id.hpp"
#include "device/unary_device_operation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/pool/downsample/device/downsample_op.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/ternary/where.hpp"
#include "ttnn/operations/eltwise/unary/tanh_accurate/tanh_accurate.hpp"

namespace ttnn::operations::unary {

namespace detail {

inline Tensor unary_impl(
    QueueId queue_id,
    const Tensor& input_tensor,
    const std::vector<UnaryWithParam>& op_chain,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    TT_FATAL(op_chain.size() > 0, "Op chain cannot be empty");
    DataType output_dtype = (op_chain[0].op_type == UnaryOpType::TYPECAST)
                                ? static_cast<DataType>(op_chain[0].params[1])
                                : input_tensor.get_dtype();
    auto arch = input_tensor.device()->arch();
    bool preserve_fp32_precision = (arch != tt::ARCH::GRAYSKULL) and (input_tensor.get_dtype() == DataType::FLOAT32);
    bool fp32_dest_acc_en = preserve_fp32_precision or output_dtype == DataType::UINT32 or
                            output_dtype == DataType::INT32 or output_dtype == DataType::FLOAT32 or
                            input_tensor.get_dtype() == DataType::UINT32 or input_tensor.get_dtype() == DataType::INT32;
    bool bfp8_pack_precise = (op_chain[0].op_type == UnaryOpType::TYPECAST && output_dtype == DataType::BFLOAT8_B);

    auto output_memory_config = optional_output_tensor.has_value()
                                    ? optional_output_tensor.value().memory_config()
                                    : memory_config.value_or(input_tensor.memory_config());
    return prim::unary(
        queue_id,
        input_tensor,
        op_chain,
        output_dtype,
        output_memory_config,
        fp32_dest_acc_en,
        preserve_fp32_precision,
        bfp8_pack_precise,
        optional_output_tensor);
}

}  // namespace detail

template <UnaryOpType... unary_op_types>
Tensor ExecuteUnary<unary_op_types...>::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return detail::unary_impl(
        queue_id, input_tensor, {UnaryWithParam{unary_op_types}...}, memory_config, optional_output_tensor);
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
template struct ExecuteUnary<UnaryOpType::ATAN>;
template struct ExecuteUnary<UnaryOpType::COS>;
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
template struct ExecuteUnary<UnaryOpType::LOG>;
template struct ExecuteUnary<UnaryOpType::LOG10>;
template struct ExecuteUnary<UnaryOpType::LOG2>;
template struct ExecuteUnary<UnaryOpType::LOG1P>;
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
template struct ExecuteUnary<UnaryOpType::SQRT>;
template struct ExecuteUnary<UnaryOpType::SQUARE>;
template struct ExecuteUnary<UnaryOpType::TAN>;
template struct ExecuteUnary<UnaryOpType::TANH>;
template struct ExecuteUnary<UnaryOpType::TILED_PROD>;
template struct ExecuteUnary<UnaryOpType::BITWISE_NOT>;
template struct ExecuteUnary<UnaryOpType::ALT_COMPLEX_ROTATE90>;

template <UnaryOpType unary_op_type>
Tensor ExecuteUnaryWithFastAndApproximateMode<unary_op_type>::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const bool parameter,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return detail::unary_impl(
        queue_id,
        input_tensor,
        {UnaryWithParam{unary_op_type, static_cast<float>(parameter)}},
        memory_config,
        optional_output_tensor);
}

template struct ExecuteUnaryWithFastAndApproximateMode<UnaryOpType::EXP>;
template struct ExecuteUnaryWithFastAndApproximateMode<UnaryOpType::ERF>;
template struct ExecuteUnaryWithFastAndApproximateMode<UnaryOpType::ERFC>;
template struct ExecuteUnaryWithFastAndApproximateMode<UnaryOpType::GELU>;
template struct ExecuteUnaryWithFastAndApproximateMode<UnaryOpType::RSQRT>;

template <UnaryOpType unary_op_type>
Tensor ExecuteUnaryWithVectorAndFastAndApproximateMode<unary_op_type>::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const int mode,
    const bool parameter,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return detail::unary_impl(
        queue_id,
        input_tensor,
        {UnaryWithParam{unary_op_type, {static_cast<float>(mode), static_cast<float>(parameter)}}},
        memory_config,
        optional_output_tensor);
}

template struct ExecuteUnaryWithVectorAndFastAndApproximateMode<UnaryOpType::SIGMOID>;

template <UnaryOpType unary_op_type>
Tensor ExecuteUnaryWithFloatParameter<unary_op_type>::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const float parameter,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return detail::unary_impl(
        queue_id,
        input_tensor,
        {UnaryWithParam{unary_op_type, static_cast<float>(parameter)}},
        memory_config,
        optional_output_tensor);
}

template struct ExecuteUnaryWithFloatParameter<UnaryOpType::ELU>;
template struct ExecuteUnaryWithFloatParameter<UnaryOpType::RSUB>;
template struct ExecuteUnaryWithFloatParameter<UnaryOpType::HEAVISIDE>;
template struct ExecuteUnaryWithFloatParameter<UnaryOpType::LEAKY_RELU>;
template struct ExecuteUnaryWithFloatParameter<UnaryOpType::RELU_MAX>;
template struct ExecuteUnaryWithFloatParameter<UnaryOpType::RELU_MIN>;
template struct ExecuteUnaryWithFloatParameter<UnaryOpType::REMAINDER>;
template struct ExecuteUnaryWithFloatParameter<UnaryOpType::FMOD>;
template struct ExecuteUnaryWithFloatParameter<UnaryOpType::FILL>;
template struct ExecuteUnaryWithFloatParameter<UnaryOpType::UNARY_GT>;
template struct ExecuteUnaryWithFloatParameter<UnaryOpType::UNARY_LT>;
template struct ExecuteUnaryWithFloatParameter<UnaryOpType::UNARY_NE>;
template struct ExecuteUnaryWithFloatParameter<UnaryOpType::UNARY_EQ>;
template struct ExecuteUnaryWithFloatParameter<UnaryOpType::MAXIMUM>;
template struct ExecuteUnaryWithFloatParameter<UnaryOpType::MINIMUM>;

Tensor Sigmoid_accurate::invoke(
    QueueId queue_id,
    const Tensor& input,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return detail::unary_impl(
        queue_id,
        input,
        {UnaryWithParam(UnaryOpType::NEG),
         UnaryWithParam(UnaryOpType::EXP, 1.0f),
         UnaryWithParam(UnaryOpType::ADD_UNARY_SFPU, 1.0f),
         UnaryWithParam(UnaryOpType::RECIP)},
        memory_config,
        optional_output_tensor);
}

template struct ExecuteUnary<UnaryOpType::SIGMOID, UnaryOpType::LOG>;
Tensor LogSigmoid::invoke(
    QueueId queue_id,
    const Tensor& input,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return detail::unary_impl(
        queue_id,
        input,
        {UnaryWithParam(UnaryOpType::SIGMOID, {(int)VecMode::RC, false}), UnaryWithParam(UnaryOpType::LOG)},
        memory_config,
        optional_output_tensor);
}

Tensor Eqz::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    UnaryOpType op_type = UnaryOpType::EQZ;
    return detail::unary_impl(queue_id, input_tensor, {UnaryWithParam{op_type}}, memory_config, optional_output_tensor);
}

Tensor Unary_chain::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const std::vector<UnaryWithParam>& ops_chain,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    TT_FATAL(ops_chain.size() > 0, "Op chain cannot be empty");
    return detail::unary_impl(queue_id, input_tensor, ops_chain, memory_config, optional_output_tensor);
}

Tensor Softplus::invoke(
    QueueId queue_id,
    const Tensor& input,
    const float beta,
    const float threshold,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    TT_ASSERT(input.device()->arch() != tt::ARCH::GRAYSKULL, "Softplus is not currently supported on Grayskull");
    return detail::unary_impl(
        queue_id,
        input,
        {UnaryWithParam{UnaryOpType::SOFTPLUS, {beta, threshold}}},
        memory_config,
        optional_output_tensor);
}

// tanh[x] = (exp[2x] - 1) / (exp[2x] + 1)
Tensor Tanh::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor,
    bool accuracy) {
    UnaryOpType op_type = UnaryOpType::TANH;
    if (!accuracy) {
        return detail::unary_impl(
            queue_id, input_tensor, {UnaryWithParam{op_type}}, memory_config, optional_output_tensor);
    } else {
        return ttnn::tanh_accurate(queue_id, input_tensor, memory_config, optional_output_tensor);
    }
}

Tensor Prelu::invoke(
    QueueId queue_id,
    const Tensor& input,
    float value,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return detail::unary_impl(
        queue_id, input, {UnaryWithParam{UnaryOpType::PRELU_SFPU, value}}, memory_config, optional_output_tensor);
}

Tensor Identity::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    UnaryOpType op_type = UnaryOpType::IDENTITY;
    if (input_tensor.get_dtype() == DataType::UINT32) {
        op_type = UnaryOpType::IDENTITY_UINT32;
    }

    return detail::unary_impl(queue_id, input_tensor, {UnaryWithParam{op_type}}, memory_config, optional_output_tensor);
}

Tensor Abs::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    UnaryOpType op_type = UnaryOpType::ABS;
    if (input_tensor.get_dtype() == DataType::INT32) {
        op_type = UnaryOpType::ABS_INT32;
    }
    return detail::unary_impl(queue_id, input_tensor, {UnaryWithParam{op_type}}, memory_config, optional_output_tensor);
}

Tensor Abs::invoke(const ComplexTensor& input_tensor, const MemoryConfig& output_mem_config) {
    return ttnn::hypot(input_tensor[0], input_tensor[1], output_mem_config);
}

Tensor Floor::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    UnaryOpType op_type = UnaryOpType::FLOOR;
    if (input_tensor.get_dtype() == DataType::FLOAT32) {
        op_type = UnaryOpType::FLOOR_FLOAT32;
    }

    return detail::unary_impl(queue_id, input_tensor, {UnaryWithParam{op_type}}, memory_config, optional_output_tensor);
}

Tensor Ceil::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    UnaryOpType op_type = UnaryOpType::CEIL;
    if (input_tensor.get_dtype() == DataType::FLOAT32) {
        op_type = UnaryOpType::CEIL_FLOAT32;
    }

    return detail::unary_impl(queue_id, input_tensor, {UnaryWithParam{op_type}}, memory_config, optional_output_tensor);
}

Tensor Mish::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    UnaryOpType op_type = UnaryOpType::MISH;

    return detail::unary_impl(queue_id, input_tensor, {UnaryWithParam{op_type}}, memory_config, optional_output_tensor);
}

template <UnaryOpType unary_op_type, typename T>
Tensor ExecuteUnaryWithIntegerParameter<unary_op_type, T>::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    T parameter,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return detail::unary_impl(
        queue_id,
        input_tensor,
        {UnaryWithParam{unary_op_type, static_cast<float>(parameter)}},
        memory_config,
        optional_output_tensor);
}

template struct ExecuteUnaryWithIntegerParameter<UnaryOpType::POWER, uint32_t>;
template struct ExecuteUnaryWithIntegerParameter<UnaryOpType::LEFT_SHIFT, int32_t>;
template struct ExecuteUnaryWithIntegerParameter<UnaryOpType::RIGHT_SHIFT, int32_t>;
template struct ExecuteUnaryWithIntegerParameter<UnaryOpType::ROUND, int32_t>;
template struct ExecuteUnaryWithIntegerParameter<UnaryOpType::BITWISE_AND, int32_t>;
template struct ExecuteUnaryWithIntegerParameter<UnaryOpType::BITWISE_OR, int32_t>;
template struct ExecuteUnaryWithIntegerParameter<UnaryOpType::BITWISE_XOR, int32_t>;

template <UnaryOpType unary_op_type, typename T>
Tensor SymmetricBinop<unary_op_type, T>::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    T param,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return detail::unary_impl(
        queue_id,
        input_tensor,
        {UnaryWithParam(unary_op_type, static_cast<float>(param))},
        memory_config,
        optional_output_tensor);
}

template <UnaryOpType unary_op_type, typename T>
Tensor SymmetricBinop<unary_op_type, T>::invoke(
    QueueId queue_id,
    T param,
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return detail::unary_impl(
        queue_id,
        input_tensor,
        {UnaryWithParam(unary_op_type, static_cast<float>(param))},
        memory_config,
        optional_output_tensor);
}

// Explicit template instantiation
template struct SymmetricBinop<UnaryOpType::ADD_UNARY_SFPU>;
template struct SymmetricBinop<UnaryOpType::MUL_UNARY_SFPU>;

template <UnaryOpType unary_op_type, UnaryOpType unary_op_rev_type>
Tensor AsymmetricBinop<unary_op_type, unary_op_rev_type>::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    float param,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return detail::unary_impl(
        queue_id,
        input_tensor,
        {UnaryWithParam(unary_op_type, static_cast<float>(param))},
        memory_config,
        optional_output_tensor);
}

template <UnaryOpType unary_op_type, UnaryOpType unary_op_rev_type>
Tensor AsymmetricBinop<unary_op_type, unary_op_rev_type>::invoke(
    QueueId queue_id,
    float param,
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return detail::unary_impl(
        queue_id,
        input_tensor,
        {UnaryWithParam(unary_op_rev_type, static_cast<float>(param))},
        memory_config,
        optional_output_tensor);
}

template struct AsymmetricBinop<UnaryOpType::SUB_UNARY_SFPU, UnaryOpType::RSUB>;
template struct AsymmetricBinop<UnaryOpType::DIV_UNARY_SFPU, UnaryOpType::RDIV>;

}  // namespace ttnn::operations::unary
