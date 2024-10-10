// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unary.hpp"

#include "ttnn/common/constants.hpp"
#include "device/unary_device_operation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/pool/downsample/device/downsample_op.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"

namespace ttnn::operations::unary {

namespace detail {

inline Tensor unary_impl(
    uint8_t queue_id,
    const Tensor& input_tensor,
    const std::vector<UnaryWithParam>& op_chain,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    DataType output_dtype = (op_chain[0].op_type == UnaryOpType::TYPECAST) ? static_cast<DataType>(op_chain[0].params[1]) : input_tensor.get_dtype();
    bool preserve_fp32_precision = (op_chain[0].op_type == UnaryOpType::TYPECAST) and (input_tensor.get_dtype() == DataType::FLOAT32);
    bool fp32_dest_acc_en = preserve_fp32_precision or
                            output_dtype == DataType::UINT32 or
                            output_dtype == DataType::INT32 or
                            output_dtype == DataType::FLOAT32 or
                            input_tensor.get_dtype() == DataType::UINT32 or
                            input_tensor.get_dtype() == DataType::INT32;  // MT: Currently only uint32/int32 is moved to
                                                                          // DST directly, fp32 is converted to fp16b

    auto output_memory_config = optional_output_tensor.has_value() ? optional_output_tensor.value().memory_config() : memory_config.value_or(input_tensor.memory_config());
    return prim::unary(queue_id, input_tensor, op_chain, output_dtype, output_memory_config, fp32_dest_acc_en, preserve_fp32_precision, optional_output_tensor);
}

}  // namespace detail

template <UnaryOpType... unary_op_types>
Tensor ExecuteUnary<unary_op_types...>::invoke(
    uint8_t queue_id,
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return detail::unary_impl(
        queue_id, input_tensor, {UnaryWithParam{unary_op_types}...}, memory_config, optional_output_tensor);
}

template <UnaryOpType... unary_op_types>
Tensor ExecuteUnary<unary_op_types...>::invoke(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return detail::unary_impl(
        DefaultQueueId, input_tensor, {UnaryWithParam{unary_op_types}...}, memory_config, optional_output_tensor);
}

template <>
Tensor ExecuteUnary<UnaryOpType::ABS>::invoke(
    const ComplexTensor& input_tensor,
    const MemoryConfig& output_mem_config) {
    return ttnn::hypot(input_tensor[0],input_tensor[1],output_mem_config);
}

template <>
ComplexTensor ExecuteUnary<UnaryOpType::RECIP>::invoke(
    const ComplexTensor& input,
    const MemoryConfig& output_mem_config) {
    Tensor a_plus_b = ttnn::add(input[0],input[1],std::nullopt,output_mem_config);
    Tensor a_minus_b = ttnn::subtract(input[0],input[1],std::nullopt,output_mem_config);
    Tensor asqr_plus_bsqr = ttnn::add(ttnn::square(input[0],output_mem_config),ttnn::square(input[1],output_mem_config),
                                std::nullopt,output_mem_config);
    Tensor inv_dr = ttnn::reciprocal( asqr_plus_bsqr, output_mem_config );
    Tensor conj_im = ttnn::multiply( ttnn::neg(input[1],output_mem_config), inv_dr, std::nullopt, output_mem_config);
    Tensor conj_re = ttnn::multiply( input[0], inv_dr, std::nullopt, output_mem_config);
    return ComplexTensor({ conj_re, conj_im});

}

template struct ExecuteUnary<UnaryOpType::ABS>;
template struct ExecuteUnary<UnaryOpType::ACOS>;
template struct ExecuteUnary<UnaryOpType::ASIN>;
template struct ExecuteUnary<UnaryOpType::ATAN>;
template struct ExecuteUnary<UnaryOpType::COS>;
template struct ExecuteUnary<UnaryOpType::ERFINV>;
template struct ExecuteUnary<UnaryOpType::EXP2>;
template struct ExecuteUnary<UnaryOpType::EXPM1>;
template struct ExecuteUnary<UnaryOpType::EQZ>;
template struct ExecuteUnary<UnaryOpType::FLOOR>;
template struct ExecuteUnary<UnaryOpType::CEIL>;
template struct ExecuteUnary<UnaryOpType::GEZ>;
template struct ExecuteUnary<UnaryOpType::GTZ>;
template struct ExecuteUnary<UnaryOpType::I0>;
template struct ExecuteUnary<UnaryOpType::ISFINITE>;
template struct ExecuteUnary<UnaryOpType::ISINF>;
template struct ExecuteUnary<UnaryOpType::ISNAN>;
template struct ExecuteUnary<UnaryOpType::ISNEGINF>;
template struct ExecuteUnary<UnaryOpType::ISPOSINF>;
template struct ExecuteUnary<UnaryOpType::LEZ>;
template struct ExecuteUnary<UnaryOpType::LOG>;
template struct ExecuteUnary<UnaryOpType::LOG10>;
template struct ExecuteUnary<UnaryOpType::LOG2>;
template struct ExecuteUnary<UnaryOpType::LOGICAL_NOT_UNARY>;
template struct ExecuteUnary<UnaryOpType::LTZ>;
template struct ExecuteUnary<UnaryOpType::NEG>;
template struct ExecuteUnary<UnaryOpType::NEZ>;
template struct ExecuteUnary<UnaryOpType::RECIP>;
template struct ExecuteUnary<UnaryOpType::RELU>;
template struct ExecuteUnary<UnaryOpType::RELU6>;
template struct ExecuteUnary<UnaryOpType::SIGMOID>;
template struct ExecuteUnary<UnaryOpType::SIGN>;
template struct ExecuteUnary<UnaryOpType::SIGNBIT>;
template struct ExecuteUnary<UnaryOpType::SILU>;
template struct ExecuteUnary<UnaryOpType::SIN>;
template struct ExecuteUnary<UnaryOpType::SQRT>;
template struct ExecuteUnary<UnaryOpType::SQUARE>;
template struct ExecuteUnary<UnaryOpType::TAN>;
template struct ExecuteUnary<UnaryOpType::TANH>;
template struct ExecuteUnary<UnaryOpType::SIGMOID, UnaryOpType::LOG>;
template struct ExecuteUnary<UnaryOpType::TILED_PROD>;

template <UnaryOpType unary_op_type>
Tensor ExecuteUnaryWithFastAndApproximateMode<unary_op_type>::invoke(
    uint8_t queue_id,
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

template <UnaryOpType unary_op_type>
Tensor ExecuteUnaryWithFastAndApproximateMode<unary_op_type>::invoke(
    const Tensor& input_tensor,
    const bool parameter,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return detail::unary_impl(
        DefaultQueueId,
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
Tensor ExecuteUnaryWithFloatParameter<unary_op_type>::invoke(
    uint8_t queue_id,
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

template <UnaryOpType unary_op_type>
Tensor ExecuteUnaryWithFloatParameter<unary_op_type>::invoke(
    const Tensor& input_tensor,
    const float parameter,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return detail::unary_impl(
        DefaultQueueId,
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

Tensor Sigmoid_accurate::invoke(
    uint8_t queue_id,
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

Tensor Sigmoid_accurate::invoke(
    const Tensor& input,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return detail::unary_impl(
        DefaultQueueId,
        input,
        {UnaryWithParam(UnaryOpType::NEG),
         UnaryWithParam(UnaryOpType::EXP, 1.0f),
         UnaryWithParam(UnaryOpType::ADD_UNARY_SFPU, 1.0f),
         UnaryWithParam(UnaryOpType::RECIP)},
        memory_config,
        optional_output_tensor);
}

Tensor Unary_chain::invoke(
    uint8_t queue_id,
    const Tensor& input_tensor,
    const std::vector<UnaryWithParam>& ops_chain,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    TT_FATAL(ops_chain.size() > 0, "Op chain cannot be empty");
    return detail::unary_impl(queue_id, input_tensor, ops_chain, memory_config, optional_output_tensor);
}

Tensor Unary_chain::invoke(
    const Tensor& input_tensor,
    const std::vector<UnaryWithParam>& ops_chain,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    TT_FATAL(ops_chain.size() > 0, "Op chain cannot be empty");
    return detail::unary_impl(DefaultQueueId, input_tensor, ops_chain, memory_config, optional_output_tensor);
}

Tensor Softplus::invoke(
    uint8_t queue_id,
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

Tensor Softplus::invoke(
    const Tensor& input,
    const float beta,
    const float threshold,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    TT_ASSERT(input.device()->arch() != tt::ARCH::GRAYSKULL, "Softplus is not currently supported on Grayskull");
    return detail::unary_impl(
        DefaultQueueId,
        input,
        {UnaryWithParam{UnaryOpType::SOFTPLUS, {beta, threshold}}},
        memory_config,
        optional_output_tensor);
}

Tensor Identity::invoke(
    uint8_t queue_id,
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    UnaryOpType op_type = UnaryOpType::IDENTITY;
    if (input_tensor.get_dtype() == DataType::UINT32) {
        op_type = UnaryOpType::IDENTITY_UINT32;
    }

    return detail::unary_impl(
        queue_id, input_tensor, {UnaryWithParam{op_type}}, memory_config, optional_output_tensor);
}

Tensor Identity::invoke(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    UnaryOpType op_type = UnaryOpType::IDENTITY;
    if (input_tensor.get_dtype() == DataType::UINT32) {
        op_type = UnaryOpType::IDENTITY_UINT32;
    }

    return detail::unary_impl(
        DefaultQueueId, input_tensor, {UnaryWithParam{op_type}}, memory_config, optional_output_tensor);
}

Tensor Dropout::invoke(
    const Tensor& input,
    const uint32_t seed,
    const float probability,
    const float scale,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    TT_ASSERT(input.device()->arch() != tt::ARCH::GRAYSKULL, "Dropout is not currently supported on Grayskull");
    return detail::unary_impl(
        DefaultQueueId,
        input,
        {UnaryWithParam{UnaryOpType::DROPOUT, {static_cast<float>(seed), probability, scale}}},
        memory_config,
        optional_output_tensor);
}

Tensor Dropout::invoke(
    uint8_t queue_id,
    const Tensor& input,
    const uint32_t seed,
    const float probability,
    const float scale,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    TT_ASSERT(input.device()->arch() != tt::ARCH::GRAYSKULL, "Dropout is not currently supported on Grayskull");
    return detail::unary_impl(
        queue_id,
        input,
        {UnaryWithParam{UnaryOpType::DROPOUT, {static_cast<float>(seed), probability, scale}}},
        memory_config,
        optional_output_tensor);
}

template <UnaryOpType unary_op_type, typename T>
Tensor ExecuteUnaryWithIntegerParameter<unary_op_type, T>::invoke(
    uint8_t queue_id,
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

template <UnaryOpType unary_op_type, typename T>
Tensor ExecuteUnaryWithIntegerParameter<unary_op_type, T>::invoke(
    const Tensor& input_tensor,
    T parameter,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return detail::unary_impl(
        DefaultQueueId,
        input_tensor,
        {UnaryWithParam{unary_op_type, static_cast<float>(parameter)}},
        memory_config,
        optional_output_tensor);
}

template struct ExecuteUnaryWithIntegerParameter<UnaryOpType::POWER, uint32_t>;
template struct ExecuteUnaryWithIntegerParameter<UnaryOpType::LEFT_SHIFT, int32_t>;
template struct ExecuteUnaryWithIntegerParameter<UnaryOpType::RIGHT_SHIFT, int32_t>;
template struct ExecuteUnaryWithIntegerParameter<UnaryOpType::BITWISE_AND, int32_t>;
template struct ExecuteUnaryWithIntegerParameter<UnaryOpType::BITWISE_OR, int32_t>;
template struct ExecuteUnaryWithIntegerParameter<UnaryOpType::BITWISE_XOR, int32_t>;
template struct ExecuteUnaryWithIntegerParameter<UnaryOpType::BITWISE_NOT, int32_t>;


template <UnaryOpType unary_op_type, typename T>
Tensor SymmetricBinop<unary_op_type, T>::invoke(
    uint8_t queue_id,
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
    uint8_t queue_id,
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

template <UnaryOpType unary_op_type, typename T>
Tensor SymmetricBinop<unary_op_type, T>::invoke(
    const Tensor& input_tensor,
    T param,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return detail::unary_impl(
        DefaultQueueId,
        input_tensor,
        {UnaryWithParam(unary_op_type, static_cast<float>(param))},
        memory_config,
        optional_output_tensor);
}

template <UnaryOpType unary_op_type, typename T>
Tensor SymmetricBinop<unary_op_type, T>::invoke(
    T param,
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return detail::unary_impl(
        DefaultQueueId,
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
    uint8_t queue_id,
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
    uint8_t queue_id,
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

template <UnaryOpType unary_op_type, UnaryOpType unary_op_rev_type>
Tensor AsymmetricBinop<unary_op_type, unary_op_rev_type>::invoke(
    const Tensor& input_tensor,
    float param,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return detail::unary_impl(
        DefaultQueueId,
        input_tensor,
        {UnaryWithParam(unary_op_type, static_cast<float>(param))},
        memory_config,
        optional_output_tensor);
}

template <UnaryOpType unary_op_type, UnaryOpType unary_op_rev_type>
Tensor AsymmetricBinop<unary_op_type, unary_op_rev_type>::invoke(
    float param,
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return detail::unary_impl(
        DefaultQueueId,
        input_tensor,
        {UnaryWithParam(unary_op_rev_type, static_cast<float>(param))},
        memory_config,
        optional_output_tensor);
}

template struct AsymmetricBinop<UnaryOpType::SUB_UNARY_SFPU, UnaryOpType::RSUB>;
template struct AsymmetricBinop<UnaryOpType::DIV_UNARY_SFPU, UnaryOpType::RDIV>;

}
