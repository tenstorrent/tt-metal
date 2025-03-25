// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "quantization.hpp"
#include "ttnn/operations/copy.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.hpp"

namespace ttnn::operations::quantization {

Tensor QuantOp::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const float scale,
    const int32_t zero_point,
    const std::optional<int32_t> axis,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    const ttnn::DataType a_dtype = input_tensor.get_dtype();
    constexpr ttnn::DataType c_dtype = ttnn::DataType::INT32;

    TT_FATAL(tt::tt_metal::is_floating_point(a_dtype), "Quantize only takes floating-point number inputs");
    TT_FATAL(output_dtype.value_or(c_dtype) == c_dtype, "Quantize only supports int32 outputs for now");
    if (optional_output_tensor.has_value()) {
        TT_FATAL(optional_output_tensor->dtype() == c_dtype, "Quantize only supports int32 outputs for now");
    }

    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations{};
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations{};
    const std::array post_activations{
        ttnn::operations::unary::UnaryWithParam{unary::UnaryOpType::ZERO_POINT, static_cast<float>(zero_point)}};

    // LLK quant kernel expects the reciprocal of the actual scale to avoid doing div on the device
    return ttnn::prim::binary_ng(
        queue_id,
        tt::tt_metal::is_block_float(a_dtype) ? ttnn::typecast(input_tensor, DataType::BFLOAT16) : input_tensor,
        1.0f / scale,
        binary_ng::BinaryOpType::QUANT,
        c_dtype,
        memory_config,
        optional_output_tensor,
        lhs_activations,
        rhs_activations,
        post_activations);
}

Tensor QuantOp::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const Tensor& scale,
    const int32_t zero_point,
    const std::optional<int32_t> axis,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    const ttnn::DataType a_dtype = input_tensor.get_dtype();
    const ttnn::DataType b_dtype = scale.get_dtype();
    constexpr ttnn::DataType c_dtype = ttnn::DataType::INT32;

    TT_FATAL(tt::tt_metal::is_floating_point(a_dtype), "Quantize only takes floating-point number inputs");
    TT_FATAL(tt::tt_metal::is_floating_point(b_dtype), "Quantize only takes floating-point number scales");
    TT_FATAL(!tt::tt_metal::is_block_float(b_dtype), "Unsupported scale tensor format");
    TT_FATAL(scale.get_logical_volume() == 1u, "Per-tensor quantize only takes a single scale value");
    TT_FATAL(output_dtype.value_or(c_dtype) == c_dtype, "Quantize only supports int32 outputs for now");
    if (optional_output_tensor.has_value()) {
        TT_FATAL(optional_output_tensor->dtype() == c_dtype, "Quantize only supports int32 outputs for now");
    }

    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations{};
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations{};
    const std::array post_activations{
        ttnn::operations::unary::UnaryWithParam{unary::UnaryOpType::ZERO_POINT, static_cast<float>(zero_point)}};

    // TODO: do reciprocal using an activation containing an UnaryWithParam that does rdiv(scale, 1), benchmark it
    return ttnn::prim::binary_ng(
        queue_id,
        tt::tt_metal::is_block_float(a_dtype) ? ttnn::typecast(input_tensor, DataType::BFLOAT16) : input_tensor,
        ttnn::reciprocal(scale),
        binary_ng::BinaryOpType::QUANT,
        c_dtype,
        memory_config,
        optional_output_tensor,
        lhs_activations,
        rhs_activations,
        post_activations);
}

Tensor RequantOp::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const float in_scale,
    const int32_t in_zero_point,
    const float out_scale,
    const int32_t out_zero_point,
    const std::optional<int32_t> axis,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    const ttnn::DataType a_dtype = input_tensor.get_dtype();
    constexpr ttnn::DataType c_dtype = ttnn::DataType::INT32;

    TT_FATAL(a_dtype == ttnn::DataType::INT32, "Requantize only supports int32 inputs for now");
    TT_FATAL(output_dtype.value_or(c_dtype) == c_dtype, "Requantize only supports int32 outputs for now");
    if (optional_output_tensor.has_value()) {
        TT_FATAL(optional_output_tensor->dtype() == c_dtype, "Requantize only supports int32 outputs for now");
    }

    // Expansion of q' = [(q - z_in) * s_in] / s_out + z_out
    const float scale = out_scale / in_scale;
    // ZP is passed to and consumed by the kernel as f32 anyway, might as well preserve some accuracy here
    const float zero_point = out_zero_point - in_zero_point / scale;

    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations{};
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations{};
    const std::array post_activations{
        ttnn::operations::unary::UnaryWithParam{unary::UnaryOpType::ZERO_POINT, zero_point}};

    return ttnn::prim::binary_ng(
        queue_id,
        input_tensor,
        1.0f / scale,
        binary_ng::BinaryOpType::REQUANT,
        c_dtype,
        memory_config,
        optional_output_tensor,
        lhs_activations,
        rhs_activations,
        post_activations);
}

Tensor RequantOp::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const Tensor& in_scale,
    const int32_t in_zero_point,
    const float out_scale,
    const int32_t out_zero_point,
    const std::optional<int32_t> axis,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    const ttnn::DataType b_dtype = in_scale.get_dtype();

    TT_FATAL(tt::tt_metal::is_floating_point(b_dtype), "Requantize only takes floating-point number scales");
    TT_FATAL(!tt::tt_metal::is_block_float(b_dtype), "Unsupported scale tensor format");
    TT_FATAL(in_scale.get_logical_volume() == 1u, "Per-tensor requantize only takes a single input scale value");

    const float in_scale_scalar = [](const DataType b_dtype, const Tensor& in_scale) {
        if (b_dtype == DataType::FLOAT32) {
            return in_scale.to_vector<float>()[0];
        } else {
            return in_scale.to_vector<bfloat16>()[0].to_float();
        }
    }(b_dtype, in_scale);

    return invoke(
        queue_id,
        input_tensor,
        in_scale_scalar,
        in_zero_point,
        out_scale,
        out_zero_point,
        axis,
        output_dtype,
        memory_config,
        optional_output_tensor);
}

Tensor RequantOp::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const float in_scale,
    const int32_t in_zero_point,
    const Tensor& out_scale,
    const int32_t out_zero_point,
    const std::optional<int32_t> axis,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    const ttnn::DataType b_dtype = out_scale.get_dtype();

    TT_FATAL(tt::tt_metal::is_floating_point(b_dtype), "Requantize only takes floating-point number scales");
    TT_FATAL(!tt::tt_metal::is_block_float(b_dtype), "Unsupported scale tensor format");
    TT_FATAL(out_scale.get_logical_volume() == 1u, "Per-tensor requantize only takes a single output scale value");

    const float out_scale_scalar = [](const DataType b_dtype, const Tensor& out_scale) {
        if (b_dtype == DataType::FLOAT32) {
            return out_scale.to_vector<float>()[0];
        } else {
            return out_scale.to_vector<bfloat16>()[0].to_float();
        }
    }(b_dtype, out_scale);

    return invoke(
        queue_id,
        input_tensor,
        in_scale,
        in_zero_point,
        out_scale_scalar,
        out_zero_point,
        axis,
        output_dtype,
        memory_config,
        optional_output_tensor);
}

Tensor RequantOp::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const Tensor& in_scale,
    const int32_t in_zero_point,
    const Tensor& out_scale,
    const int32_t out_zero_point,
    const std::optional<int32_t> axis,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    const ttnn::DataType b_dtype = out_scale.get_dtype();

    TT_FATAL(tt::tt_metal::is_floating_point(b_dtype), "Requantize only takes floating-point number scales");
    TT_FATAL(!tt::tt_metal::is_block_float(b_dtype), "Unsupported scale tensor format");
    TT_FATAL(in_scale.get_logical_volume() == 1u, "Per-tensor requantize only takes a single input scale value");
    TT_FATAL(out_scale.get_logical_volume() == 1u, "Per-tensor requantize only takes a single output scale value");

    const float out_scale_scalar = [](const DataType b_dtype, const Tensor& out_scale) {
        if (b_dtype == DataType::FLOAT32) {
            return out_scale.to_vector<float>()[0];
        } else {
            return out_scale.to_vector<bfloat16>()[0].to_float();
        }
    }(b_dtype, out_scale);

    return invoke(
        queue_id,
        input_tensor,
        in_scale,
        in_zero_point,
        out_scale_scalar,
        out_zero_point,
        axis,
        output_dtype,
        memory_config,
        optional_output_tensor);
}

Tensor DequantOp::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const float scale,
    const int32_t zero_point,
    const std::optional<int32_t> axis,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    const ttnn::DataType a_dtype = input_tensor.get_dtype();
    ttnn::DataType c_dtype = ttnn::DataType::BFLOAT16;
    if (output_dtype.has_value() && optional_output_tensor.has_value()) {
        TT_FATAL(
            output_dtype.value() == optional_output_tensor->dtype(),
            "Mismatching output_dtype and output tensor dtype");
        c_dtype = output_dtype.value();
    } else if (output_dtype.has_value()) {
        c_dtype = output_dtype.value();
    } else if (optional_output_tensor.has_value()) {
        c_dtype = optional_output_tensor->dtype();
    }
    TT_FATAL(a_dtype == ttnn::DataType::INT32, "Dequantize only supports int32 inputs for now");

    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations{};
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations{};
    // LLK dequant kernel does addition, so we need to negate zero_point
    const std::array post_activations{
        ttnn::operations::unary::UnaryWithParam{unary::UnaryOpType::ZERO_POINT, static_cast<float>(-zero_point)}};

    return ttnn::prim::binary_ng(
        queue_id,
        input_tensor,
        scale,
        binary_ng::BinaryOpType::DEQUANT,
        c_dtype,
        memory_config,
        optional_output_tensor,
        lhs_activations,
        rhs_activations,
        post_activations);
}

Tensor DequantOp::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const Tensor& scale,
    const int32_t zero_point,
    const std::optional<int32_t> axis,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    const ttnn::DataType a_dtype = input_tensor.get_dtype();
    const ttnn::DataType b_dtype = scale.get_dtype();
    ttnn::DataType c_dtype = ttnn::DataType::BFLOAT16;
    if (output_dtype.has_value() && optional_output_tensor.has_value()) {
        TT_FATAL(
            output_dtype.value() == optional_output_tensor->dtype(),
            "Mismatching output_dtype and output tensor dtype");
        c_dtype = output_dtype.value();
    } else if (output_dtype.has_value()) {
        c_dtype = output_dtype.value();
    } else if (optional_output_tensor.has_value()) {
        c_dtype = optional_output_tensor->dtype();
    }
    TT_FATAL(a_dtype == ttnn::DataType::INT32, "Dequantize only supports int32 inputs for now");
    TT_FATAL(tt::tt_metal::is_floating_point(b_dtype), "Quantize only takes floating-point number scales");
    TT_FATAL(!tt::tt_metal::is_block_float(b_dtype), "Unsupported scale tensor format");
    TT_FATAL(scale.get_logical_volume() == 1u, "Per-tensor dequantize only takes a single scale value");

    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations{};
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations{};
    // LLK dequant kernel does addition, so we need to negate zero_point
    const std::array post_activations{
        ttnn::operations::unary::UnaryWithParam{unary::UnaryOpType::ZERO_POINT, static_cast<float>(-zero_point)}};

    return ttnn::prim::binary_ng(
        queue_id,
        input_tensor,
        scale,
        binary_ng::BinaryOpType::DEQUANT,
        c_dtype,
        memory_config,
        optional_output_tensor,
        lhs_activations,
        rhs_activations,
        post_activations);
}

}  // namespace ttnn::operations::quantization
