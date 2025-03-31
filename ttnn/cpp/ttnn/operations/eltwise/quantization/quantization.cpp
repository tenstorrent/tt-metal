// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "quantization.hpp"
#include "ttnn/operations/copy.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"

ttnn::DataType get_output_dtype(
    const std::optional<const ttnn::DataType>& output_dtype,
    const std::optional<ttnn::Tensor>& output_tensor,
    const ttnn::DataType default_dtype) {
    if (output_dtype.has_value() && output_tensor.has_value()) {
        TT_FATAL(output_dtype.value() == output_tensor->dtype(), "Mismatching output_dtype and output tensor dtype");
        return output_dtype.value();
    } else if (output_dtype.has_value()) {
        return output_dtype.value();
    } else if (output_tensor.has_value()) {
        return output_tensor->dtype();
    } else {
        return default_dtype;
    }
}

void check_scale_tensor_dtype(const ttnn::Tensor& scale) {
    const auto dtype = scale.get_dtype();
    TT_FATAL(tt::tt_metal::is_floating_point(dtype), "Quantization only takes floating-point number scales");
    TT_FATAL(!tt::tt_metal::is_block_float(dtype), "Unsupported quantization scale data type");
    TT_FATAL(scale.get_logical_volume() == 1u, "Per-tensor quantization only takes scalar-tensor scales");
}

namespace ttnn::operations::quantization {

Tensor QuantOp::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const std::variant<Tensor, float>& scale,
    const int32_t zero_point,
    const std::optional<int32_t> axis,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    const DataType a_dtype = input_tensor.get_dtype();
    constexpr DataType c_dtype = DataType::INT32;

    TT_FATAL(tt::tt_metal::is_floating_point(a_dtype), "Quantize only takes floating-point number inputs");
    TT_FATAL(output_dtype.value_or(c_dtype) == c_dtype, "Quantize only supports int32 outputs for now");
    if (optional_output_tensor.has_value()) {
        TT_FATAL(optional_output_tensor->dtype() == c_dtype, "Quantize only supports int32 outputs for now");
    }

    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations{};
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations{};
    const std::array post_activations{
        ttnn::operations::unary::UnaryWithParam{unary::UnaryOpType::ZERO_POINT, static_cast<float>(zero_point)}};

    return std::visit(
        [&](auto&& scale_v) -> Tensor {
            constexpr bool scale_is_scalar = std::is_same_v<float, std::decay_t<decltype(scale_v)>>;
            if constexpr (scale_is_scalar) {
                // LLK quant kernel expects the reciprocal of the actual scale to avoid doing div on the device
                return ttnn::prim::binary_ng(
                    queue_id,
                    tt::tt_metal::is_block_float(a_dtype) ? ttnn::typecast(input_tensor, DataType::BFLOAT16)
                                                          : input_tensor,
                    1.0f / scale_v,
                    binary_ng::BinaryOpType::QUANT,
                    c_dtype,
                    memory_config,
                    optional_output_tensor,
                    lhs_activations,
                    rhs_activations,
                    post_activations);
            } else {
                check_scale_tensor_dtype(scale_v);
                // TODO: do reciprocal using an activation containing an UnaryWithParam that does rdiv(scale, 1),
                // benchmark it
                return ttnn::prim::binary_ng(
                    queue_id,
                    tt::tt_metal::is_block_float(a_dtype) ? ttnn::typecast(input_tensor, DataType::BFLOAT16)
                                                          : input_tensor,
                    ttnn::reciprocal(scale_v),
                    binary_ng::BinaryOpType::QUANT,
                    c_dtype,
                    memory_config,
                    optional_output_tensor,
                    lhs_activations,
                    rhs_activations,
                    post_activations);
            }
        },
        scale);
}

Tensor RequantOp::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const std::variant<Tensor, float>& in_scale,
    const int32_t in_zero_point,
    const std::variant<Tensor, float>& out_scale,
    const int32_t out_zero_point,
    const std::optional<int32_t> axis,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    const DataType a_dtype = input_tensor.get_dtype();
    constexpr DataType c_dtype = DataType::INT32;

    TT_FATAL(a_dtype == DataType::INT32, "Requantize only supports int32 inputs for now");
    TT_FATAL(output_dtype.value_or(c_dtype) == c_dtype, "Requantize only supports int32 outputs for now");
    if (optional_output_tensor.has_value()) {
        TT_FATAL(optional_output_tensor->dtype() == c_dtype, "Requantize only supports int32 outputs for now");
    }

    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations{};
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations{};

    return std::visit(
        [&](auto&& in_scale_v, auto&& out_scale_v) -> Tensor {
            constexpr bool in_scale_is_scalar = std::is_same_v<float, std::decay_t<decltype(in_scale_v)>>;
            constexpr bool out_scale_is_scalar = std::is_same_v<float, std::decay_t<decltype(out_scale_v)>>;

            if constexpr (!in_scale_is_scalar) {
                check_scale_tensor_dtype(in_scale_v);
            }
            if constexpr (!out_scale_is_scalar) {
                check_scale_tensor_dtype(out_scale_v);
            }

            // Enable fast path only when both scales are scalars, otherwise fallback to a composite op
            if constexpr (in_scale_is_scalar && out_scale_is_scalar) {
                // Expansion of q' = [(q - z_in) * s_in] / s_out + z_out
                const float scale_recip = in_scale_v / out_scale_v;
                // z is passed to and consumed by the LLK as f32 anyway, might as well preserve some accuracy here
                const float zero_point = out_zero_point - in_zero_point * scale_recip;

                const std::array post_activations{
                    ttnn::operations::unary::UnaryWithParam{unary::UnaryOpType::ZERO_POINT, zero_point}};
                return ttnn::prim::binary_ng(
                    queue_id,
                    input_tensor,
                    scale_recip,
                    binary_ng::BinaryOpType::REQUANT,
                    c_dtype,
                    memory_config,
                    optional_output_tensor,
                    lhs_activations,
                    rhs_activations,
                    post_activations);
            } else {
                // The composite op fallback, very generic but uses quite a few ops, and the accuracy is not as good as
                // the fast path
                Tensor scale_recip;
                if constexpr (in_scale_is_scalar) {
                    scale_recip = ttnn::rdiv(out_scale_v, in_scale_v);
                } else {
                    scale_recip = ttnn::div(in_scale_v, out_scale_v);
                }

                const Tensor zero_point = ttnn::add(ttnn::multiply(scale_recip, -in_zero_point), out_zero_point);

                // TODO: at the time of writing, there is no easy way to do tensor i32 .* f32 -> i32/f32
                // 1. ttnn::multiply can't handle integer tensors and has implicit broadcasting bugs
                // 2. binary_ng::MUL uses FPU for everything except f32 .* f32, so it also can't handle i32 tensors
                // 3. Passing a unary TYPCAST op as an activation is buggy on both ttnn::multiply & binary_ng::MUL
                // WA: use dequant with zero-point == 0 to do i32 * f32 -> f32 on the SFPU
                const Tensor input_scaled =
                    DequantOp::invoke(queue_id, input_tensor, scale_recip, 0, std::nullopt, DataType::FLOAT32);

                // TODO: ttnn::add also has implicit broadcasting bugs, use binary_ng::ADD here
                tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> post_activations{};
                const Tensor input_shifted = ttnn::prim::binary_ng(
                    queue_id,
                    input_scaled,
                    zero_point,
                    binary_ng::BinaryOpType::ADD,
                    DataType::FLOAT32,
                    std::nullopt,
                    std::nullopt,
                    lhs_activations,
                    rhs_activations,
                    post_activations);
                return ttnn::typecast(input_shifted, c_dtype);
            }
        },
        in_scale,
        out_scale);
}

Tensor DequantOp::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const std::variant<Tensor, float>& scale,
    const int32_t zero_point,
    const std::optional<int32_t> axis,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    const DataType a_dtype = input_tensor.get_dtype();
    const DataType c_dtype = get_output_dtype(output_dtype, optional_output_tensor, DataType::BFLOAT16);

    TT_FATAL(a_dtype == DataType::INT32, "Dequantize only supports int32 inputs for now");
    TT_FATAL(
        c_dtype == DataType::FLOAT32 || c_dtype == DataType::BFLOAT16,
        "Dequantize only supports bf16/f32 outputs for now");

    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations{};
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations{};
    // LLK dequant kernel does addition, so we need to negate zero_point
    const std::array post_activations{
        ttnn::operations::unary::UnaryWithParam{unary::UnaryOpType::ZERO_POINT, static_cast<float>(-zero_point)}};

    return std::visit(
        [&](auto&& scale_v) -> Tensor {
            constexpr bool scale_is_scalar = std::is_same_v<float, std::decay_t<decltype(scale_v)>>;
            if constexpr (scale_is_scalar) {
                return ttnn::prim::binary_ng(
                    queue_id,
                    input_tensor,
                    scale_v,
                    binary_ng::BinaryOpType::DEQUANT,
                    c_dtype,
                    memory_config,
                    optional_output_tensor,
                    lhs_activations,
                    rhs_activations,
                    post_activations);
            } else {
                check_scale_tensor_dtype(scale_v);
                return ttnn::prim::binary_ng(
                    queue_id,
                    input_tensor,
                    scale_v,
                    binary_ng::BinaryOpType::DEQUANT,
                    c_dtype,
                    memory_config,
                    optional_output_tensor,
                    lhs_activations,
                    rhs_activations,
                    post_activations);
            }
        },
        scale);
}

}  // namespace ttnn::operations::quantization
