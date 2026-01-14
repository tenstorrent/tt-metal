// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "quantization.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.hpp"

#include <cassert>

#include <tt_stl/overloaded.hpp>

/*
 * TODO: Improve the composite op fallback and/or make binary eltwise ops more flexible.
 *
 * At the time of writing, the binary eltwise op framework that this implementation rely on is not
 * flexible enough to handle all the needs of the quantization ops:
 * 0. The only supported integer type is i32.
 * 1. Implicit broadcasting is only well supported in BinaryNg (so we pass use_legacy=false).
 * 2. Support for mixed data type A op B -> C is incomplete, especially when the data types have
 *    different sizes (e.g. bf16 .* f32 -> i32) and/or when implicit broadcasting is involved.
 * 3. There're some issues in the activation mechanism, which could have been used to do reciprocal
 *    and typecast ops to simplify the workflow.
 *
 * Therefore, the composite op fallback paths explicitly call multiple BinaryNg ops and do lots of
 * typecasting, reducing accuracy/performance/maintainability. The user will be surprised when the
 * op is suddenly 2-3x slower just because the scale and/or zero-point is passed as a scalar-tensor.
 *
 * Possible improvements:
 * 0. Support more integer types (i8/u8/i16/u16), the packer & unpacker are fully capable, just not
 *    programmed to do so yet.
 * 1. Improve BinaryNg's activation mechanism so we can pass the unary TYPECAST/RECIP as activations
 *    to simplify the composite ops.
 * 2. Improve BinaryNg's mixed datatype support so we can remove typecasts all together.
 * 3. Let TR0 get the scalar value of the scale/zero-points from the scalar-tensor to avoid the
 *    composite op fallback all together (but too specific to quantization?).
 * 4. Per-channel quantization ops probably need their own set of LLKs and support from BinaryNg,
 *    or maybe it's better to handle them as ternary ops?
 */

namespace {

ttnn::DataType get_output_dtype(
    const std::optional<const ttnn::DataType>& output_dtype,
    const std::optional<ttnn::Tensor>& output_tensor,
    const ttnn::DataType default_dtype) {
    if (output_dtype.has_value() && output_tensor.has_value()) {
        TT_FATAL(output_dtype.value() == output_tensor->dtype(), "Mismatching output_dtype and output tensor dtype");
        return output_dtype.value();
    }
    if (output_dtype.has_value()) {
        return output_dtype.value();
    }
    if (output_tensor.has_value()) {
        return output_tensor->dtype();
    }
    return default_dtype;
}

void check_per_tensor_scale(const ttnn::Tensor& scale) {
    const auto dtype = scale.dtype();
    TT_FATAL(tt::tt_metal::is_floating_point(dtype), "Quantization only takes floating-point number scales");
    TT_FATAL(!tt::tt_metal::is_block_float(dtype), "Unsupported quantization scale data type");
    TT_FATAL(scale.logical_volume() == 1u, "Per-tensor quantization only takes scalar-tensor scales");
}

// Explicitly delete variant overloads to prevent misusage
template <typename... Ts>
void check_per_tensor_scale(const std::variant<Ts...>&) = delete;

// Ignore all other types of inputs
template <typename T>
void check_per_tensor_scale(const T&) {}

void check_per_tensor_zero_point(const ttnn::Tensor& zero_point) {
    const auto dtype = zero_point.dtype();
    TT_FATAL(dtype == ttnn::DataType::INT32, "Quantization only takes int32 zero-points for now");
    TT_FATAL(zero_point.logical_volume() == 1u, "Per-tensor quantization only takes scalar-tensor zero-points");
}

template <typename... Ts>
void check_per_tensor_zero_point(const std::variant<Ts...>&) = delete;

template <typename T>
void check_per_tensor_zero_point(const T&) {}

void check_per_channel_tensor_args(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor* scale_p,
    const ttnn::Tensor* zero_point_p,
    const int32_t axis,
    const int32_t rank) {
    TT_FATAL(
        scale_p != nullptr && scale_p->logical_shape().rank() == 1,
        "Per-channel quantization expects 1D scale tensors");
    TT_FATAL(
        zero_point_p != nullptr && zero_point_p->logical_shape().rank() == 1,
        "Per-channel quantization expects 1D zero-point tensors");
    TT_FATAL(
        scale_p->logical_shape() == zero_point_p->logical_shape(),
        "Per-channel quantization expects scale & zero-point tensors of matching shapes");
    TT_FATAL(axis >= -rank && axis < rank, "Axis {} is outside the range [{}, {}]", axis, -rank, rank - 1);
    TT_FATAL(
        input_tensor.logical_shape()[axis] == scale_p->logical_volume(),
        "Size of the scale tensor doesn't match the size of the input tensor along the given axis");
    TT_FATAL(
        input_tensor.logical_shape()[axis] == zero_point_p->logical_volume(),
        "Size of the zero-point tensor doesn't match the size of the input tensor along the given axis");

    const auto scale_dtype = scale_p->dtype();
    TT_FATAL(tt::tt_metal::is_floating_point(scale_dtype), "Quantization only takes floating-point number scales");
    TT_FATAL(!tt::tt_metal::is_block_float(scale_dtype), "Unsupported quantization scale data type");

    const auto zero_point_dtype = zero_point_p->dtype();
    TT_FATAL(zero_point_dtype == ttnn::DataType::INT32, "Quantization only takes int32 zero-points for now");
}

void check_scale_tensor_args(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor* scale_p,
    const int32_t axis,
    const int32_t rank,
    bool is_per_channel) {
    TT_FATAL(scale_p != nullptr, "Scale tensor cannot be null");
    TT_FATAL(axis >= -rank && axis < rank, "Axis {} is outside the range [{}, {}]", axis, -rank, rank - 1);

    const auto scale_dtype = scale_p->dtype();
    TT_FATAL(tt::tt_metal::is_floating_point(scale_dtype), "Quantization only takes floating-point number scales");
    TT_FATAL(!tt::tt_metal::is_block_float(scale_dtype), "Unsupported quantization scale data type");

    if (is_per_channel) {
        TT_FATAL(scale_p->logical_shape().rank() == 1, "Per-channel quantization expects 1D scale tensors");
        TT_FATAL(
            input_tensor.logical_shape()[axis] == scale_p->logical_volume(),
            "Size of the scale tensor doesn't match the size of the input tensor along the given axis");
    } else {
        TT_FATAL(scale_p->logical_volume() == 1u, "Per-tensor quantization only takes scalar-tensor scales");
    }
}

void check_zero_point_tensor_args(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor* zero_point_p,
    const int32_t axis,
    const int32_t rank,
    bool is_per_channel) {
    TT_FATAL(zero_point_p != nullptr, "Zero-point tensor cannot be null");
    TT_FATAL(axis >= -rank && axis < rank, "Axis {} is outside the range [{}, {}]", axis, -rank, rank - 1);

    const auto zero_point_dtype = zero_point_p->dtype();
    TT_FATAL(zero_point_dtype == ttnn::DataType::INT32, "Quantization only takes int32 zero-points for now");

    if (is_per_channel) {
        TT_FATAL(zero_point_p->logical_shape().rank() == 1, "Per-channel quantization expects 1D zero-point tensors");
        TT_FATAL(
            input_tensor.logical_shape()[axis] == zero_point_p->logical_volume(),
            "Size of the zero-point tensor doesn't match the size of the input tensor along the given axis");
    } else {
        TT_FATAL(zero_point_p->logical_volume() == 1u, "Per-tensor quantization only takes scalar-tensor zero-points");
    }
}

ttnn::Tensor reshape_per_channel_vector_args(
    const ttnn::Tensor& vector, ttnn::Shape tensor_shape, const int32_t axis, const ttnn::DataType out_dtype) {
    // This function is internal use only, use asserts instead of TT_FATAL to convey intented usage
    const int32_t rank = static_cast<int32_t>(tensor_shape.rank());
    assert(axis >= -rank && axis < rank);
    assert(vector.logical_shape().rank() == 1);
    assert(vector.logical_volume() == tensor_shape[axis]);
    const int32_t axis_normalized = (axis + rank) % rank;
    for (int32_t i = 0; i < rank; i++) {
        if (i != axis_normalized) {
            tensor_shape[i] = 1;
        }
    }
    const ttnn::Tensor result = ttnn::reshape(ttnn::typecast(vector, out_dtype), tensor_shape);
    assert(result.logical_shape().rank() == rank);
    assert(result.logical_shape()[axis] == vector.logical_volume());
    assert(result.logical_volume() == vector.logical_volume());
    return result;
}

}  // anonymous namespace

namespace ttnn::operations::quantization {

Tensor QuantOp::invoke(
    const Tensor& input_tensor,
    const std::variant<Tensor, float>& scale,
    const std::variant<Tensor, int32_t>& zero_point,
    const std::optional<int32_t> axis,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    const Tensor input_a = tt::tt_metal::is_block_float(input_tensor.dtype())
                               ? ttnn::typecast(input_tensor, DataType::BFLOAT16)
                               : input_tensor;

    const DataType a_dtype = input_a.dtype();
    constexpr DataType c_dtype = DataType::INT32;

    TT_FATAL(tt::tt_metal::is_floating_point(a_dtype), "Quantize only takes floating-point number inputs");
    TT_FATAL(output_dtype.value_or(c_dtype) == c_dtype, "Quantize only supports int32 outputs for now");
    if (optional_output_tensor.has_value()) {
        TT_FATAL(optional_output_tensor->dtype() == c_dtype, "Quantize only supports int32 outputs for now");
    }

    constexpr tt::stl::Span<const unary::EltwiseUnaryWithParam> none{};

    const bool is_per_channel = axis.has_value();
    if (is_per_channel) {
        const Tensor* scale_p = std::get_if<Tensor>(&scale);
        const Tensor* zero_point_p = std::get_if<Tensor>(&zero_point);

        const int32_t axis_v = axis.value();
        const ttnn::Shape& input_shape = input_a.logical_shape();

        check_per_channel_tensor_args(input_a, scale_p, zero_point_p, axis_v, input_shape.rank());

        const Tensor scale_full = reshape_per_channel_vector_args(*scale_p, input_shape, axis_v, a_dtype);
        const Tensor zero_point_full = reshape_per_channel_vector_args(*zero_point_p, input_shape, axis_v, a_dtype);
        const Tensor input_scaled =
            ttnn::divide(input_a, scale_full, a_dtype, std::nullopt, std::nullopt, none, none, none, false);
        return ttnn::typecast(
            ttnn::add(
                input_scaled,
                zero_point_full,
                std::nullopt,
                memory_config,
                optional_output_tensor,
                none,
                none,
                none,
                false),
            c_dtype);
    }

    return std::visit(
        tt::stl::overloaded{
            [&](const float scale, const int32_t zero_point) {
                const std::array post_activation{
                    unary::EltwiseUnaryWithParam{unary::UnaryOpType::ZERO_POINT, static_cast<float>(zero_point)}};

                // LLK quant kernel expects the reciprocal of the actual scale to avoid doing div on the device
                return ttnn::prim::binary_ng(
                    input_a,
                    1.0f / scale,
                    binary::BinaryOpType::QUANT,
                    c_dtype,
                    memory_config,
                    optional_output_tensor,
                    /*fast_and_approximate_mode*/ false,
                    none,
                    none,
                    post_activation,
                    std::nullopt);
            },
            [&](const Tensor& scale, const int32_t zero_point) {
                check_per_tensor_scale(scale);
                const std::array post_activation{
                    unary::EltwiseUnaryWithParam{unary::UnaryOpType::ZERO_POINT, static_cast<float>(zero_point)}};

                return ttnn::prim::binary_ng(
                    input_a,
                    ttnn::reciprocal(scale),
                    binary::BinaryOpType::QUANT,
                    c_dtype,
                    memory_config,
                    optional_output_tensor,
                    /*fast_and_approximate_mode*/ false,
                    none,
                    none,
                    post_activation,
                    std::nullopt);
            },
            [&](const float scale, const Tensor& zero_point) {
                check_per_tensor_zero_point(zero_point);
                const Tensor input_scaled =
                    ttnn::divide(input_a, scale, std::nullopt, std::nullopt, std::nullopt, none, none, none, false);
                return ttnn::typecast(
                    ttnn::add(
                        input_scaled,
                        zero_point.dtype() == a_dtype ? zero_point : ttnn::typecast(zero_point, a_dtype),
                        a_dtype,
                        std::nullopt,
                        std::nullopt,
                        none,
                        none,
                        none,
                        false),
                    c_dtype);
            },
            [&](const Tensor& scale, const Tensor& zero_point) {
                check_per_tensor_scale(scale);
                check_per_tensor_zero_point(zero_point);
                const Tensor input_scaled = ttnn::divide(
                    input_a,
                    scale.dtype() == a_dtype ? scale : ttnn::typecast(scale, a_dtype),
                    a_dtype,
                    std::nullopt,
                    std::nullopt,
                    none,
                    none,
                    none,
                    false);
                return ttnn::typecast(
                    ttnn::add(
                        input_scaled,
                        zero_point.dtype() == a_dtype ? zero_point : ttnn::typecast(zero_point, a_dtype),
                        a_dtype,
                        std::nullopt,
                        std::nullopt,
                        none,
                        none,
                        none,
                        false),
                    c_dtype);
            }},
        scale,
        zero_point);
}

Tensor RequantOp::invoke(
    const Tensor& input_tensor,
    const std::variant<Tensor, float>& in_scale,
    const std::variant<Tensor, int32_t>& in_zero_point,
    const std::variant<Tensor, float>& out_scale,
    const std::variant<Tensor, int32_t>& out_zero_point,
    const std::optional<int32_t> axis,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    const DataType a_dtype = input_tensor.dtype();
    constexpr DataType c_dtype = DataType::INT32;

    TT_FATAL(a_dtype == DataType::INT32, "Requantize only supports int32 inputs for now");
    TT_FATAL(output_dtype.value_or(c_dtype) == c_dtype, "Requantize only supports int32 outputs for now");
    if (optional_output_tensor.has_value()) {
        TT_FATAL(optional_output_tensor->dtype() == c_dtype, "Requantize only supports int32 outputs for now");
    }

    constexpr tt::stl::Span<const unary::EltwiseUnaryWithParam> none{};

    const bool has_axis = axis.has_value();

    const Tensor* in_scale_p = std::get_if<Tensor>(&in_scale);
    const Tensor* in_zero_point_p = std::get_if<Tensor>(&in_zero_point);
    const Tensor* out_scale_p = std::get_if<Tensor>(&out_scale);
    const Tensor* out_zero_point_p = std::get_if<Tensor>(&out_zero_point);

    // Use the fused per-channel tensor path when axis is provided AND all params are tensors.
    // Mixed scalar/tensor cases will fall through to composite fallback.
    // NOTE: Benchmarks show this path is not faster; it's selected
    //       for simpler shape handling (explicit per-channel expansion).
    const bool all_params_are_tensors = in_scale_p && in_zero_point_p && out_scale_p && out_zero_point_p;

    if (has_axis && all_params_are_tensors) {
        const int32_t axis_v = axis.value();
        const ttnn::Shape& input_shape = input_tensor.logical_shape();
        const int32_t rank = input_shape.rank();

        const bool in_scale_is_full_size = in_scale_p->logical_volume() == input_shape[axis_v];
        const bool in_zero_point_is_full_size = in_zero_point_p->logical_volume() == input_shape[axis_v];
        const bool out_scale_is_full_size = out_scale_p->logical_volume() == input_shape[axis_v];
        const bool out_zero_point_is_full_size = out_zero_point_p->logical_volume() == input_shape[axis_v];

        TT_FATAL(
            in_scale_is_full_size == in_zero_point_is_full_size,
            "Input scale and input zero-point must both be per-channel or both be per-tensor, but got: "
            "input scale {} per-channel, input zero-point {} per-channel",
            in_scale_is_full_size ? "is" : "is not",
            in_zero_point_is_full_size ? "is" : "is not");

        TT_FATAL(
            out_scale_is_full_size == out_zero_point_is_full_size,
            "Output scale and output zero-point must both be per-channel or both be per-tensor, but got: "
            "output scale {} per-channel, output zero-point {} per-channel",
            out_scale_is_full_size ? "is" : "is not",
            out_zero_point_is_full_size ? "is" : "is not");

        // Validate tensor shapes.
        check_scale_tensor_args(input_tensor, in_scale_p, axis_v, rank, in_scale_is_full_size);
        check_zero_point_tensor_args(input_tensor, in_zero_point_p, axis_v, rank, in_zero_point_is_full_size);
        check_scale_tensor_args(input_tensor, out_scale_p, axis_v, rank, out_scale_is_full_size);
        check_zero_point_tensor_args(input_tensor, out_zero_point_p, axis_v, rank, out_zero_point_is_full_size);

        // Shape expansion and typecasting for the scale and zero-point tensors.
        auto expand_or_cast = [&](const Tensor& v, bool is_full_size, DataType dt) -> Tensor {
            return is_full_size ? reshape_per_channel_vector_args(v, input_shape, axis_v, dt) : ttnn::typecast(v, dt);
        };

        const Tensor in_scale_full = expand_or_cast(*in_scale_p, in_scale_is_full_size, DataType::FLOAT32);
        const Tensor in_zero_point_full =
            expand_or_cast(*in_zero_point_p, in_zero_point_is_full_size, DataType::FLOAT32);
        const Tensor out_scale_full = expand_or_cast(*out_scale_p, out_scale_is_full_size, DataType::FLOAT32);
        const Tensor out_zero_point_full =
            expand_or_cast(*out_zero_point_p, out_zero_point_is_full_size, DataType::FLOAT32);

        const Tensor scale_recip_full = ttnn::divide(
            in_scale_full, out_scale_full, std::nullopt, std::nullopt, std::nullopt, none, none, none, false);
        const Tensor in_zero_point_scaled_full = ttnn::multiply(
            in_zero_point_full, scale_recip_full, std::nullopt, std::nullopt, std::nullopt, none, none, none, false);
        const Tensor zero_point_full = ttnn::subtract(
            out_zero_point_full,
            in_zero_point_scaled_full,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            none,
            none,
            none,
            false);

        const Tensor input_scaled = ttnn::multiply(
            ttnn::typecast(input_tensor, DataType::FLOAT32),
            scale_recip_full,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            none,
            none,
            none,
            false);
        return ttnn::typecast(
            ttnn::add(
                input_scaled,
                zero_point_full,
                std::nullopt,
                memory_config,
                optional_output_tensor,
                none,
                none,
                none,
                false),
            c_dtype);
    }

    return std::visit(
        tt::stl::overloaded{
            // Enable fast path for all scalar scales & zero-points, fallback to composite ops otherwise.
            [&](const float in_scale,
                const int32_t in_zero_point,
                const float out_scale,
                const int32_t out_zero_point) {
                // Expansion of q' = [(q - z_in) * s_in] / s_out + z_out
                const float scale_recip = in_scale / out_scale;
                // z is passed to and consumed by the LLK as f32 anyway, might as well preserve some accuracy here.
                const float zero_point = out_zero_point - (in_zero_point * scale_recip);

                const std::array post_activation{
                    unary::EltwiseUnaryWithParam{unary::UnaryOpType::ZERO_POINT, zero_point}};
                return ttnn::prim::binary_ng(
                    input_tensor,
                    scale_recip,
                    binary::BinaryOpType::REQUANT,
                    c_dtype,
                    memory_config,
                    optional_output_tensor,
                    /*fast_and_approximate_mode*/ false,
                    none,
                    none,
                    post_activation,
                    std::nullopt);
            },
            [&](const auto& in_scale, const auto& in_zero_point, const auto& out_scale, const auto& out_zero_point) {
                // Pass axis only to operations that have tensor parameters.
                constexpr bool has_tensor_in_scale = std::is_same_v<std::decay_t<decltype(in_scale)>, Tensor>;
                constexpr bool has_tensor_out_scale = std::is_same_v<std::decay_t<decltype(out_scale)>, Tensor>;

                const std::optional<int> dequant_axis = has_tensor_in_scale ? axis : std::nullopt;
                const std::optional<int> quant_axis = has_tensor_out_scale ? axis : std::nullopt;

                const Tensor dequantized = DequantOp::invoke(
                    input_tensor, in_scale, in_zero_point, dequant_axis, std::nullopt, std::nullopt, std::nullopt);
                return QuantOp::invoke(
                    dequantized, out_scale, out_zero_point, quant_axis, c_dtype, memory_config, optional_output_tensor);
            }},
        in_scale,
        in_zero_point,
        out_scale,
        out_zero_point);
}

Tensor DequantOp::invoke(
    const Tensor& input_tensor,
    const std::variant<Tensor, float>& scale,
    const std::variant<Tensor, int32_t>& zero_point,
    const std::optional<int32_t> axis,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    const DataType a_dtype = input_tensor.dtype();
    const DataType c_dtype = get_output_dtype(output_dtype, optional_output_tensor, DataType::BFLOAT16);

    TT_FATAL(a_dtype == DataType::INT32, "Dequantize only supports int32 inputs for now");
    TT_FATAL(
        c_dtype == DataType::FLOAT32 || c_dtype == DataType::BFLOAT16,
        "Dequantize only supports bf16/f32 outputs for now");

    constexpr tt::stl::Span<const unary::EltwiseUnaryWithParam> none{};

    const bool is_per_channel = axis.has_value();
    if (is_per_channel) {
        const Tensor* scale_p = std::get_if<Tensor>(&scale);
        const Tensor* zero_point_p = std::get_if<Tensor>(&zero_point);

        const int32_t axis_v = axis.value();
        const ttnn::Shape& input_shape = input_tensor.logical_shape();

        check_per_channel_tensor_args(input_tensor, scale_p, zero_point_p, axis_v, input_shape.rank());

        const Tensor scale_full = reshape_per_channel_vector_args(*scale_p, input_shape, axis_v, DataType::FLOAT32);
        const Tensor zero_point_full =
            reshape_per_channel_vector_args(*zero_point_p, input_shape, axis_v, DataType::FLOAT32);
        const Tensor input_shifted = ttnn::subtract(
            ttnn::typecast(input_tensor, DataType::FLOAT32),
            zero_point_full,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            none,
            none,
            none,
            false);
        return ttnn::typecast(
            ttnn::multiply(
                input_shifted,
                scale_full,
                std::nullopt,
                memory_config,
                optional_output_tensor,
                none,
                none,
                none,
                false),
            c_dtype);
    }

    return std::visit(
        tt::stl::overloaded{
            [&](const float scale, const int32_t zero_point) {
                // LLK dequant kernel does addition, so we need to negate zero_point
                const std::array post_activation{
                    unary::EltwiseUnaryWithParam{unary::UnaryOpType::ZERO_POINT, static_cast<float>(-zero_point)}};
                return ttnn::prim::binary_ng(
                    input_tensor,
                    scale,
                    binary::BinaryOpType::DEQUANT,
                    c_dtype,
                    memory_config,
                    optional_output_tensor,
                    /*fast_and_approximate_mode*/ false,
                    none,
                    none,
                    post_activation,
                    std::nullopt);
            },
            [&](const Tensor& scale, const int32_t zero_point) {
                check_per_tensor_scale(scale);
                const std::array post_activation{
                    unary::EltwiseUnaryWithParam{unary::UnaryOpType::ZERO_POINT, static_cast<float>(-zero_point)}};
                return ttnn::prim::binary_ng(
                    input_tensor,
                    scale,
                    binary::BinaryOpType::DEQUANT,
                    c_dtype,
                    memory_config,
                    optional_output_tensor,
                    /*fast_and_approximate_mode*/ false,
                    none,
                    none,
                    post_activation,
                    std::nullopt);
            },
            [&](const float scale, const Tensor& zero_point) {
                check_per_tensor_zero_point(zero_point);
                const Tensor input_shifted = ttnn::typecast(
                    ttnn::subtract(
                        input_tensor, zero_point, std::nullopt, std::nullopt, std::nullopt, none, none, none, false),
                    c_dtype);
                return ttnn::multiply(
                    input_shifted, scale, c_dtype, memory_config, optional_output_tensor, none, none, none, false);
            },
            [&](const Tensor& scale, const Tensor& zero_point) {
                check_per_tensor_scale(scale);
                check_per_tensor_zero_point(zero_point);
                const Tensor input_shifted = ttnn::typecast(
                    ttnn::subtract(
                        input_tensor, zero_point, std::nullopt, std::nullopt, std::nullopt, none, none, none, false),
                    c_dtype);
                return ttnn::multiply(
                    input_shifted,
                    scale.dtype() == c_dtype ? scale : ttnn::typecast(scale, c_dtype),
                    c_dtype,
                    memory_config,
                    optional_output_tensor,
                    none,
                    none,
                    none,
                    false);
            }},
        scale,
        zero_point);
}

}  // namespace ttnn::operations::quantization
