// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <reflect>

#include "ternary_composite_op.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/data_movement/copy/copy.hpp"

namespace ttnn {

namespace operations::ternary {

// addcmul(input,tensor1,tensor2,value)=input+value×tensor1×tensor2
Tensor _addcmul(
    const Tensor& input_a,
    const Tensor& input_b,
    const Tensor& input_c,
    float value,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& output_tensor) {
    TT_FATAL(
        input_a.storage_type() == StorageType::DEVICE && input_b.storage_type() == StorageType::DEVICE &&
            input_c.storage_type() == StorageType::DEVICE,
        "Ternary operation requires input tensors to be on Device.");

    // Associate as (value * input_b) * input_c, matching both the LLK kernel
    // (ckernel_sfpu_addcmul.h) and the registered golden torch.addcmul, which computes
    // self + scalar_val * t1_val * t2_val left-to-right. Scaling by `value` first (rather
    // than last) avoids spurious overflow/underflow when `value` is chosen to keep the
    // product in range (e.g. Adam's addcmul(avg_sq, grad, grad, value=1-beta2)).
    Tensor t_scaled = ttnn::multiply(input_b, value, std::nullopt, output_mem_config);
    Tensor t_factor = ttnn::multiply(t_scaled, input_c, std::nullopt, output_mem_config);
    t_scaled.deallocate();
    Tensor result = ttnn::add(input_a, t_factor, std::nullopt, output_mem_config);
    if (output_tensor.has_value()) {
        ttnn::assign(result, output_tensor.value());
        return output_tensor.value();
    }
    return result;
}

// addcdiv(input,tensor1,tensor2,value)=input+(value×tensor1)/tensor2
Tensor _addcdiv(
    const Tensor& input_a,
    const Tensor& input_b,
    const Tensor& input_c,
    float value,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& output_tensor) {
    TT_FATAL(
        input_a.storage_type() == StorageType::DEVICE && input_b.storage_type() == StorageType::DEVICE &&
            input_c.storage_type() == StorageType::DEVICE,
        "Ternary operation requires input tensors to be on Device.");

    auto write_output = [&](const Tensor& result) {
        if (!output_tensor.has_value()) {
            return result;
        }
        ttnn::assign(result, output_tensor.value());
        return output_tensor.value();
    };

    Tensor t_factor = ttnn::multiply(input_b, value, std::nullopt, output_mem_config);
    Tensor t_div = ttnn::div(t_factor, input_c, false, std::nullopt, std::nullopt, output_mem_config);
    t_factor.deallocate();

    if (t_div.dtype() != DataType::FLOAT32) {
        // For non-FP32 (BF16 / BFLOAT8_B): t_factor can underflow to 0 in BF16 when |value * input_b| is small,
        // and non-FP32 division at input_c == 0 does not preserve IEEE 0/0 -> NaN or signed-zero divisor signs.
        // Fix up the quotient t_div at input_c == 0 before adding input_a so that:
        //   1. value == 0, input_b == 0, or isnan(input_b) at input_c == 0 yields NaN,
        //   2. input_c == -0.0 flips the pole infinity sign via signbit(input_c), and
        //   3. adding input_a afterwards preserves input_a = NaN and (-inf) + (+inf) = NaN.
        const float t_nan = std::nanf("");
        if (value == 0.0f) {
            t_div = ttnn::where(ttnn::eqz(input_c, output_mem_config), t_nan, t_div, output_mem_config);
        } else {
            const float signed_inf = std::copysign(std::numeric_limits<float>::infinity(), value);
            Tensor sign_b = ttnn::sign(input_b, output_mem_config);
            Tensor c_zero_sign =
                ttnn::where(ttnn::signbit(input_c, output_mem_config), -1.0f, 1.0f, output_mem_config);
            Tensor pole_sign = ttnn::multiply(sign_b, c_zero_sign, std::nullopt, output_mem_config);
            sign_b.deallocate();
            c_zero_sign.deallocate();
            Tensor t_inf = ttnn::multiply(pole_sign, signed_inf, std::nullopt, output_mem_config);
            pole_sign.deallocate();
            Tensor is_nan_pole = ttnn::logical_or(
                ttnn::eqz(input_b, output_mem_config),
                ttnn::isnan(input_b, output_mem_config),
                std::nullopt,
                output_mem_config);
            Tensor pole_val = ttnn::where(is_nan_pole, t_nan, t_inf, output_mem_config);
            is_nan_pole.deallocate();
            t_inf.deallocate();
            t_div = ttnn::where(ttnn::eqz(input_c, output_mem_config), pole_val, t_div, output_mem_config);
            pole_val.deallocate();
        }
    }

    Tensor result = ttnn::add(input_a, t_div, std::nullopt, output_mem_config);
    t_div.deallocate();
    return write_output(result);
}

// Fallback composite implementation for lerp (with scalar weight)
// lerp(input, end, weight) = start + weight * (end - start)
Tensor _lerp_overload(
    const Tensor& input,
    const Tensor& end,
    float weight,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& output_tensor) {
    TT_FATAL(input.dtype() == end.dtype(), "Expected the same dtype as start (input), for end");

    // Validate output tensor dtype if provided
    if (output_tensor.has_value()) {
        auto input_dtype = input.dtype();
        auto output_dtype = output_tensor->dtype();

        bool valid_dtype = (output_dtype == input_dtype) ||
                           (output_dtype == DataType::FLOAT32 && input_dtype == DataType::BFLOAT16) ||
                           (output_dtype == DataType::BFLOAT16 && input_dtype == DataType::FLOAT32);

        TT_FATAL(
            valid_dtype,
            "Output tensor dtype must match input dtype, or be FLOAT32 when inputs are BFLOAT16, or be BFLOAT16 when "
            "inputs are FLOAT32. "
            "Got input dtype: {}, output dtype: {}",
            input_dtype,
            output_dtype);
    }

    Tensor t_diff = ttnn::subtract(end, input, std::nullopt, output_mem_config);
    Tensor t_mul = ttnn::multiply(t_diff, weight, std::nullopt, output_mem_config);
    Tensor result = ttnn::add(input, t_mul, std::nullopt, output_mem_config);

    // Handle optional output tensor
    if (output_tensor.has_value()) {
        ttnn::assign(result, output_tensor.value());
        return output_tensor.value();
    }

    return result;
}

// LLK implementation for lerp (with tensor weight)
Tensor _lerp(
    const Tensor& input,
    const Tensor& end,
    const Tensor& weight,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& output_tensor) {
    TT_FATAL(input.dtype() == end.dtype(), "Expected the same dtype as start (input), for end");
    TT_FATAL(input.dtype() == weight.dtype(), "Expected the same dtype as start (input), for weight");

    // Validate output tensor dtype if provided
    if (output_tensor.has_value()) {
        auto input_dtype = input.dtype();
        auto output_dtype = output_tensor->dtype();

        bool valid_dtype = (output_dtype == input_dtype) ||
                           (output_dtype == DataType::FLOAT32 && input_dtype == DataType::BFLOAT16) ||
                           (output_dtype == DataType::BFLOAT16 && input_dtype == DataType::FLOAT32);

        TT_FATAL(
            valid_dtype,
            "Output tensor dtype must match input dtype, or be FLOAT32 when inputs are BFLOAT16, or be BFLOAT16 when "
            "inputs are FLOAT32. "
            "Got input dtype: {}, output dtype: {}",
            input_dtype,
            output_dtype);
    }

    Tensor t_diff = ttnn::multiply(
        ttnn::subtract(end, input, std::nullopt, output_mem_config), weight, std::nullopt, output_mem_config);
    Tensor result = ttnn::add(input, t_diff, std::nullopt, output_mem_config);

    // Handle optional output tensor
    if (output_tensor.has_value()) {
        ttnn::assign(result, output_tensor.value());
        return output_tensor.value();
    }

    return result;
}

}  // namespace operations::ternary

// Fallback composite implementations for mac: y = a * b + c.
// Used when the native LLK path cannot serve the request (unsupported broadcast,
// block-float subtile broadcast, or integer dtypes, which the SFPU mac path does
// not handle - it computes in the FP32 SFPU accumulator).
Tensor _mac(const Tensor& a, const Tensor& b, const Tensor& c, const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::add(ttnn::multiply(a, b, std::nullopt, output_mem_config), c, std::nullopt, output_mem_config);
}

// TTS: a * b + scalar
Tensor _mac(const Tensor& a, const Tensor& b, float c, const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::add(ttnn::multiply(a, b, std::nullopt, output_mem_config), c, std::nullopt, output_mem_config);
}

// TST: a * scalar + c
Tensor _mac(const Tensor& a, float b, const Tensor& c, const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::add(ttnn::multiply(a, b, std::nullopt, output_mem_config), c, std::nullopt, output_mem_config);
}

// TSS: a * scalar1 + scalar2
Tensor _mac(const Tensor& a, float b, float c, const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::add(ttnn::multiply(a, b, std::nullopt, output_mem_config), c, std::nullopt, output_mem_config);
}

}  // namespace ttnn
