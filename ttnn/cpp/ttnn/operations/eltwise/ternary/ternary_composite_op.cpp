// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ternary_composite_op.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/data_movement/copy/copy.hpp"

namespace ttnn::operations::ternary {

// addcmul(input,tensor1,tensor2,value)=input+value×tensor1×tensor2
Tensor _addcmul(
    const Tensor& input_a,
    const Tensor& input_b,
    const Tensor& input_c,
    float value,
    const std::optional<MemoryConfig>& output_mem_config) {
    TT_FATAL(
        input_a.storage_type() == StorageType::DEVICE && input_b.storage_type() == StorageType::DEVICE &&
            input_c.storage_type() == StorageType::DEVICE,
        "Ternary operation requires input tensors to be on Device.");

    Tensor t_mul = ttnn::multiply(input_b, input_c, std::nullopt, output_mem_config);
    Tensor t_factor = ttnn::multiply(t_mul, value, std::nullopt, output_mem_config);
    t_mul.deallocate();
    Tensor result = ttnn::add(input_a, t_factor, std::nullopt, output_mem_config);
    return result;
}

// addcdiv(input,tensor1,tensor2,value)=input+(value×tensor1)/tensor2
Tensor _addcdiv(
    const Tensor& input_a,
    const Tensor& input_b,
    const Tensor& input_c,
    float value,
    const std::optional<MemoryConfig>& output_mem_config) {
    TT_FATAL(
        input_a.storage_type() == StorageType::DEVICE && input_b.storage_type() == StorageType::DEVICE &&
            input_c.storage_type() == StorageType::DEVICE,
        "Ternary operation requires input tensors to be on Device.");

    Tensor t_factor = ttnn::multiply(input_b, value, std::nullopt, output_mem_config);
    Tensor t_div = ttnn::div(t_factor, input_c, false, std::nullopt, std::nullopt, output_mem_config);
    Tensor result = ttnn::add(input_a, t_div, std::nullopt, output_mem_config);

    if (result.dtype() == DataType::FLOAT32) {
        return result;
    }

    // For non-FP32: 0.5 * inf != inf but 1.7014e+3 and 0/0 = 0
    // To match Torch behavior after golden normalization, we explicitly
    // inject an "infinite term" whenever input_c == 0:
    //   (value * input_b) / 0  ->  sign(input_b) * (value * inf)
    // If input_b == 0 and input_c ==0: force sign = +1 so the result becomes (value * inf)
    // Note: We intentionally avoid using sign(t_factor) here, since (value * input_b) can underflow to 0 in BF16.
    const float signed_inf = value * std::numeric_limits<float>::infinity();  // value * torch.tensor(float("inf"))
    Tensor sign_b = ttnn::sign(input_b, output_mem_config);
    Tensor sign_or_one = ttnn::where(ttnn::eqz(input_b, output_mem_config), 1.0f, sign_b, output_mem_config);
    Tensor t_inf = ttnn::multiply(sign_or_one, signed_inf, std::nullopt, output_mem_config);
    result = ttnn::where(ttnn::eqz(input_c, output_mem_config), t_inf, result, output_mem_config);
    return result;
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

// Function: MAC
// compute multiply-accumulate: y = a * b + c,  over various 8 combinations of a, b, c
// being a scalar or tensor
Tensor _mac(const Tensor& a, const Tensor& b, const Tensor& c, const std::optional<MemoryConfig>& output_mem_config) {
    bool a_is_scalar = a.is_scalar();
    bool b_is_scalar = b.is_scalar();
    bool c_is_scalar = c.is_scalar();

    // When 'a' is a tensor, compute a * b + c regardless of whether b and c are scalars or tensors
    if (!a_is_scalar) {
        return ttnn::add(ttnn::multiply(a, b, std::nullopt, output_mem_config), c, std::nullopt, output_mem_config);
    }
    if (a_is_scalar && !b_is_scalar) {
        // a - scalar, b - tensor, c - scalar or tensor
        return ttnn::add(ttnn::multiply(b, a, std::nullopt, output_mem_config), c, std::nullopt, output_mem_config);
    }
    if (a_is_scalar && b_is_scalar && !c_is_scalar) {
        // a - scalar, b - scalar, c - is tensor
        return ttnn::add(c, ttnn::multiply(a, b, std::nullopt, output_mem_config), std::nullopt, output_mem_config);
    }

    // all scalars
    // a - scalar, b - scalar, c - is scalar
    TT_ASSERT(a_is_scalar && b_is_scalar && c_is_scalar);
    return ttnn::add(ttnn::multiply(a, b), c);
}

// y = a * b + c
Tensor _mac_overload(const Tensor& a, float b, float c, const std::optional<MemoryConfig>& output_mem_config) {
    return ttnn::add(ttnn::multiply(a, b, std::nullopt, output_mem_config), c, std::nullopt, output_mem_config);
}

}  // namespace ttnn::operations::ternary
