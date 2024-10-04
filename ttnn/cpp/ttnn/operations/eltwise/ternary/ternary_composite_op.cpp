// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "ternary_composite_op.hpp"
#include "ttnn/operations/creation.hpp"

namespace ttnn::operations::ternary{

// addcmul(input,tensor1,tensor2,value)=input+value×tensor1×tensor2
Tensor _addcmul(
    const Tensor& input_a,
    const Tensor& input_b,
    const Tensor& input_c,
    float value,
    const std::optional<MemoryConfig>& output_mem_config) {
    Tensor t_mul = ttnn::multiply(input_b, input_c, std::nullopt, output_mem_config);
    Tensor t_factor = ttnn::multiply(t_mul, value, std::nullopt, output_mem_config);
    t_mul.deallocate();
    Tensor result = ttnn::add(input_a, t_factor, std::nullopt, output_mem_config);
    return result;
}

// addcdiv(input,tensor1,tensor2,value)=input+value×tensor1/tensor2
Tensor _addcdiv(
    const Tensor& input_a,
    const Tensor& input_b,
    const Tensor& input_c,
    float value,
    const std::optional<MemoryConfig>& output_mem_config) {
    Tensor t_div = ttnn::multiply(input_b, ttnn::reciprocal(input_c, output_mem_config), std::nullopt, output_mem_config);
    Tensor t_factor = ttnn::multiply(t_div, value, std::nullopt, output_mem_config);
    t_div.deallocate();
    Tensor result = ttnn::add(input_a, t_factor, std::nullopt, output_mem_config);
    Tensor t_inf = ttnn::full_like(input_a, std::numeric_limits<float>::infinity());
    Tensor t_nan = ttnn::full_like(input_a, std::nanf(""));
    return ttnn::where(
        ttnn::eqz(input_c, output_mem_config),
        (value == 0) ? t_nan
                     : ttnn::where(
                           ttnn::eqz(input_b, output_mem_config),
                           t_nan,
                           ttnn::multiply(t_inf, ttnn::sign(input_b, output_mem_config), std::nullopt, output_mem_config)),
        result,
        output_mem_config);
}

// lerp(input, end, weight) = start   weight * (end - start)
Tensor _lerp_overload(const Tensor& input_a, const Tensor& input_b, float value, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor t_diff = ttnn::subtract(input_b, input_a, std::nullopt, output_mem_config);
    Tensor t_mul = ttnn::multiply(t_diff, value, std::nullopt, output_mem_config);
    Tensor result = ttnn::add(input_a, t_mul, std::nullopt, output_mem_config);
    return result;
}

Tensor _lerp(const Tensor& input_a, const Tensor& input_b, const Tensor& input_c, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor t_diff = ttnn::multiply(
        ttnn::subtract(input_b, input_a, std::nullopt, output_mem_config), input_c, std::nullopt, output_mem_config);
    Tensor result = ttnn::add(input_a, t_diff, std::nullopt, output_mem_config);
    return result;
}

// Function: MAC
// compute multiply-accumulate: y = a * b + c,  over various 8 combinations of a, b, c
// being a scalar or tensor
Tensor _mac(const Tensor& a, const Tensor& b, const Tensor& c, const std::optional<MemoryConfig>& output_mem_config) {
    bool a_is_scalar = a.is_scalar();
    bool b_is_scalar = b.is_scalar();
    bool c_is_scalar = c.is_scalar();

    if (!a_is_scalar && !b_is_scalar && !c_is_scalar) {
        // all tensors
        return ttnn::add(ttnn::multiply(a, b, std::nullopt, output_mem_config), c, std::nullopt, output_mem_config);
    } else if (!a_is_scalar && !b_is_scalar && c_is_scalar) {
        // a - tensor, b - tensor, c - is scalar
        return ttnn::add(
            ttnn::multiply(a, b, std::nullopt, output_mem_config), c, std::nullopt, output_mem_config);
    } else if (!a_is_scalar && b_is_scalar && !c_is_scalar) {
        // a - tensor, b - scalar, c - is tensor
        return ttnn::add(ttnn::multiply(a, b, std::nullopt, output_mem_config), c, std::nullopt, output_mem_config);
    } else if (!a_is_scalar && b_is_scalar && c_is_scalar) {
        // a - tensor, b - scalar, c - is scalar
        return ttnn::add(
            ttnn::multiply(a, b, std::nullopt, output_mem_config), c, std::nullopt, output_mem_config);
    } else if (a_is_scalar && !b_is_scalar && !c_is_scalar) {
        // a - scalar, b - tensor, c - tensor
        return ttnn::add(ttnn::multiply(b, a, std::nullopt, output_mem_config), c, std::nullopt, output_mem_config);
    } else if (a_is_scalar && !b_is_scalar && c_is_scalar) {
        // a - scalar, b - tensor, c - is scalar
        return ttnn::add(
            ttnn::multiply(b, a, std::nullopt, output_mem_config), c, std::nullopt, output_mem_config);
    } else if (a_is_scalar && b_is_scalar && !c_is_scalar) {
        // a - scalar, b - scalar, c - is tensor
        return ttnn::add(
            c, ttnn::multiply(a, b, std::nullopt, output_mem_config), std::nullopt, output_mem_config);
    }

    // all scalars
    // a - scalar, b - scalar, c - is scalar
    TT_ASSERT(a_is_scalar && b_is_scalar && c_is_scalar);
    return ttnn::add(ttnn::multiply(a, b), c);
}

Tensor _mac_overload(const Tensor& a, float b, float c, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor t_b = ttnn::operations::creation::create_scalar(b, a.get_dtype(), Layout::TILE, a.device());
    Tensor t_c = ttnn::operations::creation::create_scalar(c, a.get_dtype(), Layout::TILE, a.device());
    Tensor return_tensor = _mac(a, t_b, t_c, output_mem_config);
    t_b.deallocate();
    t_c.deallocate();
    return return_tensor;
}

} // namespace ttnn::operations::ternary
