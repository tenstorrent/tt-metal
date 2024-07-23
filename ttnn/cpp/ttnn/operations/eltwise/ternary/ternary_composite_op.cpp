// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "ternary_composite_op.hpp"

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

} // namespace ttnn::operations::ternary
