// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "third_party/magic_enum/magic_enum.hpp"
#include "ttnn/experimental/tt_numpy/functions.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/experimental/tt_dnn/op_library/composite/composite_ops.hpp"
#include "ternary_composite_op.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/types.hpp"
#include "tt_metal/common/bfloat16.hpp"

namespace ttnn::operations::ternary{

// addcmul(input,tensor1,tensor2,value)=input+value×tensor1×tensor2
Tensor _addcmul(
    const Tensor& input_a,
    const Tensor& input_b,
    const Tensor& input_c,
    float value,
    const MemoryConfig& output_mem_config) {
    Tensor t_value =
        ttnn::operations::creation::create_scalar(value, input_a.get_dtype(), Layout::TILE, input_a.device());
    Tensor t_mul = ttnn::multiply(input_b, input_c, std::nullopt, output_mem_config);
    Tensor t_factor = ttnn::multiply(t_mul, t_value, std::nullopt, output_mem_config);
    t_mul.deallocate();
    t_value.deallocate();
    Tensor result = ttnn::add(input_a, t_factor, std::nullopt, output_mem_config);
    return result;
}

// addcdiv(input,tensor1,tensor2,value)=input+value×tensor1/tensor2
Tensor _addcdiv(
    const Tensor& input_a,
    const Tensor& input_b,
    const Tensor& input_c,
    float value,
    const MemoryConfig& output_mem_config) {
    Tensor t_value =
        ttnn::operations::creation::create_scalar(value, input_a.get_dtype(), Layout::TILE, input_a.device());
    Tensor t_div = ttnn::multiply(input_b, ttnn::reciprocal(input_c, output_mem_config), std::nullopt, output_mem_config);
    Tensor t_factor = ttnn::multiply(t_div, t_value, std::nullopt, output_mem_config);
    t_div.deallocate();
    t_value.deallocate();
    Tensor result = ttnn::add(input_a, t_factor, std::nullopt, output_mem_config);
    Tensor t_inf = full_like(input_a, std::numeric_limits<float>::infinity());
    Tensor t_nan = full_like(input_a, std::nanf(""));
    return where(
        ttnn::eqz(input_c, output_mem_config),
        (value == 0) ? t_nan
                     : where(
                           ttnn::eqz(input_b, output_mem_config),
                           t_nan,
                           ttnn::multiply(t_inf, ttnn::sign(input_b, output_mem_config), std::nullopt, output_mem_config)),
        result,
        output_mem_config);
}

} // namespace ttnn::operations::binary
