// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/data_movement/bcast/bcast.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/ternary/where.hpp"

#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"
#include "ttnn/operations/eltwise/ternary_backward/ternary_backward.hpp"
#include "third_party/magic_enum/magic_enum.hpp"

namespace ttnn::operations::ternary_backward {

std::vector<Tensor> AddcmulBackwardOperation::invoke(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& tensor1,
    const Tensor& tensor2,
    float value,
    const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    Tensor grad_a = ttnn::multiply(ttnn::multiply(grad, tensor2, std::nullopt, output_mem_config), value, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_a);
    Tensor grad_b = ttnn::multiply(ttnn::multiply(grad, tensor1, std::nullopt, output_mem_config), value, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}

std::vector<Tensor> AddcdivBackwardOperation::invoke(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& tensor1,
    const Tensor& tensor2,
    float value,
    const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    Tensor t_inf = ttnn::operations::creation::full_like(input, std::numeric_limits<float>::infinity(), input.get_dtype(), input.get_layout(), std::nullopt, output_mem_config);
    Tensor t_nan = ttnn::operations::creation::full_like(input, std::nanf(""), input.get_dtype(), input.get_layout(), std::nullopt, output_mem_config);
    Tensor grad_a = ttnn::multiply(ttnn::multiply(grad, value, std::nullopt, output_mem_config), ttnn::reciprocal(tensor2, output_mem_config));
    grad_tensor.emplace_back(where(
        ttnn::eqz(tensor2, output_mem_config),
        where(ttnn::eqz(grad, output_mem_config), t_nan, t_inf, output_mem_config),
        grad_a,
        output_mem_config));
    Tensor tmp = ttnn::multiply(
        ttnn::multiply(ttnn::neg(grad, output_mem_config), value, std::nullopt, output_mem_config), tensor1, std::nullopt, output_mem_config);
    Tensor grad_b =
        ttnn::multiply(tmp, ttnn::reciprocal(ttnn::square(tensor2, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(where(
        ttnn::eqz(tensor2, output_mem_config),
        where(ttnn::eqz(grad, output_mem_config), t_nan, ttnn::neg(t_inf, output_mem_config), output_mem_config),
        grad_b,
        output_mem_config));
    return grad_tensor;
}

std::vector<OptionalTensor> WhereBackwardOperation::invoke(
    uint8_t queue_id,
    const Tensor& grad,
    const Tensor& condition,
    const Tensor& input,
    const Tensor& other,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::vector<bool>& are_required_outputs,
    OptionalTensor input_grad,
    OptionalTensor other_grad) {
    std::vector<OptionalTensor> result;
    if (are_required_outputs.at(0)) {
        if(input_grad.has_value()){
            where(queue_id, condition, grad, 0.0f, output_mem_config, input_grad);
        } else {
            input_grad = where(queue_id, condition, grad, 0.0f, output_mem_config);
        }
        result.emplace_back(input_grad);
    } else {
        result.emplace_back(std::nullopt);
    }
    if (are_required_outputs.at(1)) {
        if(other_grad.has_value()){
            where(queue_id, condition, 0.0f, grad, output_mem_config, other_grad);
        } else {
            other_grad = where(queue_id, condition, 0.0f, grad, output_mem_config);
        }
        result.emplace_back(other_grad);
    } else {
        result.emplace_back(std::nullopt);
    }
    return std::move(result);
}

// lerp(input, end, weight) = self: grad * (1 - weight), end: grad * weight
std::vector<Tensor> LerpBackwardOperation::invoke(
    const Tensor& grad, const Tensor& input, const Tensor& end, const Tensor& weight, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result_1 = ttnn::multiply(grad, ttnn::subtract(ttnn::operations::creation::full_like(weight, 1.0), weight, std::nullopt, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result_1);
    Tensor result_2 = ttnn::multiply(grad, weight, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result_2);
    Tensor zero = ttnn::multiply(grad, ttnn::subtract(end, input, std::nullopt, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(zero);
    return grad_tensor;
}

std::vector<Tensor> LerpBackwardOperation::invoke(
    const Tensor& grad, const Tensor& input, const Tensor& end, float weight, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float sub_scalar = 1.0f - weight;
    Tensor result_1 = ttnn::multiply(grad, sub_scalar, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result_1);
    Tensor result_2 = ttnn::multiply(grad, weight, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result_2);
    return grad_tensor;
}


}  // namespace ttnn::operations::ternary
