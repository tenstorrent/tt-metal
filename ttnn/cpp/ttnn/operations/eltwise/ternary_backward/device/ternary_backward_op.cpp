// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/eltwise/ternary_backward/device/ternary_backward_op.hpp"

#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/experimental/tt_dnn/op_library/bcast/bcast_op.hpp"
#include "ttnn/experimental/tt_dnn/op_library/unpad/unpad_op.hpp"

#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"

#include "third_party/magic_enum/magic_enum.hpp"

namespace ttnn::operations::ternary_backward {

std::vector<Tensor> _addcmul_bw(
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

std::vector<Tensor> _addcdiv_bw(
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
    Tensor grad_a = ttnn::multiply(ttnn::multiply(grad, value, std::nullopt, output_mem_config), recip(tensor2, output_mem_config));
    grad_tensor.emplace_back(where(
        eqz(tensor2, output_mem_config),
        where(eqz(grad, output_mem_config), t_nan, t_inf, output_mem_config),
        grad_a,
        output_mem_config));
    Tensor tmp = ttnn::multiply(
        ttnn::multiply(ttnn::neg(grad, output_mem_config), value, std::nullopt, output_mem_config), tensor1, std::nullopt, output_mem_config);
    Tensor grad_b =
        ttnn::multiply(tmp, ttnn::reciprocal(ttnn::square(tensor2, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(where(
        eqz(tensor2, output_mem_config),
        where(eqz(grad, output_mem_config), t_nan, ttnn::neg(t_inf, output_mem_config), output_mem_config),
        grad_b,
        output_mem_config));
    return grad_tensor;
}

std::vector<std::optional<Tensor>> _where_bw(
    uint8_t queue_id,
    const Tensor& grad,
    const Tensor& condition,
    const Tensor& input,
    const Tensor& other,
    const MemoryConfig& output_mem_config,
    const std::vector<bool>& are_required_outputs,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad) {
    std::vector<std::optional<Tensor>> result;
    if (are_required_outputs.at(0)) {
        if(input_grad.has_value()){
            tt::tt_metal::where(queue_id, condition, grad, 0.0f, output_mem_config, input_grad);
        } else {
            input_grad = tt::tt_metal::where(queue_id, condition, grad, 0.0f, output_mem_config);
        }
        result.emplace_back(input_grad);
    } else {
        result.emplace_back(std::nullopt);
    }
    if (are_required_outputs.at(1)) {
        if(other_grad.has_value()){
            tt::tt_metal::where(queue_id, condition, 0.0f, grad, output_mem_config, other_grad);
        } else {
            other_grad = tt::tt_metal::where(queue_id, condition, 0.0f, grad, output_mem_config);
        }
        result.emplace_back(other_grad);
    } else {
        result.emplace_back(std::nullopt);
    }
    return std::move(result);
}

std::vector<std::optional<Tensor>> _where_bw_overload(
    const Tensor& grad,
    const Tensor& condition,
    const Tensor& input,
    const Tensor& other,
    const MemoryConfig& output_mem_config,
    const std::vector<bool>& are_required_outputs,
    std::optional<Tensor> input_grad,
    std::optional<Tensor> other_grad) {
    uint8_t default_queue_id = 0;
    return _where_bw(default_queue_id, condition, grad, input, other, output_mem_config, are_required_outputs, input_grad, other_grad);
}

// lerp(input, end, weight) = self: grad * (1 - weight), end: grad * weight
std::vector<Tensor> _lerp_overload(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& end,
    const Tensor& weight,
    const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result_1 = ttnn::multiply(grad, ttnn::subtract(ttnn::operations::creation::full_like(weight, 1.0), weight, std::nullopt, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result_1);
    Tensor result_2 = ttnn::multiply(grad, weight, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result_2);
    return grad_tensor;
}

std::function<std::vector<Tensor>(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const MemoryConfig&)> TernaryBackwardFunction::get_function_type0(TernaryBackwardOpType OpType){
    switch (OpType) {
        case TernaryBackwardOpType::LERP_BW:
            return _lerp_overload;
        default:
            TT_ASSERT(false && "Undefined op type");
            return 0;
    }
}

std::function<std::vector<Tensor>(const Tensor&, const Tensor&, const Tensor&, const Tensor&, float, const MemoryConfig&)> TernaryBackwardFunction::get_function_type(TernaryBackwardOpType OpType){
    switch (OpType) {
        case TernaryBackwardOpType::ADDCMUL_BW:
            return _addcmul_bw;
        case TernaryBackwardOpType::ADDCDIV_BW:
            return _addcdiv_bw;
        default:
            TT_ASSERT(false && "Undefined op type");
            return 0;
    }
}

std::function<std::vector<std::optional<Tensor>>(uint8_t , const Tensor&, const Tensor&, const Tensor&, const Tensor&, const MemoryConfig&, const std::vector<bool>&, std::optional<Tensor>, std::optional<Tensor>)> TernaryBackwardFunction::get_function_type_opt(TernaryBackwardOpType OpType){
    switch (OpType) {
        case TernaryBackwardOpType::WHERE_BW:
            return _where_bw;
        default:
            TT_ASSERT(false && "Undefined op type");
            return 0;
    }
}

std::function<std::vector<std::optional<Tensor>>(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const MemoryConfig&, const std::vector<bool>&, std::optional<Tensor>, std::optional<Tensor>)> TernaryBackwardFunction::get_function_type_opt_wo_qid(TernaryBackwardOpType OpType){
    switch (OpType) {
        case TernaryBackwardOpType::WHERE_BW:
            return _where_bw_overload;
        default:
            TT_ASSERT(false && "Undefined op type");
            return 0;
    }
}

}  // namespace ttnn::operations::ternary
