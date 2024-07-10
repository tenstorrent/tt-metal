// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/eltwise/unary_backward/device/unary_backward_op.hpp"

#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_eager/tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tt_eager/tt_dnn/op_library/composite/composite_ops.hpp"
#include "tt_eager/tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_eager/tt_dnn/op_library/unpad/unpad_op.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"

namespace ttnn::operations::unary_backward {

std::vector<ttnn::Tensor> _mul_bw(
    const Tensor& grad, const Tensor& input, float scalar, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = mul_unary(grad, scalar, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<Tensor> _clamp_min_bw(
    const Tensor& grad, const Tensor& input, float min, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor minT = gte_unary(input, min, output_mem_config);
    Tensor result = ttnn::multiply(grad, minT, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<Tensor> _clamp_bw(
    const Tensor& grad, const Tensor& input, float min, float max, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor minT = gte_unary(input, min, output_mem_config);
    Tensor maxT = lte_unary(input, max, output_mem_config);
    Tensor result = ttnn::logical_and(minT, maxT, std::nullopt, output_mem_config);
    result = ttnn::multiply(grad, result, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<Tensor> _assign_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    return grad_tensor;
}

std::vector<Tensor> _multigammaln_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor digamma_result = ttnn::multiply(grad, tt::tt_metal::digamma(input, output_mem_config), std::nullopt, output_mem_config);
    Tensor digamma_result_2 = ttnn::multiply(
        grad, tt::tt_metal::digamma(add_unary(-0.5, input, output_mem_config), output_mem_config), std::nullopt, output_mem_config);

    Tensor grad_result = ttnn::add(digamma_result, digamma_result_2, std::nullopt, output_mem_config);

    digamma_result = ttnn::multiply(
        grad, tt::tt_metal::digamma(add_unary(-1.0, input, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    grad_result = ttnn::add(grad_result, digamma_result, std::nullopt, output_mem_config);

    digamma_result = ttnn::multiply(
        grad, tt::tt_metal::digamma(add_unary(-1.5, input, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    grad_result = ttnn::add(grad_result, digamma_result, std::nullopt, output_mem_config);

    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

std::vector<Tensor> _add_bw(
    const Tensor& grad, const Tensor& input, float alpha, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    return grad_tensor;
}

std::vector<Tensor> _unary_comp_bw(const Tensor& grad, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor zero_grad = tt::tt_metal::zeros_like(grad, output_mem_config);
    grad_tensor.emplace_back(zero_grad);
    return grad_tensor;
}

std::vector<Tensor> _eq_bw(
    const Tensor& grad, const Tensor& input, float other, const MemoryConfig& output_mem_config) {
    return _unary_comp_bw(grad, output_mem_config);
}

std::vector<Tensor> _lgamma_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = ttnn::multiply(grad, tt::tt_metal::digamma(input, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

std::vector<Tensor> _sub_bw(const Tensor& grad, const Tensor& input, float alpha, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    return grad_tensor;
}

std::vector<Tensor> _frac_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    return grad_tensor;
}

std::vector<Tensor> _trunc_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = tt::tt_metal::zeros_like(grad, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// return: grad_output * (max_deriv - sign * (z / (1 + z)))
// z = exp(-abs(input))
std::vector<Tensor> _log_sigmoid_bw(const Tensor& grad, const Tensor& input, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor max_deriv = where(ttnn::ltz(input, output_mem_config), 1, 0, output_mem_config);
    Tensor in_sign = where(ttnn::ltz(input, output_mem_config), 1, -1, output_mem_config);
    Tensor in_abs = ttnn::abs(input, output_mem_config);
    Tensor z = ttnn::exp(ttnn::neg(in_abs, output_mem_config), false, output_mem_config);

    Tensor mul_z = ttnn::multiply(z, ttnn::reciprocal((ttnn::add(z, 1.0f, std::nullopt, output_mem_config)), output_mem_config), std::nullopt, output_mem_config);

    Tensor mul_sign = ttnn::multiply(in_sign, mul_z, std::nullopt, output_mem_config);
    Tensor sub_max = ttnn::subtract(max_deriv, mul_sign, std::nullopt, output_mem_config);

    Tensor grad_result = ttnn::multiply(grad, sub_max, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

std::function<std::vector<ttnn::Tensor>(const Tensor&, const Tensor&, const MemoryConfig&)> UnaryBackwardFunction::get_function_type1(UnaryBackwardOpType OpType){
    switch (OpType) {
        case UnaryBackwardOpType::ASSIGN_BW:
            return _assign_bw;
        case UnaryBackwardOpType::MULTIGAMMALN_BW:
            return _multigammaln_bw;
        case UnaryBackwardOpType::LGAMMA_BW:
            return _lgamma_bw;
        case UnaryBackwardOpType::FRAC_BW:
            return _frac_bw;
        case UnaryBackwardOpType::TRUNC_BW:
            return _trunc_bw;
        case UnaryBackwardOpType::LOG_SIGMOID_BW:
            return _log_sigmoid_bw;
        default:
            TT_ASSERT(false && "Undefined op type");
            return 0;
    }
}

std::function<std::vector<ttnn::Tensor>(const Tensor&, const Tensor&, float, const MemoryConfig&)> UnaryBackwardFunction::get_function_type1_w_float(UnaryBackwardOpType OpType){
    switch (OpType) {
        case UnaryBackwardOpType::MUL_BW:
            return _mul_bw;
        case UnaryBackwardOpType::CLAMP_MIN_BW:
            return _clamp_min_bw;
        case UnaryBackwardOpType::ADD_BW:
            return _add_bw;
        case UnaryBackwardOpType::EQ_BW:
            return _eq_bw;
        case UnaryBackwardOpType::SUB_BW:
            return _sub_bw;
        default:
            TT_ASSERT(false && "Undefined op type");
            return 0;
    }
}

std::function<std::vector<ttnn::Tensor>(const Tensor&, const Tensor&, float, float, const MemoryConfig&)> UnaryBackwardFunction::get_function_type1_w_two_float(UnaryBackwardOpType OpType){
    switch (OpType) {
        case UnaryBackwardOpType::CLAMP_BW:
            return _clamp_bw;
        default:
            TT_ASSERT(false && "Undefined op type");
            return 0;
    }
}

}  // namespace ttnn::operations::unary
