// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/eltwise/ternary_backward/device/ternary_backward_op.hpp"

#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_eager/tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tt_eager/tt_dnn/op_library/composite/composite_ops.hpp"
#include "tt_eager/tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_eager/tt_dnn/op_library/unpad/unpad_op.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"

namespace ttnn::operations::ternary_backward {

namespace utils {


std::vector<Tensor> _addcmul_bw(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& tensor1,
    const Tensor& tensor2,
    float value,
    const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    Tensor grad_a = mul_unary(ttnn::multiply(grad, tensor2, std::nullopt, output_mem_config), value, output_mem_config);
    grad_tensor.emplace_back(grad_a);
    Tensor grad_b = mul_unary(ttnn::multiply(grad, tensor1, std::nullopt, output_mem_config), value, output_mem_config);
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
    Tensor grad_a = ttnn::multiply(mul_unary(grad, value, output_mem_config), recip(tensor2, output_mem_config));
    grad_tensor.emplace_back(where(
        eqz(tensor2, output_mem_config),
        where(eqz(grad, output_mem_config), t_nan, t_inf, output_mem_config),
        grad_a,
        output_mem_config));
    Tensor tmp = ttnn::multiply(
        mul_unary(neg(grad, output_mem_config), value, output_mem_config), tensor1, std::nullopt, output_mem_config);
    Tensor grad_b =
        ttnn::multiply(tmp, recip(ttnn::square(tensor2, output_mem_config), output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(where(
        eqz(tensor2, output_mem_config),
        where(eqz(grad, output_mem_config), t_nan, neg(t_inf, output_mem_config), output_mem_config),
        grad_b,
        output_mem_config));
    return grad_tensor;
}

std::function<std::vector<ttnn::Tensor>(const Tensor&, const Tensor&, const Tensor&, const Tensor&, float, const MemoryConfig&)> get_function_type(TernaryBackwardOpType OpType){
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


}


}  // namespace ttnn::operations::ternary
