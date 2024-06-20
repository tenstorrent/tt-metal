// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "third_party/magic_enum/magic_enum.hpp"

#include "tt_eager/tt_dnn/op_library/bcast/bcast_op.hpp"

#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"
#include "ttnn/operations/eltwise/binary_backward/device/binary_backward_op.hpp"


namespace ttnn::operations::binary_backward {

namespace utils {
//}
// using namespace tt::tt_metal;

std::vector<Tensor> _atan2_bw(
    const Tensor& grad, const Tensor& input, const Tensor& other, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float t_nan = std::nanf("");
    UnaryWithParam op1{UnaryOpType::SQUARE};
    UnaryWithParam op2{UnaryOpType::RECIP};
    Tensor recip_mul =
        ttnn::multiply(grad, unary_chain(hypot(input, other), {op1, op2}, output_mem_config), std::nullopt, output_mem_config);
    Tensor grad_a = ttnn::multiply(other, recip_mul, std::nullopt, output_mem_config);
    Tensor cond = ttnn::logical_and(eqz(input, output_mem_config), eqz(other, output_mem_config));
    grad_a = where(cond, t_nan, grad_a, output_mem_config);
    grad_tensor.emplace_back(grad_a);
    Tensor grad_b = ttnn::multiply(neg(input), recip_mul, std::nullopt, output_mem_config);
    grad_b = where(cond, t_nan, grad_b, output_mem_config);
    recip_mul.deallocate();
    cond.deallocate();
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}


std::vector<Tensor> _embedding_bw(
    const Tensor& grad, const Tensor& input, const Tensor& weight, const MemoryConfig& output_mem_config) {
    TT_FATAL(input.get_dtype() == DataType::UINT32, "Input must be UINT32");
    TT_FATAL(
        grad.get_legacy_shape()[0] == 1 && grad.get_legacy_shape()[1] == 1,
        "First two dimensions for the grad must be 1");
    TT_FATAL(
        input.get_legacy_shape()[1] == 1 && input.get_legacy_shape()[2] == 1,
        "Only dim 0 && 3 for the input can be non 1");
    std::vector<Tensor> grad_tensor;
    Tensor grad_a = embeddings(input, grad, false);
    grad_tensor.emplace_back(grad_a);

    return grad_tensor;
}

std::function<std::vector<Tensor>(const Tensor&, const Tensor&, const Tensor&, const MemoryConfig&)> get_function(BinaryBackwardOpType OpType){
    switch (OpType) {
        case BinaryBackwardOpType::ATAN2_BW:
            return _atan2_bw;
        case BinaryBackwardOpType::EMBEDDING_BW:
            return _embedding_bw;
        default: TT_ASSERT(false && "Undefined op type");
    }
}

}


}  // namespace ttnn::operations::binary
