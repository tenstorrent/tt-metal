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

namespace ttnn::operations::unary_backward {

namespace utils {


std::vector<ttnn::Tensor> _unary_mul_bw(
    const Tensor& grad, const Tensor& input, float scalar, const MemoryConfig& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = mul_unary(grad, scalar, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}


std::function<std::vector<ttnn::Tensor>(const Tensor&, const Tensor&, const MemoryConfig&)> get_function_type1(UnaryBackwardOpType OpType){
    switch (OpType) {
        default:
            TT_ASSERT(false && "Undefined op type");
            return 0;
    }
}

std::function<std::vector<ttnn::Tensor>(const Tensor&, const Tensor&, float, const MemoryConfig&)> get_function_type1_w_float(UnaryBackwardOpType OpType){
    switch (OpType) {
        case UnaryBackwardOpType::UNARY_MUL_BW:
            return _unary_mul_bw;
        default:
            TT_ASSERT(false && "Undefined op type");
            return 0;
    }
}

}

}  // namespace ttnn::operations::unary
