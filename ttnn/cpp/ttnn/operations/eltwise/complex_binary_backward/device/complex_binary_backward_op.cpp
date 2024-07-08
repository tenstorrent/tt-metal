// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "third_party/magic_enum/magic_enum.hpp"

#include "tt_eager/tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tt_eager/tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_eager/tt_dnn/op_library/composite/composite_ops.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"
#include "ttnn/operations/eltwise/complex_binary_backward/device/complex_binary_backward_op.hpp"
#include "tt_eager/tt_dnn/op_library/unpad/unpad_op.hpp"


namespace ttnn::operations::complex_binary_backward {

namespace utils {


// complex add
// self: grad, other: grad * alpha
std::vector<ComplexTensor> _complex_add_bw(const ComplexTensor& grad, const ComplexTensor& input, const ComplexTensor& other, float alpha, const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    ComplexTensor grad_a = grad;
    grad_tensor.emplace_back(grad_a);
    const Tensor& grad_r = grad.real();
    const Tensor& grad_i = grad.imag();
    ComplexTensor grad_b = ComplexTensor({mul_unary(grad_r, alpha, output_mem_config), mul_unary(grad_i, alpha, output_mem_config)});
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}

// complex sub
// self: grad, other: -grad * alpha
std::vector<ComplexTensor> _complex_sub_bw(const ComplexTensor& grad, const ComplexTensor& input, const ComplexTensor& other, float alpha, const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    ComplexTensor grad_a = grad;
    grad_tensor.emplace_back(grad);
    const Tensor& grad_r = grad.real();
    const Tensor& grad_i = grad.imag();
    UnaryWithParam op1 {UnaryOpType::NEG};
    UnaryWithParam op2 {UnaryOpType::MUL_UNARY_SFPU, alpha};
    ComplexTensor grad_b = ComplexTensor({tt::tt_metal::unary_chain( grad_r, {op1, op2}, output_mem_config), tt::tt_metal::unary_chain( grad_i, {op1, op2}, output_mem_config)});
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}


// complex mul
// grad_input = grad * other.conj()
// grad_other = grad * input.conj()
std::vector<ComplexTensor> _complex_mul_bw(const ComplexTensor& grad, const ComplexTensor& input, const ComplexTensor& other, const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    ComplexTensor grad_a = tt::tt_metal::complex_mul(grad, conj(other,output_mem_config), output_mem_config);
    grad_tensor.emplace_back(grad_a);
    ComplexTensor grad_b = tt::tt_metal::complex_mul(grad, conj(input,output_mem_config), output_mem_config);
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}

std::function<std::vector<ComplexTensor>(const ComplexTensor&, const ComplexTensor&, const ComplexTensor&, float, const MemoryConfig&)> get_function_type1(ComplexBinaryBackwardOpType OpType){
    switch (OpType) {
        case ComplexBinaryBackwardOpType::COMPLEX_ADD_BW:
            return _complex_add_bw;
        case ComplexBinaryBackwardOpType::COMPLEX_SUB_BW:
            return _complex_sub_bw;
        default:
            TT_ASSERT(false && "Undefined op type");
            return 0;
    }
}

std::function<std::vector<ComplexTensor>(const ComplexTensor&, const ComplexTensor&, const ComplexTensor&, const MemoryConfig&)> get_function_type2(ComplexBinaryBackwardOpType OpType){
    switch (OpType) {
        case ComplexBinaryBackwardOpType::COMPLEX_MUL_BW:
            return _complex_mul_bw;
        default:
            TT_ASSERT(false && "Undefined op type");
            return 0;
    }
}



}

}  // namespace ttnn::operations::binary
