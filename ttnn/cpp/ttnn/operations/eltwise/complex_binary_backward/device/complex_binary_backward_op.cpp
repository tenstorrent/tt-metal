// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "complex_binary_backward_op.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"
#include "ttnn/operations/eltwise/complex_unary/device/complex_unary_op.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/complex_binary/device/complex_binary_op.hpp"
#include "ttnn/operations/eltwise/complex_unary/complex_unary.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/ternary/where.hpp"
#include "ttnn/operations/creation.hpp"


namespace ttnn::operations::complex_binary_backward {

// complex add
// self: grad, other: grad * alpha
std::vector<ComplexTensor> _complex_add_bw(const ComplexTensor& grad, const ComplexTensor& input, const ComplexTensor& other, float alpha, const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    ComplexTensor grad_a = grad;
    grad_tensor.emplace_back(grad_a);
    const Tensor& grad_r = grad.real();
    const Tensor& grad_i = grad.imag();
    ComplexTensor grad_b = ComplexTensor({ttnn::multiply(grad_r, alpha, std::nullopt, output_mem_config), ttnn::multiply(grad_i, alpha, std::nullopt, output_mem_config)});
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
    using ttnn::operations::unary::UnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<UnaryWithParam> ops_chain = {
    UnaryWithParam{UnaryOpType::NEG},
    UnaryWithParam{UnaryOpType::MUL_UNARY_SFPU, alpha} };
    ComplexTensor grad_b = ComplexTensor({ttnn::unary_chain( grad_r, ops_chain, output_mem_config), ttnn::unary_chain( grad_i, ops_chain, output_mem_config)});
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}


}  // namespace ttnn::operations::complex_binary_backward
