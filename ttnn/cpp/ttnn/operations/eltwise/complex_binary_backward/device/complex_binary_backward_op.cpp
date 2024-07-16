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
#include "ttnn/operations/eltwise/complex_unary/device/complex_unary_op.hpp"
#include "ttnn/cpp/ttnn/operations/eltwise/binary/binary.hpp"

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

// complex mul
// grad_input = grad * other.conj()
// grad_other = grad * input.conj()
std::vector<ComplexTensor> _complex_mul_bw(const ComplexTensor& grad, const ComplexTensor& input, const ComplexTensor& other, const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    ComplexTensor grad_a = complex_mul(grad, conj(other,output_mem_config), output_mem_config);
    grad_tensor.emplace_back(grad_a);
    ComplexTensor grad_b = complex_mul(grad, conj(input,output_mem_config), output_mem_config);
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}

//  complex div
//  self: grad / other.conj();
//  other: -grad * ((self / other) / other).conj();
std::vector<ComplexTensor> _complex_div_bw(const ComplexTensor& grad, const ComplexTensor& input, const ComplexTensor& other, const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    Tensor condition_nan = ttnn::logical_and(ttnn::eqz(other.real(),output_mem_config), ttnn::eqz(other.imag(),output_mem_config), std::nullopt, output_mem_config);
    ComplexTensor grad_a = complex_div(grad, conj(other,output_mem_config), output_mem_config);
    Tensor grad_a_r = where(condition_nan, ttnn::operations::creation::full_like(grad.real(), std::nanf(""), std::nullopt, std::nullopt, std::nullopt, output_mem_config), ttnn::operations::complex_unary::_real(grad_a,output_mem_config),  output_mem_config);
    Tensor grad_a_i = where(condition_nan, ttnn::operations::creation::full_like(grad.imag(), std::nanf(""), std::nullopt, std::nullopt, std::nullopt, output_mem_config), ttnn::operations::complex_unary::_imag(grad_a,output_mem_config),  output_mem_config);
    grad_a = ComplexTensor({grad_a_r, grad_a_i});
    grad_a_r.deallocate();
    grad_a_i.deallocate();
    grad_tensor.emplace_back(grad_a);
    ComplexTensor neg_grad = ComplexTensor({ttnn::neg(grad.real(),output_mem_config), ttnn::neg(grad.imag(),output_mem_config)});
    ComplexTensor grad_b = complex_mul(neg_grad, conj(complex_div(complex_div(input, other, output_mem_config), other, output_mem_config ),output_mem_config), output_mem_config);
    neg_grad.deallocate();
    Tensor grad_b_r = where(condition_nan, ttnn::operations::creation::full_like(grad.real(), std::nanf(""), std::nullopt, std::nullopt, std::nullopt, output_mem_config), ttnn::operations::complex_unary::_real(grad_b,output_mem_config),  output_mem_config);
    Tensor grad_b_i = where(condition_nan, ttnn::operations::creation::full_like(grad.imag(), std::nanf(""), std::nullopt, std::nullopt, std::nullopt, output_mem_config), ttnn::operations::complex_unary::_imag(grad_b,output_mem_config),  output_mem_config);
    grad_b = ComplexTensor({grad_b_r, grad_b_i});
    grad_b_r.deallocate();
    grad_b_i.deallocate();
    condition_nan.deallocate();
    grad_tensor.emplace_back(grad_b);
    return grad_tensor;
}

}  // namespace ttnn::operations::complex_binary_backward
