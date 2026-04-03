// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "complex_unary_backward.hpp"

#include "ttnn/operations/eltwise/complex_unary_backward/device/complex_unary_backward_op.hpp"

namespace ttnn {

std::vector<ComplexTensor> polar_bw(
    const ComplexTensor& grad_tensor, const ComplexTensor& input_tensor, const MemoryConfig& memory_config) {
    return operations::complex_unary_backward::_polar_bw(grad_tensor, input_tensor, memory_config);
}

std::vector<ComplexTensor> conj_bw(
    const ComplexTensor& grad_tensor, const ComplexTensor& input_tensor, const MemoryConfig& memory_config) {
    return operations::complex_unary_backward::_conj_bw(grad_tensor, input_tensor, memory_config);
}

std::vector<ComplexTensor> imag_bw(
    const Tensor& grad_tensor, const ComplexTensor& input_tensor, const MemoryConfig& memory_config) {
    return operations::complex_unary_backward::_imag_bw(grad_tensor, input_tensor, memory_config);
}

std::vector<ComplexTensor> real_bw(
    const Tensor& grad_tensor, const ComplexTensor& input_tensor, const MemoryConfig& memory_config) {
    return operations::complex_unary_backward::_real_bw(grad_tensor, input_tensor, memory_config);
}

std::vector<ComplexTensor> angle_bw(
    const Tensor& grad_tensor, const ComplexTensor& input_tensor, const MemoryConfig& memory_config) {
    return operations::complex_unary_backward::_angle_bw(grad_tensor, input_tensor, memory_config);
}

}  // namespace ttnn
