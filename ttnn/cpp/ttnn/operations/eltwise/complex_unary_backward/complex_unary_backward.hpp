// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/eltwise/complex/complex.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

std::vector<ComplexTensor> polar_bw(
    const ComplexTensor& grad_tensor_arg, const ComplexTensor& input_tensor_arg, const MemoryConfig& memory_config);

std::vector<ComplexTensor> conj_bw(
    const ComplexTensor& grad_tensor_arg, const ComplexTensor& input_tensor_arg, const MemoryConfig& memory_config);

std::vector<ComplexTensor> imag_bw(
    const Tensor& grad_tensor_arg, const ComplexTensor& input_tensor_arg, const MemoryConfig& memory_config);

std::vector<ComplexTensor> real_bw(
    const Tensor& grad_tensor_arg, const ComplexTensor& input_tensor_arg, const MemoryConfig& memory_config);

std::vector<ComplexTensor> angle_bw(
    const Tensor& grad_tensor_arg, const ComplexTensor& input_tensor_arg, const MemoryConfig& memory_config);

}  // namespace ttnn
