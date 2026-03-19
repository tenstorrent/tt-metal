// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/memory_config/memory_config.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::complex_unary_backward {

std::vector<ComplexTensor> _polar_bw(
    const ComplexTensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config);
std::vector<ComplexTensor> _conj_bw(
    const ComplexTensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config);
std::vector<ComplexTensor> _complex_recip_bw(
    const ComplexTensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config);

std::vector<ComplexTensor> _imag_bw(
    const Tensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config);
std::vector<ComplexTensor> _real_bw(
    const Tensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config);
std::vector<ComplexTensor> _angle_bw(
    const Tensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config);
std::vector<ComplexTensor> _complex_abs_bw(
    const Tensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config);

}  // namespace ttnn::operations::complex_unary_backward
