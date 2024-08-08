// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"

namespace ttnn::operations::complex_unary_backward {

enum class ComplexUnaryBackwardOpType {
    POLAR_BW,
    IMAG_BW,
    REAL_BW,
    ANGLE_BW,
    CONJ_BW,
    COMPLEX_ABS_BW,
    COMPLEX_RECIP_BW,
};

std::vector<ComplexTensor> _polar_bw(const ComplexTensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config);
std::vector<ComplexTensor> _conj_bw(const ComplexTensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config);
std::vector<ComplexTensor> _complex_recip_bw(const ComplexTensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config);

std::vector<ComplexTensor> _imag_bw(const Tensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config);
std::vector<ComplexTensor> _real_bw(const Tensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config);
std::vector<ComplexTensor> _angle_bw(const Tensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config);
std::vector<ComplexTensor> _complex_abs_bw(const Tensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config);

template <ComplexUnaryBackwardOpType OpType>
struct OpHandler;

template <>
struct OpHandler<ComplexUnaryBackwardOpType::POLAR_BW> {
    static std::vector<ComplexTensor> handle( const ComplexTensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config ) {
        return _polar_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<ComplexUnaryBackwardOpType::CONJ_BW> {
    static std::vector<ComplexTensor> handle( const ComplexTensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config ) {
        return _conj_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<ComplexUnaryBackwardOpType::COMPLEX_RECIP_BW> {
    static std::vector<ComplexTensor> handle( const ComplexTensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config ) {
        return _complex_recip_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<ComplexUnaryBackwardOpType::IMAG_BW> {
    static std::vector<ComplexTensor> handle( const Tensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config ) {
        return _imag_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<ComplexUnaryBackwardOpType::REAL_BW> {
    static std::vector<ComplexTensor> handle( const Tensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config ) {
        return _real_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<ComplexUnaryBackwardOpType::ANGLE_BW> {
    static std::vector<ComplexTensor> handle( const Tensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config ) {
        return _angle_bw(grad, input, output_mem_config);
    }
};

template <>
struct OpHandler<ComplexUnaryBackwardOpType::COMPLEX_ABS_BW> {
    static std::vector<ComplexTensor> handle( const Tensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config ) {
        return _complex_abs_bw(grad, input, output_mem_config);
    }
};

}  // namespace ttnn::operations::complex_unary_backward
