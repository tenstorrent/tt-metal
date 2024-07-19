// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include "tensor/tensor.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include "ttnn/experimental/tt_dnn/op_library/complex/complex_ops.hpp"

namespace ttnn::operations::complex_unary {

constexpr uint8_t DefaultQueueId = 0;
enum class ComplexUnaryOpType {
    REAL,
    IMAG,
    ANGLE,
    IS_IMAG,
    IS_REAL,
    ABS,
};

Tensor _real(const ComplexTensor& input, const MemoryConfig& output_mem_config);
Tensor _imag(const ComplexTensor& input, const MemoryConfig& output_mem_config);
Tensor _angle(const ComplexTensor& input, const MemoryConfig& output_mem_config);
Tensor _is_imag(const ComplexTensor& input, const MemoryConfig& output_mem_config);
Tensor _is_real(const ComplexTensor& input, const MemoryConfig& output_mem_config);
Tensor _abs(const ComplexTensor& input, const MemoryConfig& output_mem_config);

template <ComplexUnaryOpType OpType>
struct OpHandler_complex_type1;

template <>
struct OpHandler_complex_type1<ComplexUnaryOpType::REAL> {
    static Tensor handle( const ComplexTensor& input, const MemoryConfig& output_mem_config ) {
        return _real(input, output_mem_config);
    }
};

template <>
struct OpHandler_complex_type1<ComplexUnaryOpType::IMAG> {
    static Tensor handle( const ComplexTensor& input, const MemoryConfig& output_mem_config ) {
        return _imag(input, output_mem_config);
    }
};

template <>
struct OpHandler_complex_type1<ComplexUnaryOpType::ANGLE> {
    static Tensor handle( const ComplexTensor& input, const MemoryConfig& output_mem_config ) {
        return _angle(input, output_mem_config);
    }
};

template <>
struct OpHandler_complex_type1<ComplexUnaryOpType::IS_IMAG> {
    static Tensor handle( const ComplexTensor& input, const MemoryConfig& output_mem_config ) {
        return _is_imag(input, output_mem_config);
    }
};

template <>
struct OpHandler_complex_type1<ComplexUnaryOpType::IS_REAL> {
    static Tensor handle( const ComplexTensor& input, const MemoryConfig& output_mem_config ) {
        return _is_real(input, output_mem_config);
    }
};

template <>
struct OpHandler_complex_type1<ComplexUnaryOpType::ABS> {
    static Tensor handle( const ComplexTensor& input, const MemoryConfig& output_mem_config ) {
        return _abs(input, output_mem_config);
    }
};

template <ComplexUnaryOpType OpType>
auto get_function_complex_unary() {
    return &OpHandler_complex_type1<OpType>::handle;
}


}  // namespace ttnn::operations::complex_unary
