// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/complex/complex_ops.hpp"
#include "ttnn/operations/eltwise/complex_binary/device/complex_binary_op.hpp"

namespace ttnn::operations::complex_unary {
using ComplexTensor = complex_binary::ComplexTensor;

constexpr uint8_t DefaultQueueId = 0;
enum class ComplexUnaryOpType {
    REAL,
    IMAG,
    ANGLE,
    IS_IMAG,
    IS_REAL,
    ABS,
    CONJ,
    RECIPROCAL,
    POLAR,
};

//OpHandler_complex_type1 = get_function_complex_unary --> Tensor return type
Tensor _real(const ComplexTensor& input, const MemoryConfig& output_mem_config);
Tensor _imag(const ComplexTensor& input, const MemoryConfig& output_mem_config);
Tensor _angle(const ComplexTensor& input, const MemoryConfig& output_mem_config);
Tensor _is_imag(const ComplexTensor& input, const MemoryConfig& output_mem_config);
Tensor _is_real(const ComplexTensor& input, const MemoryConfig& output_mem_config);
Tensor _abs(const ComplexTensor& input, const MemoryConfig& output_mem_config);

//OpHandler_complex_type2 = get_function_complex_unary_type2 --> ComplexTensor return type
ComplexTensor _conj(const ComplexTensor& input, const MemoryConfig& output_mem_config);
ComplexTensor _reciprocal(const ComplexTensor& input, const MemoryConfig& output_mem_config);
ComplexTensor _polar(const ComplexTensor& input, const MemoryConfig& output_mem_config);

template <ComplexUnaryOpType OpType>
struct OpHandler;

template <>
struct OpHandler<ComplexUnaryOpType::REAL> {
    static Tensor handle( const ComplexTensor& input, const MemoryConfig& output_mem_config ) {
        return _real(input, output_mem_config);
    }
};

template <>
struct OpHandler<ComplexUnaryOpType::IMAG> {
    static Tensor handle( const ComplexTensor& input, const MemoryConfig& output_mem_config ) {
        return _imag(input, output_mem_config);
    }
};

template <>
struct OpHandler<ComplexUnaryOpType::ANGLE> {
    static Tensor handle( const ComplexTensor& input, const MemoryConfig& output_mem_config ) {
        return _angle(input, output_mem_config);
    }
};

template <>
struct OpHandler<ComplexUnaryOpType::IS_IMAG> {
    static Tensor handle( const ComplexTensor& input, const MemoryConfig& output_mem_config ) {
        return _is_imag(input, output_mem_config);
    }
};

template <>
struct OpHandler<ComplexUnaryOpType::IS_REAL> {
    static Tensor handle( const ComplexTensor& input, const MemoryConfig& output_mem_config ) {
        return _is_real(input, output_mem_config);
    }
};

template <>
struct OpHandler<ComplexUnaryOpType::ABS> {
    static Tensor handle( const ComplexTensor& input, const MemoryConfig& output_mem_config ) {
        return _abs(input, output_mem_config);
    }
};

template <>
struct OpHandler<ComplexUnaryOpType::CONJ> {
    static ComplexTensor handle( const ComplexTensor& input, const MemoryConfig& output_mem_config ) {
        return _conj(input, output_mem_config);
    }
};

template <>
struct OpHandler<ComplexUnaryOpType::RECIPROCAL> {
    static ComplexTensor handle( const ComplexTensor& input, const MemoryConfig& output_mem_config ) {
        return _reciprocal(input, output_mem_config);
    }
};

template <>
struct OpHandler<ComplexUnaryOpType::POLAR> {
    static ComplexTensor handle( const ComplexTensor& input, const MemoryConfig& output_mem_config ) {
        return _polar(input, output_mem_config);
    }
};

template <ComplexUnaryOpType OpType>
auto get_function_complex_unary() {
    return &OpHandler<OpType>::handle;
}

template <ComplexUnaryOpType OpType>
auto get_function_complex_unary_type2() {
    return &OpHandler<OpType>::handle;
}


}  // namespace ttnn::operations::complex_unary
