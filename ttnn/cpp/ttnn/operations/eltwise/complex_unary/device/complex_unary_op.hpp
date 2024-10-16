// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include <magic_enum.hpp>
#include "ttnn/operations/eltwise/complex/complex.hpp"

namespace ttnn::operations::complex_unary {

enum class ComplexUnaryOpType {
    REAL,
    IMAG,
    ANGLE,
    IS_IMAG,
    IS_REAL,
    CONJ,
    POLAR,
};

// Tensor return type
Tensor _real(const ComplexTensor& input, const MemoryConfig& output_mem_config);
Tensor _imag(const ComplexTensor& input, const MemoryConfig& output_mem_config);
Tensor _angle(const ComplexTensor& input, const MemoryConfig& output_mem_config);
Tensor _is_imag(const ComplexTensor& input, const MemoryConfig& output_mem_config);
Tensor _is_real(const ComplexTensor& input, const MemoryConfig& output_mem_config);

// ComplexTensor return type
ComplexTensor _conj(const ComplexTensor& input, const MemoryConfig& output_mem_config);
ComplexTensor _polar(const ComplexTensor& input, const MemoryConfig& output_mem_config);

template <ComplexUnaryOpType OpType>
struct OpHandler;

template <>
struct OpHandler<ComplexUnaryOpType::REAL> {
    static Tensor handle(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
        return _real(input, output_mem_config);
    }
};

template <>
struct OpHandler<ComplexUnaryOpType::IMAG> {
    static Tensor handle(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
        return _imag(input, output_mem_config);
    }
};

template <>
struct OpHandler<ComplexUnaryOpType::ANGLE> {
    static Tensor handle(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
        return _angle(input, output_mem_config);
    }
};

template <>
struct OpHandler<ComplexUnaryOpType::IS_IMAG> {
    static Tensor handle(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
        return _is_imag(input, output_mem_config);
    }
};

template <>
struct OpHandler<ComplexUnaryOpType::IS_REAL> {
    static Tensor handle(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
        return _is_real(input, output_mem_config);
    }
};

template <>
struct OpHandler<ComplexUnaryOpType::CONJ> {
    static ComplexTensor handle(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
        return _conj(input, output_mem_config);
    }
};

template <>
struct OpHandler<ComplexUnaryOpType::POLAR> {
    static ComplexTensor handle(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
        return _polar(input, output_mem_config);
    }
};

}  // namespace ttnn::operations::complex_unary
