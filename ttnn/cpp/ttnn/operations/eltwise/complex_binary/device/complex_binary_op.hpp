// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include <magic_enum.hpp>
#include "ttnn/operations/eltwise/complex/complex.hpp"

namespace ttnn::operations::complex_binary {

enum class ComplexBinaryOpType {
    ADD,
    SUB,
    MUL,
    DIV,
};

// OpHandler_complex_binary_type1 = get_function_complex_binary
ComplexTensor _add(const ComplexTensor& input_a, const ComplexTensor& input_b, const MemoryConfig& output_mem_config);
ComplexTensor _sub(const ComplexTensor& input_a, const ComplexTensor& input_b, const MemoryConfig& output_mem_config);
ComplexTensor _mul(const ComplexTensor& input_a, const ComplexTensor& input_b, const MemoryConfig& output_mem_config);
ComplexTensor _div(const ComplexTensor& input_a, const ComplexTensor& input_b, const MemoryConfig& output_mem_config);

template <ComplexBinaryOpType OpType>
struct OpHandler;

template <>
struct OpHandler<ComplexBinaryOpType::ADD> {
    static ComplexTensor handle( const ComplexTensor& input_a, const ComplexTensor& input_b, const MemoryConfig& output_mem_config ) {
        return _add(input_a, input_b, output_mem_config);
    }
};

template <>
struct OpHandler<ComplexBinaryOpType::SUB> {
    static ComplexTensor handle( const ComplexTensor& input_a, const ComplexTensor& input_b, const MemoryConfig& output_mem_config ) {
        return _sub(input_a, input_b, output_mem_config);
    }
};

template <>
struct OpHandler<ComplexBinaryOpType::MUL> {
    static ComplexTensor handle( const ComplexTensor& input_a, const ComplexTensor& input_b, const MemoryConfig& output_mem_config ) {
        return _mul(input_a, input_b, output_mem_config);
    }
};

template <>
struct OpHandler<ComplexBinaryOpType::DIV> {
    static ComplexTensor handle( const ComplexTensor& input_a, const ComplexTensor& input_b, const MemoryConfig& output_mem_config ) {
        return _div(input_a, input_b, output_mem_config);
    }
};

template <ComplexBinaryOpType OpType>
auto get_function_complex_binary() {
    return &OpHandler<OpType>::handle;
}

}  // namespace ttnn::operations::complex_binary
