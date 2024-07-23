// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/complex/complex_ops.hpp"

namespace ttnn::operations::complex_binary {

constexpr uint8_t DefaultQueueId = 0;
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
struct OpHandler_complex_binary_type1;

template <>
struct OpHandler_complex_binary_type1<ComplexBinaryOpType::ADD> {
    static ComplexTensor handle( const ComplexTensor& input_a, const ComplexTensor& input_b, const MemoryConfig& output_mem_config ) {
        return _add(input_a, input_b, output_mem_config);
    }
};

template <>
struct OpHandler_complex_binary_type1<ComplexBinaryOpType::SUB> {
    static ComplexTensor handle( const ComplexTensor& input_a, const ComplexTensor& input_b, const MemoryConfig& output_mem_config ) {
        return _sub(input_a, input_b, output_mem_config);
    }
};

template <>
struct OpHandler_complex_binary_type1<ComplexBinaryOpType::MUL> {
    static ComplexTensor handle( const ComplexTensor& input_a, const ComplexTensor& input_b, const MemoryConfig& output_mem_config ) {
        return _mul(input_a, input_b, output_mem_config);
    }
};

template <>
struct OpHandler_complex_binary_type1<ComplexBinaryOpType::DIV> {
    static ComplexTensor handle( const ComplexTensor& input_a, const ComplexTensor& input_b, const MemoryConfig& output_mem_config ) {
        return _div(input_a, input_b, output_mem_config);
    }
};

template <ComplexBinaryOpType OpType>
auto get_function_complex_binary() {
    return &OpHandler_complex_binary_type1<OpType>::handle;
}

}  // namespace ttnn::operations::complex_binary
