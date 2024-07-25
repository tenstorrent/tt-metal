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

namespace ttnn::operations::complex_binary_backward {
using ComplexTensor = complex_binary::ComplexTensor;

constexpr uint8_t DefaultQueueId = 0;
enum class ComplexBinaryBackwardOpType {
    COMPLEX_ADD_BW,
    COMPLEX_SUB_BW,
    COMPLEX_MUL_BW,
    COMPLEX_DIV_BW,
};

std::vector<ComplexTensor> _complex_add_bw(const ComplexTensor& grad, const ComplexTensor& input, const ComplexTensor& other, float alpha, const MemoryConfig& output_mem_config);
std::vector<ComplexTensor> _complex_sub_bw(const ComplexTensor& grad, const ComplexTensor& input, const ComplexTensor& other, float alpha, const MemoryConfig& output_mem_config);
std::vector<ComplexTensor> _complex_mul_bw(const ComplexTensor& grad, const ComplexTensor& input, const ComplexTensor& other, const MemoryConfig& output_mem_config);
std::vector<ComplexTensor> _complex_div_bw(const ComplexTensor& grad, const ComplexTensor& input, const ComplexTensor& other, const MemoryConfig& output_mem_config);

template <ComplexBinaryBackwardOpType OpType>
struct OpHandler;

template <>
struct OpHandler<ComplexBinaryBackwardOpType::COMPLEX_MUL_BW> {
    static std::vector<ComplexTensor> handle( const ComplexTensor& grad, const ComplexTensor& input, const ComplexTensor& other, const MemoryConfig& output_mem_config ) {
        return _complex_mul_bw(grad, input, other, output_mem_config);
    }
};

template <>
struct OpHandler<ComplexBinaryBackwardOpType::COMPLEX_DIV_BW> {
    static std::vector<ComplexTensor> handle( const ComplexTensor& grad, const ComplexTensor& input, const ComplexTensor& other, const MemoryConfig& output_mem_config ) {
        return _complex_div_bw(grad, input, other, output_mem_config);
    }
};

template <>
struct OpHandler<ComplexBinaryBackwardOpType::COMPLEX_ADD_BW> {
    static std::vector<ComplexTensor> handle( const ComplexTensor& grad, const ComplexTensor& input, const ComplexTensor& other, float alpha, const MemoryConfig& output_mem_config ) {
        return _complex_add_bw(grad, input, other, alpha, output_mem_config);
    }
};

template <>
struct OpHandler<ComplexBinaryBackwardOpType::COMPLEX_SUB_BW> {
    static std::vector<ComplexTensor> handle( const ComplexTensor& grad, const ComplexTensor& input, const ComplexTensor& other, float alpha, const MemoryConfig& output_mem_config ) {
        return _complex_sub_bw(grad, input, other, alpha, output_mem_config);
    }
};

template <ComplexBinaryBackwardOpType OpType>
auto get_function_wo_float() {
    return &OpHandler<OpType>::handle;
}

template <ComplexBinaryBackwardOpType OpType>
auto get_function_w_float() {
    return &OpHandler<OpType>::handle;
}

}  // namespace ttnn::operations::complex_binary_backward
