// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include "tensor/tensor.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_eager/tt_dnn/op_library/complex/complex_ops.hpp"

namespace ttnn::operations::complex_unary_backward {

constexpr uint8_t DefaultQueueId = 0;
enum class ComplexUnaryBackwardOpType {
    POLAR_BW,
};

//OpHandler_complex : get_function_complex
std::vector<ComplexTensor> _polar_bw(const ComplexTensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config);

template <ComplexUnaryBackwardOpType OpType>
struct OpHandler_complex;

template <>
struct OpHandler_complex<ComplexUnaryBackwardOpType::POLAR_BW> {
    static std::vector<ComplexTensor> handle( const ComplexTensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config ) {
        return _polar_bw(grad, input, output_mem_config);
    }
};

template <ComplexUnaryBackwardOpType OpType>
auto get_function_complex() {
    return &OpHandler_complex<OpType>::handle;
}

}

}  // namespace ttnn::operations::complex_unary_backward
