
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/complex_unary_op.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn {

namespace operations::complex_unary {

template <ComplexUnaryOpType complex_unary_op_type>
struct ExecuteComplexUnaryTensor {

    //Type 1: 1 input tensor
    static Tensor invoke(const ComplexTensor &input_tensor_arg, const MemoryConfig &memory_config) {
        return OpHandler<complex_unary_op_type>::handle(input_tensor_arg, memory_config);
    }
};

template <ComplexUnaryOpType complex_unary_op_type>
struct ExecuteComplexUnaryComplexTensor {
    static ComplexTensor invoke(const ComplexTensor &input_tensor_arg, const MemoryConfig &memory_config) {
        return OpHandler<complex_unary_op_type>::handle(input_tensor_arg, memory_config);
    }
};

}

constexpr auto real = ttnn::register_operation<
    "ttnn::real",
    operations::complex_unary::ExecuteComplexUnaryTensor<operations::complex_unary::ComplexUnaryOpType::REAL>>();
constexpr auto imag = ttnn::register_operation<
    "ttnn::imag",
    operations::complex_unary::ExecuteComplexUnaryTensor<operations::complex_unary::ComplexUnaryOpType::IMAG>>();
constexpr auto angle = ttnn::register_operation<
    "ttnn::angle",
    operations::complex_unary::ExecuteComplexUnaryTensor<operations::complex_unary::ComplexUnaryOpType::ANGLE>>();
constexpr auto is_imag = ttnn::register_operation<
    "ttnn::is_imag",
    operations::complex_unary::ExecuteComplexUnaryTensor<operations::complex_unary::ComplexUnaryOpType::IS_IMAG>>();
constexpr auto is_real = ttnn::register_operation<
    "ttnn::is_real",
    operations::complex_unary::ExecuteComplexUnaryTensor<operations::complex_unary::ComplexUnaryOpType::IS_REAL>>();

constexpr auto conj = ttnn::register_operation<
    "ttnn::conj",
    operations::complex_unary::ExecuteComplexUnaryComplexTensor<operations::complex_unary::ComplexUnaryOpType::CONJ>>();
constexpr auto polar = ttnn::register_operation<
    "ttnn::polar",
    operations::complex_unary::ExecuteComplexUnaryComplexTensor<operations::complex_unary::ComplexUnaryOpType::POLAR>>();
}
