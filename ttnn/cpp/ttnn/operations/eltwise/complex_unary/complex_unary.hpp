
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/complex_unary_op.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement.hpp"

namespace ttnn {

namespace operations::complex_unary {
using ComplexTensor = complex_binary::ComplexTensor;

//OpHandler_complex_type1 = get_function_complex_unary --> Tensor return type
template <ComplexUnaryOpType complex_unary_op_type>
struct ExecuteComplexUnaryType1 {

    //Type 1: 1 input tensor
    static Tensor operator()(const ComplexTensor &input_tensor_arg, const MemoryConfig &memory_config) {
        return OpHandler<complex_unary_op_type>::handle(input_tensor_arg, memory_config);
    }
};

//OpHandler_complex_type2 = get_function_complex_unary_type2 --> ComplexTensor return type
template <ComplexUnaryOpType complex_unary_op_type>
struct ExecuteComplexUnaryType2 {
    static ComplexTensor operator()(const ComplexTensor &input_tensor_arg, const MemoryConfig &memory_config) {
        return OpHandler<complex_unary_op_type>::handle(input_tensor_arg, memory_config);
    }
};

}

//OpHandler_complex_type1 = get_function_complex_unary --> Tensor return type
constexpr auto real = ttnn::register_operation<
    "ttnn::real",
    operations::complex_unary::ExecuteComplexUnaryType1<operations::complex_unary::ComplexUnaryOpType::REAL>>();
constexpr auto imag = ttnn::register_operation<
    "ttnn::imag",
    operations::complex_unary::ExecuteComplexUnaryType1<operations::complex_unary::ComplexUnaryOpType::IMAG>>();
constexpr auto angle = ttnn::register_operation<
    "ttnn::angle",
    operations::complex_unary::ExecuteComplexUnaryType1<operations::complex_unary::ComplexUnaryOpType::ANGLE>>();
constexpr auto is_imag = ttnn::register_operation<
    "ttnn::is_imag",
    operations::complex_unary::ExecuteComplexUnaryType1<operations::complex_unary::ComplexUnaryOpType::IS_IMAG>>();
constexpr auto is_real = ttnn::register_operation<
    "ttnn::is_real",
    operations::complex_unary::ExecuteComplexUnaryType1<operations::complex_unary::ComplexUnaryOpType::IS_REAL>>();

//OpHandler_complex_type2 = get_function_complex_unary_type2 --> ComplexTensor return type
constexpr auto conj = ttnn::register_operation<
    "ttnn::conj",
    operations::complex_unary::ExecuteComplexUnaryType2<operations::complex_unary::ComplexUnaryOpType::CONJ>>();
constexpr auto polar = ttnn::register_operation<
    "ttnn::polar",
    operations::complex_unary::ExecuteComplexUnaryType2<operations::complex_unary::ComplexUnaryOpType::POLAR>>();
}
