
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
        auto op_type = get_function_complex_unary<complex_unary_op_type>();
        return op_type(input_tensor_arg, memory_config);
    }
};

//OpHandler_complex_type2 = get_function_complex_unary_type2 --> ComplexTensor return type
template <ComplexUnaryOpType complex_unary_op_type>
struct ExecuteComplexUnaryType2 {
    static ComplexTensor operator()(const ComplexTensor &input_tensor_arg, const MemoryConfig &memory_config) {
        auto op_type = get_function_complex_unary_type2<complex_unary_op_type>();
        return op_type(input_tensor_arg, memory_config);
    }
};

}

//OpHandler_complex_type1 = get_function_complex_unary --> Tensor return type
constexpr auto real = ttnn::register_operation<operations::complex_unary::ExecuteComplexUnaryType1<operations::complex_unary::ComplexUnaryOpType::REAL>>("ttnn::real");
constexpr auto imag = ttnn::register_operation<operations::complex_unary::ExecuteComplexUnaryType1<operations::complex_unary::ComplexUnaryOpType::IMAG>>("ttnn::imag");
constexpr auto angle = ttnn::register_operation<operations::complex_unary::ExecuteComplexUnaryType1<operations::complex_unary::ComplexUnaryOpType::ANGLE>>("ttnn::angle");
constexpr auto is_imag = ttnn::register_operation<operations::complex_unary::ExecuteComplexUnaryType1<operations::complex_unary::ComplexUnaryOpType::IS_IMAG>>("ttnn::is_imag");
constexpr auto is_real = ttnn::register_operation<operations::complex_unary::ExecuteComplexUnaryType1<operations::complex_unary::ComplexUnaryOpType::IS_REAL>>("ttnn::is_real");

//OpHandler_complex_type2 = get_function_complex_unary_type2 --> ComplexTensor return type
constexpr auto conj = ttnn::register_operation<operations::complex_unary::ExecuteComplexUnaryType2<operations::complex_unary::ComplexUnaryOpType::CONJ>>("ttnn::conj");
constexpr auto polar = ttnn::register_operation<operations::complex_unary::ExecuteComplexUnaryType2<operations::complex_unary::ComplexUnaryOpType::POLAR>>("ttnn::polar");

}
