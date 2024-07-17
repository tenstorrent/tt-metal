
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/complex_unary_op.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement.hpp"

namespace ttnn {

namespace operations::complex_unary {

template <ComplexUnaryOpType complex_unary_op_type>
struct ExecuteComplexUnaryType1 {

    //Type 1: 1 input tensor

    static Tensor execute_on_main_thread(
        const ComplexTensor &input_tensor_arg,
        const MemoryConfig &memory_config) {

        auto op_type = get_function_type1<complex_unary_op_type>();
        return op_type(input_tensor_arg, memory_config);
        }

};

}
constexpr auto real = ttnn::register_operation<operations::complex_unary::ExecuteComplexUnaryType1<operations::complex_unary::ComplexUnaryOpType::REAL>>("ttnn::real");
constexpr auto imag = ttnn::register_operation<operations::complex_unary::ExecuteComplexUnaryType1<operations::complex_unary::ComplexUnaryOpType::IMAG>>("ttnn::imag");
constexpr auto angle = ttnn::register_operation<operations::complex_unary::ExecuteComplexUnaryType1<operations::complex_unary::ComplexUnaryOpType::ANGLE>>("ttnn::angle");

}
