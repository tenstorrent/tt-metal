
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/complex_unary_backward_op.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement.hpp"

namespace ttnn {

namespace operations::complex_unary_backward {

//OpHandler_complex : get_function_complex
template <ComplexUnaryBackwardOpType complex_unary_backward_op_type>
struct ExecuteComplexUnaryBackward {

    static std::vector<ComplexTensor> execute_on_main_thread(
        const ComplexTensor &grad_tensor_arg,
        const ComplexTensor &input_tensor_arg,
        const MemoryConfig& memory_config) {

        auto op_type = get_function_complex<complex_unary_backward_op_type>();
        return op_type(grad_tensor_arg, input_tensor_arg, memory_config);
        }

};

//OpHandler_tensor_complex : get_function_tensor_complex
template <ComplexUnaryBackwardOpType complex_unary_backward_op_type>
struct ExecuteComplexUnaryBackwardTensor {

    static std::vector<ComplexTensor> execute_on_main_thread(
        const Tensor &grad_tensor_arg,
        const ComplexTensor &input_tensor_arg,
        const MemoryConfig& memory_config) {

        auto op_type = get_function_tensor_complex<complex_unary_backward_op_type>();
        return op_type(grad_tensor_arg, input_tensor_arg, memory_config);
        }

};

}

//OpHandler_complex : get_function_complex
constexpr auto polar_bw = ttnn::register_operation<operations::complex_unary_backward::ExecuteComplexUnaryBackward<operations::complex_unary_backward::ComplexUnaryBackwardOpType::POLAR_BW>>("ttnn::polar_bw");
constexpr auto conj_bw = ttnn::register_operation<operations::complex_unary_backward::ExecuteComplexUnaryBackward<operations::complex_unary_backward::ComplexUnaryBackwardOpType::CONJ_BW>>("ttnn::conj_bw");

//OpHandler_tensor_complex : get_function_tensor_complex
constexpr auto imag_bw = ttnn::register_operation<operations::complex_unary_backward::ExecuteComplexUnaryBackwardTensor<operations::complex_unary_backward::ComplexUnaryBackwardOpType::IMAG_BW>>("ttnn::imag_bw");
constexpr auto real_bw = ttnn::register_operation<operations::complex_unary_backward::ExecuteComplexUnaryBackwardTensor<operations::complex_unary_backward::ComplexUnaryBackwardOpType::REAL_BW>>("ttnn::real_bw");
constexpr auto angle_bw = ttnn::register_operation<operations::complex_unary_backward::ExecuteComplexUnaryBackwardTensor<operations::complex_unary_backward::ComplexUnaryBackwardOpType::ANGLE_BW>>("ttnn::angle_bw");

}
