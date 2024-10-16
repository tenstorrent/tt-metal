
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/complex_unary_backward_op.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn {

namespace operations::complex_unary_backward {

template <ComplexUnaryBackwardOpType complex_unary_backward_op_type>
struct ExecuteComplexUnaryBackward {
    static std::vector<ComplexTensor> invoke(const ComplexTensor &grad_tensor_arg,
                                             const ComplexTensor &input_tensor_arg,
                                             const MemoryConfig &memory_config) {
        return OpHandler<complex_unary_backward_op_type>::handle(grad_tensor_arg, input_tensor_arg, memory_config);
    }
};

template <ComplexUnaryBackwardOpType complex_unary_backward_op_type>
struct ExecuteComplexUnaryBackwardTensor {
    static std::vector<ComplexTensor> invoke(const Tensor &grad_tensor_arg,
                                             const ComplexTensor &input_tensor_arg,
                                             const MemoryConfig &memory_config) {
        return OpHandler<complex_unary_backward_op_type>::handle(grad_tensor_arg, input_tensor_arg, memory_config);
    }
};

}  // namespace operations::complex_unary_backward

constexpr auto polar_bw =
    ttnn::register_operation<"ttnn::polar_bw",
                             operations::complex_unary_backward::ExecuteComplexUnaryBackward<
                                 operations::complex_unary_backward::ComplexUnaryBackwardOpType::POLAR_BW>>();
constexpr auto conj_bw =
    ttnn::register_operation<"ttnn::conj_bw",
                             operations::complex_unary_backward::ExecuteComplexUnaryBackward<
                                 operations::complex_unary_backward::ComplexUnaryBackwardOpType::CONJ_BW>>();

constexpr auto imag_bw =
    ttnn::register_operation<"ttnn::imag_bw",
                             operations::complex_unary_backward::ExecuteComplexUnaryBackwardTensor<
                                 operations::complex_unary_backward::ComplexUnaryBackwardOpType::IMAG_BW>>();
constexpr auto real_bw =
    ttnn::register_operation<"ttnn::real_bw",
                             operations::complex_unary_backward::ExecuteComplexUnaryBackwardTensor<
                                 operations::complex_unary_backward::ComplexUnaryBackwardOpType::REAL_BW>>();
constexpr auto angle_bw =
    ttnn::register_operation<"ttnn::angle_bw",
                             operations::complex_unary_backward::ExecuteComplexUnaryBackwardTensor<
                                 operations::complex_unary_backward::ComplexUnaryBackwardOpType::ANGLE_BW>>();

}  // namespace ttnn
