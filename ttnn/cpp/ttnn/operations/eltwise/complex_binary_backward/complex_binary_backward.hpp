
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/complex_binary_backward_op.cpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement.hpp"

namespace ttnn {

namespace operations::complex_binary_backward {

template <ComplexBinaryBackwardOpType complex_binary_backward_op_type>
struct ExecuteComplexBinaryBackward {

    //Type 1: 2 inputs, 1 grad tensor

    static std::vector<ComplexTensor> execute_on_main_thread(
        const ComplexTensor &grad_tensor_arg,
        const ComplexTensor &input_tensor_a_arg,
        const ComplexTensor &input_tensor_b_arg,
        float alpha,
        const MemoryConfig &memory_config) {

        auto op_type = utils::get_function_type1(complex_binary_backward_op_type);
        return op_type(grad_tensor_arg, input_tensor_a_arg, input_tensor_b_arg, alpha, memory_config);
        }

};

}
constexpr auto complex_add_bw = ttnn::register_operation<operations::complex_binary_backward::ExecuteComplexBinaryBackward<operations::complex_binary_backward::ComplexBinaryBackwardOpType::COMPLEX_ADD_BW>>("ttnn::complex_add_bw");

}
