
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/complex_binary_backward_op.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement.hpp"

namespace ttnn {

namespace operations::complex_binary_backward {

template <ComplexBinaryBackwardOpType complex_binary_backward_op_type>
struct ExecuteComplexBinaryBackwardWFloat {

    static std::vector<ComplexTensor> operator()(
        const ComplexTensor &grad_tensor_arg,
        const ComplexTensor &input_tensor_a_arg,
        const ComplexTensor &input_tensor_b_arg,
        float alpha,
        const MemoryConfig &memory_config) {
        return OpHandler<complex_binary_backward_op_type>::handle(grad_tensor_arg, input_tensor_a_arg, input_tensor_b_arg, alpha, memory_config);
    }
};

template <ComplexBinaryBackwardOpType complex_binary_backward_op_type>
struct ExecuteComplexBinaryBackwardWoFloat {

    static std::vector<ComplexTensor> operator()(
        const ComplexTensor &grad_tensor_arg,
        const ComplexTensor &input_tensor_a_arg,
        const ComplexTensor &input_tensor_b_arg,
        const MemoryConfig &memory_config) {
        return OpHandler<complex_binary_backward_op_type>::handle(grad_tensor_arg, input_tensor_a_arg, input_tensor_b_arg, memory_config);
    }
};

}

}
