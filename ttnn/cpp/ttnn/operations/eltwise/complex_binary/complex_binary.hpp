
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/complex_binary_op.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn {

namespace operations::complex_binary {

//OpHandler_complex_binary_type1 = get_function_complex_binary
template <ComplexBinaryOpType complex_binary_op_type>
struct ExecuteComplexBinaryType1 {

    //Type 1: 1 input tensor
    static ComplexTensor invoke(
        const ComplexTensor &input_tensor_a_arg,
        const ComplexTensor &input_tensor_b_arg,
        const MemoryConfig &memory_config) {
        return OpHandler<complex_binary_op_type>::handle(input_tensor_a_arg, input_tensor_b_arg, memory_config);
    }
};

} //namespace operations::complex_binary

} //namespace ttnn
