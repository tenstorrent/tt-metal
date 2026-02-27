// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "complex_binary.hpp"

namespace ttnn::operations::complex_binary {

ComplexTensor add(
    const ComplexTensor& input_tensor_a, const ComplexTensor& input_tensor_b, const MemoryConfig& memory_config) {
    return _add(input_tensor_a, input_tensor_b, memory_config);
}

ComplexTensor subtract(
    const ComplexTensor& input_tensor_a, const ComplexTensor& input_tensor_b, const MemoryConfig& memory_config) {
    return _sub(input_tensor_a, input_tensor_b, memory_config);
}

ComplexTensor multiply(
    const ComplexTensor& input_tensor_a, const ComplexTensor& input_tensor_b, const MemoryConfig& memory_config) {
    return _mul(input_tensor_a, input_tensor_b, memory_config);
}

ComplexTensor divide(
    const ComplexTensor& input_tensor_a, const ComplexTensor& input_tensor_b, const MemoryConfig& memory_config) {
    return _div(input_tensor_a, input_tensor_b, memory_config);
}

}  // namespace ttnn::operations::complex_binary
