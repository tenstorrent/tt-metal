// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "complex_unary.hpp"
#include "device/complex_unary_op.hpp"
#include "ttnn/types.hpp"
#include "ttnn/graph/composite_trace.hpp"

namespace ttnn {

Tensor real(const ComplexTensor& input_tensor, const std::optional<MemoryConfig>& memory_config) {
    TT_OP_SCOPE("ttnn::real");
    auto output_mem_config = memory_config.value_or(input_tensor.real().memory_config());
    return operations::complex_unary::_real(input_tensor, output_mem_config);
}
Tensor imag(const ComplexTensor& input_tensor, const std::optional<MemoryConfig>& memory_config) {
    TT_OP_SCOPE("ttnn::imag");
    auto output_mem_config = memory_config.value_or(input_tensor.real().memory_config());
    return operations::complex_unary::_imag(input_tensor, output_mem_config);
}
Tensor angle(const ComplexTensor& input_tensor, const std::optional<MemoryConfig>& memory_config) {
    TT_OP_SCOPE("ttnn::angle");
    auto output_mem_config = memory_config.value_or(input_tensor.real().memory_config());
    return operations::complex_unary::_angle(input_tensor, output_mem_config);
}
Tensor is_imag(const ComplexTensor& input_tensor, const std::optional<MemoryConfig>& memory_config) {
    TT_OP_SCOPE("ttnn::is_imag");
    auto output_mem_config = memory_config.value_or(input_tensor.real().memory_config());
    return operations::complex_unary::_is_imag(input_tensor, output_mem_config);
}
Tensor is_real(const ComplexTensor& input_tensor, const std::optional<MemoryConfig>& memory_config) {
    TT_OP_SCOPE("ttnn::is_real");
    auto output_mem_config = memory_config.value_or(input_tensor.real().memory_config());
    return operations::complex_unary::_is_real(input_tensor, output_mem_config);
}
ComplexTensor conj(const ComplexTensor& input_tensor, const std::optional<MemoryConfig>& memory_config) {
    TT_OP_SCOPE("ttnn::conj");
    auto output_mem_config = memory_config.value_or(input_tensor.real().memory_config());
    return operations::complex_unary::_conj(input_tensor, output_mem_config);
}
ComplexTensor polar(const ComplexTensor& input_tensor, const std::optional<MemoryConfig>& memory_config) {
    TT_OP_SCOPE("ttnn::polar");
    auto output_mem_config = memory_config.value_or(input_tensor.real().memory_config());
    return operations::complex_unary::_polar(input_tensor, output_mem_config);
}
}  // namespace ttnn
