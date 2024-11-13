// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "complex_unary_op.hpp"
#include "ttnn/operations/data_movement/bcast/bcast.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"

namespace ttnn::operations::complex_unary {

Tensor _real(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return input[0];
}

Tensor _imag(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return input[1];
}

Tensor _angle(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return ttnn::neg( atan2(input[1],input[0],output_mem_config), output_mem_config );
}

Tensor _is_imag(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return ttnn::eqz( input[0], output_mem_config);
}

Tensor _is_real(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return ttnn::eqz( input[1], output_mem_config);
}

ComplexTensor _conj(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return ComplexTensor({input[0], ttnn::neg(input[1],output_mem_config)});
}

ComplexTensor _polar(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    const Tensor& input_a = input.real();
    const Tensor& input_b = input.imag();
    Tensor c = ttnn::cos(input_b,output_mem_config);
    Tensor r = ttnn::multiply(input_a,c,std::nullopt,output_mem_config);
    c.deallocate();

    Tensor s = ttnn::sin(input_b,output_mem_config);
    Tensor i = ttnn::multiply(input_a,s,std::nullopt,output_mem_config);
    s.deallocate();

    return ComplexTensor({r,i});
}

}  // namespace ttnn::operations::complex_unary
