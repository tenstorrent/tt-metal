// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "complex_unary_op.hpp"
#include "ttnn/operations/data_movement/bcast/bcast.hpp"
#include "tools/profiler/op_profiler.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"
#include "ttnn/operations/core/to_memory_config/to_memory_config_op.hpp"

namespace ttnn::operations::complex_unary {

namespace {

// real and imag hand back a component of the input instead of computing a tensor, so the result is a
// view of it. to_memory_config returns the component untouched when it already satisfies the request,
// which keeps the default path and matching requests zero-copy. A host component has no config to
// satisfy and cannot be relocated by a device op, so it is handed back as it is today.
Tensor place_component(const Tensor& component, const MemoryConfig& output_mem_config) {
    if (!ttnn::get_memory_config(component).has_value()) {
        return component;
    }
    return ttnn::to_memory_config(component, output_mem_config);
}

}  // namespace

Tensor _real(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return place_component(input[0], output_mem_config);
}

Tensor _imag(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return place_component(input[1], output_mem_config);
}

Tensor _angle(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return atan2(input[1], input[0], output_mem_config);
}

Tensor _is_imag(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return ttnn::eqz(input[0], output_mem_config);
}

Tensor _is_real(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return ttnn::eqz(input[1], output_mem_config);
}

ComplexTensor _conj(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    return ComplexTensor({input[0], ttnn::neg(input[1], output_mem_config)});
}

ComplexTensor _polar(const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    const Tensor& input_a = input.real();
    const Tensor& input_b = input.imag();
    Tensor c = ttnn::cos(input_b, output_mem_config);
    Tensor r = ttnn::multiply(input_a, c, std::nullopt, output_mem_config);
    c.deallocate();

    Tensor s = ttnn::sin(input_b, output_mem_config);
    Tensor i = ttnn::multiply(input_a, s, std::nullopt, output_mem_config);
    s.deallocate();

    return ComplexTensor({r, i});
}

}  // namespace ttnn::operations::complex_unary
