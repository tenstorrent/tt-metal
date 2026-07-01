// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Device-op skeleton for ttnn::prim::complex_mul — fused ROW_MAJOR
// elementwise complex multiply.  See complex_mul_device_operation_types.hpp
// for the kernel semantics.

#pragma once

#include <tuple>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"

#include "complex_mul_device_operation_types.hpp"
#include "complex_mul_factory.hpp"

namespace ttnn::experimental::prim {

struct ComplexMulDeviceOperation {
    using operation_attributes_t = ComplexMulParams;
    using tensor_args_t          = ComplexMulTensorArgs;

    using spec_return_value_t   = std::tuple<TensorSpec, TensorSpec>;
    using tensor_return_value_t = std::tuple<Tensor, Tensor>;

    using program_factory_t = std::variant<ComplexMulFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(
        const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(
        const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t&, const tensor_args_t&);
    static tt::stl::hash::hash_t compute_program_hash(
        const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

// Public entry point used by the op-API layer (complex_mul.cpp).
//   All four input tensors must share the same shape, dtype (fp32 or
//   bf16), and ROW_MAJOR layout.  Returns (out_real, out_imag) with
//   the same spec as the inputs.
std::tuple<Tensor, Tensor> complex_mul(
    const Tensor& a_real,
    const Tensor& a_imag,
    const Tensor& b_real,
    const Tensor& b_imag);

}  // namespace ttnn::prim
