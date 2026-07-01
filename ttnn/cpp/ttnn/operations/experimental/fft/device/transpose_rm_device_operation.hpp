// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Device-op skeleton for ttnn::prim::transpose_rm — swap the last two
// dims of a ROW_MAJOR fp32/bf16 tensor.

#pragma once

#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"

#include "transpose_rm_device_operation_types.hpp"
#include "transpose_rm_factory.hpp"

namespace ttnn::experimental::prim {

struct TransposeRmDeviceOperation {
    using operation_attributes_t = TransposeRmParams;
    using tensor_args_t          = TransposeRmTensorArgs;

    // Single-tensor return (just the transposed tensor).
    using spec_return_value_t   = TensorSpec;
    using tensor_return_value_t = Tensor;

    using program_factory_t = std::variant<TransposeRmFactory>;

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

// Public entry point.  Swap last two dims of `input`.  Requires A, C ≥ 32
// and multiples of 32; fp32 or bf16; ROW_MAJOR.
Tensor transpose_rm(const Tensor& input);

}  // namespace ttnn::prim
