// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Device-op skeleton for ttnn::prim::rebank_rm_merge — inverse page-size
// conversion for ROW_MAJOR fp32/bf16 tensors.

#pragma once

#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"

#include "rebank_rm_merge_device_operation_types.hpp"
#include "rebank_rm_merge_factory.hpp"

namespace ttnn::experimental::prim {

struct RebankRmMergeDeviceOperation {
    using operation_attributes_t = RebankRmMergeParams;
    using tensor_args_t          = RebankRmMergeTensorArgs;

    using spec_return_value_t   = TensorSpec;
    using tensor_return_value_t = Tensor;

    using program_factory_t = std::variant<RebankRmMergeFactory>;

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

// Public entry point.  Merges (B*N2, N1) → (B, N1*N2).
// chunks_per_merge = N2 (must be pow-2; input rows must be divisible by it).
Tensor rebank_rm_merge(const Tensor& input, uint32_t chunks_per_merge);

}  // namespace ttnn::prim
