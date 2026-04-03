// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "patchable_generic_op_types.hpp"
#include "patchable_generic_op_program_factory.hpp"

namespace ttnn::operations::experimental::generic {

// Like GenericOpDeviceOperation, but uses address-slot scanning
// (PR 39972 pattern) for cache-hit overrides instead of full RT arg copy.
struct PatchableGenericOpDeviceOperation {
    using operation_attributes_t = patchable_operation_attributes_t;
    using tensor_args_t = patchable_tensor_args_t;
    using spec_return_value_t = patchable_spec_return_value_t;
    using tensor_return_value_t = patchable_tensor_return_value_t;
    using program_factory_t = std::variant<program::PatchableGenericMeshProgramFactory>;

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::generic

namespace ttnn::prim {
ttnn::operations::experimental::generic::patchable_tensor_return_value_t patchable_generic_op(
    const std::vector<Tensor>& io_tensors,
    const ttnn::operations::experimental::generic::patchable_operation_attributes_t& operation_attributes);
}  // namespace ttnn::prim
