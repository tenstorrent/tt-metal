// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fusion_dispatch_op_types.hpp"
#include "fusion_dispatch_op_program_factory.hpp"

namespace ttnn::operations::experimental::fusion {

struct FusionDispatchOpDeviceOperation {
    using operation_attributes_t = fusion_dispatch_operation_attributes_t;
    using tensor_args_t = fusion_dispatch_tensor_args_t;
    using spec_return_value_t = fusion_dispatch_spec_return_value_t;
    using tensor_return_value_t = fusion_dispatch_tensor_return_value_t;
    using program_factory_t = std::variant<program::FusionDispatchMeshProgramFactory>;

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::fusion

namespace ttnn::prim {
ttnn::operations::experimental::fusion::fusion_dispatch_tensor_return_value_t fusion_dispatch_op(
    const std::vector<Tensor>& io_tensors,
    const ttnn::operations::experimental::fusion::fusion_dispatch_operation_attributes_t& operation_attributes);
}  // namespace ttnn::prim
