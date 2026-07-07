// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "depth_to_space_pad_device_operation_types.hpp"
#include "depth_to_space_pad_program_factory.hpp"

namespace ttnn::experimental::prim {

struct DepthToSpacePadDeviceOperation {
    using operation_attributes_t = DepthToSpacePadParams;
    using tensor_args_t = DepthToSpacePadInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<DepthToSpacePadMeshWorkloadFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

// Primitive entry point: fused depth-to-space + pad into a newly-allocated padded buffer (interior
// only; border left for a later halo + halo_scatter border_only). Returns the padded buffer.
Tensor depth_to_space_pad(const Tensor& conv_out, const ttnn::experimental::prim::DepthToSpacePadParams& params);

}  // namespace ttnn::prim
