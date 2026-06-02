// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/device_operation.hpp"
#include "fusion_dispatch_op_device_operation.hpp"

#include <tt_stl/reflection.hpp>

// Reuse the hash function from generic_op
namespace ttnn::operations::generic {
ttsl::hash::hash_t compute_program_descriptor_hash(const tt::tt_metal::ProgramDescriptor& program_descriptor);
}

namespace ttnn::operations::experimental::fusion {

using namespace tt::tt_metal;

void FusionDispatchOpDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t&, const tensor_args_t&) {}

void FusionDispatchOpDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t&, const tensor_args_t&) {}

fusion_dispatch_spec_return_value_t FusionDispatchOpDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    return tensor_args.output_tensor.tensor_spec();
}

fusion_dispatch_tensor_return_value_t FusionDispatchOpDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    return tensor_args.output_tensor;
}

ttsl::hash::hash_t FusionDispatchOpDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t&) {
    // Must differ from GenericOpDeviceOperation::compute_program_hash — same descriptor would
    // otherwise hit the wrong cached_mesh_workload_t layout (segfault in override).
    size_t hash = ttsl::hash::type_hash<FusionDispatchOpDeviceOperation>;
    for (const auto& [mesh_coord_range, program_descriptor] : operation_attributes.mesh_programs) {
        ttsl::hash::hash_combine(hash, mesh_coord_range);
        ttsl::hash::hash_combine(hash, ttnn::operations::generic::compute_program_descriptor_hash(program_descriptor));
    }
    return hash;
}

ProgramDescriptor FusionDispatchOpDeviceOperation::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/) {
    // Python pre-patches the descriptor's runtime args and CB buffers in-place
    // before invoking the primitive, so we can return the (single) descriptor
    // as-is.  In practice mesh_programs always carries exactly one entry that
    // spans the full mesh; if more than one is present the first is used and
    // the rest are currently ignored (matching the assumption baked into the
    // Python entry points in fusion_dispatch_op_nanobind.cpp).
    const auto& mesh_programs = operation_attributes.mesh_programs;
    TT_FATAL(!mesh_programs.empty(), "fusion_dispatch_op: mesh_programs must not be empty");
    return mesh_programs.front().second;
}

}  // namespace ttnn::operations::experimental::fusion

namespace ttnn::prim {
ttnn::operations::experimental::fusion::fusion_dispatch_tensor_return_value_t fusion_dispatch_op(
    const std::vector<Tensor>& io_tensors,
    const ttnn::operations::experimental::fusion::fusion_dispatch_operation_attributes_t& operation_attributes) {
    using OperationType = ttnn::operations::experimental::fusion::FusionDispatchOpDeviceOperation;
    TT_FATAL(
        io_tensors.size() >= 2,
        "io_tensors must contain at least one input tensor and one output tensor, got {} tensors.",
        io_tensors.size());

    auto tensor_args = OperationType::tensor_args_t{.io_tensors = io_tensors, .output_tensor = io_tensors.back()};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
