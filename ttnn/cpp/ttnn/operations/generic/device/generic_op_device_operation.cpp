// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/device_operation.hpp"
#include "generic_op_device_operation.hpp"
#include "generic_op_device_operation_types.hpp"

#include <tt-metalium/program_descriptors.hpp>
#include <tt_stl/reflection.hpp>
#include <unordered_set>

namespace ttnn::operations::generic {

using namespace tt::tt_metal;

void verify_no_duplicate_mesh_coord_ranges(
    const tt::tt_metal::experimental::MeshProgramDescriptor::MeshPrograms& mesh_programs) {
    std::unordered_set<ttnn::MeshCoordinateRange> seen;
    seen.reserve(mesh_programs.size());
    for (const auto& [range, _] : mesh_programs) {
        auto [it, inserted] = seen.insert(range);
        TT_FATAL(inserted, "Duplicate MeshCoordinateRange found in MeshProgramDescriptor: {}", range);
    }
}

GenericOpDeviceOperation::program_factory_t GenericOpDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return program::GenericMeshProgramFactory{};
}

void GenericOpDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& /*tensor_args*/) {
    verify_no_duplicate_mesh_coord_ranges(attributes.mesh_programs);
}

void GenericOpDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& /*tensor_args*/) {
    verify_no_duplicate_mesh_coord_ranges(attributes.mesh_programs);
}

spec_return_value_t GenericOpDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    // User has to do this. Just referencing last element (preallocated output tensor).
    return tensor_args.output_tensor.tensor_spec();
}

tensor_return_value_t GenericOpDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    // Don't create anything, user is passing output tensor.
    return tensor_args.output_tensor;
}

tt::stl::hash::hash_t GenericOpDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    size_t hash = 0;
    for (const auto& [mesh_coord_range, program_descriptor] : operation_attributes.mesh_programs) {
        ttsl::hash::hash_combine(hash, mesh_coord_range);
        ttsl::hash::hash_combine(hash, compute_program_descriptor_hash(program_descriptor));
    }
    return hash;
}

}  // namespace ttnn::operations::generic

namespace ttnn::prim {
ttnn::operations::generic::tensor_return_value_t generic_op(
    const std::vector<Tensor>& io_tensors,
    const ttnn::operations::generic::operation_attributes_t& operation_attributes) {
    using OperationType = ttnn::operations::generic::GenericOpDeviceOperation;
    TT_FATAL(
        io_tensors.size() >= 2,
        "io_tensors must contain at least one input tensor and one output tensor, got {} tensors.",
        io_tensors.size());

    auto tensor_args = OperationType::tensor_args_t{.io_tensors = io_tensors, .output_tensor = io_tensors.back()};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
