// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/device_operation.hpp"
#include "generic_op_device_operation.hpp"
#include "generic_op_device_operation_types.hpp"

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

tt::stl::hash::hash_t compute_program_descriptor_hash(const tt::tt_metal::ProgramDescriptor& program_descriptor) {
    if (program_descriptor.custom_program_hash) {
        return *program_descriptor.custom_program_hash;
    }

    auto hash_kernel = [&](const KernelDescriptor& kernel) -> size_t {
        return ttsl::hash::hash_objects_with_default_seed(
            kernel.kernel_source,
            kernel.source_type,
            kernel.core_ranges,
            kernel.compile_time_args,
            kernel.defines,
            kernel.common_runtime_args.size(),
            kernel.runtime_args.size(),
            kernel.config.index(),
            kernel.config);
    };

    auto hash_cb_format_descriptor = [&](const CBFormatDescriptor& format_descriptor) -> size_t {
        return ttsl::hash::hash_objects_with_default_seed(
            format_descriptor.buffer_index,
            format_descriptor.data_format,
            format_descriptor.page_size,
            format_descriptor.tile);
    };

    auto hash_circular_buffer = [&](const CBDescriptor& cb) -> size_t {
        size_t hash = cb.core_ranges.size();
        for (const auto& core_range : cb.core_ranges.ranges()) {
            ttsl::hash::hash_combine(hash, core_range);
        }
        ttsl::hash::hash_combine(hash, cb.format_descriptors.size());
        for (const auto& format_descriptor : cb.format_descriptors) {
            ttsl::hash::hash_combine(hash, hash_cb_format_descriptor(format_descriptor));
        }
        ttsl::hash::hash_combine(hash, cb.remote_format_descriptors.size());
        for (const auto& format_descriptor : cb.remote_format_descriptors) {
            ttsl::hash::hash_combine(hash, hash_cb_format_descriptor(format_descriptor));
        }
        ttsl::hash::hash_combine(hash, cb.buffer != nullptr);
        ttsl::hash::hash_combine(hash, cb.global_circular_buffer != nullptr);
        return hash;
    };

    auto hash_semaphore = [&](const SemaphoreDescriptor& semaphore) -> size_t {
        return ttsl::hash::hash_objects_with_default_seed(
            semaphore.core_ranges, semaphore.core_type, semaphore.initial_value);
    };

    size_t hash = 0;
    for (const auto& kernel : program_descriptor.kernels) {
        ttsl::hash::hash_combine(hash, hash_kernel(kernel));
    }
    for (const auto& cb : program_descriptor.cbs) {
        ttsl::hash::hash_combine(hash, hash_circular_buffer(cb));
    }
    for (const auto& semaphore : program_descriptor.semaphores) {
        ttsl::hash::hash_combine(hash, hash_semaphore(semaphore));
    }
    return hash;
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
