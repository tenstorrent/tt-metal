// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/device_operation.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"
#include "generic_op_device_operation.hpp"
#include "generic_op_device_operation_types.hpp"

#include <tt_stl/reflection.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_cache.hpp>
#include <tt-metalium/mesh_device.hpp>
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

ttsl::hash::hash_t compute_program_descriptor_hash(const tt::tt_metal::ProgramDescriptor& program_descriptor) {
    if (program_descriptor.custom_program_hash) {
        return *program_descriptor.custom_program_hash;
    }

    auto hash_kernel = [&](const KernelDescriptor& kernel) -> size_t {
        return ttsl::hash::hash_objects_with_default_seed(
            kernel.kernel_source,
            kernel.source_type,
            kernel.core_ranges,
            kernel.compile_time_args,
            kernel.named_compile_time_args,
            kernel.defines,
            kernel.common_runtime_args.size(),
            kernel.named_common_runtime_args.size(),
            kernel.named_per_core_runtime_args.size(),
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
        size_t hash = cb.total_size;
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

ttsl::hash::hash_t GenericOpDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    size_t hash = 0;
    for (const auto& [mesh_coord_range, program_descriptor] : operation_attributes.mesh_programs) {
        ttsl::hash::hash_combine(hash, mesh_coord_range);
        ttsl::hash::hash_combine(hash, compute_program_descriptor_hash(program_descriptor));
    }
    return hash;
}

ProgramCompileInfo get_program_compile_info(
    const std::vector<Tensor>& io_tensors, const tt::tt_metal::ProgramDescriptor& program_descriptor) {
    TT_FATAL(!io_tensors.empty(), "io_tensors must not be empty");
    auto* mesh_device = io_tensors.front().device();
    TT_FATAL(mesh_device != nullptr, "Tensor must be on a device");

    // Build the same MeshProgramDescriptor that the SPMD generic_op path creates
    operation_attributes_t attrs;
    attrs.mesh_programs.emplace_back(ttnn::MeshCoordinateRange(mesh_device->shape()), program_descriptor);

    tensor_args_t tensor_args{.io_tensors = io_tensors, .output_tensor = io_tensors.back()};

    // Compute the exact cache hash used by the device operation
    using Adapter = ttnn::device_operation::MeshDeviceOperationAdapter<GenericOpDeviceOperation>;
    auto hash = Adapter::compute_mesh_workload_hash(mesh_device, attrs, tensor_args);

    // Look up the cached program
    auto& cache = mesh_device->get_program_cache();
    TT_FATAL(
        cache.contains(hash),
        "Program not found in cache. Call ttnn.generic_op(io_tensors, program_descriptor) first.");
    auto& factory = cache.get(hash);

    // Extract the Program from the type-erased cache
    using cached_t = program::GenericMeshProgramFactory::cached_mesh_workload_t;
    auto& cached_workload = factory.cached_program.get<cached_t>();
    auto& programs = cached_workload.workload.get_programs();
    TT_FATAL(!programs.empty(), "Cached workload has no programs");
    auto& program = programs.begin()->second;

    // Read ProgramConfig for TENSIX core type (index 0 — always the first programmable core type)
    constexpr uint32_t tensix_idx = 0;
    auto pc = tt::tt_metal::detail::get_program_config_info(program, tensix_idx);

    // Read kernel metadata with binary sizes (pass a device to get packed_size)
    auto devices = mesh_device->get_devices();
    tt::tt_metal::IDevice* device = devices.empty() ? nullptr : devices.front();
    auto kernel_metas = tt::tt_metal::detail::collect_kernel_meta(program, device);

    // Read per-core-type config sizes
    auto config_sizes = tt::tt_metal::detail::get_program_config_sizes(program);

    return ProgramCompileInfo{
        .rta_offset = pc.rta_offset,
        .sem_offset = pc.sem_offset,
        .sem_size = pc.sem_size,
        .cb_offset = pc.cb_offset,
        .cb_size = pc.cb_size,
        .dfb_offset = pc.dfb_offset,
        .dfb_size = pc.dfb_size,
        .local_cb_size = pc.local_cb_size,
        .kernel_text_offset = pc.kernel_text_offset,
        .kernel_text_size = pc.kernel_text_size,
        .program_config_sizes = std::move(config_sizes),
        .kernel_metas = std::move(kernel_metas),
    };
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
