// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/device_operation.hpp"
#include "generic_op_device_operation.hpp"
#include "generic_op_device_operation_types.hpp"

#include <tt_stl/reflection.hpp>

#include <algorithm>
#include <cmath>
#include <variant>
#include <vector>

namespace ttnn::operations::generic {

using namespace tt::tt_metal;

void verify_no_duplicate_mesh_coord_ranges(
    const tt::tt_metal::experimental::MeshProgramDescriptor::MeshPrograms& mesh_programs) {
    for (size_t i = 0; i < mesh_programs.size(); ++i) {
        const auto& range_i = mesh_programs[i].first;
        for (size_t j = i + 1; j < mesh_programs.size(); ++j) {
            const auto& range_j = mesh_programs[j].first;
            TT_FATAL(
                !range_i.intersects(range_j),
                "Overlapping MeshCoordinateRange found in MeshProgramDescriptor: {} overlaps {}",
                range_i,
                range_j);
        }
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

tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t>
GenericOpDeviceOperation::create_op_performance_model(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using PerformanceModel = tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t>;

    // Largest declared value over the devices' programs, as in MeshProgramDescriptor::attribute_values
    const ProgramDescriptor* first_declaring_program = nullptr;
    ProgramPerformanceDescriptor declared;
    for (const auto& [range, program] : operation_attributes.mesh_programs) {
        if (!program.performance) {
            continue;
        }
        if (first_declaring_program == nullptr) {
            first_declaring_program = &program;
            declared.math_fidelity = program.performance->math_fidelity;
        }
        declared.dram_bytes_read = std::max(declared.dram_bytes_read, program.performance->dram_bytes_read);
        declared.dram_bytes_written = std::max(declared.dram_bytes_written, program.performance->dram_bytes_written);
        declared.flops = std::max(declared.flops, program.performance->flops);
    }
    const auto& output = tensor_args.output_tensor;
    if (first_declaring_program == nullptr || output.storage_type() != StorageType::DEVICE) {
        return PerformanceModel{};
    }

    MathFidelity math_fidelity = MathFidelity::LoFi;
    if (declared.math_fidelity) {
        math_fidelity = *declared.math_fidelity;
    } else {
        for (const auto& kernel : first_declaring_program->kernels) {
            if (const auto* config = std::get_if<ComputeConfigDescriptor>(&kernel.config)) {
                math_fidelity = config->math_fidelity;
                break;
            }
        }
    }

    // Same peak as the matmul model: 4096 FLOPs per core per cycle at LoFi on the full compute grid
    constexpr double tensix_flops_per_cycle_lofi = 4096;
    const CoreCoord compute_grid = output.device()->compute_with_storage_grid_size();
    const double num_cores = static_cast<double>(compute_grid.x * compute_grid.y);
    const int ideal_compute_cycles = static_cast<int>(std::ceil(
        static_cast<double>(declared.flops) / (num_cores * tensix_flops_per_cycle_lofi) *
        PerformanceModel::fidelity_multiplier(math_fidelity)));

    std::vector<Tensor> input_tensors(tensor_args.io_tensors.begin(), tensor_args.io_tensors.end() - 1);
    PerformanceModel model(input_tensors, tensor_return_value, ideal_compute_cycles);

    const double dram_bytes = static_cast<double>(declared.dram_bytes_read + declared.dram_bytes_written);
    model.ideal_bandwidth_ns = static_cast<int>(
        std::ceil(dram_bytes / tt::tt_metal::operation::peak_dram_bandwidth_gb_s(output.device()->arch())));
    model.ideal_ns = std::max(model.ideal_compute_ns, model.ideal_bandwidth_ns);
    return model;
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
            kernel.opt_level.has_value(),
            kernel.opt_level.value_or(KernelBuildOptLevel{}),
            kernel.compiler_include_paths,
            kernel.common_runtime_args.size(),
            // Blaze-only experimental named args (issue #50953): hash the FULL named-RT-arg schema
            // (names/lengths/dispatch across all 4 variants), NOT values. Replaces the previous
            // partial hashing that used .size() of only 3 of 4 variants and never the names.
            tt::tt_metal::experimental::blaze::hash_named_args_schema(kernel.blaze_named_args),
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
    ttsl::hash::hash_combine(hash, tt::tt_metal::internal::hash_reload_table(program_descriptor.reload_table));
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

}  // namespace ttnn::operations::generic

namespace ttnn::prim {
namespace {

ttnn::operations::generic::GenericOpDeviceOperation::tensor_args_t make_tensor_args(
    const std::vector<Tensor>& io_tensors) {
    TT_FATAL(
        io_tensors.size() >= 2,
        "io_tensors must contain at least one input tensor and one output tensor, got {} tensors.",
        io_tensors.size());
    return {.io_tensors = io_tensors, .output_tensor = io_tensors.back()};
}

}  // namespace

ttnn::operations::generic::tensor_return_value_t generic_op(
    const std::vector<Tensor>& io_tensors,
    const ttnn::operations::generic::operation_attributes_t& operation_attributes) {
    using OperationType = ttnn::operations::generic::GenericOpDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(operation_attributes, make_tensor_args(io_tensors));
}

void prepare_generic_op(
    const std::vector<Tensor>& io_tensors,
    const ttnn::operations::generic::operation_attributes_t& operation_attributes) {
    using OperationType = ttnn::operations::generic::GenericOpDeviceOperation;
    ttnn::device_operation::prepare<OperationType>(operation_attributes, make_tensor_args(io_tensors));
}
}  // namespace ttnn::prim
