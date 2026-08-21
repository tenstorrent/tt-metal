// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
    const operation_attributes_t& attributes, const tensor_args_t& /*tensor_args*/) {
    if (attributes.is_spec()) {
        return program::GenericSpecFactory{};
    }
    return program::GenericMeshDescriptorFactory{};
}

void GenericOpDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    if (!attributes.is_spec()) {
        verify_no_duplicate_mesh_coord_ranges(attributes.mesh_program_descriptor().mesh_programs);
        return;
    }
    const auto& spec_program = attributes.spec_program();
    for (const auto& param : spec_program.spec.tensor_parameters) {
        TT_FATAL(
            spec_program.tensor_arg_indices.contains(param.unique_id),
            "TensorParameter '{}' has no entry in tensor_args",
            param.unique_id);
    }
    for (const auto& [name, io_index] : spec_program.tensor_arg_indices) {
        TT_FATAL(
            io_index < tensor_args.io_tensors.size(),
            "tensor argument '{}' maps to io_tensors index {}, but only {} io tensors were supplied",
            name,
            io_index,
            tensor_args.io_tensors.size());
    }
}

void GenericOpDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attributes, tensor_args);
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
    return hash;
}

namespace {

namespace m2 = tt::tt_metal::experimental;

// Every field below is structural: it changes the compiled program or its resource layout.
// Anything that varies per dispatch (runtime arg VALUES, tensor addresses) must stay out.

size_t hash_nodes(const m2::Nodes& nodes) {
    return std::visit([](const auto& n) { return ttsl::hash::hash_objects_with_default_seed(n); }, nodes);
}

size_t hash_source(const std::variant<std::filesystem::path, m2::KernelSpec::SourceCode>& source) {
    return std::visit(
        ttsl::overloaded{
            [](const std::filesystem::path& p) { return ttsl::hash::hash_objects_with_default_seed(p.string()); },
            [](const m2::KernelSpec::SourceCode& c) { return ttsl::hash::hash_objects_with_default_seed(c.code); }},
        source);
}

size_t hash_hw_config(const std::variant<m2::DataMovementHardwareConfig, m2::ComputeHardwareConfig>& hw_config) {
    return std::visit(
        ttsl::overloaded{
            [](const m2::DataMovementHardwareConfig& dm) {
                return std::visit(
                    ttsl::overloaded{
                        [](const m2::DataMovementGen1Config& c) {
                            return ttsl::hash::hash_objects_with_default_seed(0, c.processor, c.noc, c.noc_mode);
                        },
                        [](const m2::DataMovementGen2Config& c) {
                            return ttsl::hash::hash_objects_with_default_seed(
                                1, c.disable_dfb_implicit_sync_for, c.disable_dfb_implicit_sync_for_all);
                        }},
                    dm);
            },
            [](const m2::ComputeHardwareConfig& compute) {
                return std::visit(
                    ttsl::overloaded{
                        [](const m2::ComputeGen1Config& c) {
                            return ttsl::hash::hash_objects_with_default_seed(
                                2,
                                c.fpu_math_fidelity,
                                c.sfpu_precision_mode,
                                c.bfp_pack_precision_mode,
                                c.enable_32_bit_dest,
                                c.double_buffer_dest,
                                c.unpack_modes);
                        },
                        [](const m2::ComputeGen2Config& c) {
                            return ttsl::hash::hash_objects_with_default_seed(
                                3,
                                c.fpu_math_fidelity,
                                c.sfpu_precision_mode,
                                c.enable_32_bit_dest,
                                c.double_buffer_dest,
                                c.unpack_modes,
                                c.enable_2x_src_register);
                        }},
                    compute);
            }},
        hw_config);
}

size_t hash_kernel_spec(const m2::KernelSpec& kernel) {
    size_t hash = ttsl::hash::hash_objects_with_default_seed(*kernel.unique_id, kernel.num_threads);
    ttsl::hash::hash_combine(hash, hash_source(kernel.source));

    for (const auto& include_path : kernel.compiler_options.include_paths) {
        ttsl::hash::hash_combine(hash, include_path.string());
    }
    ttsl::hash::hash_combine(hash, kernel.compiler_options.defines);
    ttsl::hash::hash_combine(hash, kernel.compiler_options.opt_level);

    for (const auto& b : kernel.dfb_bindings) {
        ttsl::hash::hash_combine(
            hash,
            ttsl::hash::hash_objects_with_default_seed(
                *b.dfb_spec_name, b.accessor_name, b.endpoint_type, b.access_pattern));
    }
    for (const auto& b : kernel.semaphore_bindings) {
        ttsl::hash::hash_combine(
            hash, ttsl::hash::hash_objects_with_default_seed(*b.semaphore_spec_name, b.accessor_name));
    }
    for (const auto& b : kernel.scratchpad_bindings) {
        ttsl::hash::hash_combine(
            hash, ttsl::hash::hash_objects_with_default_seed(*b.scratchpad_spec_name, b.accessor_name));
    }
    for (const auto& b : kernel.tensor_bindings) {
        ttsl::hash::hash_combine(
            hash, ttsl::hash::hash_objects_with_default_seed(*b.tensor_parameter_name, b.accessor_name));
    }

    // Arg NAMES are structural (they become generated accessors); arg VALUES are not.
    ttsl::hash::hash_combine(hash, kernel.compile_time_args);
    for (const auto& name : kernel.runtime_arg_schema.runtime_arg_names) {
        ttsl::hash::hash_combine(hash, name);
    }
    for (const auto& name : kernel.runtime_arg_schema.common_runtime_arg_names) {
        ttsl::hash::hash_combine(hash, name);
    }

    ttsl::hash::hash_combine(hash, hash_hw_config(kernel.hw_config));
    ttsl::hash::hash_combine(
        hash,
        ttsl::hash::hash_objects_with_default_seed(
            kernel.advanced_options.num_runtime_varargs, kernel.advanced_options.num_common_runtime_varargs));
    return hash;
}

size_t hash_dataflow_buffer(const m2::DataflowBufferSpec& dfb) {
    size_t hash = ttsl::hash::hash_objects_with_default_seed(
        *dfb.unique_id,
        dfb.entry_size,
        dfb.num_entries,
        dfb.data_format_metadata,
        dfb.tile_format_metadata,
        dfb.advanced_options.allow_instance_multi_binding);
    ttsl::hash::hash_combine(hash, dfb.unpack_face_geometry_metadata.has_value());
    if (dfb.unpack_face_geometry_metadata.has_value()) {
        ttsl::hash::hash_combine(hash, *dfb.unpack_face_geometry_metadata);
    }
    ttsl::hash::hash_combine(hash, dfb.borrowed_from.has_value());
    if (dfb.borrowed_from.has_value()) {
        ttsl::hash::hash_combine(hash, **dfb.borrowed_from);
    }
    for (const auto& alias : dfb.advanced_options.alias_with) {
        ttsl::hash::hash_combine(hash, *alias);
    }
    return hash;
}

}  // namespace

ttsl::hash::hash_t compute_program_spec_hash(const m2::ProgramSpec& spec) {
    size_t hash = ttsl::hash::hash_objects_with_default_seed(spec.name);

    for (const auto& kernel : spec.kernels) {
        ttsl::hash::hash_combine(hash, hash_kernel_spec(kernel));
    }
    for (const auto& dfb : spec.dataflow_buffers) {
        ttsl::hash::hash_combine(hash, hash_dataflow_buffer(dfb));
    }
    for (const auto& semaphore : spec.semaphores) {
        ttsl::hash::hash_combine(hash, ttsl::hash::hash_objects_with_default_seed(*semaphore.unique_id));
        ttsl::hash::hash_combine(hash, hash_nodes(semaphore.target_nodes));
    }
    for (const auto& scratchpad : spec.scratchpads) {
        ttsl::hash::hash_combine(
            hash, ttsl::hash::hash_objects_with_default_seed(*scratchpad.unique_id, scratchpad.size_per_node));
    }
    for (const auto& param : spec.tensor_parameters) {
        ttsl::hash::hash_combine(
            hash,
            ttsl::hash::hash_objects_with_default_seed(
                *param.unique_id,
                param.spec,
                param.relaxations.match_padded_shape_only,
                param.relaxations.dynamic_tensor_shape));
    }
    for (const auto& work_unit : spec.work_units) {
        ttsl::hash::hash_combine(hash, ttsl::hash::hash_objects_with_default_seed(work_unit.name));
        for (const auto& kernel_name : work_unit.kernels) {
            ttsl::hash::hash_combine(hash, *kernel_name);
        }
        ttsl::hash::hash_combine(hash, hash_nodes(work_unit.target_nodes));
    }
    return hash;
}

ttsl::hash::hash_t GenericOpDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    size_t hash = 0;
    if (operation_attributes.is_spec()) {
        const auto& spec_program = operation_attributes.spec_program();
        hash = compute_program_spec_hash(spec_program.spec);
        for (const auto& [name, io_index] : spec_program.tensor_arg_indices) {
            ttsl::hash::hash_combine(hash, ttsl::hash::hash_objects_with_default_seed(*name, io_index));
        }
        return hash;
    }
    for (const auto& [mesh_coord_range, program_descriptor] :
         operation_attributes.mesh_program_descriptor().mesh_programs) {
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
    // Structural, not semantic: the only thing this op needs from io_tensors is that back() names
    // the output tensor, since tensor_return_value_t is a Tensor. A program that reads no tensor
    // (a generator: fill/iota/random) or reads and writes one (in-place) legitimately passes a
    // single tensor. A caller who forgot to pre-allocate an output still gets a precise error
    // downstream: their tensor_args entry for the output names an out-of-range io_tensors index.
    TT_FATAL(
        !io_tensors.empty(),
        "io_tensors must contain at least the output tensor as its last element, got {} tensors.",
        io_tensors.size());

    auto tensor_args = OperationType::tensor_args_t{.io_tensors = io_tensors, .output_tensor = io_tensors.back()};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
