// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/device_operation.hpp"
#include "generic_op_device_operation.hpp"

#include "ttnn/metal2_field_count.hpp"
#include "generic_op_device_operation_types.hpp"

#include <filesystem>
#include <variant>

#include <tt_stl/overloaded.hpp>
#include <tt_stl/reflection.hpp>

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
    ttsl::hash::hash_combine(hash, tt::tt_metal::internal::hash_reload_table(program_descriptor.reload_table));
    return hash;
}

namespace {

namespace m2 = tt::tt_metal::experimental;
using ttsl::hash::hash_combine;
using ttsl::hash::hash_objects_with_default_seed;

// Every field below is structural: it changes the compiled program or its resource layout.
// Anything that varies per dispatch (runtime arg VALUES, tensor addresses) must stay out.
// Each helper enumerates every field of its struct; the TTNN_M2_ASSERT_FIELD_COUNT next to it
// fails the build when tt_metal adds a field, so the hash cannot silently fall behind the API
// (a missing field means two different programs share one cache key).

template <typename Range>
void hash_combine_each(size_t& hash, const Range& range) {
    for (const auto& item : range) {
        hash_combine(hash, item);
    }
}

template <typename Range>
void hash_combine_each_name(size_t& hash, const Range& names) {
    for (const auto& name : names) {
        hash_combine(hash, *name);
    }
}

size_t hash_nodes(const m2::Nodes& nodes) {
    return std::visit([](const auto& n) { return hash_objects_with_default_seed(n); }, nodes);
}

TTNN_M2_ASSERT_FIELD_COUNT(m2::TensorSpecRelaxations, 4);
size_t hash_relaxations(const m2::TensorSpecRelaxations& r) {
    return hash_objects_with_default_seed(
        r.match_padded_shape_only, r.dynamic_tensor_shape, r.relax_logical_rank, r.match_page_size);
}

TTNN_M2_ASSERT_FIELD_COUNT(m2::TensorParameter, 3);
size_t hash_tensor_parameter(const m2::TensorParameter& param) {
    size_t hash = hash_objects_with_default_seed(*param.unique_id, param.spec);
    hash_combine(hash, hash_relaxations(param.relaxations));
    return hash;
}

TTNN_M2_ASSERT_FIELD_COUNT(m2::DFBAdvancedOptions, 3);
TTNN_M2_ASSERT_FIELD_COUNT(m2::DataflowBufferSpec, 7);
size_t hash_dataflow_buffer(const m2::DataflowBufferSpec& dfb) {
    size_t hash = hash_objects_with_default_seed(
        *dfb.unique_id, dfb.entry_size, dfb.num_entries, dfb.data_format_metadata, dfb.tile_format_metadata);
    hash_combine(hash, dfb.borrowed_from.has_value());
    if (dfb.borrowed_from.has_value()) {
        hash_combine(hash, **dfb.borrowed_from);
    }
    hash_combine_each_name(hash, dfb.advanced_options.alias_with);
    hash_combine(hash, dfb.advanced_options.allow_instance_multi_binding);
    hash_combine_each_name(hash, dfb.advanced_options.prefetcher_pipe_relays);
    return hash;
}

// SemaphoreAdvancedOptions::initial_value is deprecated and always 0 from this op's callers.
TTNN_M2_ASSERT_FIELD_COUNT(m2::SemaphoreAdvancedOptions, 2);
TTNN_M2_ASSERT_FIELD_COUNT(m2::SemaphoreSpec, 3);
size_t hash_semaphore(const m2::SemaphoreSpec& semaphore) {
    size_t hash = hash_objects_with_default_seed(*semaphore.unique_id, semaphore.advanced_options.max_value);
    hash_combine(hash, hash_nodes(semaphore.target_nodes));
    return hash;
}

TTNN_M2_ASSERT_FIELD_COUNT(m2::ScratchpadSpec, 4);
size_t hash_scratchpad(const m2::ScratchpadSpec& scratchpad) {
    return hash_objects_with_default_seed(
        *scratchpad.unique_id,
        scratchpad.size_per_node,
        scratchpad.data_format_metadata,
        scratchpad.tile_format_metadata);
}

TTNN_M2_ASSERT_FIELD_COUNT(m2::DataMovementHardwareConfig::DataMovement1XXConfig, 4);
TTNN_M2_ASSERT_FIELD_COUNT(m2::DataMovementHardwareConfig::DataMovement2XXConfig, 2);
TTNN_M2_ASSERT_FIELD_COUNT(m2::DataMovementHardwareConfig, 2);
TTNN_M2_ASSERT_FIELD_COUNT(m2::ComputeHardwareConfig::Compute1XXConfig, 1);
TTNN_M2_ASSERT_FIELD_COUNT(m2::ComputeHardwareConfig::Compute2XXConfig, 0);
TTNN_M2_ASSERT_FIELD_COUNT(m2::ComputeHardwareConfig, 7);
size_t hash_hw_config(const std::variant<m2::DataMovementHardwareConfig, m2::ComputeHardwareConfig>& hw_config) {
    return std::visit(
        ttsl::overloaded{
            [](const m2::DataMovementHardwareConfig& dm) {
                size_t hash = hash_objects_with_default_seed(0);
                if (const auto& c = dm.config_1xx) {
                    hash_combine(
                        hash,
                        hash_objects_with_default_seed(1, c->processor, c->noc, c->noc_mode, c->dedicated_noc_sharing));
                }
                if (const auto& c = dm.config_2xx) {
                    hash_combine(hash, hash_objects_with_default_seed(2, c->disable_dfb_implicit_sync_for_all));
                    hash_combine_each_name(hash, c->disable_dfb_implicit_sync_for);
                }
                return hash;
            },
            [](const m2::ComputeHardwareConfig& c) {
                size_t hash = hash_objects_with_default_seed(
                    3,
                    c.fpu_math_fidelity,
                    c.sfpu_precision_mode,
                    c.enable_32_bit_dest,
                    c.double_buffer_dest,
                    c.unpack_modes);
                if (c.config_1xx) {
                    hash_combine(hash, hash_objects_with_default_seed(4, c.config_1xx->bfp_pack_precision_mode));
                }
                if (c.config_2xx) {
                    hash_combine(hash, hash_objects_with_default_seed(5));
                }
                return hash;
            }},
        hw_config);
}

TTNN_M2_ASSERT_FIELD_COUNT(m2::KernelSpec::SourceCode, 1);
size_t hash_source(const std::variant<std::filesystem::path, m2::KernelSpec::SourceCode>& source) {
    return std::visit(
        ttsl::overloaded{
            [](const std::filesystem::path& p) { return hash_objects_with_default_seed(0, p.string()); },
            [](const m2::KernelSpec::SourceCode& c) { return hash_objects_with_default_seed(1, c.code); }},
        source);
}

TTNN_M2_ASSERT_FIELD_COUNT(m2::KernelSpec::CompilerOptions, 3);
size_t hash_compiler_options(const m2::KernelSpec::CompilerOptions& options) {
    size_t hash = hash_objects_with_default_seed(options.defines, options.opt_level);
    for (const auto& include_path : options.include_paths) {
        hash_combine(hash, include_path.string());
    }
    return hash;
}

TTNN_M2_ASSERT_FIELD_COUNT(m2::DFBBinding, 5);
TTNN_M2_ASSERT_FIELD_COUNT(m2::SemaphoreBinding, 2);
TTNN_M2_ASSERT_FIELD_COUNT(m2::ScratchpadBinding, 2);
TTNN_M2_ASSERT_FIELD_COUNT(m2::TensorBinding, 2);
TTNN_M2_ASSERT_FIELD_COUNT(m2::KernelSpec::RuntimeArgSchema, 2);
TTNN_M2_ASSERT_FIELD_COUNT(m2::KernelAdvancedOptions::TensorBindingSequence, 2);
TTNN_M2_ASSERT_FIELD_COUNT(m2::KernelAdvancedOptions::PrefetcherPipeBinding, 2);
// KernelAdvancedOptions::num_runtime_varargs_per_node is deprecated and not hashed.
TTNN_M2_ASSERT_FIELD_COUNT(m2::KernelAdvancedOptions, 6);
size_t hash_kernel_advanced_options(const m2::KernelAdvancedOptions& options) {
    size_t hash = hash_objects_with_default_seed(options.num_runtime_varargs, options.num_common_runtime_varargs);
    hash_combine_each(hash, options.compile_time_varargs);
    for (const auto& sequence : options.tensor_binding_sequences) {
        hash_combine(hash, sequence.sequence_name);
        hash_combine_each(hash, sequence.members);
    }
    for (const auto& binding : options.prefetcher_pipe_bindings) {
        hash_combine(hash, binding.accessor_name);
        hash_combine_each_name(hash, binding.pipe_parameter_names);
    }
    return hash;
}

TTNN_M2_ASSERT_FIELD_COUNT(m2::KernelSpec, 12);
size_t hash_kernel_spec(const m2::KernelSpec& kernel) {
    size_t hash = hash_objects_with_default_seed(*kernel.unique_id, kernel.num_threads);
    hash_combine(hash, hash_source(kernel.source));
    hash_combine(hash, hash_compiler_options(kernel.compiler_options));
    for (const auto& b : kernel.dfb_bindings) {
        hash_combine(
            hash,
            hash_objects_with_default_seed(
                *b.dfb_spec_name, b.accessor_name, b.endpoint_type, b.access_pattern, b.block_size));
    }
    for (const auto& b : kernel.semaphore_bindings) {
        hash_combine(hash, hash_objects_with_default_seed(*b.semaphore_spec_name, b.accessor_name));
    }
    for (const auto& b : kernel.scratchpad_bindings) {
        hash_combine(hash, hash_objects_with_default_seed(*b.scratchpad_spec_name, b.accessor_name));
    }
    for (const auto& b : kernel.tensor_bindings) {
        hash_combine(hash, hash_objects_with_default_seed(*b.tensor_parameter_name, b.accessor_name));
    }
    // Arg NAMES are structural (they become generated accessors); runtime arg VALUES are not.
    hash_combine(hash, kernel.compile_time_args);
    hash_combine_each(hash, kernel.runtime_arg_schema.runtime_arg_names);
    hash_combine_each(hash, kernel.runtime_arg_schema.common_runtime_arg_names);
    hash_combine(hash, hash_hw_config(kernel.hw_config));
    hash_combine(hash, hash_kernel_advanced_options(kernel.advanced_options));
    return hash;
}

TTNN_M2_ASSERT_FIELD_COUNT(m2::WorkUnitSpec, 3);
size_t hash_work_unit(const m2::WorkUnitSpec& work_unit) {
    size_t hash = hash_objects_with_default_seed(work_unit.name);
    hash_combine_each_name(hash, work_unit.kernels);
    hash_combine(hash, hash_nodes(work_unit.target_nodes));
    return hash;
}

TTNN_M2_ASSERT_FIELD_COUNT(m2::PrefetcherPipeParameter, 4);
TTNN_M2_ASSERT_FIELD_COUNT(m2::ProgramAdvancedOptions, 1);
size_t hash_program_advanced_options(const m2::ProgramAdvancedOptions& options) {
    size_t hash = hash_objects_with_default_seed(options.prefetcher_pipe_parameters.size());
    for (const auto& pipe : options.prefetcher_pipe_parameters) {
        hash_combine(hash, hash_objects_with_default_seed(*pipe.unique_id, pipe.ring_size, pipe.entry_size));
        hash_combine(hash, hash_nodes(pipe.receivers));
    }
    return hash;
}

}  // namespace

TTNN_M2_ASSERT_FIELD_COUNT(m2::ProgramSpec, 8);
ttsl::hash::hash_t compute_program_spec_hash(const m2::ProgramSpec& spec) {
    size_t hash = hash_objects_with_default_seed(spec.name);
    for (const auto& kernel : spec.kernels) {
        hash_combine(hash, hash_kernel_spec(kernel));
    }
    for (const auto& dfb : spec.dataflow_buffers) {
        hash_combine(hash, hash_dataflow_buffer(dfb));
    }
    for (const auto& semaphore : spec.semaphores) {
        hash_combine(hash, hash_semaphore(semaphore));
    }
    for (const auto& scratchpad : spec.scratchpads) {
        hash_combine(hash, hash_scratchpad(scratchpad));
    }
    for (const auto& param : spec.tensor_parameters) {
        hash_combine(hash, hash_tensor_parameter(param));
    }
    for (const auto& work_unit : spec.work_units) {
        hash_combine(hash, hash_work_unit(work_unit));
    }
    hash_combine(hash, hash_program_advanced_options(spec.advanced_options));
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
namespace {

ttnn::operations::generic::GenericOpDeviceOperation::tensor_args_t make_tensor_args(
    const std::vector<Tensor>& io_tensors) {
    // Structural, not semantic: the only thing this op needs from io_tensors is that back() names
    // the output tensor, since tensor_return_value_t is a Tensor. A program that reads no tensor
    // (a generator: fill/iota/random) or reads and writes one (in-place) legitimately passes a
    // single tensor. A caller who forgot to pre-allocate an output still gets a precise error
    // downstream: their tensor_args entry for the output names an out-of-range io_tensors index.
    TT_FATAL(
        !io_tensors.empty(),
        "io_tensors must contain at least the output tensor as its last element, got {} tensors.",
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
