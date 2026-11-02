// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "generic_spec_op_device_operation.hpp"

#include <algorithm>
#include <filesystem>
#include <optional>
#include <string>
#include <variant>

#include <tt_stl/assert.hpp>
#include <tt_stl/reflection.hpp>

namespace ttnn::operations::generic {

using namespace tt::tt_metal::experimental;

namespace {

// Rebuilt on every dispatch: tensor references must point at this call's io_tensors.
ProgramRunArgs make_program_run_args(
    const GenericSpecOpDeviceOperation::operation_attributes_t& operation_attributes,
    const GenericSpecOpDeviceOperation::tensor_args_t& tensor_args) {
    ProgramRunArgs run_args;
    run_args.kernel_run_args = operation_attributes.kernel_run_args;
    for (const auto& [tensor_parameter_name, io_index] : operation_attributes.tensor_bindings) {
        run_args.tensor_args.emplace(
            tensor_parameter_name, TensorArgument{tensor_args.io_tensors.at(io_index).mesh_tensor()});
    }
    return run_args;
}

// Hashes everything in a ProgramSpec that shapes the compiled program. Of the advanced options, only
// DFBAdvancedOptions.alias_with and .borrowed_offset are settable from Python; both are hashed, and the
// rest keep their defaults. Arg values go through std::hash, which covers both ArgValue and the plain
// uint32_t of older tt-metal.
std::size_t hash_program_spec(const ProgramSpec& spec) {
    std::size_t hash = 0;
    auto combine = [&hash](const auto&... values) { (ttsl::hash::hash_combine(hash, values), ...); };
    // An unset format must not hash like DataFormat::Float32 (value 0).
    auto combine_format = [&combine](const std::optional<tt::DataFormat>& format) {
        combine(format.has_value());
        if (format.has_value()) {
            combine(*format);
        }
    };
    combine(spec.name);
    for (const auto& kernel : spec.kernels) {
        combine(*kernel.unique_id, kernel.source.index(), kernel.num_threads, kernel.compiler_options.opt_level);
        if (const auto* path = std::get_if<std::filesystem::path>(&kernel.source)) {
            combine(path->string());
        } else {
            combine(std::get<KernelSpec::SourceCode>(kernel.source).code);
        }
        for (const auto& include_path : kernel.compiler_options.include_paths) {
            combine(include_path.string());
        }
        for (const auto& [name, value] : kernel.compiler_options.defines) {
            combine(name, value);
        }
        for (const auto& binding : kernel.dfb_bindings) {
            combine(*binding.dfb_spec_name, binding.accessor_name, binding.endpoint_type, binding.access_pattern);
            combine(binding.block_size);
        }
        for (const auto& binding : kernel.semaphore_bindings) {
            combine(*binding.semaphore_spec_name, binding.accessor_name);
        }
        for (const auto& binding : kernel.scratchpad_bindings) {
            combine(*binding.scratchpad_spec_name, binding.accessor_name);
        }
        for (const auto& binding : kernel.tensor_bindings) {
            combine(*binding.tensor_parameter_name, binding.accessor_name);
        }
        for (const auto& [name, value] : kernel.compile_time_args) {
            combine(name, value);
        }
        for (const auto& name : kernel.runtime_arg_schema.runtime_arg_names) {
            combine(name);
        }
        for (const auto& name : kernel.runtime_arg_schema.common_runtime_arg_names) {
            combine(name);
        }
        combine(kernel.hw_config.index());
        if (const auto* dm = std::get_if<DataMovementHardwareConfig>(&kernel.hw_config)) {
            combine(dm->config_1xx.has_value(), dm->config_2xx.has_value());
            if (dm->config_1xx.has_value()) {
                combine(dm->config_1xx->processor, dm->config_1xx->noc, dm->config_1xx->noc_mode);
                combine(dm->config_1xx->dedicated_noc_sharing);
            }
            if (dm->config_2xx.has_value()) {
                combine(dm->config_2xx->disable_dfb_implicit_sync_for_all);
                for (const auto& name : dm->config_2xx->disable_dfb_implicit_sync_for) {
                    combine(*name);
                }
            }
        } else {
            const auto& compute = std::get<ComputeHardwareConfig>(kernel.hw_config);
            combine(compute.fpu_math_fidelity, compute.sfpu_precision_mode, compute.enable_32_bit_dest);
            combine(compute.double_buffer_dest);
            for (const auto& [name, mode] : compute.unpack_modes) {
                combine(*name, mode);
            }
            combine(compute.config_1xx.has_value(), compute.config_2xx.has_value());
            if (compute.config_1xx.has_value()) {
                combine(compute.config_1xx->bfp_pack_precision_mode);
            }
            // The RVV options exist only in newer tt-metal versions.
            [&](const auto& config) {
                if constexpr (requires { config.config_2xx->enable_unpack_rvv; }) {
                    combine(config.config_2xx.has_value() && config.config_2xx->enable_unpack_rvv);
                }
                if constexpr (requires { config.config_12x; }) {
                    combine(config.config_12x.has_value() && config.config_12x->enable_pack_rvv);
                }
            }(compute);
        }
    }
    for (const auto& dfb : spec.dataflow_buffers) {
        combine(*dfb.unique_id, dfb.entry_size, dfb.num_entries);
        combine_format(dfb.data_format_metadata);
        combine(dfb.tile_format_metadata.has_value());
        if (dfb.tile_format_metadata.has_value()) {
            combine(dfb.tile_format_metadata->get_height(), dfb.tile_format_metadata->get_width());
        }
        combine(dfb.borrowed_from.has_value());
        if (dfb.borrowed_from.has_value()) {
            combine(**dfb.borrowed_from);
        }
        combine(dfb.advanced_options.borrowed_offset);
        // Aliasing changes the DFB layout; the order of the names does not.
        std::vector<std::string> alias_with;
        for (const auto& name : dfb.advanced_options.alias_with) {
            alias_with.push_back(*name);
        }
        std::sort(alias_with.begin(), alias_with.end());
        combine(alias_with.size());
        for (const auto& name : alias_with) {
            combine(name);
        }
    }
    for (const auto& semaphore : spec.semaphores) {
        combine(*semaphore.unique_id, ttsl::hash::hash_objects_with_default_seed(semaphore.target_nodes));
    }
    for (const auto& scratchpad : spec.scratchpads) {
        combine(*scratchpad.unique_id, scratchpad.size_per_node);
        combine_format(scratchpad.data_format_metadata);
        combine(scratchpad.tile_format_metadata.has_value());
        if (scratchpad.tile_format_metadata.has_value()) {
            combine(scratchpad.tile_format_metadata->get_height(), scratchpad.tile_format_metadata->get_width());
        }
    }
    for (const auto& tensor_parameter : spec.tensor_parameters) {
        combine(*tensor_parameter.unique_id, ttsl::hash::hash_objects_with_default_seed(tensor_parameter.spec));
    }
    for (const auto& work_unit : spec.work_units) {
        combine(work_unit.name, ttsl::hash::hash_objects_with_default_seed(work_unit.target_nodes));
        for (const auto& kernel : work_unit.kernels) {
            combine(*kernel);
        }
    }
    return hash;
}

}  // namespace

void GenericSpecOpDeviceOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    TT_FATAL(!tensor_args.io_tensors.empty(), "generic_op: io_tensors must contain at least the output tensor");
    for (const auto& [tensor_parameter_name, io_index] : operation_attributes.tensor_bindings) {
        TT_FATAL(
            io_index < tensor_args.io_tensors.size(),
            "generic_op: TensorParameter '{}' is bound to io_tensors[{}], but only {} io tensors were given",
            tensor_parameter_name,
            io_index,
            tensor_args.io_tensors.size());
    }
}

void GenericSpecOpDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
}

void GenericSpecOpDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
}

GenericSpecOpDeviceOperation::spec_return_value_t GenericSpecOpDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    return tensor_args.io_tensors.back().tensor_spec();
}

GenericSpecOpDeviceOperation::tensor_return_value_t GenericSpecOpDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    // The caller preallocates the output, as for generic_op.
    return tensor_args.io_tensors.back();
}

ttsl::hash::hash_t GenericSpecOpDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (operation_attributes.custom_program_hash.has_value()) {
        return *operation_attributes.custom_program_hash;
    }
    // Runtime arg values are reapplied on every hit, so only their schema (names and nodes) is hashed.
    std::size_t run_args_schema_hash = 0;
    for (const auto& kernel_run_args : operation_attributes.kernel_run_args) {
        ttsl::hash::hash_combine(run_args_schema_hash, *kernel_run_args.kernel);
        for (const auto& [name, values_per_node] : kernel_run_args.runtime_arg_values) {
            ttsl::hash::hash_combine(run_args_schema_hash, name);
            for (const auto& [node, value] : values_per_node) {
                ttsl::hash::hash_combine(run_args_schema_hash, node);
            }
        }
        for (const auto& [name, value] : kernel_run_args.common_runtime_arg_values) {
            ttsl::hash::hash_combine(run_args_schema_hash, name);
        }
    }
    // Tensor specs are hashed because a cached program bakes in the bound tensors' layouts.
    std::vector<tt::tt_metal::TensorSpec> io_tensor_specs;
    io_tensor_specs.reserve(tensor_args.io_tensors.size());
    for (const auto& tensor : tensor_args.io_tensors) {
        io_tensor_specs.push_back(tensor.tensor_spec());
    }
    std::size_t tensor_bindings_hash = 0;
    for (const auto& [tensor_parameter_name, io_index] : operation_attributes.tensor_bindings) {
        ttsl::hash::hash_combine(tensor_bindings_hash, *tensor_parameter_name);
        ttsl::hash::hash_combine(tensor_bindings_hash, io_index);
    }
    return ttsl::hash::hash_objects_with_default_seed(
        hash_program_spec(operation_attributes.spec), tensor_bindings_hash, run_args_schema_hash, io_tensor_specs);
}

ttnn::device_operation::ProgramArtifacts
GenericSpecOpDeviceOperation::GenericSpecProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& /*tensor_return_value*/) {
    return ttnn::device_operation::ProgramArtifacts{
        .spec = operation_attributes.spec, .run_params = make_program_run_args(operation_attributes, tensor_args)};
}

ProgramRunArgs GenericSpecOpDeviceOperation::GenericSpecProgramFactory::override_runtime_arguments(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    return make_program_run_args(operation_attributes, tensor_args);
}

}  // namespace ttnn::operations::generic

namespace ttnn::prim {

Tensor generic_spec_op(
    const std::vector<Tensor>& io_tensors,
    const ttnn::operations::generic::GenericSpecOpDeviceOperation::operation_attributes_t& operation_attributes) {
    using OperationType = ttnn::operations::generic::GenericSpecOpDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        operation_attributes, OperationType::tensor_args_t{.io_tensors = io_tensors});
}

}  // namespace ttnn::prim
