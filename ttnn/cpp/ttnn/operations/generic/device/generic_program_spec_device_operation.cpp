// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "generic_program_spec_device_operation.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/tensor_spec_relaxations.hpp>
#include <tt-metalium/experimental/prefetcher_pipe.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/overloaded.hpp>
#include <tt_stl/reflection.hpp>

#include "ttnn/config.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/mesh_device_operation_utils.hpp"

namespace ttnn::operations::generic {

namespace m2 = tt::tt_metal::experimental;

namespace {

using SpecAttributes = GenericProgramSpecDeviceOperation::operation_attributes_t;

bool nodes_contain(const m2::Nodes& nodes, const m2::NodeCoord& node) {
    return std::visit(
        ttsl::overloaded{
            [&](const m2::NodeCoord& coord) { return coord == node; },
            [&](const m2::NodeRange& range) { return range.contains(node); },
            [&](const m2::NodeRangeSet& range_set) { return range_set.contains(node); }},
        nodes);
}

// The nodes a kernel runs on: those of every work unit that lists it.
std::unordered_set<m2::NodeCoord> nodes_of_kernel(const m2::ProgramSpec& spec, const m2::KernelSpecName& kernel) {
    std::unordered_set<m2::NodeCoord> nodes;
    const auto add_range = [&nodes](const m2::NodeRange& range) {
        for (const auto& node : range) {
            nodes.insert(node);
        }
    };
    for (const auto& work_unit : spec.work_units) {
        if (std::ranges::find(work_unit.kernels, kernel) == work_unit.kernels.end()) {
            continue;
        }
        std::visit(
            ttsl::overloaded{
                [&](const m2::NodeCoord& node) { nodes.insert(node); },
                [&](const m2::NodeRange& range) { add_range(range); },
                [&](const m2::NodeRangeSet& range_set) {
                    for (const auto& range : range_set.ranges()) {
                        add_range(range);
                    }
                }},
            work_unit.target_nodes);
    }
    return nodes;
}

// A kernel's runtime vararg count on a node: num_runtime_varargs, unless the deprecated per-node counts name the
// node.
uint32_t runtime_vararg_count(const m2::KernelAdvancedOptions& options, const m2::NodeCoord& node) {
    for (const auto& [nodes, count] : options.num_runtime_varargs_per_node) {
        if (nodes_contain(nodes, node)) {
            return count;
        }
    }
    return options.num_runtime_varargs;
}

bool has_runtime_arguments(const m2::KernelSpec& kernel) {
    const auto& schema = kernel.runtime_arg_schema;
    const auto& options = kernel.advanced_options;
    return !schema.runtime_arg_names.empty() || !schema.common_runtime_arg_names.empty() ||
           options.num_runtime_varargs > 0 || options.num_common_runtime_varargs > 0 ||
           std::ranges::any_of(
               options.num_runtime_varargs_per_node, [](const auto& entry) { return entry.second > 0; });
}

template <typename Value>
std::vector<std::string> names_of(const m2::Table<std::string, Value>& values) {
    std::vector<std::string> names;
    names.reserve(values.size());
    for (const auto& [name, value] : values) {
        names.push_back(name);
    }
    return names;
}

std::string join_names(const std::vector<std::string>& names) {
    std::string joined;
    for (const auto& name : names) {
        joined += joined.empty() ? name : ", " + name;
    }
    return joined;
}

template <typename Value>
bool has_exactly(const m2::Table<std::string, Value>& values, const m2::Group<std::string>& names) {
    return values.size() == names.size() &&
           std::ranges::all_of(names, [&values](const std::string& name) { return values.contains(name); });
}

void check_kernel_run_args(
    const m2::ProgramSpec& spec, const m2::KernelSpec& kernel, const m2::KernelRunArgs& run_args, bool check_nodes) {
    const auto& schema = kernel.runtime_arg_schema;
    TT_FATAL(
        has_exactly(run_args.runtime_arg_values, schema.runtime_arg_names),
        "generic_op: kernel '{}' declares the runtime arguments [{}], but its run args give [{}]",
        kernel.unique_id,
        join_names(schema.runtime_arg_names),
        join_names(names_of(run_args.runtime_arg_values)));
    TT_FATAL(
        has_exactly(run_args.common_runtime_arg_values, schema.common_runtime_arg_names),
        "generic_op: kernel '{}' declares the common runtime arguments [{}], but its run args give [{}]",
        kernel.unique_id,
        join_names(schema.common_runtime_arg_names),
        join_names(names_of(run_args.common_runtime_arg_values)));

    // Metal writes varargs by position on a later call without checking their count, so every call checks it.
    const auto& options = kernel.advanced_options;
    const auto& varargs = run_args.advanced_options;
    TT_FATAL(
        varargs.common_runtime_varargs.size() == options.num_common_runtime_varargs,
        "generic_op: kernel '{}' declares {} common runtime varargs, but its run args give {}",
        kernel.unique_id,
        options.num_common_runtime_varargs,
        varargs.common_runtime_varargs.size());
    for (const auto& [node, values] : varargs.runtime_varargs) {
        const uint32_t count = runtime_vararg_count(options, node);
        TT_FATAL(
            values.size() == count,
            "generic_op: kernel '{}' declares {} runtime varargs on node {}, but its run args give {}",
            kernel.unique_id,
            count,
            node.str(),
            values.size());
    }

    if (!check_nodes) {
        return;
    }
    // A later call keeps the previous value of a node it leaves out, and Metal does not report it.
    const auto nodes = nodes_of_kernel(spec, kernel.unique_id);
    for (const auto& [name, values] : run_args.runtime_arg_values) {
        const bool covers_nodes =
            values.size() == nodes.size() &&
            std::ranges::all_of(values, [&nodes](const auto& entry) { return nodes.contains(entry.first); });
        TT_FATAL(
            covers_nodes,
            "generic_op: kernel '{}' runs on {} nodes, but runtime argument '{}' does not have exactly one value "
            "for each of them",
            kernel.unique_id,
            nodes.size(),
            name);
    }
    for (const auto& node : nodes) {
        TT_FATAL(
            runtime_vararg_count(options, node) == 0 || varargs.runtime_varargs.contains(node),
            "generic_op: kernel '{}' declares runtime varargs, but its run args give none for node {}",
            kernel.unique_id,
            node.str());
    }
    for (const auto& [node, values] : varargs.runtime_varargs) {
        TT_FATAL(
            nodes.contains(node),
            "generic_op: kernel '{}' does not run on node {}, but its run args give runtime varargs for it",
            kernel.unique_id,
            node.str());
    }
}

void check_tensor_args(
    const m2::ProgramSpec& spec, const m2::ProgramRunArgs& run_args, const std::vector<Tensor>& io_tensors) {
    TT_FATAL(
        run_args.tensor_args.size() == spec.tensor_parameters.size(),
        "generic_op: the ProgramSpec declares {} tensor parameters, but run_args.tensor_args gives {} tensors",
        spec.tensor_parameters.size(),
        run_args.tensor_args.size());
    std::vector<bool> claimed(io_tensors.size(), false);
    for (const auto& parameter : spec.tensor_parameters) {
        const auto tensor_arg = run_args.tensor_args.get(parameter.unique_id);
        TT_FATAL(
            tensor_arg.has_value(),
            "generic_op: tensor parameter '{}' has no tensor in run_args.tensor_args",
            parameter.unique_id);
        // A tensor argument refers to a MeshTensor, which every copy of a ttnn::Tensor shares.
        const auto* mesh_tensor = &m2::mesh_tensor_of(*tensor_arg);
        std::size_t index = 0;
        while (index < io_tensors.size() && (claimed[index] || &io_tensors[index].mesh_tensor() != mesh_tensor)) {
            ++index;
        }
        TT_FATAL(
            index < io_tensors.size(),
            "generic_op: the tensor of tensor parameter '{}' is not in io_tensors. Each tensor parameter needs its "
            "own io_tensors entry, so list a tensor once for each parameter it is bound to",
            parameter.unique_id);
        claimed[index] = true;
        const auto& tensor_spec = mesh_tensor->tensor_spec();
        TT_FATAL(
            tensor_spec == parameter.spec ||
                m2::tensorspecs_match_with_relaxation(tensor_spec, parameter.spec, parameter.relaxations),
            "generic_op: the tensor of tensor parameter '{}' has the spec {}, which does not match the parameter's "
            "spec {}",
            parameter.unique_id,
            tensor_spec,
            parameter.spec);
    }
}

void check_prefetcher_pipe_args(const m2::ProgramSpec& spec, const m2::ProgramRunArgs& run_args) {
    const auto& parameters = spec.advanced_options.prefetcher_pipe_parameters;
    const auto& pipes = run_args.advanced_options.prefetcher_pipe_args;
    for (const auto& parameter : parameters) {
        TT_FATAL(
            pipes.contains(parameter.unique_id),
            "generic_op: PrefetcherPipeParameter '{}' has no pipe in run_args.advanced_options.prefetcher_pipe_args",
            parameter.unique_id);
    }
    for (const auto& [name, pipe] : pipes) {
        TT_FATAL(
            std::ranges::find(parameters, name, &m2::PrefetcherPipeParameter::unique_id) != parameters.end(),
            "generic_op: run_args.advanced_options.prefetcher_pipe_args has a pipe for '{}', which the ProgramSpec "
            "does not declare",
            name);
    }
}

// On a later call Metal checks DFB size overrides only with validate_program_args on. Without these checks, a bad entry
// fails after the entries before it have resized their buffers, which then keep sizes their allocations do not match.
// Metal still checks some sizes only after setting them: divisibility by the number of producers and consumers, the
// capacity limit, Quasar's ring limits, and whether a borrowed DFB fits its tensor.
void check_dfb_run_overrides(const m2::ProgramSpec& spec, const m2::ProgramRunArgs& run_args) {
    const auto& overrides = run_args.dfb_run_overrides;
    for (auto entry = overrides.begin(); entry != overrides.end(); ++entry) {
        const auto dfb = std::ranges::find(spec.dataflow_buffers, entry->dfb, &m2::DataflowBufferSpec::unique_id);
        TT_FATAL(
            dfb != spec.dataflow_buffers.end(),
            "generic_op: run_args.dfb_run_overrides names DFB '{}', which the ProgramSpec does not declare",
            entry->dfb);
        TT_FATAL(
            std::ranges::find(overrides.begin(), entry, entry->dfb, &m2::DFBRunOverrides::dfb) == entry,
            "generic_op: run_args.dfb_run_overrides lists DFB '{}' more than once",
            entry->dfb);
        TT_FATAL(
            entry->entry_size.value_or(1) > 0 && entry->num_entries.value_or(1) > 0,
            "generic_op: run_args.dfb_run_overrides sets a size of 0 for DFB '{}'",
            entry->dfb);
        TT_FATAL(
            dfb->advanced_options.prefetcher_pipe_relays.empty() ||
                (!entry->entry_size.has_value() && !entry->num_entries.has_value()),
            "generic_op: run_args.dfb_run_overrides resizes DFB '{}', which relays a PrefetcherPipe and keeps the "
            "pipe's size",
            entry->dfb);
    }
}

// The checks every call runs before Metal sees the run args. Metal checks them only on the first call, and on later
// calls only with validate_program_args on, and even then it keeps the previous value of anything left out.
void check_program(
    const m2::ProgramSpec& spec,
    const m2::ProgramRunArgs& run_args,
    const std::vector<Tensor>& io_tensors,
    bool check_nodes) {
    std::vector<bool> has_run_args(spec.kernels.size(), false);
    for (const auto& kernel_run_args : run_args.kernel_run_args) {
        const auto kernel = std::ranges::find(spec.kernels, kernel_run_args.kernel, &m2::KernelSpec::unique_id);
        TT_FATAL(
            kernel != spec.kernels.end(),
            "generic_op: run_args.kernel_run_args names kernel '{}', which the ProgramSpec does not declare",
            kernel_run_args.kernel);
        const auto index = static_cast<std::size_t>(kernel - spec.kernels.begin());
        TT_FATAL(
            !has_run_args[index],
            "generic_op: run_args.kernel_run_args lists kernel '{}' more than once",
            kernel_run_args.kernel);
        has_run_args[index] = true;
        check_kernel_run_args(spec, *kernel, kernel_run_args, check_nodes);
    }
    for (std::size_t i = 0; i < spec.kernels.size(); ++i) {
        TT_FATAL(
            has_run_args[i] || !has_runtime_arguments(spec.kernels[i]),
            "generic_op: kernel '{}' takes runtime arguments, but run_args.kernel_run_args has no entry for it",
            spec.kernels[i].unique_id);
    }
    check_tensor_args(spec, run_args, io_tensors);
    check_prefetcher_pipe_args(spec, run_args);
    check_dfb_run_overrides(spec, run_args);
}

// Only the first call needs this: the ranges are part of the cache key.
void check_mesh_ranges(const SpecAttributes& attributes) {
    const auto& programs = attributes.programs;
    for (std::size_t i = 0; i < programs.size(); ++i) {
        for (std::size_t j = i + 1; j < programs.size(); ++j) {
            if (programs[i].range.has_value() && programs[j].range.has_value()) {
                TT_FATAL(
                    !programs[i].range->intersects(*programs[j].range),
                    "generic_op: mesh ranges {} and {} overlap",
                    *programs[i].range,
                    *programs[j].range);
            }
        }
    }
}

// Every call needs this: the cache key holds the devices of only one tensor.
void check_tensors_on_program_devices(const SpecAttributes& attributes, const tensor_args_t& tensor_args) {
    namespace utils = ttnn::device_operation::mesh_device_operation_utils;
    if (utils::all_tensors_have_uniform_storage(tensor_args)) {
        return;
    }
    const auto& programs = attributes.programs;
    // The devices the programs run on. A program without a range runs where ttnn's framework places it: on the
    // devices of one tensor, which the framework compares with the other tensors only by count. So every tensor is
    // checked against every one of these devices here.
    std::vector<ttnn::MeshCoordinate> coords;
    for (const auto& program : programs) {
        if (program.range.has_value()) {
            for (const auto& coord : *program.range) {
                coords.push_back(coord);
            }
        } else {
            for (const auto& coord :
                 utils::extract_tensor_coordinates(tensor_args, tensor_args.output_tensor.device())) {
                coords.push_back(coord);
            }
        }
    }
    for (std::size_t i = 0; i < tensor_args.io_tensors.size(); ++i) {
        const auto& coords_of_tensor = tensor_args.io_tensors[i].device_storage().get_coords();
        const std::unordered_set<ttnn::MeshCoordinate> held(coords_of_tensor.begin(), coords_of_tensor.end());
        for (const auto& coord : coords) {
            TT_FATAL(
                held.contains(coord),
                "generic_op: a program runs on device {}, which does not hold io_tensors[{}]",
                coord,
                i);
        }
    }
}

// ttsl hashes a std::optional of a std-hashable type with std::hash, and libc++ hashes an empty one to 0, the hash of
// DataFormat::Float32. So with libc++ an unset data_format_metadata would hash like Float32; the spec hash records
// which of them are set.
ttsl::hash::hash_t hash_data_formats_set(const m2::ProgramSpec& spec) {
    ttsl::hash::hash_t hash = 0;
    for (const auto& dfb : spec.dataflow_buffers) {
        hash = ttsl::hash::hash_objects(hash, dfb.data_format_metadata.has_value());
    }
    for (const auto& scratchpad : spec.scratchpads) {
        hash = ttsl::hash::hash_objects(hash, scratchpad.data_format_metadata.has_value());
    }
    return hash;
}

// Everything that run args refer to by name, the nodes each kernel runs on, and each kernel's vararg counts. On a later
// call Metal writes run args into the cached program by these names, keeps whatever they leave out, and writes varargs
// by position into a buffer sized from the counts. So the key includes them even when a custom hash replaces the spec
// hash, and a cached program whose arguments, tensor parameters, buffers, pipes, nodes or vararg counts differ from the
// spec's is never reused.
ttsl::hash::hash_t hash_run_args_schema(const m2::ProgramSpec& spec) {
    ttsl::hash::hash_t hash = spec.kernels.size();
    for (const auto& kernel : spec.kernels) {
        const auto& options = kernel.advanced_options;
        hash = ttsl::hash::hash_objects(
            hash,
            kernel.unique_id,
            kernel.runtime_arg_schema,
            options.num_runtime_varargs,
            options.num_common_runtime_varargs,
            options.num_runtime_varargs_per_node);
    }
    hash = ttsl::hash::hash_objects(hash, spec.tensor_parameters.size());
    for (const auto& parameter : spec.tensor_parameters) {
        hash = ttsl::hash::hash_objects(hash, parameter.unique_id);
    }
    hash = ttsl::hash::hash_objects(hash, spec.dataflow_buffers.size());
    for (const auto& dfb : spec.dataflow_buffers) {
        hash = ttsl::hash::hash_objects(hash, dfb.unique_id);
    }
    const auto& pipe_parameters = spec.advanced_options.prefetcher_pipe_parameters;
    hash = ttsl::hash::hash_objects(hash, pipe_parameters.size());
    for (const auto& parameter : pipe_parameters) {
        hash = ttsl::hash::hash_objects(hash, parameter.unique_id);
    }
    // The nodes each kernel runs on: a cached program keeps the values of nodes the run args do not cover.
    hash = ttsl::hash::hash_objects(hash, spec.work_units.size());
    for (const auto& work_unit : spec.work_units) {
        hash = ttsl::hash::hash_objects(hash, work_unit.kernels, work_unit.target_nodes);
    }
    return hash;
}

// The PrefetcherPipes a program's run args supply. Metal ties a program to the first pipe of each parameter and
// rejects a different one later, so a call with other pipes needs its own cache entry.
ttsl::hash::hash_t hash_prefetcher_pipes(const m2::ProgramSpec& spec, const m2::ProgramRunArgs& run_args) {
    ttsl::hash::hash_t hash = 0;
    for (const auto& parameter : spec.advanced_options.prefetcher_pipe_parameters) {
        const auto pipe = run_args.advanced_options.prefetcher_pipe_args.get(parameter.unique_id);
        hash = ttsl::hash::hash_objects(hash, pipe.has_value() ? *pipe->get().identity() : 0);
    }
    return hash;
}

}  // namespace

ttsl::hash::hash_t compute_program_spec_hash(const m2::ProgramSpec& spec) {
    // Tripwires: these three types are hashed field by field below. Every other type goes through ttsl's reflection
    // hash, which picks up a new field by itself.
    static_assert(
        reflect::size<m2::ProgramSpec>() == 8,
        "ProgramSpec gained or lost a field: hash it in compute_program_spec_hash, then update this count.");
    static_assert(
        reflect::size<m2::WorkUnitSpec>() == 3,
        "WorkUnitSpec gained or lost a field: hash it in compute_program_spec_hash, then update this count.");
    static_assert(
        reflect::size<m2::TensorParameter>() == 3,
        "TensorParameter gained or lost a field: hash it in compute_program_spec_hash, then update this count.");

    // ProgramSpec::name and WorkUnitSpec::name are left out: they are for messages only. Lists are hashed with their
    // lengths, so an element cannot move from one list to the next without changing the hash.
    ttsl::hash::hash_t hash = ttsl::hash::hash_objects_with_default_seed(
        spec.kernels, spec.dataflow_buffers, spec.semaphores, spec.scratchpads, spec.advanced_options);
    // A tensor parameter's spec is hashed under its relaxations, so a relaxed field such as a dynamic shape does not
    // change the hash.
    hash = ttsl::hash::hash_objects(hash, spec.tensor_parameters.size());
    for (const auto& parameter : spec.tensor_parameters) {
        hash = ttsl::hash::hash_objects(
            hash,
            parameter.unique_id,
            parameter.relaxations,
            m2::hash_tensorspec_with_relaxation(parameter.spec, parameter.relaxations));
    }
    hash = ttsl::hash::hash_objects(hash, spec.work_units.size());
    for (const auto& work_unit : spec.work_units) {
        hash = ttsl::hash::hash_objects(hash, work_unit.kernels, work_unit.target_nodes);
    }
    return ttsl::hash::hash_objects(hash, hash_data_formats_set(spec));
}

GenericProgramSpecDeviceOperation::spec_return_value_t GenericProgramSpecDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    return tensor_args.output_tensor.tensor_spec();
}

GenericProgramSpecDeviceOperation::tensor_return_value_t GenericProgramSpecDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    // The caller preallocates the output.
    return tensor_args.output_tensor;
}

void GenericProgramSpecDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    check_mesh_ranges(operation_attributes);
    check_tensors_on_program_devices(operation_attributes, tensor_args);
    // SetProgramRunArgs checks the nodes too, but only after the kernels compile.
    for (const auto& program : operation_attributes.programs) {
        check_program(*program.spec, *program.run_args, tensor_args.io_tensors, /*check_nodes=*/true);
    }
}

void GenericProgramSpecDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    check_tensors_on_program_devices(operation_attributes, tensor_args);
    const bool check_nodes = ttnn::CONFIG.get<"validate_program_args">();
    for (const auto& program : operation_attributes.programs) {
        check_program(*program.spec, *program.run_args, tensor_args.io_tensors, check_nodes);
    }
}

ttsl::hash::hash_t GenericProgramSpecDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    // The framework adds the op's type to the key. Both forms combine the spec hash, or the custom hash in its place,
    // with the run-args schema, so passing compute_program_spec_hash(spec) as the custom hash gives the default key.
    ttsl::hash::hash_t hash = 0;
    for (const auto& program : operation_attributes.programs) {
        const auto& spec = *program.spec;
        const ttsl::hash::hash_t spec_hash = operation_attributes.custom_program_hash.has_value()
                                                 ? *operation_attributes.custom_program_hash
                                                 : compute_program_spec_hash(spec);
        hash = ttsl::hash::hash_objects(
            hash, program.range, spec_hash, hash_run_args_schema(spec), hash_prefetcher_pipes(spec, *program.run_args));
    }
    return hash;
}

ttnn::device_operation::MeshWorkloadArtifacts
GenericProgramSpecDeviceOperation::ProgramFactory::create_mesh_workload_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    ttnn::device_operation::MeshWorkloadArtifacts artifacts;
    for (const auto& program : operation_attributes.programs) {
        if (program.range.has_value()) {
            artifacts.programs.push_back(
                {.range = *program.range, .spec = *program.spec, .run_params = *program.run_args});
            continue;
        }
        for (const auto& range : tensor_coords.ranges()) {
            artifacts.programs.push_back({.range = range, .spec = *program.spec, .run_params = *program.run_args});
        }
    }
    return artifacts;
}

const tt::tt_metal::experimental::ProgramRunArgs&
GenericProgramSpecDeviceOperation::ProgramFactory::override_runtime_arguments(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/,
    const ttnn::MeshCoordinateRange& range) {
    for (const auto& program : operation_attributes.programs) {
        if (!program.range.has_value() || *program.range == range) {
            return *program.run_args;
        }
    }
    TT_THROW("generic_op: no run args for mesh range {}", range);
}

}  // namespace ttnn::operations::generic

namespace ttnn::prim {

namespace {

using GenericProgramSpecOperation = ttnn::operations::generic::GenericProgramSpecDeviceOperation;

GenericProgramSpecOperation::tensor_args_t make_program_spec_tensor_args(const std::vector<Tensor>& io_tensors) {
    TT_FATAL(!io_tensors.empty(), "generic_op: io_tensors must hold at least the output tensor");
    for (std::size_t i = 0; i < io_tensors.size(); ++i) {
        TT_FATAL(
            io_tensors[i].storage_type() == ttnn::StorageType::DEVICE && io_tensors[i].is_allocated(),
            "generic_op: io_tensors[{}] must be a tensor allocated on a device",
            i);
    }
    // The program runs on one MeshDevice; a tensor on another would be bound by an address that means nothing there.
    for (std::size_t i = 0; i < io_tensors.size(); ++i) {
        TT_FATAL(
            io_tensors[i].device() == io_tensors.back().device(),
            "generic_op: io_tensors[{}] is on a different MeshDevice than the output tensor",
            i);
    }
    return {.io_tensors = io_tensors, .output_tensor = io_tensors.back()};
}

}  // namespace

Tensor generic_op(
    const std::vector<Tensor>& io_tensors,
    const GenericProgramSpecOperation::operation_attributes_t& operation_attributes) {
    return ttnn::device_operation::launch<GenericProgramSpecOperation>(
        operation_attributes, make_program_spec_tensor_args(io_tensors));
}

void prepare_generic_op(
    const std::vector<Tensor>& io_tensors,
    const GenericProgramSpecOperation::operation_attributes_t& operation_attributes) {
    ttnn::device_operation::prepare<GenericProgramSpecOperation>(
        operation_attributes, make_program_spec_tensor_args(io_tensors));
}

}  // namespace ttnn::prim
