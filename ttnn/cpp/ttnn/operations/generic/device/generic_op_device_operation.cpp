// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/device_operation.hpp"
#include "generic_op_device_operation.hpp"

#include "generic_op_device_operation_types.hpp"
#include "ttnn/config.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iterator>
#include <string>
#include <variant>

#include <reflect>
#include <tt-metalium/experimental/metal2_host_api/tensor_spec_relaxations.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
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

namespace {

namespace m2 = tt::tt_metal::experimental;

// Nodes a kernel runs on: the union of the work units that list it.
CoreRangeSet kernel_nodes(const m2::ProgramSpec& spec, const m2::KernelSpecName& kernel) {
    CoreRangeSet nodes;
    for (const auto& unit : spec.work_units) {
        if (std::find(unit.kernels.begin(), unit.kernels.end(), kernel) == unit.kernels.end()) {
            continue;
        }
        nodes = nodes.merge(std::visit(
            ttsl::overloaded{
                [](const CoreCoord& c) { return CoreRangeSet(CoreRange(c)); },
                [](const CoreRange& r) { return CoreRangeSet(r); },
                [](const CoreRangeSet& set) { return set; }},
            unit.target_nodes));
    }
    return nodes;
}

template <typename Range, typename T>
bool contains(const Range& range, const T& value) {
    return std::find(std::begin(range), std::end(range), value) != std::end(range);
}

// The spec path rebuilds the program's run args from this call's run_args on every launch, a cache
// hit included. Metal 2.0's UpdateProgramRunArgs reads an omitted kernel, arg or node as "keep the
// previous call's value", and with validate_program_args off (the default) it checks nothing on a
// hit, so the completeness the miss path enforces is enforced here for both paths: kernel and arg
// names and vararg counts always, per-node coverage under validate_program_args.
void validate_run_args(const m2::ProgramSpec& spec, const m2::ProgramRunArgs& run_args, bool check_nodes) {
    TT_FATAL(
        run_args.tensor_args.empty(),
        "run_args.tensor_args must be empty: generic_op binds the tensor parameters from its tensor_args "
        "(parameter name -> io_tensors index)");
    for (const auto& kernel_args : run_args.kernel_run_args) {
        TT_FATAL(
            std::any_of(
                spec.kernels.begin(),
                spec.kernels.end(),
                [&](const m2::KernelSpec& kernel) { return kernel.unique_id == kernel_args.kernel; }),
            "run_args carry runtime args for kernel '{}', which the ProgramSpec does not declare",
            kernel_args.kernel);
    }
    for (const auto& kernel : spec.kernels) {
        const auto& schema = kernel.runtime_arg_schema;
        const auto& advanced = kernel.advanced_options;
        const bool expects_args = !schema.runtime_arg_names.empty() || !schema.common_runtime_arg_names.empty() ||
                                  advanced.num_runtime_varargs > 0 || advanced.num_common_runtime_varargs > 0;
        const auto it = std::find_if(
            run_args.kernel_run_args.begin(), run_args.kernel_run_args.end(), [&](const m2::KernelRunArgs& args) {
                return args.kernel == kernel.unique_id;
            });
        if (it == run_args.kernel_run_args.end()) {
            TT_FATAL(
                !expects_args,
                "Kernel '{}' declares runtime args but run_args carry no KernelRunArgs for it; on a cache hit Metal "
                "2.0 would keep the previous call's values",
                kernel.unique_id);
            continue;
        }
        const auto& args = *it;
        for (const auto& name : schema.runtime_arg_names) {
            TT_FATAL(
                args.runtime_arg_values.contains(name),
                "Kernel '{}' declares runtime arg '{}' but run_args do not set it",
                kernel.unique_id,
                name);
        }
        for (const auto& [name, values] : args.runtime_arg_values) {
            TT_FATAL(
                contains(schema.runtime_arg_names, name),
                "Kernel '{}' is given runtime arg '{}', which its schema does not declare",
                kernel.unique_id,
                name);
        }
        for (const auto& name : schema.common_runtime_arg_names) {
            TT_FATAL(
                args.common_runtime_arg_values.contains(name),
                "Kernel '{}' declares common runtime arg '{}' but run_args do not set it",
                kernel.unique_id,
                name);
        }
        for (const auto& [name, value] : args.common_runtime_arg_values) {
            TT_FATAL(
                contains(schema.common_runtime_arg_names, name),
                "Kernel '{}' is given common runtime arg '{}', which its schema does not declare",
                kernel.unique_id,
                name);
        }
        TT_FATAL(
            args.advanced_options.common_runtime_varargs.size() == advanced.num_common_runtime_varargs,
            "Kernel '{}' expects {} common runtime varargs, run_args carry {}",
            kernel.unique_id,
            advanced.num_common_runtime_varargs,
            args.advanced_options.common_runtime_varargs.size());
        // The deprecated per-node vararg counts are left to Metal 2.0's own checks.
        const bool uniform_varargs = advanced.num_runtime_varargs_per_node.empty();
        if (uniform_varargs) {
            for (const auto& [node, varargs] : args.advanced_options.runtime_varargs) {
                TT_FATAL(
                    varargs.size() == advanced.num_runtime_varargs,
                    "Kernel '{}' expects {} runtime varargs per node, run_args carry {} for node {}",
                    kernel.unique_id,
                    advanced.num_runtime_varargs,
                    varargs.size(),
                    node.str());
            }
        }
        if (!check_nodes) {
            continue;
        }
        const CoreRangeSet nodes = kernel_nodes(spec, kernel.unique_id);
        const uint32_t num_nodes = nodes.num_cores();
        for (const auto& [name, values] : args.runtime_arg_values) {
            for (const auto& [node, value] : values) {
                TT_FATAL(
                    nodes.contains(node),
                    "Kernel '{}' is given runtime arg '{}' for node {}, on which no work unit runs it",
                    kernel.unique_id,
                    name,
                    node.str());
            }
            TT_FATAL(
                values.size() == num_nodes,
                "Kernel '{}' runs on {} nodes but run_args set runtime arg '{}' on {} of them; a node left out keeps "
                "the previous call's value on a cache hit",
                kernel.unique_id,
                num_nodes,
                name,
                values.size());
        }
        if (uniform_varargs && advanced.num_runtime_varargs > 0) {
            for (const auto& [node, varargs] : args.advanced_options.runtime_varargs) {
                TT_FATAL(
                    nodes.contains(node),
                    "Kernel '{}' is given runtime varargs for node {}, on which no work unit runs it",
                    kernel.unique_id,
                    node.str());
            }
            TT_FATAL(
                args.advanced_options.runtime_varargs.size() == num_nodes,
                "Kernel '{}' runs on {} nodes but run_args set runtime varargs on {} of them",
                kernel.unique_id,
                num_nodes,
                args.advanced_options.runtime_varargs.size());
        }
    }
}

}  // namespace

void GenericOpDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    if (!attributes.is_spec()) {
        verify_no_duplicate_mesh_coord_ranges(attributes.mesh_program_descriptor().mesh_programs);
        return;
    }
    const auto& spec_program = attributes.spec_program();
    const auto& spec = spec_program.spec;
    const auto& indices = spec_program.tensor_arg_indices;
    const auto& io_tensors = tensor_args.io_tensors;

    for (const auto& [name, io_index] : indices) {
        TT_FATAL(
            std::any_of(
                spec.tensor_parameters.begin(),
                spec.tensor_parameters.end(),
                [&](const m2::TensorParameter& param) { return param.unique_id == name; }),
            "tensor_args name '{}', which is not a TensorParameter of the ProgramSpec",
            name);
        TT_FATAL(
            io_index < io_tensors.size(),
            "tensor argument '{}' maps to io_tensors index {}, but only {} io tensors were supplied",
            name,
            io_index,
            io_tensors.size());
    }
    // The framework resolves each TensorArgument to its own, previously unclaimed, slot in the io
    // tensor enumeration (one slot per io_tensors entry), so two parameters cannot share an index.
    for (auto first = indices.begin(); first != indices.end(); ++first) {
        for (auto second = std::next(first); second != indices.end(); ++second) {
            TT_FATAL(
                first->second != second->second,
                "tensor_args map '{}' and '{}' both to io_tensors index {}; each parameter takes its own entry, so "
                "list the tensor once per parameter in io_tensors",
                first->first,
                second->first,
                first->second);
        }
    }
    for (const auto& param : spec.tensor_parameters) {
        const auto it = indices.find(param.unique_id);
        TT_FATAL(it != indices.end(), "TensorParameter '{}' has no entry in tensor_args", param.unique_id);
        const auto& tensor_spec = io_tensors[it->second].tensor_spec();
        TT_FATAL(
            m2::tensorspecs_match_with_relaxation(param.spec, tensor_spec, param.relaxations),
            "io_tensors[{}] (shape {}, {}, {}, {}) does not match TensorParameter '{}' (shape {}, {}, {}, {}) under "
            "the parameter's relaxations",
            it->second,
            tensor_spec.logical_shape(),
            tensor_spec.data_type(),
            tensor_spec.layout(),
            tensor_spec.memory_config(),
            param.unique_id,
            param.spec.logical_shape(),
            param.spec.data_type(),
            param.spec.layout(),
            param.spec.memory_config());
    }
    // supports_per_core_allocation above is the descriptor path's promise (its addresses come from
    // the caller). A ProgramSpec binds a tensor by one address on every node, so a per-core buffer
    // would be read as though every core shared the first core's allocation.
    for (size_t index = 0; index < io_tensors.size(); ++index) {
        TT_FATAL(
            !tt::tt_metal::experimental::per_core_allocation::is_per_core_allocation(
                io_tensors[index].mesh_buffer().device_local_config().sharding_args),
            "io_tensors[{}] is per-core allocated, which the ProgramSpec form of generic_op does not support; build "
            "the tensor with a lockstep memory config",
            index);
    }
    validate_run_args(spec, spec_program.run_args, ttnn::CONFIG.get<"validate_program_args">());
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

using ttsl::hash::hash_combine;
using ttsl::hash::hash_objects_with_default_seed;

// Every field below is structural: it changes the compiled program or its resource layout.
// Anything that varies per dispatch (runtime arg VALUES, tensor addresses) stays out, and so do
// the two free-form names, ProgramSpec::name and WorkUnitSpec::name: the headers define them as
// debug/messaging only, and hashing them would give a per-call unique name its own cache entry.
// Each helper enumerates every field of its struct; the TTNN_M2_ASSERT_FIELD_COUNT next to it
// fails the build when tt_metal adds a field, so the hash cannot silently fall behind the API
// (a missing field means two different programs share one cache key).
#define TTNN_M2_ASSERT_FIELD_COUNT(TYPE, N)                                                          \
    static_assert(                                                                                   \
        reflect::size<TYPE>() == (N),                                                                \
        #TYPE                                                                                        \
        " gained or lost a field in tt_metal: hash it below (or document why it is excluded), then " \
        "update this count")

// Lists fold as (tag, length, elements...). The length keeps an element from migrating across a
// list boundary and the tag keeps same-shaped lists apart, so {runtime_arg_names = [x, y],
// common_runtime_arg_names = []} and {[x], [y]} get different keys. Tables (unordered by
// definition) go through std::hash<Table>, which is order-independent.
enum class List : uint32_t {
    Kernels,
    DataflowBuffers,
    Semaphores,
    Scratchpads,
    TensorParameters,
    WorkUnits,
    PrefetcherPipeParameters,
    DfbAliasWith,
    DfbPrefetcherPipeRelays,
    DisableDfbImplicitSyncFor,
    IncludePaths,
    KernelDfbBindings,
    KernelSemaphoreBindings,
    KernelScratchpadBindings,
    KernelTensorBindings,
    KernelRuntimeArgNames,
    KernelCommonRuntimeArgNames,
    KernelCompileTimeVarargs,
    KernelTensorBindingSequences,
    TensorBindingSequenceMembers,
    KernelPrefetcherPipeBindings,
    PrefetcherPipeBindingNames,
    KernelVarargsPerNode,
    WorkUnitKernels,
};

template <typename Range, typename Hasher>
void hash_list(size_t& hash, List tag, const Range& range, Hasher hasher) {
    hash_combine(hash, static_cast<uint32_t>(tag));
    hash_combine(hash, static_cast<uint64_t>(std::size(range)));
    for (const auto& item : range) {
        hash_combine(hash, hasher(item));
    }
}

constexpr auto hash_name = [](const auto& name) { return hash_objects_with_default_seed(*name); };
constexpr auto hash_string = [](const std::string& value) { return hash_objects_with_default_seed(value); };
constexpr auto hash_u32 = [](uint32_t value) { return hash_objects_with_default_seed(value); };

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
    // The declared spec counts only as far as the relaxations make it pertinent: under
    // dynamic_tensor_shape the program does not depend on the shape, so neither does its key.
    return hash_objects_with_default_seed(
        *param.unique_id,
        m2::hash_tensorspec_with_relaxation(param.spec, param.relaxations),
        hash_relaxations(param.relaxations));
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
    hash_list(hash, List::DfbAliasWith, dfb.advanced_options.alias_with, hash_name);
    hash_combine(hash, dfb.advanced_options.allow_instance_multi_binding);
    hash_list(hash, List::DfbPrefetcherPipeRelays, dfb.advanced_options.prefetcher_pipe_relays, hash_name);
    return hash;
}

// initial_value is deprecated but the builder still honours it (semaphore.cpp), so it is keyed.
TTNN_M2_ASSERT_FIELD_COUNT(m2::SemaphoreAdvancedOptions, 2);
TTNN_M2_ASSERT_FIELD_COUNT(m2::SemaphoreSpec, 3);
size_t hash_semaphore(const m2::SemaphoreSpec& semaphore) {
    size_t hash = hash_objects_with_default_seed(
        *semaphore.unique_id, semaphore.advanced_options.max_value, semaphore.advanced_options.initial_value);
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
                    hash_list(hash, List::DisableDfbImplicitSyncFor, c->disable_dfb_implicit_sync_for, hash_name);
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
    hash_list(hash, List::IncludePaths, options.include_paths, [](const std::filesystem::path& path) {
        return hash_objects_with_default_seed(path.string());
    });
    return hash;
}

TTNN_M2_ASSERT_FIELD_COUNT(m2::KernelAdvancedOptions::TensorBindingSequence, 2);
TTNN_M2_ASSERT_FIELD_COUNT(m2::KernelAdvancedOptions::PrefetcherPipeBinding, 2);
// num_runtime_varargs_per_node is deprecated but the builder still honours it (kernel_lowering.cpp),
// so it is keyed.
TTNN_M2_ASSERT_FIELD_COUNT(m2::KernelAdvancedOptions, 6);
size_t hash_kernel_advanced_options(const m2::KernelAdvancedOptions& options) {
    size_t hash = hash_objects_with_default_seed(options.num_runtime_varargs, options.num_common_runtime_varargs);
    hash_list(hash, List::KernelCompileTimeVarargs, options.compile_time_varargs, hash_u32);
    hash_list(hash, List::KernelTensorBindingSequences, options.tensor_binding_sequences, [](const auto& sequence) {
        size_t sequence_hash = hash_objects_with_default_seed(sequence.sequence_name);
        hash_list(sequence_hash, List::TensorBindingSequenceMembers, sequence.members, hash_string);
        return sequence_hash;
    });
    hash_list(hash, List::KernelPrefetcherPipeBindings, options.prefetcher_pipe_bindings, [](const auto& binding) {
        size_t binding_hash = hash_objects_with_default_seed(binding.accessor_name);
        hash_list(binding_hash, List::PrefetcherPipeBindingNames, binding.pipe_parameter_names, hash_name);
        return binding_hash;
    });
    hash_list(hash, List::KernelVarargsPerNode, options.num_runtime_varargs_per_node, [](const auto& entry) {
        return hash_objects_with_default_seed(hash_nodes(entry.first), entry.second);
    });
    return hash;
}

TTNN_M2_ASSERT_FIELD_COUNT(m2::DFBBinding, 5);
TTNN_M2_ASSERT_FIELD_COUNT(m2::SemaphoreBinding, 2);
TTNN_M2_ASSERT_FIELD_COUNT(m2::ScratchpadBinding, 2);
TTNN_M2_ASSERT_FIELD_COUNT(m2::TensorBinding, 2);
TTNN_M2_ASSERT_FIELD_COUNT(m2::KernelSpec::RuntimeArgSchema, 2);
TTNN_M2_ASSERT_FIELD_COUNT(m2::KernelSpec, 12);
size_t hash_kernel_spec(const m2::KernelSpec& kernel) {
    size_t hash = hash_objects_with_default_seed(*kernel.unique_id, kernel.num_threads);
    hash_combine(hash, hash_source(kernel.source));
    hash_combine(hash, hash_compiler_options(kernel.compiler_options));
    hash_list(hash, List::KernelDfbBindings, kernel.dfb_bindings, [](const m2::DFBBinding& b) {
        return hash_objects_with_default_seed(
            *b.dfb_spec_name, b.accessor_name, b.endpoint_type, b.access_pattern, b.block_size);
    });
    hash_list(hash, List::KernelSemaphoreBindings, kernel.semaphore_bindings, [](const m2::SemaphoreBinding& b) {
        return hash_objects_with_default_seed(*b.semaphore_spec_name, b.accessor_name);
    });
    hash_list(hash, List::KernelScratchpadBindings, kernel.scratchpad_bindings, [](const m2::ScratchpadBinding& b) {
        return hash_objects_with_default_seed(*b.scratchpad_spec_name, b.accessor_name);
    });
    hash_list(hash, List::KernelTensorBindings, kernel.tensor_bindings, [](const m2::TensorBinding& b) {
        return hash_objects_with_default_seed(*b.tensor_parameter_name, b.accessor_name);
    });
    // Arg NAMES are structural (they become generated accessors); runtime arg VALUES are not.
    hash_combine(hash, kernel.compile_time_args);
    hash_list(hash, List::KernelRuntimeArgNames, kernel.runtime_arg_schema.runtime_arg_names, hash_string);
    hash_list(hash, List::KernelCommonRuntimeArgNames, kernel.runtime_arg_schema.common_runtime_arg_names, hash_string);
    hash_combine(hash, hash_hw_config(kernel.hw_config));
    hash_combine(hash, hash_kernel_advanced_options(kernel.advanced_options));
    return hash;
}

TTNN_M2_ASSERT_FIELD_COUNT(m2::WorkUnitSpec, 3);
size_t hash_work_unit(const m2::WorkUnitSpec& work_unit) {
    size_t hash = hash_objects_with_default_seed(0);  // work_unit.name: debug only, not keyed
    hash_list(hash, List::WorkUnitKernels, work_unit.kernels, hash_name);
    hash_combine(hash, hash_nodes(work_unit.target_nodes));
    return hash;
}

TTNN_M2_ASSERT_FIELD_COUNT(m2::PrefetcherPipeParameter, 4);
TTNN_M2_ASSERT_FIELD_COUNT(m2::ProgramAdvancedOptions, 1);
size_t hash_program_advanced_options(const m2::ProgramAdvancedOptions& options) {
    size_t hash = hash_objects_with_default_seed(0);
    hash_list(
        hash,
        List::PrefetcherPipeParameters,
        options.prefetcher_pipe_parameters,
        [](const m2::PrefetcherPipeParameter& pipe) {
            size_t pipe_hash = hash_objects_with_default_seed(*pipe.unique_id, pipe.ring_size, pipe.entry_size);
            hash_combine(pipe_hash, hash_nodes(pipe.receivers));
            return pipe_hash;
        });
    return hash;
}

TTNN_M2_ASSERT_FIELD_COUNT(m2::ProgramSpec, 8);
#undef TTNN_M2_ASSERT_FIELD_COUNT

}  // namespace

ttsl::hash::hash_t compute_program_spec_hash(const m2::ProgramSpec& spec) {
    size_t hash = hash_objects_with_default_seed(0);  // spec.name: debug only, not keyed
    hash_list(hash, List::Kernels, spec.kernels, hash_kernel_spec);
    hash_list(hash, List::DataflowBuffers, spec.dataflow_buffers, hash_dataflow_buffer);
    hash_list(hash, List::Semaphores, spec.semaphores, hash_semaphore);
    hash_list(hash, List::Scratchpads, spec.scratchpads, hash_scratchpad);
    hash_list(hash, List::TensorParameters, spec.tensor_parameters, hash_tensor_parameter);
    hash_list(hash, List::WorkUnits, spec.work_units, hash_work_unit);
    hash_combine(hash, hash_program_advanced_options(spec.advanced_options));
    return hash;
}

ttsl::hash::hash_t GenericOpDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // The variant index keeps the descriptor and spec keys in disjoint spaces.
    size_t hash = hash_objects_with_default_seed(operation_attributes.program.index());
    if (operation_attributes.is_spec()) {
        const auto& spec_program = operation_attributes.spec_program();
        hash_combine(hash, compute_program_spec_hash(spec_program.spec));
        // std::hash<Table> is order-independent: {a: 0, b: 1} and {b: 1, a: 0} are one binding.
        hash_combine(hash, spec_program.tensor_arg_indices);
        // The bound tensors' specs, as far as each parameter's relaxations make them pertinent: a
        // spec reused with tensors of another shape, dtype or memory layout must miss, since the
        // compiled program bakes those in. An index this cannot resolve is reported by validation.
        const auto& io_tensors = tensor_args.io_tensors;
        for (const auto& param : spec_program.spec.tensor_parameters) {
            const auto it = spec_program.tensor_arg_indices.find(param.unique_id);
            if (it == spec_program.tensor_arg_indices.end() || it->second >= io_tensors.size()) {
                continue;
            }
            hash_combine(
                hash, m2::hash_tensorspec_with_relaxation(io_tensors[it->second].tensor_spec(), param.relaxations));
        }
        return hash;
    }
    for (const auto& [mesh_coord_range, program_descriptor] :
         operation_attributes.mesh_program_descriptor().mesh_programs) {
        hash_combine(hash, mesh_coord_range);
        hash_combine(hash, compute_program_descriptor_hash(program_descriptor));
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
